"""MAT improvement by replaying its own highest-scoring episodes."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .environment import Kitchen, KitchenBatch, find_scene
from .evaluate import score_policy, summarize, write_evaluation
from .model import MultiAgentTransformer
from .train import evaluation_rank, load_checkpoint, policy_checkpoint, positive, record, save_checkpoint


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    score: float


def keep_rewarded_prefix(elite, spawns, observations, actions, rewards, score_cap):
    rewards = np.asarray(rewards, dtype=np.float32)
    positive_steps = np.flatnonzero(rewards > 0)
    if not len(positive_steps):
        return
    score = float(rewards.sum())
    end = int(positive_steps[-1]) + 1
    previous = elite.get(spawns)
    if previous is None or (score, -end) > (previous.score, -len(previous.actions)):
        returns = np.cumsum(rewards[:end][::-1])[::-1] / score_cap
        elite[spawns] = Trajectory(np.stack(observations[:end]), np.stack(actions[:end]), returns, score)


def collect_elites(model, scene, elite, seed, envs, episodes, probability, device, score_cap):
    rng = np.random.default_rng(seed)
    observations = [[] for _ in range(envs)]
    actions = [[] for _ in range(envs)]
    rewards = [[] for _ in range(envs)]
    completed = []
    steps = 0
    model.eval()
    with KitchenBatch(scene, envs, seed, subprocesses=True) as batch:
        inputs = batch.reset()
        while len(completed) < episodes:
            greedy = rng.random() >= probability
            batch_actions = model.act(torch.as_tensor(inputs, device=device), greedy=greedy, temperature=0.7)[0]
            joint_actions = batch_actions.cpu().numpy()
            transitions = batch.step(joint_actions)
            steps += envs
            for i, transition in enumerate(transitions):
                observations[i].append(inputs[i])
                actions[i].append(joint_actions[i])
                rewards[i].append(transition.reward)
                if not transition.done:
                    continue
                episode = transition.episode
                completed.append(episode)
                keep_rewarded_prefix(elite, tuple(episode["spawns"]), observations[i], actions[i], rewards[i], score_cap)
                observations[i].clear()
                actions[i].clear()
                rewards[i].clear()
            inputs = np.stack([transition.observation for transition in transitions])
    return steps, summarize(completed, score_cap)


def replay_player_permutations(scene, elite, seed, score_cap):
    reference_spawns, reference = max(elite.items(), key=lambda item: (item[1].score, -len(item[1].actions)))
    game = Kitchen(scene, seed)
    seen = set()
    verified = []
    steps = 0
    for _ in range(600):
        observation = game.reset()
        if game.spawns in seen:
            continue
        seen.add(game.spawns)
        actor_order = [reference_spawns.index(position) for position in game.spawns]
        observations, actions, rewards = [], [], []
        for reference_action in reference.actions:
            action = reference_action[actor_order]
            observations.append(observation)
            actions.append(action)
            result = game.step(action, restart=False)
            rewards.append(result.reward)
            steps += 1
            observation = result.observation
            if result.done:
                break
        score = float(sum(rewards))
        verified.append({"seed": game.seed, "spawns": game.spawns, "score": score})
        if score == score_cap:
            elite.pop(game.spawns, None)
        keep_rewarded_prefix(elite, game.spawns, observations, actions, rewards, score_cap)
        if len(seen) == 6:
            break
    return steps, verified


def replay_elites(model, optimizer, elite, epochs, device, focus_errors=False):
    states = torch.as_tensor(np.concatenate([trajectory.observations for trajectory in elite.values()]), device=device)
    actions = torch.as_tensor(np.concatenate([trajectory.actions for trajectory in elite.values()]), device=device)
    returns = torch.as_tensor(
        np.concatenate([trajectory.returns for trajectory in elite.values()]), device=device
    ).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        for indexes in torch.randperm(len(actions), device=device).split(128):
            log_probabilities, values, _ = model(states[indexes], actions[indexes], detach_value_encoder=True)
            target = returns[indexes]
            if focus_errors:
                weights = 1.0 + 4.0 * (log_probabilities.detach().exp() < 0.6)
            else:
                weights = target
            actor_loss = -(weights * log_probabilities).mean()
            value_loss = (values - target).square().mean()
            loss = actor_loss + 0.25 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
    return float(actor_loss.detach()), float(value_loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--env-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=positive, default=8)
    parser.add_argument("--passes", type=positive, default=6)
    parser.add_argument("--episodes-per-pass", type=positive, default=64)
    parser.add_argument("--epochs", type=positive, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--relabel-players", action="store_true", help="verify successful actions under every assignment of chef IDs")
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=positive, default=12)
    args = parser.parse_args()
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning rate must be finite and positive")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    scene = find_scene(args.env_root)
    description = Kitchen(scene, args.seed).describe()
    score_cap = description["score_cap"]
    model = MultiAgentTransformer(description["observation_size"], description["agents"], description["actions"]).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, description)["weights"])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    initial = score_policy(scene, args.eval_seed, args.eval_episodes, device, model=model)
    write_evaluation(args.run_dir / "initial-eval.json", initial)
    record(args.run_dir, {"event": "initial-evaluation", **initial["summary"]})
    baseline = policy_checkpoint(model, description, "self-imitation-start", 0, 0, initial)
    best = baseline
    elite = {}
    steps, initial_sample = collect_elites(model, scene, elite, 1000 + args.seed, args.envs, 24, 0.0, device, score_cap)
    record(args.run_dir, {"event": "greedy-samples", "steps": steps, **initial_sample})
    if args.relabel_players and elite:
        replay_steps, verified = replay_player_permutations(scene, elite, 1200 + args.seed, score_cap)
        steps += replay_steps
        record(args.run_dir, {"event": "player-relabeling", "steps": steps, "verified_rollouts": verified})
    exploration = (0.05, 0.12, 0.25, 0.10, 0.18, 0.08)
    for iteration in range(1, args.passes + 1):
        probability = exploration[(iteration - 1) % len(exploration)]
        all_full = len(elite) == 6 and all(item.score == score_cap for item in elite.values())
        if not all_full:
            collected, metrics = collect_elites(
                model, scene, elite, 2000 + args.seed + 1000 * iteration, args.envs,
                args.episodes_per_pass, probability, device, score_cap,
            )
            steps += collected
            record(args.run_dir, {
                "event": "samples", "iteration": iteration, "steps": steps, "exploration_probability": probability,
                "elite_scores": [{"spawns": key, "score": value.score} for key, value in sorted(elite.items())],
                **metrics,
            })
        if not elite:
            continue
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
        actor_loss, value_loss = replay_elites(model, optimizer, elite, args.epochs, device, focus_errors=all_full)
        result = score_policy(scene, args.eval_seed, args.eval_episodes, device, model=model)
        summary = result["summary"]
        record(args.run_dir, {
            "event": "evaluation", "iteration": iteration, "steps": steps,
            "actor_loss": actor_loss, "value_loss": value_loss, **summary,
        })
        checkpoint = policy_checkpoint(model, description, "self-imitation-mat", iteration, steps, result)
        save_checkpoint(args.run_dir / "latest.pt", checkpoint)
        write_evaluation(args.run_dir / "latest-eval.json", result)
        if evaluation_rank(result) > evaluation_rank(best["evaluation"]):
            best = checkpoint
            save_checkpoint(args.run_dir / "best.pt", best)
            write_evaluation(args.run_dir / "best-eval.json", result)
        else:
            model.load_state_dict(best["weights"])
        if summary["full_score_rate"] == 1 and summary["spawn_assignments"] == 6:
            break


if __name__ == "__main__":
    main()

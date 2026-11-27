"""Train MAT with coordinated demonstrations and online PPO in continuous SharedKitchen."""

import argparse
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np
import torch

from .evaluate_shared import INITIAL_CHECKPOINT, evaluate_batch, prepare_scene, summarize_episodes
from .environment import Kitchen, KitchenBatch
from .evaluate import write_evaluation
from .model import MultiAgentTransformer
from .self_imitation import Trajectory, replay_elites
from .train import (
    FORMAT_VERSION, collect, policy_checkpoint, positive, ppo_update, record, save_checkpoint,
)


def coordinated_actions(controller, env):
    from overcooked_ai_py.cooperative.shared import INTERACT, STAY
    from overcooked_ai_py.cooperative.shared.controller import navigation_action
    from overcooked_ai_py.cooperative.shared.layouts import COOKER_PORTS, STATIONS

    actions = controller.actions(env)
    state = env.state
    cooker, table = state.cooker, state.assembly
    side, server = controller.side, controller.server

    if state.chefs[side].held is None and state.cutting.item is None and table.side is not None:
        actions[side] = navigation_action(env, side, (5, 1), STATIONS["L"], ((4, 1), (4, 2)))

    cooker_needs_both = (
        cooker.item is not None and env.order(cooker.item.order_id).kind == "joint" and table.main is None
    )
    if cooker_needs_both and state.chefs[server].held is None and (table.plate is None or table.plate.kind != "meal"):
        action = navigation_action(env, server, (2, 2), STATIONS["G"])
        actions[server] = STAY if action == INTERACT and cooker.remaining else action

    if cooker_needs_both and cooker.remaining == 0:
        unloaders = []
        for chef_id in env.active:
            chef = state.chefs[chef_id]
            target = tuple(p + d for p, d in zip(chef.position, chef.orientation, strict=True))
            if actions[chef_id] == INTERACT and chef.held is None and chef.position in COOKER_PORTS and target == STATIONS["G"]:
                unloaders.append(chef_id)
        if len(unloaders) == 1:
            actions[unloaders[0]] = STAY

    return actions


def demonstrations(scene, seed):
    game = Kitchen(scene, seed)
    from overcooked_ai_py.cooperative.shared.controller import ScriptedController
    from overcooked_ai_py.cooperative.shared.layouts import SPAWNS

    examples = {}
    metrics = []
    for _ in range(120):
        observation = game.reset()
        if game.spawns in examples:
            continue
        controller = ScriptedController(game.env.core)
        observations, actions, rewards = [], [], []
        while True:
            joint_action = coordinated_actions(controller, game.env.core)
            observations.append(observation)
            actions.append(joint_action)
            result = game.step(joint_action, restart=False)
            observation = result.observation
            rewards.append(result.reward)
            if result.done:
                break
        returns = np.cumsum(np.asarray(rewards[::-1], dtype=np.float32))[::-1].copy()
        score = result.episode["team_score"]
        examples[game.spawns] = Trajectory(np.stack(observations), np.asarray(actions), returns, score)
        metrics.append(result.episode)
        if len(examples) == 6:
            break
    if len(examples) != 6:
        raise RuntimeError(f"Expected six assignments of the chefs to {SPAWNS}.")

    reward_scale = max(entry.score for entry in examples.values())
    if reward_scale <= 0:
        raise RuntimeError("The demonstrations produced no deliveries.")
    for entry in examples.values():
        entry.returns /= reward_scale
    return examples, reward_scale, metrics


def rank(result):
    summary = result["summary"]
    return (
        summary["mean_team_score"],
        summary["min_team_score"],
        summary["mean_joint_attempt_success_rate"],
        -summary["mean_joint_wait_steps"],
    )


def validate_policy(scene, model):
    model.eval()
    episodes = evaluate_batch(scene, model, 100000, 12)
    episodes.sort(key=lambda row: row["seed"])
    return {"summary": summarize_episodes(episodes), "episodes": episodes}


def train_online(args, scene, model, description, commit, initialization):
    if initialization["scene"] != description or initialization["scene_commit"] != commit:
        raise ValueError("Initialize PPO with a checkpoint trained on this continuous kitchen.")
    reward_scale = initialization["training"]["reward_scale"]
    options = SimpleNamespace(
        epochs=2, minibatch=128, temperature=1.0, entropy=0.0001,
        gamma=1.0, gae_lambda=0.95, detach_value_encoder=True,
    )
    device = torch.device("cpu")
    envs, ticks = 6, 150
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    training = {
        "seed": args.seed,
        "envs": envs,
        "rollout_ticks": ticks,
        "reward_scale": reward_scale,
        "initialize": str(args.initialize),
        "learning_rate": args.learning_rate,
        **vars(options),
    }
    best = validate_policy(scene, model)
    write_evaluation(args.run_dir / "best-eval.json", best)
    save_checkpoint(args.run_dir / "best.pt", initialization)
    record(args.run_dir, {"event": "initial-evaluation", **training, **best["summary"]})
    with KitchenBatch(scene, envs, args.seed, subprocesses=False) as batch:
        observation = batch.reset()
        for update in range(1, args.ppo_updates + 1):
            observation, rollout, completed = collect(model, batch, observation, ticks, device, options.temperature)
            losses = ppo_update(model, optimizer, rollout, options, device, reward_scale)
            steps = update * envs * ticks
            record(args.run_dir, {
                "event": "ppo-update",
                "update": update,
                "steps": steps,
                "scoring_transitions": int(np.count_nonzero(rollout[4])),
                "episodes_completed": len(completed),
                "rollout_mean_score": mean(row["team_score"] for row in completed) if completed else None,
                **losses,
            })
            if update % args.eval_every and update != args.ppo_updates:
                continue
            result = validate_policy(scene, model)
            record(args.run_dir, {"event": "evaluation", "update": update, "steps": steps, **result["summary"]})
            checkpoint = policy_checkpoint(model, description, "continuous-mat-ppo", update, steps, result)
            checkpoint["scene_commit"] = commit
            checkpoint["training"] = training
            save_checkpoint(args.run_dir / "latest.pt", checkpoint)
            if rank(result) >= rank(best):
                best = result
                save_checkpoint(args.run_dir / "best.pt", checkpoint)
                write_evaluation(args.run_dir / "best-eval.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, default=INITIAL_CHECKPOINT)
    parser.add_argument("--env-root", type=Path)
    parser.add_argument("--epochs", type=positive, default=160)
    parser.add_argument("--ppo-updates", type=positive, help="Run online PPO from a continuous-kitchen checkpoint")
    parser.add_argument("--eval-every", type=positive, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning rate must be finite and positive")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    scene, game, manifest = prepare_scene(args.env_root, args.seed)
    commit = manifest["commit"]
    description = game.describe()
    dimensions = {key: description[key] for key in ("observation_size", "agents", "actions")}
    model = MultiAgentTransformer(**dimensions)
    initialization = torch.load(args.initialize, map_location="cpu", weights_only=True)
    if initialization["version"] != FORMAT_VERSION or any(initialization["scene"][key] != value for key, value in dimensions.items()):
        raise ValueError("The initialization checkpoint and environment use different policy interfaces.")
    model.load_state_dict(initialization["weights"])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if args.ppo_updates:
        train_online(args, scene, model, description, commit, initialization)
        return
    examples, reward_scale, teacher_episodes = demonstrations(scene, args.seed)
    teacher = {"summary": summarize_episodes(teacher_episodes), "episodes": teacher_episodes}
    write_evaluation(args.run_dir / "demonstrations.json", teacher)
    steps = sum(len(entry.actions) for entry in examples.values())
    record(args.run_dir, {
        "event": "demonstrations",
        "environment": commit,
        "initialize": str(args.initialize),
        "learning_rate": args.learning_rate,
        "steps": steps,
        "reward_scale": reward_scale,
        **teacher["summary"],
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    best = None
    for start in range(0, args.epochs, args.eval_every):
        trained_now = min(args.eval_every, args.epochs - start)
        actor_loss, value_loss = replay_elites(
            model, optimizer, examples, trained_now, torch.device("cpu"), focus_errors=True
        )
        epoch = start + trained_now
        result = validate_policy(scene, model)
        record(args.run_dir, {
            "event": "evaluation",
            "epoch": epoch,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            **result["summary"],
        })
        checkpoint = policy_checkpoint(model, description, "continuous-mat-imitation", epoch, steps, result)
        checkpoint["scene_commit"] = commit
        checkpoint["training"] = {"reward_scale": reward_scale, "initialization": str(args.initialize)}
        save_checkpoint(args.run_dir / "latest.pt", checkpoint)
        if best is None or rank(result) > rank(best):
            best = result
            save_checkpoint(args.run_dir / "best.pt", checkpoint)
            write_evaluation(args.run_dir / "best-eval.json", result)
        if result["summary"]["spawn_assignments"] == 6 and rank(result) >= rank(teacher):
            break


if __name__ == "__main__":
    main()

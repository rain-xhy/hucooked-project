"""Train MAT online with autoregressive actions and a clipped PPO objective."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .environment import Kitchen, KitchenBatch, SCENE_COMMIT, find_scene
from .evaluate import demonstrations, score_policy, summarize, write_evaluation
from .model import MultiAgentTransformer


FORMAT_VERSION = 1


def positive(raw):
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def options():
    parser = argparse.ArgumentParser(description="MAT on the original three-chef SharedKitchen")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-root", type=Path)
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    common.add_argument("--torch-threads", type=positive, default=2)
    common.add_argument("--seed", type=int, default=0)
    train = sub.add_parser("train", parents=[common])
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--envs", type=positive, default=4)
    train.add_argument("--parallel", action="store_true", help="run each game in a separate CPU process")
    train.add_argument("--steps", type=positive, default=256, help="rollout ticks per game per update")
    train.add_argument("--updates", type=positive, default=200, help="target update count, including resumed updates")
    train.add_argument("--epochs", type=positive, default=4)
    train.add_argument("--minibatch", type=positive, default=256)
    train.add_argument("--learning-rate", type=float, default=0.0005)
    train.add_argument("--entropy", type=float, default=0.01)
    train.add_argument("--temperature", type=float, default=1.0, help="logit temperature for sampling and PPO")
    train.add_argument("--detach-value-encoder", action="store_true", help="update the value head without its gradient reaching the shared encoder")
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--checkpoint-every", type=positive, default=10)
    train.add_argument("--eval-every", type=positive, default=10)
    train.add_argument("--eval-episodes", type=positive, default=12)
    train.add_argument("--eval-seed", type=int, default=100000)
    start = train.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path, help="restore optimizer, random state and unfinished games")
    start.add_argument("--initialize", type=Path, help="initialize weights, then begin a fresh online run")

    warm = sub.add_parser("warm-start", parents=[common], help="optional supervised initialization")
    warm.add_argument("--run-dir", type=Path, required=True)
    warm.add_argument("--epochs", type=positive, default=160)
    warm.add_argument("--minibatch", type=positive, default=128)
    warm.add_argument("--learning-rate", type=float, default=0.0005)
    warm.add_argument("--eval-every", type=positive, default=10)
    warm.add_argument("--eval-episodes", type=positive, default=12)
    warm.add_argument("--eval-seed", type=int, default=100000)
    warm.add_argument("--save-evaluated", action="store_true", help="keep each evaluated pretraining checkpoint")

    evaluate = sub.add_parser("evaluate", parents=[common])
    test_policy = evaluate.add_mutually_exclusive_group(required=True)
    test_policy.add_argument("--checkpoint", type=Path)
    test_policy.add_argument("--baseline", choices=("scripted", "random"))
    evaluate.add_argument("--episodes", type=positive, default=24)
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or getattr(args, "eval_seed", 0) < 0:
        parser.error("seeds must be nonnegative")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if args.command == "train" and not (0 < args.gamma <= 1 and 0 <= args.gae_lambda <= 1 and args.entropy >= 0):
        parser.error("expected 0 < gamma <= 1, 0 <= gae-lambda <= 1, and nonnegative entropy")
    if args.command == "train" and not 0 < args.temperature < float("inf"):
        parser.error("temperature must be finite and positive")
    return args


def load_checkpoint(path, scene_description):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if checkpoint["scene"] != scene_description:
        raise ValueError(f"{path} was created for a different kitchen configuration or implementation")
    return checkpoint


def save_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def policy_checkpoint(model, scene_description, stage, update, steps, evaluation=None):
    return {
        "version": FORMAT_VERSION,
        "scene": scene_description,
        "scene_commit": SCENE_COMMIT,
        "stage": stage,
        "update": update,
        "steps": steps,
        "evaluation": evaluation,
        "weights": {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()},
    }


def record(run_dir, item):
    line = json.dumps(item, allow_nan=False)
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


@torch.no_grad()
def collect(model, batch, observation, ticks, device, temperature):
    obs_buffer = np.empty((ticks, *observation.shape), dtype=np.float32)
    action_buffer = np.empty((ticks, *observation.shape[:2]), dtype=np.int64)
    log_buffer = np.empty(action_buffer.shape, dtype=np.float32)
    value_buffer = np.empty_like(log_buffer)
    rewards = np.empty(action_buffer.shape[:2], dtype=np.float32)
    ends = np.empty(rewards.shape, dtype=np.bool_)
    episodes = []
    model.eval()
    for t in range(ticks):
        obs_buffer[t] = observation
        actions, logs, values = model.act(torch.as_tensor(observation, device=device), temperature=temperature)
        action_buffer[t] = actions.cpu().numpy()
        log_buffer[t] = logs.cpu().numpy()
        value_buffer[t] = values.cpu().numpy()
        transitions = batch.step(action_buffer[t])
        observation = np.stack([result.observation for result in transitions])
        rewards[t] = [result.reward for result in transitions]
        ends[t] = [result.done for result in transitions]
        episodes.extend(result.episode for result in transitions if result.done)
    bootstrap = model.value(torch.as_tensor(observation, device=device)).cpu().numpy()
    rollout = obs_buffer, action_buffer, log_buffer, value_buffer, rewards, ends, bootstrap
    return observation, rollout, episodes


def generalized_advantage(rewards, ends, values, bootstrap, gamma, gae_lambda, score_cap):
    # Both success and the game's time limit end the finite-horizon score.
    advantage = np.empty_like(values)
    previous = np.zeros_like(bootstrap)
    next_values = bootstrap
    for t in range(len(rewards) - 1, -1, -1):
        alive = ~ends[t, :, None]
        delta = rewards[t, :, None] / score_cap + gamma * next_values * alive - values[t]
        previous = delta + gamma * gae_lambda * previous * alive
        advantage[t] = previous
        next_values = values[t]
    return advantage


def ppo_update(model, optimizer, rollout, args, device, score_cap):
    obs, actions, old_logs, values, rewards, ends, bootstrap = rollout
    advantage = generalized_advantage(rewards, ends, values, bootstrap, args.gamma, args.gae_lambda, score_cap)
    value_targets = advantage + values
    advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-6)
    tensors = [
        torch.as_tensor(array.reshape(-1, *array.shape[2:]), device=device)
        for array in (obs, actions, old_logs, values, advantage, value_targets)
    ]
    count = len(tensors[0])
    totals = torch.zeros(4, device=device)
    batches = 0
    model.train()
    for _ in range(args.epochs):
        permutation = torch.randperm(count, device=device)
        for indexes in permutation.split(args.minibatch):
            states, acts, previous_log, previous_value, adv, target = [tensor[indexes] for tensor in tensors]
            logs, predicted, entropy = model(
                states, acts, temperature=args.temperature, detach_value_encoder=args.detach_value_encoder
            )
            log_ratio = logs - previous_log
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            clipped = previous_value + (predicted - previous_value).clamp(-0.2, 0.2)
            value_loss = torch.maximum((predicted - target).square(), (clipped - target).square()).mean()
            mean_entropy = entropy.mean()
            loss = policy_loss + 0.5 * value_loss - args.entropy * mean_entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack((policy_loss, value_loss, mean_entropy, norm)).detach()
            batches += 1
    metrics = (totals / batches).tolist()
    return dict(zip(("policy_loss", "value_loss", "entropy", "gradient_norm"), metrics, strict=True))


def settings(args):
    keys = (
        "seed", "envs", "steps", "epochs", "minibatch", "learning_rate", "entropy", "gamma",
        "gae_lambda", "temperature", "detach_value_encoder",
    )
    return {name: getattr(args, name) for name in keys}


def training_checkpoint(model, description, update, steps, optimizer, batch, args, best, device):
    return {
        **policy_checkpoint(model, description, "online-mat", update, steps),
        "optimizer": optimizer.state_dict(),
        "kitchens": batch.snapshots(),
        "settings": settings(args),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "best": best,
    }


def evaluation_rank(result):
    summary = result["summary"]
    return summary["mean_score"], summary["full_score_rate"], summary["max_score"]


def assess(args, scene, model, description, update, steps, device, stage, best):
    result = score_policy(scene, args.eval_seed, args.eval_episodes, device, model=model)
    record(args.run_dir, {"event": "evaluation", "stage": stage, "update": update, "steps": steps, **result["summary"]})
    write_evaluation(args.run_dir / "latest-eval.json", result)
    previous_rank = evaluation_rank(best["evaluation"]) if best else None
    rank = evaluation_rank(result)
    is_best = previous_rank is None or rank > previous_rank or (stage == "warm-start" and rank == previous_rank)
    keep_candidate = getattr(args, "save_evaluated", False)
    if is_best or keep_candidate:
        candidate = policy_checkpoint(model, description, stage, update, steps, result)
    if keep_candidate:
        save_checkpoint(args.run_dir / f"epoch-{update:03d}.pt", candidate)
    if is_best:
        best = candidate
        save_checkpoint(args.run_dir / "best.pt", best)
        write_evaluation(args.run_dir / "best-eval.json", result)
    return best


def train(args, scene, model, description, device, checkpoint):
    step, initial_update, best = 0, 0, None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.resume:
        if checkpoint.get("stage") != "online-mat" or "optimizer" not in checkpoint:
            raise ValueError("--resume requires latest.pt from an online training run")
        previous_settings = {"temperature": 1.0, "detach_value_encoder": False, **checkpoint["settings"]}
        if previous_settings != settings(args):
            raise ValueError("Resume with the original seed, batch sizes and PPO settings")
        step, initial_update, best = checkpoint["steps"], checkpoint["update"], checkpoint["best"]
        if args.updates <= initial_update:
            raise ValueError("Set --updates to a total greater than the saved update count")
        optimizer.load_state_dict(checkpoint["optimizer"])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if best is not None:
        save_checkpoint(args.run_dir / "best.pt", best)
        write_evaluation(args.run_dir / "best-eval.json", best["evaluation"])
    record(args.run_dir, {
        "event": "start", "device": str(device), "scene": description, "settings": settings(args),
        "resume": str(args.resume) if args.resume else None,
        "initialize": str(args.initialize) if args.initialize else None,
        "update": initial_update, "steps": step,
    })
    with KitchenBatch(scene, args.envs, args.seed, args.parallel) as batch:
        if args.resume:
            observation = batch.restore(checkpoint["kitchens"])
            torch.set_rng_state(checkpoint["cpu_rng"])
            if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        else:
            observation = batch.reset()
            if args.initialize:
                initial = score_policy(scene, args.eval_seed, args.eval_episodes, device, model=model)
                record(args.run_dir, {"event": "initial-evaluation", "steps": 0, **initial["summary"]})
                write_evaluation(args.run_dir / "initial-eval.json", initial)
        for update in range(initial_update + 1, args.updates + 1):
            start = time.perf_counter()
            observation, rollout, episodes = collect(model, batch, observation, args.steps, device, args.temperature)
            sampling_seconds = time.perf_counter() - start
            metrics = ppo_update(model, optimizer, rollout, args, device, description["score_cap"])
            collected = args.steps * args.envs
            step += collected
            record(args.run_dir, {
                "event": "update", "update": update, "steps": step, "sampling_seconds": sampling_seconds,
                "steps_per_second": collected / (time.perf_counter() - start),
                "scoring_transitions": int(np.count_nonzero(rollout[4])),
                **summarize(episodes, description["score_cap"]), **metrics,
            })
            if update % args.eval_every == 0 or update == args.updates:
                best = assess(args, scene, model, description, update, step, device, "online-mat", best)
            if update % args.checkpoint_every == 0 or update % args.eval_every == 0 or update == args.updates:
                state = training_checkpoint(model, description, update, step, optimizer, batch, args, best, device)
                save_checkpoint(args.run_dir / "latest.pt", state)


def warm_start(args, scene, model, description, device):
    observations, actions, returns = demonstrations(scene, args.seed)
    inputs = torch.as_tensor(observations, device=device)
    targets = torch.as_tensor(actions, device=device)
    value_targets = torch.as_tensor(returns, device=device).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    record(args.run_dir, {"event": "demonstrations", "transitions": len(inputs), "spawn_assignments": 6})
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indexes in torch.randperm(len(inputs), device=device).split(args.minibatch):
            log_probs, values, _ = model(inputs[indexes], targets[indexes])
            imitation_loss = -log_probs.mean()
            value_loss = (values - value_targets[indexes]).square().mean()
            loss = imitation_loss + 0.5 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(args.run_dir, {
                "event": "warm-start-epoch", "epoch": epoch,
                "imitation_loss": float(imitation_loss.detach()), "value_loss": float(value_loss.detach()),
            })
            best = assess(args, scene, model, description, epoch, 0, device, "warm-start", best)


def main():
    args = options()
    scene = find_scene(args.env_root)
    description = Kitchen(scene, args.seed).describe()
    if description["score_cap"] is None:
        raise ValueError("For continuous SharedKitchen, use reproduction.mat.train_continuous or evaluate_shared.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch environment exposes no CUDA device.")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    model = None
    checkpoint = None
    source = getattr(args, "checkpoint", None) or getattr(args, "resume", None) or getattr(args, "initialize", None)
    if args.command != "evaluate" or source:
        model = MultiAgentTransformer(
            description["observation_size"], description["agents"], description["actions"]
        ).to(device)
        if source:
            checkpoint = load_checkpoint(source, description)
            model.load_state_dict(checkpoint["weights"])
    if args.command == "train":
        train(args, scene, model, description, device, checkpoint)
    elif args.command == "warm-start":
        warm_start(args, scene, model, description, device)
    else:
        result = score_policy(scene, args.seed, args.episodes, device, model=model, scripted=args.baseline == "scripted")
        if args.output:
            write_evaluation(args.output, result)
        print(json.dumps(result["summary"], indent=2), flush=True)

"""Evaluate a saved MAT policy on the current continuous-order SharedKitchen."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from .environment import Kitchen, find_scene
from .evaluate import write_evaluation
from .model import MultiAgentTransformer
from .train import FORMAT_VERSION, positive


MAT_DIRECTORY = Path(__file__).resolve().parent
INITIAL_CHECKPOINT = MAT_DIRECTORY / "checkpoints" / "fixed-8.pt"
DEFAULT_CHECKPOINT = MAT_DIRECTORY / "checkpoints" / "shared-ppo-best.pt"
METRICS = ("team_score", "meals_completed", "joint_attempt_success_rate", "joint_wait_steps")


def prepare_scene(env_root, seed):
    scene = find_scene(env_root)
    manifest = json.loads((MAT_DIRECTORY / "environment-manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["scene_git_blobs"].items():
        data = (scene / path).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if actual != expected and b"\0" not in data:
            data = data.replace(b"\r\n", b"\n")
            actual = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if actual != expected:
            raise ValueError(f"Environment file differs from commit {manifest['commit']}: {path}")
    game = Kitchen(scene, seed)
    if not game.env.core.config.continuous_orders:
        raise ValueError("Expected the environment's default continuous-order configuration.")
    return scene, game, manifest


@torch.inference_mode()
def evaluate_batch(scene, model, seed, count):
    games = [Kitchen(scene, seed + index) for index in range(count)]
    observations = [game.reset() for game in games]
    delivery_ticks = [[] for _ in games]
    episodes = []
    active = list(range(count))
    while active:
        batch = torch.from_numpy(np.stack([observations[index] for index in active]))
        actions = model.act(batch, greedy=True)[0].numpy()
        remaining = []
        for index, action in zip(active, actions, strict=True):
            game = games[index]
            transition = game.step(action, restart=False)
            observations[index] = transition.observation
            if transition.reward > 0:
                delivery_ticks[index].append(game.env.core.state.tick)
            if transition.done:
                episodes.append({**transition.episode, "delivery_ticks": delivery_ticks[index]})
            else:
                remaining.append(index)
        active = remaining
    return episodes


def summarize_episodes(episodes):
    scores = Counter(row["team_score"] for row in episodes)
    return {
        "episodes": len(episodes),
        **{f"mean_{field}": mean(row[field] for row in episodes) for field in METRICS},
        "min_team_score": min(scores),
        "max_team_score": max(scores),
        "score_counts": dict(sorted(scores.items())),
        "spawn_assignments": len({tuple(row["spawns"]) for row in episodes}),
    }


def load_policy(game, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported MAT checkpoint format.")
    observation_size = game.env.observation_space(game.agents[0]).shape[0]
    agent_count = len(game.agents)
    action_count = int(game.env.action_space(game.agents[0]).n)
    saved = checkpoint["scene"]
    saved_interface = saved["observation_size"], saved["agents"], saved["actions"]
    if saved_interface != (observation_size, agent_count, action_count):
        raise ValueError("The checkpoint and environment have different observation or action dimensions.")
    model = MultiAgentTransformer(observation_size, agent_count, action_count)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--episodes", type=positive, default=128)
    parser.add_argument("--output", type=Path, default=Path("mat-evaluation.json"))
    parser.add_argument("--env-root", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(2)
    scene, game, manifest = prepare_scene(args.env_root, args.seed)
    config = game.env.core.config

    checkpoint, model = load_policy(game, args.checkpoint)

    episodes = []
    batch_size = 32
    for offset in range(0, args.episodes, batch_size):
        count = min(batch_size, args.episodes - offset)
        episodes.extend(evaluate_batch(scene, model, args.seed + offset, count))
        print(f"Finished {len(episodes)}/{args.episodes}", flush=True)
    episodes.sort(key=lambda row: row["seed"])
    summary = summarize_episodes(episodes)
    result = {
        "environment": {
            "commit": manifest["commit"],
            "fingerprint": game.env.core.fingerprint(),
            "continuous_orders": config.continuous_orders,
            "config": asdict(config),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "training_scene_commit": checkpoint["scene_commit"],
            "stage": checkpoint["stage"],
        },
        "evaluation": {
            "first_seed": args.seed,
            "last_seed": args.seed + len(episodes) - 1,
            "action_selection": "greedy",
            "device": "cpu",
            "torch": torch.__version__,
            "metrics_source": 'terminal info["metrics"]',
        },
        "summary": summary,
        "episodes": episodes,
    }
    write_evaluation(args.output, result)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

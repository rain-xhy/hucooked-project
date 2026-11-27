"""Re-run archived evaluations and check the replay against native SharedKitchen."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import write_evaluation
from .evaluate_shared import DEFAULT_CHECKPOINT, MAT_DIRECTORY, evaluate_batch, load_policy, prepare_scene, summarize_episodes


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Mismatch in {label}")


@torch.inference_mode()
def verify_trace(model, kitchen, trace):
    observations, _ = kitchen.reset(trace["seed"])
    require_equal([list(chef.position) for chef in kitchen.state.chefs], trace["spawns"], "replay spawns")
    require_equal(len(trace["transitions"]), kitchen.config.horizon, "replay length")
    for step_number, step in enumerate(trace["transitions"], start=1):
        actions = model.act(torch.from_numpy(np.stack(observations))[None], greedy=True)[0][0].tolist()
        require_equal(actions, step["actions"], f"replay model actions at step {step_number}")
        observations, reward, _, _, info = kitchen.step(actions)
        require_equal(info["metrics"]["ticks"], step["tick"], f"replay tick {step_number}")
        require_equal(reward, step["reward"], f"replay reward at step {step_number}")
        require_equal(info["events"], step["events"], f"replay events at step {step_number}")
    require_equal(kitchen.terminated, trace["terminated"], "replay termination")
    require_equal(kitchen.truncated, trace["truncated"], "replay time limit")
    require_equal(info["metrics"], trace["metrics"], "replay terminal metrics")
    return info["metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=MAT_DIRECTORY / "results" / "ppo-128.json")
    parser.add_argument("--trace", type=Path, default=MAT_DIRECTORY / "replay" / "trace.json")
    parser.add_argument("--output", type=Path, default=Path("mat-verification.json"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    seed = reference["evaluation"]["first_seed"]
    count = reference["summary"]["episodes"]
    scene, game, manifest = prepare_scene(args.env_root, seed)
    checkpoint, model = load_policy(game, args.checkpoint)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    require_equal(digest, reference["checkpoint"]["sha256"], "evaluation checkpoint SHA-256")
    require_equal(digest, trace["checkpoint_sha256"], "replay checkpoint SHA-256")
    require_equal(manifest["commit"], reference["environment"]["commit"], "evaluation environment commit")
    require_equal(manifest["commit"], trace["scene_commit"], "replay environment commit")
    require_equal(game.env.core.fingerprint(), trace["scene_fingerprint"], "replay environment fingerprint")

    episodes = []
    for offset in range(0, count, 32):
        episodes.extend(evaluate_batch(scene, model, seed + offset, min(32, count - offset)))
        print(f"Evaluated {len(episodes)}/{count}", flush=True)
    episodes.sort(key=lambda row: row["seed"])
    summary = summarize_episodes(episodes)
    require_equal(json.loads(json.dumps(episodes)), reference["episodes"], "raw metrics for every episode")
    require_equal(json.loads(json.dumps(summary)), reference["summary"], "evaluation summary")
    replay_metrics = verify_trace(model, game.env.core, trace)
    report = {
        "environment_commit": manifest["commit"],
        "environment_files_verified": len(manifest["scene_git_blobs"]),
        "checkpoint_sha256": digest,
        "checkpoint_update": checkpoint["update"],
        "torch": torch.__version__,
        "seed_range": [seed, seed + count - 1],
        "evaluation_summary": summary,
        "episodes_matching_archive": len(episodes),
        "replay_seed": trace["seed"],
        "replay_steps_verified": len(trace["transitions"]),
        "replay_terminal_metrics": replay_metrics,
    }
    write_evaluation(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

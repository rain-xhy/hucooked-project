"""Record the native SharedKitchen rendering of a saved MAT policy."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .evaluate_shared import DEFAULT_CHECKPOINT, load_policy, prepare_scene
from .evaluate import write_evaluation
from .train import positive


def save_animation(frames, output, fps):
    sample_frames = frames[:: max(1, len(frames) // 12)]
    width, height = frames[0].size
    sample = Image.new("RGB", (width, height * len(sample_frames)))
    for index, frame in enumerate(sample_frames):
        sample.paste(frame, (0, height * index))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    duration = max(10, round(100 / fps) * 10)
    indexed[0].save(
        output, append_images=indexed[1:], save_all=True,
        duration=[duration] * (len(indexed) - 1) + [1400], loop=0, optimize=True,
    )


@torch.inference_mode()
def run_replay(model, kitchen, seed):
    observations, _ = kitchen.reset(seed)
    spawns = [chef.position for chef in kitchen.state.chefs]
    frames = [Image.fromarray(kitchen.render())]
    transitions = []
    while not (kitchen.terminated or kitchen.truncated):
        inputs = torch.from_numpy(np.stack(observations))[None]
        actions = model.act(inputs, greedy=True)[0][0].tolist()
        observations, reward, _, _, info = kitchen.step(actions)
        transitions.append({
            "tick": info["metrics"]["ticks"],
            "actions": actions,
            "reward": float(reward),
            "events": info["events"],
        })
        frames.append(Image.fromarray(kitchen.render()))
    return frames, spawns, transitions, info["metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("mat-replay"))
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--fps", type=positive, default=12)
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    torch.set_num_threads(2)
    _, game, manifest = prepare_scene(args.env_root, args.seed)
    checkpoint, model = load_policy(game, args.checkpoint)
    kitchen = game.env.core
    if (
        checkpoint["scene_commit"] != manifest["commit"]
        or checkpoint["scene"]["fingerprint"] != kitchen.fingerprint()
        or not kitchen.config.continuous_orders
    ):
        raise ValueError("The policy and replay environment must use the same continuous kitchen version.")
    frames, spawns, transitions, metrics = run_replay(model, kitchen, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    save_animation(frames, args.output / "episode.gif", args.fps)
    frames[-1].save(args.output / "final.png")
    write_evaluation(args.output / "trace.json", {
        "scene_commit": manifest["commit"],
        "scene_fingerprint": kitchen.fingerprint(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_stage": checkpoint["stage"],
        "seed": args.seed,
        "spawns": spawns,
        "action_names": ["north", "south", "east", "west", "stay", "interact"],
        "transitions": transitions,
        "metrics": metrics,
        "terminated": kitchen.terminated,
        "truncated": kitchen.truncated,
    })
    print(json.dumps({"transitions": len(transitions), "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()

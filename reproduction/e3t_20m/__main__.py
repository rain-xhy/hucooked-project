"""Run the pure E3T smoke test, staged experiment, or independent evaluation."""

import argparse
import json
from pathlib import Path

import torch
from filelock import FileLock, Timeout

from . import shared_large
from .shared_large import MODES, LargeConfig, LargeTraining, math_mode
from .shared_train import ENGINE, load, save_json

STAGES = (32_768, 1_572_864, 3_145_728, 4_718_592, 6_291_456)
DEFAULT_OUT = Path("reproduction/e3t_20m/runs/seed42")


def smoke(device):
    """One full on-policy update using the production model and kitchen."""
    config = LargeConfig(epochs=1, minibatch=5)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    trainer = LargeTraining(config, device, envs=1, workers=0)
    try:
        counts = {
            name: sum(p.numel() for p in network.parameters() if p.requires_grad)
            for name, network in (
                ("policy", trainer.policy),
                ("predictor", trainer.predictor),
            )
        }
        if sum(counts.values()) != 20_000_007:
            raise AssertionError(f"Unexpected trainable parameter count: {counts}")
        with math_mode(trainer.device, fast=True):
            batch, _ = trainer.collect(10)
            stats = trainer.optimize(batch)
        for name in counts:
            if stats[f"{name}_grad_tensors"] != stats[f"{name}_parameter_tensors"]:
                raise AssertionError(f"Missing nonzero gradient in {name}")
        print(json.dumps(dict(env_steps=trainer.steps, parameters=counts, **stats), indent=2))
    finally:
        trainer.pool.close()


def selected_mode(checkpoint):
    path = checkpoint.parent / "selection.json"
    if checkpoint.name not in {"best.pt", "best-resume.pt"} or not path.is_file():
        raise ValueError("Supply --mode or use best.pt alongside its selection.json")
    selection = json.loads(path.read_text())
    if load(checkpoint)["steps"] != selection["selected_step"] or selection["mode"] not in MODES:
        raise ValueError("Selection and checkpoint disagree")
    return selection["mode"]


def evaluate(
    checkpoint,
    out,
    device,
    workers,
    mode=None,
    episodes=100,
    seed=928401,
    parallel=100,
    log_dir=None,
):
    mode = mode or selected_mode(checkpoint)
    shared_large.main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out),
            "--mode",
            mode,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--parallel",
            str(parallel),
            "--workers",
            str(workers),
            "--device",
            device,
            *(["--log-dir", str(log_dir)] if log_dir is not None else []),
        ]
    )


def train(args):
    """Resume each official stage from the preceding stage's full checkpoint."""
    if args.steps < 128 or args.steps % 64 or args.steps > STAGES[-1]:
        raise ValueError(f"--steps must be a multiple of 64 between 128 and {STAGES[-1]}")
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(folder / ".train.lock", timeout=0):
        manifest = folder / "run.json"
        protocol = {
            "engine": ENGINE,
            "architecture": shared_large.ARCH,
            "seed": 42,
            "steps": args.steps,
        }
        if args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text()) != protocol:
                raise ValueError("Use --resume with the same output directory and step target")
        else:
            if any(p.name != ".train.lock" for p in folder.iterdir()):
                raise FileExistsError("Choose an empty output directory, or continue with --resume")
            save_json(manifest, protocol)
        latest = folder / "latest.pt"
        steps_done = int(load(latest)["steps"]) if latest.is_file() else 0
        targets = [n for n in STAGES if steps_done < n < args.steps] + [args.steps]
        if steps_done > args.steps:
            raise ValueError("Latest checkpoint exceeds the requested step target")
        for target in targets:
            command = [
                "train",
                "--steps",
                str(target),
                "--out",
                str(folder),
                "--device",
                args.device,
                "--workers",
                str(args.workers),
                "--eval-workers",
                str(args.workers),
            ]
            if latest.is_file():
                command.extend(["--resume", str(latest)])
            shared_large.main(command)
        evaluate(
            folder / "best.pt", folder / "holdout.json", args.device, args.workers, log_dir=folder
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke", help="Verify all gradients in a ten-step update")
    train_parser = sub.add_parser("train", help="Train, select on development games and evaluate")
    eval_parser = sub.add_parser("evaluate", help="Evaluate a frozen policy on normal starts")
    for p in (smoke_parser, train_parser, eval_parser):
        p.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cuda" if p is train_parser else "cpu",
        )
    train_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    train_parser.add_argument("--workers", type=int, default=11)
    train_parser.add_argument("--resume", action="store_true", help="Continue the same local run")
    train_parser.add_argument(
        "--steps",
        type=int,
        default=STAGES[-1],
        help="Cumulative target (multiple of 64)",
    )
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument(
        "--mode", choices=MODES, help="Defaults to the saved development selection"
    )
    eval_parser.add_argument("--workers", type=int, default=0)
    eval_parser.add_argument("--parallel", type=int, default=100)
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--seed", type=int, default=928401)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select a GPU runtime or pass --device cpu.")
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be nonnegative")
    if args.command == "smoke":
        smoke(args.device)
    elif args.command == "evaluate":
        evaluate(
            args.checkpoint,
            args.out,
            args.device,
            args.workers,
            args.mode,
            args.episodes,
            args.seed,
            args.parallel,
        )
    else:
        try:
            train(args)
        except Timeout:
            parser.error(f"Another training process is using {args.out}")


if __name__ == "__main__":
    main()

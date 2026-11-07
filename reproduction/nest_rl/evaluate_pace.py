"""Run the pinned upstream Overcooked evaluator on a published PACE checkpoint."""

import argparse
import os
from pathlib import Path
import runpy
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        choices=("ppo_pace_seed1", *(f"ppo_pace_two_ing_{n}_seed1" for n in (1, 2, 3))),
        default="ppo_pace_seed1",
    )
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")

    root = Path(__file__).resolve().parent
    upstream = root / "adaptive-RL"
    overcooked = upstream / "environment" / "overcooked"
    checkpoint = upstream / "logs" / "Overcooked" / args.checkpoint / "ppo" / "latest.pt"
    if not checkpoint.is_file():
        parser.error("initialize the adaptive-RL submodule to fetch the published checkpoint")

    output = root / "checkpoint-eval" / f"{args.checkpoint}-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    local_checkpoint = output / "latest.pt"
    # Upstream writes its result pickle beside the checkpoint.
    shutil.copyfile(checkpoint, local_checkpoint)
    os.chdir(upstream)
    sys.path[:0] = [str(upstream), str(overcooked), str(overcooked / "gym_cooking" / "rebar")]
    os.environ.update(
        WANDB_MODE="disabled",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )

    import evaluation_

    evaluate = evaluation_.evaluate

    def evaluate_metrics(*args, **kwargs):
        # The CLI expects a metrics dict; the current evaluator also returns recipe history.
        metrics, _ = evaluate(*args, **kwargs)
        return metrics

    evaluation_.evaluate = evaluate_metrics
    sys.argv = [
        "online_test_.py",
        "--env-name", "Overcooked",
        "--env_config", "environment/overcooked/config/fc_small_test.yaml",
        "--player-id", "0",
        "--recipe-type", "cross",
        "--train-pool-size", "3",
        "--eval-pool-size", "1",
        "--rule-based-opponents", "3",
        "--policy-dir", str(local_checkpoint),
        "--log-dir", str(output),
        "--history-size", "5",
        "--test-pass", str(args.passes),
        "--test-episodes", "5",
        "--seed", str(args.seed),
        "--no-cuda",
        "--use-dummy-vec-env",
        "--include-current-episode",
        "--self-obs-mode",
        "--separate-history",
        "--merge-encoder-computation",
        "--use-meta-episode",
    ]
    runpy.run_path("online_test_.py", run_name="__main__")


if __name__ == "__main__":
    main()

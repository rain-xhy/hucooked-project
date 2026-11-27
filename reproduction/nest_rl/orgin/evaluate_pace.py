"""Run the pinned upstream Overcooked evaluator on a published PACE checkpoint."""

import argparse
import os
import runpy
import shutil
import sys

from overcooked_setup import CONFIG, RECIPES, ROOT, UPSTREAM, activate_upstream, rule_partner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        choices=("ppo_pace_seed1", *(f"ppo_pace_two_ing_{n}_seed1" for n in (1, 2, 3))),
        default="ppo_pace_seed1",
    )
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recipe", choices=RECIPES, default="lettuce-onion")
    parser.add_argument("--partner-mode", choices=("full", "minimum"), default="full")
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")

    checkpoint = UPSTREAM / "logs" / "Overcooked" / args.checkpoint / "ppo" / "latest.pt"
    if not checkpoint.is_file():
        parser.error("initialize the adaptive-RL submodule to fetch the published checkpoint")

    output = ROOT / "checkpoint-eval" / f"{args.checkpoint}-{args.seed}-{args.partner_mode}-{args.recipe}"
    output.mkdir(parents=True, exist_ok=True)
    local_checkpoint = output / "latest.pt"
    # Upstream writes its result pickle beside the checkpoint.
    shutil.copyfile(checkpoint, local_checkpoint)
    activate_upstream()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    import evaluation_
    from environment.overcooked import policy

    evaluate = evaluation_.evaluate
    original_pool = policy.get_train_eval_pool

    def evaluate_metrics(*args, **kwargs):
        # The CLI expects a metrics dict; the current evaluator also returns recipe history.
        metrics, _ = evaluate(*args, **kwargs)
        return metrics

    def evaluation_pool(upstream_args):
        # Retain upstream pool setup and its RNG draws for paired comparisons.
        train_pool, _ = original_pool(upstream_args)
        print(f"Evaluation partner: {args.partner_mode} / {args.recipe}")
        return train_pool, [rule_partner(args.recipe, args.partner_mode)]

    evaluation_.evaluate = evaluate_metrics
    policy.get_train_eval_pool = evaluation_pool
    sys.argv = [
        "online_test_.py",
        "--env-name", "Overcooked",
        "--env_config", CONFIG,
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

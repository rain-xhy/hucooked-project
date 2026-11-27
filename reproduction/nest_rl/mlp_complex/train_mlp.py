"""Train a two-role NestRL policy on randomized soup_coordination orders."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def main():
    invocation_dir = Path.cwd()
    sys.path.insert(0, str(ROOT.parent / "mlp_standard"))
    from standard_setup import activate

    activate()
    from complex_kitchen import ComplexVecEnv, ROLES
    from train_nestrl import main as train

    sys.argv[1:1] = [
        "l1", "--player", "1", "--device", "cuda", "--env-mode", "gpu", "--history-storage", "gpu",
        "--processes", "64", "--rollout-steps", "600", "--ppo-mini-batch-size", "960",
        "--updates", "25", "--save-every-updates", "5", "--eval-every-updates", "5",
        "--trackio-project", "mlp-complex-l1-soup-coordination", "--trackio-sample-every", "40", "--profile",
    ]
    train(output_root=ROOT, recipes=ROLES, gpu_vec_class=ComplexVecEnv, invocation_dir=invocation_dir,
          trackio_dir=ROOT.parent / "mlp_standard" / "runs" / "trackio")


if __name__ == "__main__":
    main()

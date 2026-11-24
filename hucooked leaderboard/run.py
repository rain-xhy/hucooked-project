"""Train and evaluate the selected HA² leaderboard experiment."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from results import finish


LEADERBOARD = Path(__file__).resolve().parent
PROJECT = LEADERBOARD.parent
REVISIONS = {
    "original": "2cf0bd18d2523deb964b40ef99ef6bada19c8923",
    "v1": "671c17bf4ba48ecb11e056dbf1e077b0bd434b2d",
    "v2": "10aba7415fb0ce8340b2cc99025ca1f0627a92e0",
    "v3": "2d0ef3716706206eaecdc61296beccbdd23c31e5",
}
MODES = {"multi-agent": "multi", "single-agent": "single"}
DEFAULT_UPDATES = {"original": 24, "v1": 12, "v2": 12, "v3": 24}


def runtime(version):
    source = PROJECT / ".environment" / version
    marker = source / "revision"
    if not marker.is_file() or marker.read_text().strip() != REVISIONS[version]:
        raise SystemExit(f"Prepare {version} environment {REVISIONS[version]} at {source}")
    if not (source / "src/overcooked_ai_py/__init__.py").is_file():
        raise SystemExit(f"Missing environment source under {source / 'src'}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source / "src"), str(PROJECT / "hucooked haha"), str(LEADERBOARD)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stages(args, mode):
    updates = args.updates or (12 if args.version == "v3" and mode == "single-agent"
                               else DEFAULT_UPDATES[args.version])
    options = ["--device", args.device, "--seed", str(args.seed)]
    examples = ["--test-episodes", str(args.test_episodes)]
    train = ["--updates", str(updates)]
    if args.version in ("original", "v1", "v2"):
        version_options = {
            "original": [],
            "v1": ["--route-progress"],
            "v2": ["--early", "3", "--pipeline", "--synchronize", "--goal-lock"],
        }
        yield "train", f"{args.version}.train", [
            *options, *train, *examples, "--test-seed-base", str(args.test_seed),
            "--mode", MODES[mode], *version_options[args.version],
        ]
    elif mode == "single-agent":
        yield "train", "v3.single", [
            *options, *train, *examples, "--test-seed", str(args.test_seed),
            "--navigation", "shortest", "--lanes", "112",
        ]
    else:
        shared = ["--deadline-aware", "--prestage", "--lanes", "128"]
        yield "worker-pretrain", "v3.team", [
            *options, *shared, "--updates", "0", "--test-episodes", "8",
        ]
        yield "train", "v3.team", [
            *options, *shared, *train, "--test-episodes", "8",
            "--initialize", "worker-pretrain/best.pt", "--reset-manager",
        ]
        yield "ranking", "v3.team", [
            *options, *shared, *examples, "--test-seed", str(args.test_seed),
            "--mode", "evaluate", "--initialize", "train/best.pt", "--protect-supply",
        ]


def run(args, mode, env):
    directory = LEADERBOARD / args.version / mode / "runs" / args.run_id
    directory.mkdir(parents=True)
    commands = [
        [sys.executable, "-m", module, "--output", name, *flags]
        for name, module, flags in stages(args, mode)
    ]
    metadata = {
        "version": args.version, "revision": REVISIONS[args.version], "mode": mode,
        "run_id": args.run_id, "train_seed": args.seed, "test_seed": args.test_seed,
        "test_episodes_per_map_and_seat": args.test_episodes, "device": args.device,
        "commands": commands,
    }
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Running {args.version}/{mode} in {directory}", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        for command in commands:
            subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finish(directory, metadata, env)
    print(f"Completed {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=REVISIONS)
    parser.add_argument("--mode", choices=("both", *MODES), default="both")
    parser.add_argument("--run-id", default=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--seed", type=int, default=488)
    parser.add_argument("--test-seed", type=int, default=84031)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finalize", action="store_true", help="Rebuild artifacts from an existing, completed evaluation")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_id):
        parser.error("Run ID must be a single directory name")
    if args.test_episodes is None:
        args.test_episodes = 80 if args.version == "v3" else 24
    if min(args.test_seed, args.seed) < 0 or args.test_episodes < 1 or (args.updates is not None and args.updates < 1):
        parser.error("Use nonnegative seeds and positive episode and update counts")
    env = runtime(args.version)
    for mode in MODES if args.mode == "both" else (args.mode,):
        if args.finalize:
            directory = LEADERBOARD / args.version / mode / "runs" / args.run_id
            metadata = json.loads((directory / "config.json").read_text())
            finish(directory, metadata, env)
        else:
            run(args, mode, env)


if __name__ == "__main__":
    main()

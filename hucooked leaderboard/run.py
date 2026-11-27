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
    "shared": "de37e0818c2e76311e105ab8689e5162bd84133f",
}
MODES = {"multi-agent": "multi", "single-agent": "single"}
DEFAULT_UPDATES = {"original": 24, "v1": 12, "v2": 12, "v3": 24, "shared": 12}


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
    if args.pure_rl:
        resume = ["--resume", f"../{args.pure_rl_resume}/train"] if args.pure_rl_resume else []
        if args.pure_rl_resume_last:
            resume.append("--resume-last")
        if args.pure_rl_throughput:
            resume.append("--throughput")
        if args.pure_rl_manager_only:
            resume.append("--manager-only")
        if args.pure_rl_manager_width is not None:
            resume.extend(("--manager-width", str(args.pure_rl_manager_width)))
        if args.pure_rl_side_worker:
            resume.append("--side-worker")
        if args.pure_rl_prep_shaping:
            resume.append("--prep-shaping")
        if args.pure_rl_handoff_curriculum:
            resume.append("--handoff-curriculum")
        if args.pure_rl_fixed_budget:
            resume.append("--fixed-budget")
        if args.pure_rl_search:
            resume.extend(("--search-trials", str(args.pure_rl_search)))
        if args.pure_rl_focused_search:
            resume.append("--focused-search")
        yield "train", "v2.pure_rl", [
            *options, *examples, "--test-seed-base", str(args.test_seed),
            *(train if args.updates is not None else []), *resume,
        ]
    elif args.version in ("original", "v1", "v2", "shared"):
        shared_policy = ["--early", "3", "--pipeline", "--goal-lock"]
        if not args.no_sync_mask:
            shared_policy.insert(-1, "--synchronize")
        version_options = {
            "original": [],
            "v1": ["--route-progress"],
            "v2": shared_policy,
            "shared": [*shared_policy, "--profile", "continuous"],
        }
        module = "v2.train" if args.version == "shared" else f"{args.version}.train"
        yield "train", module, [
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
        ["python", "-m", module, "--output", name, *flags]
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
            subprocess.run([sys.executable, *command[1:]], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
    parser.add_argument("--no-sync-mask", action="store_true",
                        help="Allow one-sided unloading attempts in Shared training and evaluation")
    parser.add_argument("--pure-rl", action="store_true",
                        help="Train Shared Worker and Manager from rewards without demonstrations")
    parser.add_argument("--pure-rl-resume", metavar="RUN_ID",
                        help="Jointly fine-tune Manager and Worker from a previous pure RL run")
    parser.add_argument("--pure-rl-resume-last", action="store_true", help="Resume the final pure RL checkpoint")
    parser.add_argument("--pure-rl-throughput", action="store_true", help="Train against production delays")
    parser.add_argument("--pure-rl-manager-only", action="store_true", help="Freeze the Worker during resumed PPO")
    parser.add_argument("--pure-rl-manager-width", type=int, help="Widen the Manager in a resumed capacity comparison")
    parser.add_argument("--pure-rl-side-worker", action="store_true", help="Fine-tune the side role's Worker alone")
    parser.add_argument("--pure-rl-prep-shaping", action="store_true", help="Give Manager PPO feedback on side preparation")
    parser.add_argument("--pure-rl-handoff-curriculum", action="store_true", help="Mix in native handoff states during PPO")
    parser.add_argument("--pure-rl-fixed-budget", action="store_true", help="Run the full resumed PPO budget")
    parser.add_argument("--pure-rl-search", type=int, default=0, help="Number of actor parameter perturbations to try")
    parser.add_argument("--pure-rl-focused-search", action="store_true", help="Focus parameter search on production delays")
    parser.add_argument("--finalize", action="store_true", help="Rebuild artifacts from an existing, completed evaluation")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_id):
        parser.error("Run ID must be a single directory name")
    if args.no_sync_mask and args.version != "shared":
        parser.error("--no-sync-mask applies to the Shared leaderboard")
    if args.pure_rl and (args.version != "shared" or args.mode != "multi-agent"):
        parser.error("--pure-rl requires shared --mode multi-agent")
    if args.pure_rl_resume and (
        not args.pure_rl or args.pure_rl_resume == args.run_id
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.pure_rl_resume)
    ):
        parser.error("--pure-rl-resume requires --pure-rl and a different, valid run ID")
    if (args.pure_rl_resume_last or args.pure_rl_throughput or args.pure_rl_manager_only
        or args.pure_rl_search or args.pure_rl_fixed_budget or args.pure_rl_manager_width is not None
        or args.pure_rl_side_worker or args.pure_rl_prep_shaping or args.pure_rl_handoff_curriculum
    ) and not args.pure_rl_resume:
        parser.error("Pure RL fine-tuning options require --pure-rl-resume")
    if args.pure_rl_manager_width is not None and (not args.pure_rl_manager_only or args.pure_rl_manager_width < 1):
        parser.error("--pure-rl-manager-width requires --pure-rl-manager-only and a positive width")
    if args.pure_rl_search < 0:
        parser.error("--pure-rl-search must be nonnegative")
    if args.pure_rl_focused_search and not args.pure_rl_search:
        parser.error("--pure-rl-focused-search requires --pure-rl-search")
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

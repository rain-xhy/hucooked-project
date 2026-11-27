"""Compare the two leaderboard configurations on their paired evaluation seeds."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random
from statistics import mean, quantiles

from results import episodes, score
from run import LEADERBOARD, MODES, REVISIONS


def values(directory):
    final = directory / ("ranking" if (directory / "ranking").is_dir() else "train")
    files = tuple(final.glob("*evaluation.json.gz"))
    path = next((p for p in files if p.name == "best-evaluation.json.gz"), final / "evaluation.json.gz")
    with gzip.open(path, "rt") as handle:
        rows = list(episodes(json.load(handle)))
    data = defaultdict(lambda: defaultdict(list))
    for layout, _, row in rows:
        data[layout][row["seed"]].append(score(row))
    return {layout: {seed: mean(scores) for seed, scores in group.items()} for layout, group in data.items()}


def paired_difference(multi, single):
    result = {}
    for layout, reference in multi.items():
        other = single[layout]
        if reference.keys() != other.keys():
            raise ValueError(f"Mismatched evaluation seeds for {layout}")
        differences = [other[seed] - reference[seed] for seed in sorted(reference)]
        rng = random.Random(0)
        replicates = [mean(rng.choices(differences, k=len(differences))) for _ in range(4000)]
        cutoffs = quantiles(replicates, n=40, method="inclusive")
        result[layout] = {
            "paired_seeds": len(differences),
            "single_minus_multi": mean(differences),
            "bootstrap_95_percentile_interval": [cutoffs[0], cutoffs[-1]],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--version", choices=REVISIONS, action="append")
    args = parser.parse_args()
    if args.run_id != Path(args.run_id).name or args.run_id in (".", ".."):
        parser.error("Run ID must be a single directory name")
    versions = args.version or [
        version for version in REVISIONS
        if any((LEADERBOARD / version / mode / "runs" / args.run_id).is_dir() for mode in MODES)
    ]
    if not versions:
        parser.error(f"No runs found for {args.run_id}")
    report = {}
    for version in versions:
        directories = {mode: LEADERBOARD / version / mode / "runs" / args.run_id for mode in MODES}
        data = {mode: values(directory) for mode, directory in directories.items()}
        report[version] = {
            "revision": REVISIONS[version],
            "modes": {
                mode: {"directory": str(directory.relative_to(LEADERBOARD)),
                       "scores": json.loads((directory / "scores.json").read_text())}
                for mode, directory in directories.items()
            },
            "paired_comparison": paired_difference(data["multi-agent"], data["single-agent"]),
        }
    result = LEADERBOARD / "comparisons" / f"{args.run_id}.json"
    result.parent.mkdir(exist_ok=True)
    result.write_text(json.dumps(report, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()

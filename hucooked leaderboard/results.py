"""Collect scores, training series, and replay files in each run directory."""

import csv
import gzip
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys


def episodes(payload, layout="shared", seat=None):
    if isinstance(payload, list):
        for row in payload:
            yield row.get("layout", layout), row.get("learning_role", row.get("learned_role", seat)), row
    else:
        for key, rows in payload.items():
            if isinstance(rows, dict):
                yield from episodes(rows, key)
            elif key in ("sync", "counter"):
                yield from episodes(rows, key)
            else:
                yield from episodes(rows, layout, key)


def score(row):
    return row.get("score", row.get("team_score"))


def order_delivery_metrics(rows):
    """Pool released tickets and release-to-service durations across Shared episodes."""
    released = served = total_steps = 0
    for row in rows:
        for order in row["orders"]:
            if order["released_at"] < 0:
                continue
            released += 1
            if order["status"] == "served":
                duration = order["completed_at"] - order["released_at"]
                if duration < 0:
                    raise ValueError(f"Order {order['id']} was served before release")
                served += 1
                total_steps += duration
    return {
        "released_orders": released,
        "served_orders": served,
        "total_delivery_steps": total_steps,
        "order_completion_rate": served / released if released else None,
        "mean_delivery_steps": total_steps / served if served else None,
    }


def summary(rows, *, shared=False):
    values = [score(row) for row in rows]
    ticks = [row["ticks"] for row in rows if "ticks" in row]
    order_counts = {}
    for row in rows:
        counts = {name: events["delivered"] for name, events in row.get("recipes", {}).items()}
        counts.update({key: row[key] for key in ("normal_completed", "joint_completed") if key in row})
        for name, count in counts.items():
            order_counts[name] = order_counts.get(name, 0) + count
    official_metrics = (
        "meals_per_100_steps", "joint_attempt_success_rate", "joint_wait_steps",
        "three_contributor_meals", "movement_conflicts", "resource_conflicts",
    ) if "meals_per_100_steps" in rows[0] else ()
    return {
        "episodes": len(values), "mean_score": mean(values), "std_score": pstdev(values),
        "min_score": min(values), "max_score": max(values), "orders_completed": order_counts,
        **({"mean_meals_completed": mean(row["meals_completed"] for row in rows)}
           if all("meals_completed" in row for row in rows) else {}),
        **({"mean_finish_steps": mean(ticks)} if ticks else {}),
        **{key: mean(row[key] for row in rows) for key in official_metrics},
        **({"order_metrics": order_delivery_metrics(rows)}
           if shared and all("orders" in row for row in rows) else {}),
    }


def means(stats):
    if "overall" in stats:
        return means(stats["overall"])
    for name in ("mean_score", "score", "mean", "mean_return", "team_score"):
        if isinstance(stats.get(name), (int, float)):
            return {None: stats[name]}
    return {name: value for name, entry in stats.items() if isinstance(entry, dict)
            for value in means(entry).values()}


def steps_series(path):
    rows = []
    latest_update = latest_steps = 0
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        if entry.get("stage") not in ("initial", "initialized", "after_imitation", "validation",
                                      "after_recovery", "after_ppo", "ppo", "update", "ppo_update", "search_candidate"):
            continue
        latest_update = entry.get("update", latest_update)
        latest_steps = entry.get("environment_steps", entry.get("env_steps", latest_steps))
        validation = means(entry.get("results", {}))
        if isinstance(entry.get("mean_score"), (int, float)):
            validation = {None: entry["mean_score"]}
        if entry.get("stage") == "search_candidate":
            validation = {None: entry["validation_score"]}
        sampled = means(entry.get("training", {}))
        if entry.get("sampled_mean") is not None:
            sampled = {None: entry["sampled_mean"]}
        training_metric = "training"
        if entry.get("stage") == "search_candidate":
            training_metric, sampled = "screening", {None: entry["screening_score"]}
        for metric, series in ((training_metric, sampled), ("validation", validation)):
            for name, value in series.items():
                if math.isfinite(value):
                    rows.append((latest_update, latest_steps, entry["seconds"], metric, name or "all", value))
    return rows


def chart(path, rows):
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(9, 4.8), layout="constrained")
    axis = figure.subplots()
    grouped = {}
    for step, _, _, metric, map_name, value in rows:
        grouped.setdefault((metric, map_name), []).append((step, value))
    for (metric, map_name), points in sorted(grouped.items()):
        steps, scores = zip(*points)
        axis.plot(steps, scores, marker=".", linewidth=1.4,
                  linestyle="--" if metric in ("training", "screening") else "-",
                  label=f"{metric} / {map_name}")
    xlabel = "Candidate trials" if any(metric == "screening" for metric, _ in grouped) else "PPO updates"
    axis.set(xlabel=xlabel, ylabel="Mean team score", title="Team score during training")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=2, fontsize=8)
    figure.savefig(path, metadata={"Date": None})


def distribution(path, by_map):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.subplots()
    names = list(by_map)
    scores = [[score(row) for rows in by_map[name].values() for row in rows] for name in names]
    axis.boxplot(scores, showmeans=True)
    axis.set_xticks(range(1, len(names) + 1), names)
    axis.set(ylabel="Team score", title="Evaluation scores by map")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, metadata={"Date": None})


def finish(directory, metadata, env):
    selected = directory / ("ranking" if (directory / "ranking").is_dir() else "train")
    source = next((selected / name for name in ("best-evaluation.json", "evaluation.json",
                                                "best-evaluation.json.gz", "evaluation.json.gz")
                   if (selected / name).is_file()), None)
    if source is None:
        raise FileNotFoundError(f"No completed evaluation in {selected}")
    read = gzip.open if source.suffix == ".gz" else open
    with read(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    by_map = {}
    for layout, seat, row in episodes(payload):
        by_map.setdefault(layout, {}).setdefault(seat, []).append(row)
    stats, best = {}, {}
    shared = metadata["version"] == "shared"
    for map_name, groups in by_map.items():
        combined = [r for group in groups.values() for r in group]
        stats[map_name] = summary(combined, shared=shared)
        if "meals_per_100_steps" in stats[map_name]:
            reference = selected / "official-reference.json"
            if reference.is_file():
                stats[map_name]["official_reference"] = summary(json.loads(reference.read_text()), shared=shared)
        if len(groups) > 1:
            stats[map_name]["by_seat"] = {str(seat): summary(rows, shared=shared) for seat, rows in groups.items()}
        seat, row = max(((seat, row) for seat, group in groups.items() for row in group), key=lambda entry: score(entry[1]))
        best[map_name] = {"seat": seat, **row}
    (directory / "scores.json").write_text(json.dumps(stats, indent=2) + "\n")
    distribution(directory / "score-distribution.svg", by_map)
    progress = steps_series(directory / "train/progress.jsonl")
    if progress:
        with (directory / "training-curve.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("update", "environment_steps", "seconds", "metric", "map", "team_score"))
            writer.writerows(progress)
        chart(directory / "training-curve.svg", progress)
    replay = directory / "replay"
    replay.mkdir(exist_ok=True)
    for map_name, row in best.items():
        (replay / f"{map_name}.trajectory.json").write_text(json.dumps(row, separators=(",", ":")) + "\n")
    subprocess.run(
        [sys.executable, "-m", "render_replay", metadata["version"], metadata["revision"], str(replay)],
        cwd=directory, env=env, check=True,
    )
    for path in sorted(directory.rglob("*evaluation.json")):
        packed = path.with_suffix(".json.gz")
        with path.open("rb") as stream, gzip.open(packed, "wb") as compressed:
            while chunk := stream.read(1 << 20):
                compressed.write(chunk)
        path.unlink()

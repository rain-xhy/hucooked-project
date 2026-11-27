#!/usr/bin/env python3
"""Summarize PPO validation records and million-scale plateau evidence.

Run from any directory with the standard library only::

    python src/stq/scripts/summarize_ppo_training.py

By default, inputs are read from the directory containing ``experiments.json``
and outputs are written to its ``results`` subdirectory. If the raw
``metrics.jsonl`` files are absent, extract the training-records archive from
the release into that reproduction root before running this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPRO_ROOT = Path(__file__).resolve().parents[1]
PPO_EXPERIMENT_COUNT = 15
EVALUATION_EVENT = "evaluation"
SUMMARY_FIELDS = (
    "mean_team_score",
    "std_team_score",
    "mean_meals_completed",
    "mean_joint_completed",
    "mean_joint_attempt_success_rate",
    "mean_joint_wait_steps",
)
BASE_COLUMNS = (
    "experiment_id",
    "backbone",
    "model_size",
    "training_mode",
    "reward_profile",
    "result_dir",
    "event_line",
    "env_steps",
    "seed",
    "episodes",
    "reason",
    "time",
    "native_metric_episodes",
    *SUMMARY_FIELDS,
)


class SummaryError(RuntimeError):
    """Raised when the reproducibility inputs are missing or inconsistent."""


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _atomic_text_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _read_experiments(repro_root: Path) -> list[dict[str, Any]]:
    path = repro_root / "experiments.json"
    if not path.is_file():
        raise SummaryError(f"experiment manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"could not read experiment manifest {path}: {exc}") from exc
    experiments = [
        item for item in payload.get("experiments", [])
        if item.get("family") == "ppo"
    ]
    if len(experiments) != PPO_EXPERIMENT_COUNT:
        raise SummaryError(
            f"expected {PPO_EXPERIMENT_COUNT} family='ppo' entries in {path}, "
            f"found {len(experiments)}"
        )
    return experiments


def _event_row(
    experiment: dict[str, Any], event: dict[str, Any], event_line: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env_steps = event.get("env_steps")
    if not isinstance(env_steps, int) or isinstance(env_steps, bool):
        raise SummaryError(
            f"{experiment['experiment_id']}: evaluation at line {event_line} "
            "has no integer env_steps"
        )

    episode_metrics = event.get("episode_metrics")
    if not isinstance(episode_metrics, list):
        episode_metrics = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    for episode in episode_metrics:
        if not isinstance(episode, dict):
            continue
        for key, value in episode.items():
            if _finite_number(value):
                metric_values[key].append(float(value))
    native_means = {key: _mean(values) for key, values in metric_values.items()}

    row: dict[str, Any] = {
        "experiment_id": experiment.get("experiment_id"),
        "backbone": experiment.get("backbone"),
        "model_size": experiment.get("model_size"),
        "training_mode": experiment.get("training_mode"),
        "reward_profile": experiment.get("reward_profile"),
        "result_dir": experiment.get("result_dir"),
        "event_line": event_line,
        "env_steps": env_steps,
        "seed": event.get("seed"),
        "episodes": event.get("episodes", len(episode_metrics) or None),
        "reason": event.get("reason"),
        "time": event.get("time"),
        "native_metric_episodes": len(episode_metrics) or None,
    }
    for field in SUMMARY_FIELDS:
        row[field] = event.get(field)
    # Some older logs only carry per-episode native metrics rather than the
    # evaluation-level means. Derive only from the metrics actually present.
    fallback_map = {
        "mean_team_score": "team_score",
        "std_team_score": None,
        "mean_meals_completed": "meals_completed",
        "mean_joint_completed": "joint_completed",
        "mean_joint_attempt_success_rate": "joint_attempt_success_rate",
        "mean_joint_wait_steps": "joint_wait_steps",
    }
    for summary_name, native_name in fallback_map.items():
        if row[summary_name] is None and native_name is not None:
            row[summary_name] = native_means.get(native_name)
    for key, value in native_means.items():
        row[f"native_{key}_mean"] = value

    return row, {
        "env_steps": env_steps,
        "mean_team_score": row.get("mean_team_score"),
        "seed": event.get("seed"),
        "episodes": row.get("episodes"),
        "event_line": event_line,
        "reason": event.get("reason"),
    }


def _read_ppo_metrics(
    repro_root: Path, experiments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    rows: list[dict[str, Any]] = []
    records_by_id: dict[str, list[dict[str, Any]]] = {}
    native_columns: set[str] = set()

    for experiment in experiments:
        experiment_id = str(experiment.get("experiment_id", "<unknown>"))
        result_dir = experiment.get("result_dir")
        if not isinstance(result_dir, str) or not result_dir:
            raise SummaryError(f"{experiment_id}: missing result_dir in manifest")
        metrics_path = repro_root / result_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            raise SummaryError(
                f"PPO training log is missing: {metrics_path}\n"
                "Extract the Release training-records archive into this "
                "reproduction root, then rerun the summarizer. No data was generated."
            )

        records: list[dict[str, Any]] = []
        try:
            with metrics_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SummaryError(
                            f"invalid JSON in {metrics_path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(event, dict) or event.get("event") != EVALUATION_EVENT:
                        continue
                    row, record = _event_row(experiment, event, line_number)
                    rows.append(row)
                    records.append(record)
                    native_columns.update(
                        key for key in row if key.startswith("native_") and key.endswith("_mean")
                    )
        except OSError as exc:
            raise SummaryError(f"could not read {metrics_path}: {exc}") from exc
        if not records:
            raise SummaryError(f"no evaluation records found in {metrics_path}")
        records_by_id[experiment_id] = records

    return rows, records_by_id, sorted(native_columns)


def _plateau_summaries(
    experiments: list[dict[str, Any]],
    records_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        if experiment.get("model_size") != "million":
            continue
        experiment_id = str(experiment["experiment_id"])
        records = records_by_id[experiment_id]
        scored = [
            record for record in records
            if _finite_number(record.get("mean_team_score"))
        ]
        if not scored:
            raise SummaryError(f"{experiment_id}: no native team_score in eval records")
        peak_score = max(float(record["mean_team_score"]) for record in scored)
        first_peak = next(
            record for record in scored
            if float(record["mean_team_score"]) == peak_score
        )
        final_step = experiment.get("actual_env_steps")
        if not isinstance(final_step, int) or isinstance(final_step, bool):
            raise SummaryError(f"{experiment_id}: manifest lacks actual_env_steps")
        after_peak = [record for record in scored if record["env_steps"] > first_peak["env_steps"]]
        later_max = max(
            (float(record["mean_team_score"]) for record in after_peak),
            default=None,
        )
        if any(record["env_steps"] > final_step for record in scored):
            raise SummaryError(
                f"{experiment_id}: an evaluation is beyond manifest final step {final_step}"
            )
        summaries.append({
            "experiment_id": experiment_id,
            "backbone": experiment.get("backbone"),
            "model_size": experiment.get("model_size"),
            "evaluation_seed": first_peak.get("seed"),
            "evaluation_episodes": first_peak.get("episodes"),
            "score_metric": "mean_team_score",
            "first_max_score": peak_score,
            "first_max_step": first_peak["env_steps"],
            "first_max_event_line": first_peak["event_line"],
            "latest_evaluation_step": max(record["env_steps"] for record in records),
            "final_checkpoint": experiment.get("checkpoint"),
            "final_step": final_step,
            "subsequent_evaluation_count": len(after_peak),
            "highest_score_after_first_max": later_max,
            "score_breakthrough_count": sum(
                float(record["mean_team_score"]) > peak_score for record in after_peak
            ),
            "steps_after_first_max_to_final_checkpoint": final_step - first_peak["env_steps"],
            "validation_score_not_exceeded_through_final_checkpoint": (
                later_max is None or later_max <= peak_score
            ),
            "stop_reason": experiment.get("training_state"),
            "training_state": experiment.get("training_state"),
        })
    if len(summaries) != 3:
        raise SummaryError(f"expected three million-scale PPO runs, found {len(summaries)}")
    return {
        "schema_version": 1,
        "metric_scope": "fixed-seed native validation mean team_score only",
        "interpretation": (
            "A validation-score plateau means only that the recorded fixed-seed mean "
            "team_score did not exceed its prior maximum before the saved stopping "
            "checkpoint; it does not imply other metrics or policies were unchanged."
        ),
        "runs": summaries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], native_columns: list[str]) -> None:
    columns = [*BASE_COLUMNS, *native_columns]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text_write(path, buffer.getvalue())


def summarize(repro_root: Path) -> tuple[Path, Path, int, dict[str, Any]]:
    repro_root = repro_root.resolve()
    experiments = _read_experiments(repro_root)
    rows, records_by_id, native_columns = _read_ppo_metrics(repro_root, experiments)
    plateau = _plateau_summaries(experiments, records_by_id)
    results_dir = repro_root / "results"
    csv_path = results_dir / "ppo_training_evaluations.csv"
    plateau_path = results_dir / "ppo_plateau_summary.json"
    _write_csv(csv_path, rows, native_columns)
    _atomic_text_write(
        plateau_path,
        json.dumps(plateau, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return csv_path, plateau_path, len(rows), plateau


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repro-root",
        type=Path,
        default=DEFAULT_REPRO_ROOT,
        help=f"reproduction root with experiments.json (default: {DEFAULT_REPRO_ROOT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path, plateau_path, evaluation_count, plateau = summarize(args.repro_root)
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {evaluation_count} PPO evaluation rows: {csv_path}")
    print(f"wrote three-run PPO plateau summary: {plateau_path}")
    for run in plateau["runs"]:
        print(
            f"{run['backbone']}: score {run['first_max_score']:g} first at "
            f"{run['first_max_step']:,}; no score breakthrough for "
            f"{run['steps_after_first_max_to_final_checkpoint']:,} steps "
            f"through final checkpoint {run['final_step']:,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

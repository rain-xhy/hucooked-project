"""Mirror the pure E3T progress.jsonl into a local Trackio 0.45.0 dashboard."""

import argparse
import io
import json
import math
import os
import threading
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

PROJECT = "hucooked-e3t-20m-a100"
RUN = "pure-seed42"


def log_points(path, live):
    points = {}
    with path.open("rb") as source:
        for index, line in enumerate(source, 1):
            if live and not line.endswith(b"\n"):
                break
            if not line.strip():
                continue
            row = json.loads(line)
            step, kind = row["env_steps"], row["kind"]
            if (
                type(step) is not int
                or step < 0
                or not isinstance(kind, str)
                or not kind
            ):
                raise ValueError(f"{path}:{index}: invalid env_steps or kind")
            for name, value in row.items():
                if name in ("env_steps", "wall_time") or type(value) not in (
                    int,
                    float,
                ):
                    continue
                key = (step, f"{kind}/{name}")
                if not math.isfinite(value) or (key in points and points[key] != value):
                    raise ValueError(f"{path}:{index}: conflicting or nonfinite {key}")
                points[key] = value
    return points


def stored_points():
    from trackio.sqlite_storage import SQLiteStorage

    snapshot = SQLiteStorage.get_snapshot(PROJECT, RUN)
    return {
        (row["step"], name): row["value"]
        for name, series in snapshot.items()
        for row in series
    }


def sync(trackio, points):
    stored = stored_points()
    for key, value in stored.items():
        if key not in points or points[key] != value:
            raise ValueError(
                f"Log and Trackio disagree at {key}. Reimport into a fresh --data-dir."
            )
    missing = defaultdict(dict)
    for (step, name), value in points.items():
        if (step, name) not in stored:
            missing[step][name] = value
    if not missing:
        return 0
    trackio.init(
        project=PROJECT,
        name=RUN,
        resume="allow",
        embed=False,
        auto_log_cpu=False,
        auto_log_gpu=False,
        capture_console=False,
    )
    try:
        for step, metrics in sorted(missing.items()):
            trackio.log(metrics, step=step)
    finally:
        trackio.finish()
    if stored_points() != points:
        raise RuntimeError("Trackio did not persist all log points")
    return sum(map(len, missing.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reproduction/e3t_20m/runs/seed42"),
        help="training output directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Trackio database directory (default: OUT-trackio)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--follow", action="store_true", help="poll the growing log every 15 seconds"
    )
    group.add_argument(
        "--show", action="store_true", help="open the local dashboard server"
    )
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    run_dir = args.out.resolve()
    data_dir = (args.data_dir or run_dir.with_name(run_dir.name + "-trackio")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(data_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for var in (
        "TRACKIO_SPACE_ID",
        "TRACKIO_SERVER_URL",
        "TRACKIO_BUCKET_ID",
        "TRACKIO_DATASET_ID",
        "TRACKIO_WEBHOOK_URL",
    ):
        os.environ.pop(var, None)
    import trackio
    from filelock import FileLock

    if args.show:
        # Trackio checks its own loopback server during startup.
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
        with redirect_stdout(io.StringIO()):
            app, url, _, _ = trackio.show(
                project=PROJECT,
                host="127.0.0.1",
                share=False,
                open_browser=False,
                block_thread=False,
                server_port=args.port,
                mcp_server=False,
            )
        print(f"{url.rstrip('/')}/?project={PROJECT}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
        return

    progress = args.out / "progress.jsonl"
    signature = None
    while True:
        if progress.is_file() or not args.follow:
            stat = progress.stat()
            latest = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if latest != signature:
                with FileLock(data_dir / ".import.lock"):
                    points = log_points(progress, args.follow)
                    added = sync(trackio, points)
                print(f"{len(points)} log scalars, {added} newly imported", flush=True)
                signature = latest
        if not args.follow:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()

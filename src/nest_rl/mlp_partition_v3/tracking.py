"""Local Trackio storage shared by the training process and dashboard."""
import os
import sqlite3
from pathlib import Path

PROJECT = "hucooked-v3-counter-ppo"
COOPERATION_PROJECT = "hucooked-v3-cooperation"
RUNS = Path(__file__).resolve().parent / "runs"


def local_trackio():
    os.environ["TRACKIO_DIR"] = str(RUNS / "trackio")
    os.environ["TRACKIO_STORAGE_MODE"] = "sqlite"
    for key in ("TRACKIO_SPACE_ID", "TRACKIO_DATASET_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID"):
        os.environ.pop(key, None)
    import trackio
    return trackio


def open_run(name, group, config, resume=False, project=PROJECT):
    return local_trackio().init(project=project, name=name, group=group, config=config,
                                resume="allow" if resume else "never", embed=False,
                                auto_log_cpu=False, auto_log_gpu=False)


def recorded_steps(name, project=PROJECT):
    database = RUNS / "trackio" / f"{project}.db"
    if not database.exists():
        return set()
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as conn:
        return {row[0] for row in conn.execute(
            "SELECT DISTINCT step FROM metrics WHERE run_name = ?", (name,))}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=COOPERATION_PROJECT)
    args = parser.parse_args()
    local_trackio().show(project=args.project, host="127.0.0.1", server_port=7864,
                         open_browser=False, share=False, block_thread=True)

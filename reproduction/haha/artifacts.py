"""Atomic experiment artifacts and portable checkpoint metadata (no file hashes)."""

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix,
                                       dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        # Only remove the temporary file owned by this operation.
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False), encoding="utf-8")


def save_torch(path, value):
    import torch
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_info(path):
    path = Path(path)
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Checkpoint and its sidecar are both required: {path}, {metadata}")
    return read_json(metadata)


def save_policy(path, model, config):
    path = Path(path)
    model.haha_checkpoint_id = uuid.uuid4().hex
    with atomic_path(path) as temporary:
        model.save(temporary)
    write_json(path.with_suffix(".json"), {
        "artifact_id": config["artifact_id"], "stage": config["stage"],
        "checkpoint_id": model.haha_checkpoint_id,
        "timesteps": model.num_timesteps, "config": config,
    })


def check_loaded_policy(model, metadata):
    if model.num_timesteps != metadata["timesteps"]:
        raise ValueError("Checkpoint/sidecar timesteps differ; use a preceding numbered checkpoint")
    if getattr(model, "haha_artifact_id", None) != metadata["artifact_id"]:
        raise ValueError("Checkpoint/sidecar artifact ids differ")
    if getattr(model, "haha_checkpoint_id", None) != metadata["checkpoint_id"]:
        raise ValueError("Checkpoint/sidecar save ids differ; use a preceding numbered checkpoint")

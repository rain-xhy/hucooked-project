"""Resource guardrails and atomic experiment artifacts."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def setup_device(device='cuda:0', seed=1, memory_fraction=.24):
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or torch.cuda.device_count() != 1:
            raise RuntimeError('This experiment is restricted to physical GPU 1: CUDA_VISIBLE_DEVICES=1')
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    return torch.device(device)


def atomic_save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp); os.replace(temp, path)


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def source_hashes(method):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (Path(method), Path('baseline_common'), Path('src/overcooked_ai_py/cooperative'))
            for p in sorted(root.rglob('*.py'))}

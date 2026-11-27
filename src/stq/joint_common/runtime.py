"""Resource mapping and atomic artifacts for independent training jobs."""
import os
from pathlib import Path
import random

import numpy as np
import torch

from baseline_common.runtime import atomic_json, atomic_save
from baseline_common.runtime import source_hashes as baseline_source_hashes


def prepare_output(path):
    """Create the output directory selected in the experiment launch plan."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_device(device='cuda:0', seed=1, physical_gpu=1, memory_fraction=.035):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target = torch.device(device)
    if target.type == 'cuda':
        if str(physical_gpu) != os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('CUDA_VISIBLE_DEVICES must equal the configured physical_gpu')
        if torch.cuda.device_count() != 1 or target.index not in (None, 0):
            raise RuntimeError('Each experiment must expose exactly one physical GPU as cuda:0')
        if not 0. < memory_fraction <= 1.:
            raise ValueError('memory_fraction must be in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    return target


def source_hashes(method):
    return baseline_source_hashes(method, 'joint_common')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def team_inputs(observation, device):
    spatial = np.asarray(observation['spatial'])
    state = np.asarray(observation['state'])
    if spatial.ndim != 5 or spatial.shape[1:] != (3, 26, 7, 7):
        raise ValueError('Expected native spatial observation [B,3,26,7,7]')
    if state.shape != (len(spatial), 863):
        raise ValueError('Expected public team state [B,863]')
    return (torch.as_tensor(spatial[:, 0], device=device),
            torch.as_tensor(state, device=device))

"""Group native expert correction labels into one three-action sample."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def demonstration_steps(path):
    with np.load(path) as data:
        return int(data['env_steps'])


def load_joint_dataset(path):
    with np.load(path) as data:
        vector = np.asarray(data['vector'], dtype=np.float32)
        spatial = np.asarray(data['spatial'], dtype=np.float32)
        actions = np.asarray(data['actions'], dtype=np.int64)
        validation = np.asarray(data['validation'], dtype=bool)
        lengths = np.asarray(data['episode_lengths'], dtype=np.int64)
        env_steps = int(data['env_steps'])
    if vector.ndim != 2 or vector.shape[1] != 866 or len(vector) % 3:
        raise ValueError('Expected flattened native [time,chef,866] vectors')
    if spatial.shape != (len(vector), 26, 7, 7):
        raise ValueError('Expected flattened native spatial [N,26,7,7]')
    if actions.shape != (len(vector),) or validation.shape != (len(vector),):
        raise ValueError('Labels and splits must match the native chef rows')
    frames = len(vector) // 3
    vectors = vector.reshape(frames, 3, 866)
    if not (np.array_equal(vectors[:, 0, :863], vectors[:, 1, :863])
            and np.array_equal(vectors[:, 0, :863], vectors[:, 2, :863])):
        raise ValueError('The three observations must share the same public team state')
    splits = validation.reshape(frames, 3)
    if not np.all(splits == splits[:, :1]):
        raise ValueError('A joint transition cannot cross train/validation splits')
    if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0) or lengths.sum() != frames:
        raise ValueError('episode_lengths must cover all joint transitions')
    offset = 0
    for length in lengths:
        segment = splits[offset:offset + int(length), 0]
        if not np.all(segment == segment[0]):
            raise ValueError('Each full episode must belong to one dataset split')
        offset += int(length)
    if int((~splits[:, 0]).sum()) != env_steps:
        raise ValueError('env_steps must count training transitions once, without chef multiplication')
    if not np.any(splits[:, 0]) or not np.any(~splits[:, 0]):
        raise ValueError('Both training and validation episodes are required')
    if np.any(actions < 0) or np.any(actions >= 6):
        raise ValueError('Native action labels must be in [0,6)')
    return {
        'state': vectors[:, 0, :863].copy(),
        'spatial': spatial.reshape(frames, 3, 26, 7, 7)[:, 0].copy(),
        'actions': actions.reshape(frames, 3).copy(),
        'validation': splits[:, 0].copy(),
        'episode_lengths': lengths.copy(), 'env_steps': env_steps,
        'dataset_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def warm_start_joint(model, forward_logits, *, dataset_path, device, seed,
                     epochs=64, batch_size=512):
    if epochs <= 0 or batch_size < 3:
        raise ValueError('Positive epochs and at least three action labels per batch are required')
    data = load_joint_dataset(dataset_path)
    training = np.flatnonzero(~data['validation'])
    validation = np.flatnonzero(data['validation'])
    frames_per_batch = max(1, int(batch_size) // 3)
    batches = max(1, int(np.ceil(len(training) / frames_per_batch)))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(epochs):
        losses = []
        for ids in np.array_split(rng.permutation(training), batches):
            spatial = torch.as_tensor(data['spatial'][ids], device=device)
            state = torch.as_tensor(data['state'][ids], device=device)
            logits = forward_logits(spatial, state)
            labels = torch.as_tensor(data['actions'][ids], device=device)
            if logits.shape != (*labels.shape, 6):
                raise ValueError('Joint BC logits must have shape [B,3,6]')
            loss = F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint behavior cloning loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation), frames_per_batch):
            ids = validation[start:start + frames_per_batch]
            logits = forward_logits(torch.as_tensor(data['spatial'][ids], device=device),
                                    torch.as_tensor(data['state'][ids], device=device))
            labels = torch.as_tensor(data['actions'][ids], device=device)
            correct += int((logits.argmax(-1) == labels).sum().cpu())
            total += labels.numel()
    model.train()
    return {'demo_env_steps': data['env_steps'], 'epochs': int(epochs),
            'train_ce': float(np.mean(losses)), 'validation_accuracy': correct / total,
            'joint_frames_per_batch': frames_per_batch,
            'dataset_sha256': data['dataset_sha256']}

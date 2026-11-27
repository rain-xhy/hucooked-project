"""Sequence-level behavior cloning for the central hierarchical actor."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def _episodes(dataset):
    lengths = np.asarray(dataset["episode_lengths"], dtype=np.int64)
    validation = np.asarray(dataset["validation"], dtype=bool)
    n = len(dataset["actions"])
    if not len(lengths) or np.any(lengths <= 0) or int(lengths.sum()) != n:
        raise ValueError("episode_lengths must partition the complete demonstration dataset")
    episodes, start = [], 0
    for length in lengths:
        stop = start + int(length)
        split = validation[start:stop]
        if not len(split) or np.any(split != split[0]):
            raise ValueError("Each demonstration episode must belong to one data split")
        episodes.append((start, stop, bool(split[0])))
        start = stop
    return episodes


def _sequence_batch(dataset, entries, device):
    rows = dataset["spatial"]
    states = dataset["state"]
    actions = dataset["actions"]
    max_time = max(stop - start for start, stop, _ in entries)
    batch = len(entries)
    spatial = np.zeros((max_time, batch, *rows.shape[1:]), dtype=np.float32)
    state = np.zeros((max_time, batch, states.shape[-1]), dtype=np.float32)
    label = np.zeros((max_time, batch, 3), dtype=np.int64)
    valid = np.zeros((max_time, batch), dtype=bool)
    done = np.ones((max_time, batch), dtype=np.float32)
    for col, (start, stop, _) in enumerate(entries):
        length = stop - start
        spatial[:length, col] = rows[start:stop]
        state[:length, col] = states[start:stop]
        label[:length, col] = actions[start:stop]
        valid[:length, col] = True
        done[:length, col] = 0.0
        done[0, col] = 1.0
    return tuple(torch.as_tensor(value, device=device) for value in (spatial, state, label, valid, done))


def warm_start_hierarchical(model, dataset, *, device, seed, epochs=64,
                            episodes_per_batch=2):
    """Train low heads through complete joint episodes; never label high z."""
    if episodes_per_batch <= 0 or epochs <= 0:
        raise ValueError("epochs and episodes_per_batch must be positive")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    episodes = _episodes(dataset)
    train_episodes = [row for row in episodes if not row[2]]
    validation_episodes = [row for row in episodes if row[2]]
    if not train_episodes or not validation_episodes:
        raise ValueError("Joint demonstrations require train and validation episodes")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epoch_losses = []
    for _ in range(int(epochs)):
        order = rng.permutation(len(train_episodes))
        for ids in np.array_split(order, max(1, int(np.ceil(len(order) / episodes_per_batch)))):
            if not len(ids):
                continue
            group = [train_episodes[int(index)] for index in ids]
            spatial, state, labels, valid, _done = _sequence_batch(dataset, group, device)
            high_logits, _, low_logits, _, _, _ = model.unroll_complete_episodes(
                spatial, state, model.initial_state(len(group), device=device)
            )
            selected = low_logits[valid]
            target = labels[valid][..., None].expand(-1, 3, model.z_dim)
            loss = F.cross_entropy(
                selected.reshape(-1, model.action_dim), target.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite central hierarchical BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(validation_episodes), episodes_per_batch):
            group = validation_episodes[start:start + episodes_per_batch]
            spatial, state, labels, valid, _done = _sequence_batch(dataset, group, device)
            _, _, low_logits, _, _, _ = model.unroll_complete_episodes(
                spatial, state, model.initial_state(len(group), device=device)
            )
            predicted = low_logits[valid].argmax(-1)
            expected = labels[valid][..., None].expand_as(predicted)
            correct += int((predicted == expected).sum().cpu())
            total += expected.numel()
    model.train()
    return {
        "demo_env_steps": int(dataset["env_steps"]),
        "episodes": len(train_episodes), "validation_episodes": len(validation_episodes),
        "epochs": int(epochs), "sequence_mode": "complete episodes; central LSTM carried over time",
        "supervised_heads": "all 3 roles × all z low-level action heads",
        "high_level_z_supervised": False,
        "train_ce": float(np.mean(epoch_losses)) if epoch_losses else None,
        "validation_accuracy": correct / max(1, total),
    }

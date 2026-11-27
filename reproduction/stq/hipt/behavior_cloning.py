"""Sequence-aware behavior cloning for the recurrent HiPT policy."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from .model import (
    HIDDEN_SIZE,
    SPATIAL_CHANNELS,
    SPATIAL_HEIGHT,
    SPATIAL_WIDTH,
    VECTOR_DIM,
)

HIERARCHICAL_BC_EPISODES_PER_BATCH = 2


def _read_episodes(dataset_path):
    """Read the flattened [episode, time, chef] dataset and require boundaries."""
    with np.load(dataset_path) as data:
        required = {"spatial", "vector", "actions", "validation", "episode_lengths", "env_steps"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                "Sequence BC requires episode_lengths and complete SharedKitchen data; "
                f"missing {sorted(missing)}"
            )
        spatial = np.asarray(data["spatial"], dtype=np.float32)
        vector = np.asarray(data["vector"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        validation = np.asarray(data["validation"], dtype=bool)
        episode_lengths = np.asarray(data["episode_lengths"], dtype=np.int64).reshape(-1)
        demonstration_env_steps = int(data["env_steps"])

    if spatial.ndim != 4 or spatial.shape[1:] != (
        SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
    ):
        raise ValueError("Demonstration spatial data must have shape [N, 26, 7, 7]")
    if vector.shape != (len(spatial), VECTOR_DIM):
        raise ValueError("Demonstration vectors must have shape [N, 866]")
    if actions.shape != (len(spatial),) or validation.shape != (len(spatial),):
        raise ValueError("Demonstration labels and validation flags must match [N]")
    if len(episode_lengths) == 0 or np.any(episode_lengths <= 0):
        raise ValueError("episode_lengths must contain positive episode lengths")
    expected_rows = int(episode_lengths.sum()) * 3
    if expected_rows != len(spatial):
        raise ValueError(
            "episode_lengths must cover every flattened time/chef row exactly once"
        )

    episodes = []
    offset = 0
    for length in episode_lengths.tolist():
        stop = offset + int(length) * 3
        flags = validation[offset:stop]
        if not np.all(flags == flags[0]):
            raise ValueError("Each demonstration episode must belong to one split")
        episodes.append({"start": offset, "stop": stop, "length": int(length),
                         "validation": bool(flags[0])})
        offset = stop

    training_env_steps = sum(ep["length"] for ep in episodes if not ep["validation"])
    validation_count = sum(ep["validation"] for ep in episodes)
    training_count = len(episodes) - validation_count
    if training_count == 0 or validation_count == 0:
        raise ValueError("Sequence BC requires separate training and validation episodes")
    if training_env_steps != demonstration_env_steps:
        raise ValueError(
            "env_steps must count training environment transitions once, without multiplying by three chefs"
        )
    return spatial, vector, actions, episodes, demonstration_env_steps


def _episode_batches(episodes, *, rng, episodes_per_batch):
    by_length = defaultdict(list)
    for episode in episodes:
        by_length[episode["length"]].append(episode)
    batches = []
    for length in sorted(by_length):
        candidates = by_length[length]
        order = rng.permutation(len(candidates))
        for start in range(0, len(order), episodes_per_batch):
            batches.append([candidates[int(index)] for index in order[start:start + episodes_per_batch]])
    rng.shuffle(batches)
    return batches


def _sequence_tensors(model, spatial_data, vector_data, action_data, episodes, device):
    sequence_length = episodes[0]["length"]
    if any(episode["length"] != sequence_length for episode in episodes):
        raise ValueError("An episode minibatch must have one shared sequence length")
    spatial_sequences, vector_sequences, action_sequences = [], [], []
    for episode in episodes:
        start, stop = episode["start"], episode["stop"]
        spatial_sequences.append(
            spatial_data[start:stop].reshape(
                sequence_length, 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
            )
        )
        vector_sequences.append(vector_data[start:stop].reshape(sequence_length, 3, VECTOR_DIM))
        action_sequences.append(action_data[start:stop].reshape(sequence_length, 3))

    # Sequence-major layout: time, then all chef streams from each episode.
    spatial = np.stack(spatial_sequences, axis=1).reshape(
        sequence_length, len(episodes) * 3,
        SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH,
    )
    vector = np.stack(vector_sequences, axis=1).reshape(
        sequence_length, len(episodes) * 3, VECTOR_DIM
    )
    actions = np.stack(action_sequences, axis=1).reshape(
        sequence_length, len(episodes) * 3
    )
    return (
        torch.as_tensor(spatial, device=device),
        torch.as_tensor(vector, device=device),
        torch.as_tensor(actions, device=device),
    )


def _low_logits_for_sequence(model, spatial, vector):
    """Unroll one complete episode batch from zero state, carrying it over time."""
    time_steps, stream_count = spatial.shape[:2]
    flat_spatial = spatial.flatten(0, 1)
    flat_vector = vector.flatten(0, 1)
    encoded = model._encode(flat_spatial, flat_vector).reshape(
        time_steps, stream_count, HIDDEN_SIZE
    )
    initial_state = model.initial_state(stream_count, device=spatial.device,
                                        dtype=spatial.dtype)
    recurrent, _ = model.lstm(encoded, initial_state)
    return model._heads(recurrent)[2]


def warm_start_hierarchical(model, *, dataset_path, device, seed,
                            epochs: int = 64,
                            episodes_per_batch: int = HIERARCHICAL_BC_EPISODES_PER_BATCH,
                            learning_rate: float = 1e-3, max_grad_norm: float = 1.0):
    """Behavior-clone complete episodes with recurrent state carried over time.

    The high-level z head has no teacher label and receives no BC loss. Each
    low-level z head is trained against the same teacher action at every step.
    Dataset boundaries are mandatory; this helper never falls back to isolated
    zero-state observations.
    """
    if epochs <= 0 or episodes_per_batch <= 0:
        raise ValueError("epochs and episodes_per_batch must be positive")
    spatial_data, vector_data, action_data, episodes, demonstration_env_steps = _read_episodes(
        dataset_path
    )
    train_episodes = [episode for episode in episodes if not episode["validation"]]
    validation_episodes = [episode for episode in episodes if episode["validation"]]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    train_losses = []
    batch_stream_sizes, batch_frame_sizes = set(), set()
    for _ in range(epochs):
        epoch_losses = []
        for episode_batch in _episode_batches(
            train_episodes, rng=rng, episodes_per_batch=episodes_per_batch
        ):
            spatial, vector, actions = _sequence_tensors(
                model, spatial_data, vector_data, action_data, episode_batch, device
            )
            logits = _low_logits_for_sequence(model, spatial, vector)
            labels = actions.unsqueeze(-1).expand(-1, -1, model.z_dim)
            loss = F.cross_entropy(
                logits.reshape(-1, model.action_dim), labels.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite sequence BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            batch_stream_sizes.add(int(spatial.shape[1]))
            batch_frame_sizes.add(int(spatial.shape[0] * spatial.shape[1]))
        train_losses.append(float(np.mean(epoch_losses)))

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for episode_batch in _episode_batches(
            validation_episodes, rng=rng, episodes_per_batch=episodes_per_batch
        ):
            spatial, vector, actions = _sequence_tensors(
                model, spatial_data, vector_data, action_data, episode_batch, device
            )
            logits = _low_logits_for_sequence(model, spatial, vector)
            labels = actions.unsqueeze(-1).expand(-1, -1, model.z_dim)
            correct += int((logits.argmax(-1) == labels).sum().cpu())
            total += labels.numel()
    model.train()

    lengths = [episode["length"] for episode in episodes]
    return {
        "demo_env_steps": demonstration_env_steps,
        "epochs": int(epochs),
        "train_ce": train_losses[-1],
        "validation_accuracy": correct / max(1, total),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "sequence_length_min": min(lengths),
        "sequence_length_max": max(lengths),
        "episode_batch_size": int(episodes_per_batch),
        "chef_stream_batch_sizes": sorted(batch_stream_sizes),
        "frame_batch_sizes": sorted(batch_frame_sizes),
        "train_frames": sum(episode["length"] * 3 for episode in train_episodes),
        "validation_frames": sum(episode["length"] * 3 for episode in validation_episodes),
    }

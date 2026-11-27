"""Deterministic all-three-chef evaluation for trained HiPT checkpoints."""

from __future__ import annotations

import numpy as np
import torch

from .common import OptionClock, sample_durations
from .model import SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH, VECTOR_DIM


NATIVE_METRICS = (
    "team_score",
    "meals_completed",
    "normal_completed",
    "joint_completed",
    "joint_successes",
    "joint_attempt_success_rate",
    "joint_wait_steps",
    "three_contributor_meals",
    "resource_conflicts",
    "movement_conflicts",
    "ticks",
    "meals_per_100_steps",
)
EPISODE_METRICS = NATIVE_METRICS + ("training_reward", "training_shaping")


def summarize_episode_records(records):
    """Aggregate native SharedKitchen metrics while retaining each summary."""
    means, stds = {}, {}
    for key in EPISODE_METRICS:
        values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
        means[key] = float(values.mean()) if len(values) else 0.0
        stds[key] = float(values.std()) if len(values) else 0.0
    return {
        "episodes": len(records),
        "metric_means": means,
        "metric_stds": stds,
        "records": records,
    }


@torch.no_grad()
def evaluate(model, *, episodes: int = 12, num_envs: int = 4,
             workers: int = 0, seed: int = 10001,
             device: torch.device | str = "cpu", gamma: float = 0.9985,
             p_min: int = 20, p_max: int = 40):
    """Evaluate all three roles using only the learned hierarchical policy."""
    from baseline_common.envs import KitchenBatch

    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    rng = np.random.default_rng(seed)
    env = KitchenBatch(num_envs=num_envs, seed=seed, stage="standard", workers=workers,
                       train=False, gamma=gamma)
    target = episodes
    records = []
    try:
        obs = env.reset()
        state = model.initial_state(num_envs * 3, device=device)
        done_previous = np.zeros(num_envs, dtype=bool)
        clock = OptionClock((num_envs, 3))
        while len(records) < target:
            spatial = torch.as_tensor(obs["spatial"], device=device).reshape(
                num_envs * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
            )
            vector = torch.as_tensor(obs["vector"], device=device).reshape(
                num_envs * 3, VECTOR_DIM
            )
            done_roles = torch.as_tensor(
                np.repeat(done_previous[:, None], 3, axis=1).reshape(-1), device=device,
                dtype=torch.float32,
            )
            high_logits, _, low_logits, _, state = model.step(
                spatial, vector, state, done_roles
            )
            high_logits = high_logits.reshape(num_envs, 3, model.z_dim)
            low_logits = low_logits.reshape(num_envs, 3, model.z_dim, model.action_dim)
            due = clock.due
            selected = high_logits.argmax(-1).cpu().numpy()
            durations = sample_durations(rng, (num_envs, 3), p_min, p_max)
            clock.start(due, selected, durations)
            z = torch.as_tensor(clock.z.reshape(-1), device=device)
            low_selected = low_logits.reshape(-1, model.z_dim, model.action_dim).gather(
                1, z[:, None, None].expand(-1, 1, model.action_dim)
            ).squeeze(1)
            actions = low_selected.argmax(-1).cpu().numpy().reshape(num_envs, 3)
            next_obs, _, done, infos = env.step(actions)
            clock.advance(done)
            for ended, info in zip(done, infos):
                if ended and "episode_summary" in info and len(records) < target:
                    records.append(info["episode_summary"])
            obs, done_previous = next_obs, done.copy()
    finally:
        env.close()
    return summarize_episode_records(records)

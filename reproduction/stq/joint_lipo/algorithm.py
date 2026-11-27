"""PPO/GAE and selection helpers for centralized joint-action LIPO."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def compute_gae(rewards, values, dones, last_values, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    last_values = np.asarray(last_values, dtype=np.float32)
    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must share shape [time, environment]")
    if last_values.shape != (rewards.shape[1],):
        raise ValueError("last_values must have one value per environment")
    if not all(np.isfinite(array).all() for array in (rewards, values, last_values)):
        raise FloatingPointError("Non-finite rollout reward/value")
    advantages = np.zeros_like(rewards)
    next_adv = np.zeros_like(last_values)
    next_value = last_values
    for step in reversed(range(rewards.shape[0])):
        alive = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + gamma * next_value * alive - values[step]
        next_adv = delta + gamma * gae_lambda * alive * next_adv
        advantages[step] = next_adv
        next_value = values[step]
    return advantages, advantages + values


def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float,
    *,
    minimize_reward: bool = False,
) -> torch.Tensor:
    """Apply signed advantages inside the clipped PPO surrogate, as in LIPO."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages).mean()
    return surrogate if minimize_reward else -surrogate


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def select_max_compatible(scores: Sequence[float]) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one sampled XP score is required")
    if not np.isfinite(values).all():
        raise FloatingPointError("XP selection scores must all be finite")
    return int(np.argmax(values))


def role_for_xp(iteration: int, candidate_slot: int, member_id: int) -> int:
    if iteration < 0 or candidate_slot < 0 or member_id < 0:
        raise ValueError("iteration, candidate_slot, and member_id must be nonnegative")
    return int((iteration + candidate_slot + member_id) % 3)


def assert_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}")

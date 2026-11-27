"""Small algorithm helpers shared by the two HiPT training stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical


def gae(rewards, values, dones, bootstrap, gamma: float, gae_lambda: float):
    """Generalized advantages for [T, ...] trajectories with true terminals."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    bootstrap = np.asarray(bootstrap, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape[1:-1] != dones.shape[1:-1]:
        # The two uses are [T,E,R] rewards/values with [T,E] dones, or exact
        # shape equality after callers broadcast terminal flags over roles.
        if not (rewards.ndim == 3 and dones.shape == rewards.shape[:2]):
            raise ValueError("rewards, values, and dones have incompatible shapes")
    if rewards.shape[0] == 0 or bootstrap.shape != rewards.shape[1:]:
        raise ValueError("bootstrap must match a nonempty trajectory's trailing shape")
    done = dones
    if done.ndim == rewards.ndim - 1:
        done = done[..., None]
    if done.shape != rewards.shape[:done.ndim] + (1,) * (rewards.ndim - done.ndim):
        try:
            done = np.broadcast_to(done, rewards.shape)
        except ValueError as exc:
            raise ValueError("dones cannot be broadcast over trajectory roles") from exc
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(bootstrap, dtype=np.float32)
    for t in reversed(range(rewards.shape[0])):
        next_value = bootstrap if t == rewards.shape[0] - 1 else values[t + 1]
        alive = 1.0 - done[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[t] = carry
    return advantages, advantages + values


def ppo_components(new_logp: torch.Tensor, old_logp: torch.Tensor,
                   advantage: torch.Tensor, new_value: torch.Tensor,
                   old_value: torch.Tensor, returns: torch.Tensor,
                   entropy: torch.Tensor, clip: float):
    """Return clipped PPO policy/value losses and categorical entropy."""
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    pg = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
    ).mean()
    clipped_value = old_value + (new_value - old_value).clamp(-clip, clip)
    value = 0.5 * torch.maximum(
        (new_value - returns).square(), (clipped_value - returns).square()
    ).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    return pg, value, entropy.mean(), approx_kl


def sample_durations(rng: np.random.Generator, shape, p_min: int, p_max: int):
    """Uniform integer option lengths with both inclusive endpoints."""
    if p_min < 1 or p_max < p_min:
        raise ValueError("Require 1 <= p_min <= p_max")
    return rng.integers(p_min, p_max + 1, size=shape, dtype=np.int32)


class OptionClock:
    """Independent high-level z timers for every environment/chef stream."""

    def __init__(self, shape):
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("OptionClock expects shape [num_envs, num_roles]")
        self.remaining = np.zeros(shape, dtype=np.int32)
        self.z = np.zeros(shape, dtype=np.int64)

    @property
    def due(self):
        return self.remaining == 0

    def start(self, mask, z, duration):
        mask = np.asarray(mask, dtype=bool)
        z = np.asarray(z, dtype=np.int64)
        duration = np.asarray(duration, dtype=np.int32)
        if mask.shape != self.z.shape or z.shape != self.z.shape or duration.shape != self.z.shape:
            raise ValueError("Option arrays must match the clock's [env, role] shape")
        if np.any(duration[mask] <= 0):
            raise ValueError("Option durations must be positive")
        self.z[mask] = z[mask]
        self.remaining[mask] = duration[mask]

    def advance(self, episode_done):
        """Consume one primitive action and clear terminal streams immediately."""
        done = np.asarray(episode_done, dtype=bool)
        if done.shape == (self.z.shape[0],):
            done = np.repeat(done[:, None], self.z.shape[1], axis=1)
        if done.shape != self.z.shape:
            raise ValueError("episode_done must have shape [env] or [env, role]")
        active = (self.remaining > 0) & ~done
        self.remaining[active] -= 1
        self.remaining[done] = 0
        self.z[done] = 0
        return self.due.copy()


@dataclass
class PendingOption:
    """One option's stored policy data while its macro reward is accumulating."""

    history: list
    decision_index: int
    z: int
    old_logp: float
    old_value: float
    reward_sum: float = 0.0
    duration: int = 0

    def add_step(self, combined_reward: float):
        self.reward_sum += float(combined_reward)
        self.duration += 1

    def finish(self, *, bootstrap_value: float, terminal: bool):
        if self.duration <= 0:
            raise RuntimeError("Cannot complete an option with no executed actions")
        return {
            "history": self.history,
            "decision_index": self.decision_index,
            "z": self.z,
            "old_logp": self.old_logp,
            "old_value": self.old_value,
            "reward": self.reward_sum / self.duration,
            "duration": self.duration,
            "bootstrap_value": 0.0 if terminal else float(bootstrap_value),
            "terminal": bool(terminal),
        }


def high_level_gae(records_by_stream, gamma: float, gae_lambda: float):
    """Attach option-time-scale advantages/returns to completed options."""
    for records in records_by_stream:
        carry = 0.0
        for i in reversed(range(len(records))):
            record = records[i]
            alive = 0.0 if record["terminal"] else 1.0
            if alive and i + 1 < len(records):
                next_value = float(records[i + 1]["old_value"])
            else:
                next_value = float(record["bootstrap_value"])
            delta = float(record["reward"]) + gamma * next_value * alive - float(record["old_value"])
            carry = delta + gamma * gae_lambda * alive * carry
            record["advantage"] = carry
            record["return"] = carry + float(record["old_value"])
    return records_by_stream


def categorical(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits)

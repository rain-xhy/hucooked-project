"""PPO and independent-option helpers local to the central HiPT package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def gae(rewards, values, dones, bootstrap, gamma: float, gae_lambda: float):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    bootstrap = np.asarray(bootstrap, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape[:2] != dones.shape:
        raise ValueError("Expected rewards/values [T,E,3] and dones [T,E]")
    if not len(rewards) or bootstrap.shape != rewards.shape[1:]:
        raise ValueError("Bootstrap must match [E,3] for a nonempty rollout")
    advantages = np.zeros_like(rewards)
    carry = np.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        alive = (1.0 - dones[t])[:, None]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[t] = carry
    return advantages, advantages + values


def ppo_components(new_logp, old_logp, advantage, new_value, old_value, returns,
                   entropy, clip: float):
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
    ).mean()
    clipped_value = old_value + (new_value - old_value).clamp(-clip, clip)
    value = 0.5 * torch.maximum(
        (new_value - returns).square(), (clipped_value - returns).square()
    ).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    return policy, value, entropy.mean(), approx_kl


def sample_durations(rng: np.random.Generator, shape, p_min: int, p_max: int):
    if p_min < 1 or p_max < p_min:
        raise ValueError("Require 1 <= p_min <= p_max")
    return rng.integers(p_min, p_max + 1, size=shape, dtype=np.int32)


class OptionClock:
    """Three independent z clocks for each environment."""

    def __init__(self, num_envs: int):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.remaining = np.zeros((int(num_envs), 3), dtype=np.int32)
        self.z = np.zeros((int(num_envs), 3), dtype=np.int64)

    @property
    def due(self):
        return self.remaining == 0

    def start(self, mask, z, duration):
        mask, z, duration = np.asarray(mask, bool), np.asarray(z, np.int64), np.asarray(duration, np.int32)
        if mask.shape != self.z.shape or z.shape != self.z.shape or duration.shape != self.z.shape:
            raise ValueError("Option arrays must have shape [num_envs,3]")
        if np.any(duration[mask] <= 0):
            raise ValueError("Option duration must be positive")
        self.z[mask] = z[mask]
        self.remaining[mask] = duration[mask]

    def advance(self, episode_done):
        done = np.asarray(episode_done, dtype=bool)
        if done.shape != (self.z.shape[0],):
            raise ValueError("episode_done must have shape [num_envs]")
        active = (self.remaining > 0) & ~done[:, None]
        self.remaining[active] -= 1
        self.remaining[done] = 0
        self.z[done] = 0
        return self.due.copy()


@dataclass
class PendingOption:
    history: list
    decision_index: int
    role: int
    z: int
    old_logp: float
    old_value: float
    reward_sum: float = 0.0
    duration: int = 0
    started_update: int = 0

    def add_step(self, reward: float):
        self.reward_sum += float(reward)
        self.duration += 1

    def finish(self, bootstrap_value: float, terminal: bool):
        if self.duration <= 0:
            raise RuntimeError("Cannot complete an option without an executed action")
        return {
            "history": self.history,
            "decision_index": self.decision_index,
            "role": self.role,
            "z": self.z,
            "old_logp": self.old_logp,
            "old_value": self.old_value,
            "reward": self.reward_sum / self.duration,
            "duration": self.duration,
            "bootstrap_value": 0.0 if terminal else float(bootstrap_value),
            "terminal": bool(terminal),
        }


def high_level_gae(records_by_stream, gamma: float, gae_lambda: float):
    for records in records_by_stream:
        carry = 0.0
        for i in reversed(range(len(records))):
            record = records[i]
            alive = 0.0 if record["terminal"] else 1.0
            next_value = (float(records[i + 1]["old_value"])
                          if alive and i + 1 < len(records)
                          else float(record["bootstrap_value"]))
            delta = float(record["reward"]) + gamma * next_value * alive - float(record["old_value"])
            carry = delta + gamma * gae_lambda * alive * carry
            record["advantage"] = carry
            record["return"] = carry + float(record["old_value"])
    return records_by_stream

"""Central one-pass actor and objective-specific legacy critics for CoMeDi."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

OBS_DIM = 866
STATE_DIM = 863
SPATIAL_SHAPE = (26, 7, 7)
ACTION_DIM = 6
NUM_ROLES = 3
ACTOR_PARAMETER_COUNT = 119_346


class JointActor(nn.Module):
    """Encode the canonical team view and public state once for all 3 chefs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(26, 32, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5 + STATE_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, NUM_ROLES * ACTION_DIM)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.conv.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.conv.bias)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != SPATIAL_SHAPE:
            raise ValueError("canonical spatial input must have shape [B, 26, 7, 7]")
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError("public team state must have shape [B, 863]")
        if spatial.shape[0] != state.shape[0]:
            raise ValueError("spatial and state batch dimensions must match")
        visual = torch.relu(self.conv(spatial)).flatten(start_dim=1)
        features = torch.cat((visual, state), dim=-1)
        hidden = torch.relu(self.fc2(torch.relu(self.fc1(features))))
        return self.head(hidden).reshape(-1, NUM_ROLES, ACTION_DIM)

    def distribution(self, spatial: torch.Tensor, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self(spatial, state))


def build_actor(actor_state: Mapping[str, torch.Tensor] | None = None) -> JointActor:
    actor = JointActor()
    if actor_state is not None:
        actor.load_state_dict(actor_state, strict=True)
    return actor


class SpatialVectorEncoder(nn.Module):
    """Old CoMeDi critic representation, kept independently per role/objective."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(26, 32, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(32 * 5 * 5 + OBS_DIM, 64)
        self.fc2 = nn.Linear(64, 64)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.conv.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.conv.bias)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or tuple(spatial.shape[1:]) != SPATIAL_SHAPE:
            raise ValueError("critic spatial input must have shape [B, 26, 7, 7]")
        if vector.ndim != 2 or vector.shape[1] != OBS_DIM:
            raise ValueError("critic vector input must have shape [B, 866]")
        if spatial.shape[0] != vector.shape[0]:
            raise ValueError("critic spatial and vector batch dimensions must match")
        visual = torch.relu(self.conv(spatial)).flatten(start_dim=1)
        return torch.relu(self.fc2(torch.relu(self.fc1(torch.cat((visual, vector), dim=-1)))))


class ConvCritic(nn.Module):
    """One native per-role state-value estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SpatialVectorEncoder()
        self.head = nn.Linear(64, 1)

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spatial, vector)).squeeze(-1)


class ComediPolicy(nn.Module):
    """One trainable policy bundle; prior actors remain separate CPU modules."""

    def __init__(
        self,
        num_priors: int,
        *,
        lr: float = 3e-4,
        actor_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if num_priors < 0:
            raise ValueError("num_priors cannot be negative")
        self.num_priors = int(num_priors)
        self.actor = build_actor(actor_state)
        self.sp_critics = nn.ModuleList(ConvCritic() for _ in range(NUM_ROLES))
        self.mp_critics = nn.ModuleList(ConvCritic() for _ in range(NUM_ROLES))
        self.xp_critics = nn.ModuleDict({
            self.xp_key(prior, role): ConvCritic()
            for prior in range(self.num_priors)
            for role in range(NUM_ROLES)
        })
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizers: dict[str, torch.optim.Optimizer] = {}
        for role in range(NUM_ROLES):
            self.critic_optimizers[f"sp_r{role}"] = torch.optim.Adam(
                self.sp_critics[role].parameters(), lr=lr, eps=1e-5
            )
            self.critic_optimizers[f"mp_r{role}"] = torch.optim.Adam(
                self.mp_critics[role].parameters(), lr=lr, eps=1e-5
            )
        for prior in range(self.num_priors):
            for role in range(NUM_ROLES):
                key = self.xp_key(prior, role)
                self.critic_optimizers[key] = torch.optim.Adam(
                    self.xp_critics[key].parameters(), lr=lr, eps=1e-5
                )

    @staticmethod
    def xp_key(prior: int, role: int) -> str:
        return f"p{int(prior)}_r{int(role)}"

    def critic(self, phase: str, role: int, prior: int | None = None) -> ConvCritic:
        if not 0 <= int(role) < NUM_ROLES:
            raise ValueError("role must be in [0, 2]")
        if phase == "sp":
            return self.sp_critics[int(role)]
        if phase == "mp":
            return self.mp_critics[int(role)]
        if phase == "xp":
            if prior is None or not 0 <= int(prior) < self.num_priors:
                raise ValueError("XP critic requires a valid prior index")
            return self.xp_critics[self.xp_key(prior, role)]
        raise ValueError(f"unknown objective phase: {phase}")

    def critic_optimizer(
        self, phase: str, role: int, prior: int | None = None
    ) -> torch.optim.Optimizer:
        key = self.xp_key(prior, role) if phase == "xp" and prior is not None else f"{phase}_r{role}"
        try:
            return self.critic_optimizers[key]
        except KeyError as error:
            raise ValueError(f"no critic optimizer for {phase=} {role=} {prior=}") from error

    def freeze_actor(self) -> None:
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def checkpoint(self) -> dict[str, object]:
        return {
            "num_priors": self.num_priors,
            "model": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": {
                key: optimizer.state_dict() for key, optimizer in self.critic_optimizers.items()
            },
        }

    def load_checkpoint(self, payload: Mapping[str, object]) -> None:
        if int(payload["num_priors"]) != self.num_priors:
            raise ValueError("checkpoint prior count differs from policy")
        self.load_state_dict(payload["model"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        saved = payload["critic_optimizers"]
        if set(saved) != set(self.critic_optimizers):
            raise ValueError("checkpoint critic optimizer layout differs from policy")
        for key, optimizer in self.critic_optimizers.items():
            optimizer.load_state_dict(saved[key])

    def parameter_counts(self) -> dict[str, int]:
        actor = sum(parameter.numel() for parameter in self.actor.parameters())
        critics = sum(parameter.numel() for module in (
            *self.sp_critics, *self.mp_critics, *self.xp_critics.values()
        ) for parameter in module.parameters())
        return {"actor": actor, "critics": critics, "total": actor + critics}


def select_most_compatible_prior(scores: Mapping[int, Sequence[float]]) -> int:
    """Choose the prior with the largest mean score observed by this round's probe."""
    populated = {
        int(prior): float(np.mean(values))
        for prior, values in scores.items()
        if len(values) > 0
    }
    if not populated:
        raise ValueError("cannot select an XP prior without observed returns")
    return max(populated, key=lambda prior: (populated[prior], -prior))


def joint_action_batch(actions: np.ndarray, num_envs: int) -> np.ndarray:
    array = np.asarray(actions)
    if array.shape != (int(num_envs), NUM_ROLES):
        raise ValueError(f"actions must have shape [{num_envs}, {NUM_ROLES}]")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("actions must contain integer action IDs")
    if np.any(array < 0) or np.any(array >= ACTION_DIM):
        raise ValueError(f"action IDs must be in [0, {ACTION_DIM - 1}]")
    return array.astype(np.int64, copy=False)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    bootstrap = np.asarray(bootstrap, dtype=np.float32)
    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must share [time, env] shape")
    if bootstrap.shape != (rewards.shape[1],):
        raise ValueError("bootstrap must have shape [env]")
    if valid is None:
        valid = np.ones_like(dones, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != rewards.shape:
            raise ValueError("valid mask must have shape [time, env]")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for time_index in reversed(range(rewards.shape[0])):
        next_value = bootstrap if time_index == rewards.shape[0] - 1 else values[time_index + 1]
        alive = (~dones[time_index]).astype(np.float32)
        delta = rewards[time_index] + gamma * next_value * alive - values[time_index]
        carry = (delta + gamma * gae_lambda * alive * carry) * valid[time_index]
        advantages[time_index] = carry
    return advantages, advantages + values


def ppo_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float,
    weight: float = 1.0,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """Apply signed weight to advantages before clipping; entropy stays positive."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    signed_advantages = float(weight) * advantages
    loss = -torch.minimum(ratio * signed_advantages, clipped_ratio * signed_advantages).mean()
    if entropy is not None:
        loss = loss - float(entropy_coef) * entropy.mean()
    return loss


def masked_rows(values: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor):
    if torch.is_tensor(values):
        valid = torch.as_tensor(mask, dtype=torch.bool, device=values.device).reshape(-1)
        if values.shape[0] != valid.numel():
            raise ValueError("mask length must match leading tensor dimension")
        return values[valid]
    array = np.asarray(values)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    if array.shape[0] != valid.size:
        raise ValueError("mask length must match leading array dimension")
    return array[valid]

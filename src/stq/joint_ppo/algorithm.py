"""Joint-action PPO, GAE, and behavior-cloning loss helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .models import ACTION_DIM, NUM_AGENTS


def _check_joint_shapes(logits: torch.Tensor, actions: torch.Tensor | None = None) -> None:
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (NUM_AGENTS, ACTION_DIM):
        raise ValueError("joint logits must have shape [batch, 3, 6]")
    if actions is not None:
        if actions.ndim != 2 or tuple(actions.shape) != tuple(logits.shape[:2]):
            raise ValueError("joint actions must have shape [batch, 3]")
        if torch.any((actions < 0) | (actions >= ACTION_DIM)):
            raise ValueError("joint actions must be in [0, 5]")


def joint_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return the factored joint log probability, one scalar per team frame."""
    _check_joint_shapes(logits, actions)
    return Categorical(logits=logits).log_prob(actions).sum(dim=-1)


def joint_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Sum the three independent categorical entropies per team frame."""
    _check_joint_shapes(logits)
    return Categorical(logits=logits).entropy().sum(dim=-1)


def joint_bc_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Mean label-wise cross entropy while keeping each frame's 3 labels grouped."""
    _check_joint_shapes(logits, actions)
    return F.cross_entropy(logits.reshape(-1, ACTION_DIM), actions.reshape(-1))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_values: np.ndarray,
    *,
    gamma: float = 0.9985,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute team-scalar GAE over [time, environment] arrays."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)
    if rewards.ndim != 2 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must have matching [time, env] shapes")
    if bootstrap_values.shape != rewards.shape[1:]:
        raise ValueError("bootstrap values must have shape [env]")
    if not all(np.isfinite(value).all() for value in (rewards, values, bootstrap_values)):
        raise FloatingPointError("GAE inputs must be finite")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not np.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0, 1]")

    advantages = np.zeros_like(values, dtype=np.float32)
    next_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for timestep in reversed(range(rewards.shape[0])):
        next_value = bootstrap_values if timestep == rewards.shape[0] - 1 else values[timestep + 1]
        nonterminal = (~dones[timestep]).astype(np.float32)
        delta = rewards[timestep] + gamma * next_value * nonterminal - values[timestep]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[timestep] = next_advantage
    returns = advantages + values
    return advantages, returns


def _assert_finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{label} contains non-finite values")


def ppo_update(
    policy: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    *,
    device: torch.device | str,
    rng: np.random.Generator,
    epochs: int = 2,
    minibatches: int = 32,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> dict[str, float | int]:
    """Apply standard joint-action clipped PPO and independent value regression."""
    required = ("spatial", "state", "actions", "old_log_probs", "advantages", "returns")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"PPO batch missing fields: {', '.join(missing)}")
    arrays = {key: np.asarray(batch[key]) for key in required}
    count = arrays["actions"].shape[0]
    if arrays["spatial"].shape != (count, 26, 7, 7):
        raise ValueError("PPO spatial rows must have shape [frames, 26, 7, 7]")
    if arrays["state"].shape != (count, 863):
        raise ValueError("PPO state rows must have shape [frames, 863]")
    if arrays["actions"].shape != (count, NUM_AGENTS):
        raise ValueError("PPO actions must have shape [frames, 3]")
    if not np.issubdtype(arrays["actions"].dtype, np.integer):
        raise TypeError("PPO joint actions must use an integer dtype")
    for key in ("old_log_probs", "advantages", "returns"):
        if arrays[key].shape != (count,):
            raise ValueError(f"PPO {key} must have shape [frames]")
    if count < 1:
        raise ValueError("PPO batch must contain at least one joint frame")
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "b"):
        raise FloatingPointError("PPO batch contains non-finite values")
    if epochs < 1 or minibatches < 1:
        raise ValueError("epochs and minibatches must be positive")
    if not 0.0 < clip < 1.0 or entropy_coef < 0.0 or value_coef < 0.0 or max_grad_norm <= 0.0:
        raise ValueError("invalid PPO coefficient")

    # Keep the rollout buffer on host memory. Only the current joint minibatch
    # is transferred to the GPU, which leaves room for two learners per card.
    normalized_advantages = arrays["advantages"].astype(np.float32, copy=False)
    normalized_advantages = (
        normalized_advantages - normalized_advantages.mean()
    ) / (normalized_advantages.std() + 1e-8)

    actor_losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    value_losses: list[float] = []
    updates = 0
    batch_count = min(int(minibatches), count)
    for _ in range(int(epochs)):
        permutation = rng.permutation(count)
        for indices in np.array_split(permutation, batch_count):
            if not len(indices):
                continue
            spatial = torch.as_tensor(arrays["spatial"][indices], dtype=torch.float32, device=device)
            state = torch.as_tensor(arrays["state"][indices], dtype=torch.float32, device=device)
            actions = torch.as_tensor(arrays["actions"][indices], dtype=torch.long, device=device)
            old_log_probs = torch.as_tensor(arrays["old_log_probs"][indices], dtype=torch.float32, device=device)
            advantages = torch.as_tensor(normalized_advantages[indices], dtype=torch.float32, device=device)
            returns = torch.as_tensor(arrays["returns"][indices], dtype=torch.float32, device=device)
            logits = policy.actor(spatial, state)
            new_log_probs = joint_log_prob(logits, actions)
            entropy = joint_entropy(logits)
            ratio = torch.exp(new_log_probs - old_log_probs)
            objective = torch.minimum(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages,
            )
            policy_loss = -objective.mean()
            entropy_mean = entropy.mean()
            actor_loss = policy_loss - entropy_coef * entropy_mean
            _assert_finite_tensor(actor_loss, "PPO actor loss")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            actor_optimizer.step()

            # This optimizer sees only the independent critic; value loss cannot
            # change any actor-trunk parameter.
            predicted_values = policy.critic(state)
            value_loss = value_coef * 0.5 * F.mse_loss(
                predicted_values, returns
            )
            _assert_finite_tensor(value_loss, "PPO critic loss")
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.critic.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            critic_optimizer.step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            entropies.append(float(entropy_mean.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            updates += 1

    for name, module in (("actor", policy.actor), ("critic", policy.critic)):
        for parameter in module.parameters():
            _assert_finite_tensor(parameter, f"updated {name} parameter")
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "entropy": float(np.mean(entropies)),
        "value_loss": float(np.mean(value_losses)),
        "optimizer_updates": updates,
        "joint_frames": int(count),
        "joint_minibatch_size_mean": float(count / batch_count),
    }

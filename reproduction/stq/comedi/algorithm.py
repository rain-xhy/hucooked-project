"""Joint-rollout collection and independent-critic PPO updates for CoMeDi."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from .model import (
    ACTION_DIM,
    NUM_ROLES,
    OBS_DIM,
    STATE_DIM,
    SPATIAL_SHAPE,
    ComediPolicy,
    compute_gae,
    joint_action_batch,
    ppo_actor_loss,
)


# All nontrivial three-role assignments cover both 1-current:2-prior and
# 2-current:1-prior cross-play directions.
CROSSPLAY_ROLE_MASKS = np.asarray(
    ((1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    dtype=bool,
)


def _observation_tensors(observation, device):
    vectors = torch.as_tensor(observation["vector"], dtype=torch.float32, device=device)
    spatials = torch.as_tensor(observation["spatial"], dtype=torch.float32, device=device)
    states = np.asarray(observation["state"])
    if vectors.ndim != 3 or tuple(vectors.shape[1:]) != (NUM_ROLES, OBS_DIM):
        raise ValueError("vector observations must have shape [env, 3, 866]")
    if spatials.ndim != 5 or tuple(spatials.shape[1:]) != (NUM_ROLES, *SPATIAL_SHAPE):
        raise ValueError("spatial observations must have shape [env, 3, 26, 7, 7]")
    if spatials.shape[0] != vectors.shape[0]:
        raise ValueError("vector and spatial environment counts must match")
    if states.shape != (vectors.shape[0], STATE_DIM):
        raise ValueError("centralized state observations must have shape [env, 863]")
    return spatials, vectors


@torch.no_grad()
def sample_joint_actions(actor, observation, device, *, deterministic=False):
    """Sample one simultaneous integer action for each of the three roles."""
    spatial, vector = _observation_tensors(observation, device)
    num_envs = vector.shape[0]
    distribution = actor.distribution(
        spatial.reshape(num_envs * NUM_ROLES, *SPATIAL_SHAPE),
        vector.reshape(num_envs * NUM_ROLES, OBS_DIM),
    )
    actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
    log_probs = distribution.log_prob(actions)
    actions = actions.reshape(num_envs, NUM_ROLES).cpu().numpy().astype(np.int64)
    log_probs = log_probs.reshape(num_envs, NUM_ROLES).cpu().numpy().astype(np.float32)
    return joint_action_batch(actions, num_envs), log_probs


@torch.no_grad()
def _critic_grid(policy, phase, observation, device, *, prior=None):
    spatial, vector = _observation_tensors(observation, device)
    values = []
    for role in range(NUM_ROLES):
        critic = policy.critic(phase, role, prior)
        values.append(critic(spatial[:, role], vector[:, role]).cpu().numpy())
    return np.stack(values, axis=-1).astype(np.float32)


def _episode_rows(infos):
    return [info["episode_summary"] for info in infos if "episode_summary" in info]


def _bootstrap_gae(rewards, values, dones, bootstrap, gamma, gae_lambda, valid=None):
    """Compute one GAE stream per role from time/env arrays."""
    advantages = np.zeros_like(values, dtype=np.float32)
    returns = np.zeros_like(values, dtype=np.float32)
    for role in range(NUM_ROLES):
        advantages[:, :, role], returns[:, :, role] = compute_gae(
            rewards,
            values[:, :, role],
            dones,
            bootstrap[:, role],
            gamma=gamma,
            gae_lambda=gae_lambda,
            valid=valid,
        )
    return advantages, returns


def _flatten_joint(spatials, vectors, actions, log_probs, advantages, returns, valid):
    time_steps, num_envs = actions.shape[:2]
    flat_spatial = spatials.reshape(-1, *SPATIAL_SHAPE)
    flat_vector = vectors.reshape(-1, OBS_DIM)
    flat_valid = np.repeat(valid.reshape(-1), NUM_ROLES)
    return {
        "spatial": flat_spatial,
        "vector": flat_vector,
        "actions": actions.reshape(-1),
        "log_probs": log_probs.reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "roles": np.tile(np.arange(NUM_ROLES, dtype=np.int64), time_steps * num_envs),
        "valid": flat_valid,
    }


def collect_self_play(
    env,
    policy: ComediPolicy,
    vector_steps: int,
    *,
    device,
    gamma: float,
    gae_lambda: float,
    should_stop: Callable[[], bool] | None = None,
):
    """Collect shared-reward on-policy SP transitions."""
    observation = env.reset()
    num_envs = env.num_envs
    spatial_rows, vector_rows = [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    reward_rows, done_rows = [], []
    summaries = []
    actual_steps = 0

    for _ in range(int(vector_steps)):
        if should_stop is not None and should_stop():
            break
        spatial, vector = _observation_tensors(observation, device)
        actions, log_probs = sample_joint_actions(policy.actor, observation, device)
        values = _critic_grid(policy, "sp", observation, device)

        next_observation, rewards, dones, infos = env.step(actions)
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vector.cpu().numpy())
        action_rows.append(actions)
        log_prob_rows.append(log_probs)
        value_rows.append(values)
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        summaries.extend(_episode_rows(infos))
        observation = next_observation
        actual_steps += num_envs

    if actual_steps == 0:
        return None, observation, {"env_steps": 0, "episode_summaries": []}

    spatials = np.asarray(spatial_rows, dtype=np.float32)
    vectors = np.asarray(vector_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.int64)
    log_probs = np.asarray(log_prob_rows, dtype=np.float32)
    values = np.asarray(value_rows, dtype=np.float32)
    rewards = np.asarray(reward_rows, dtype=np.float32)
    dones = np.asarray(done_rows, dtype=bool)
    bootstrap = _critic_grid(policy, "sp", observation, device)
    bootstrap[dones[-1]] = 0.0
    advantages, returns = _bootstrap_gae(
        rewards, values, dones, bootstrap, gamma, gae_lambda
    )
    batch = _flatten_joint(
        spatials, vectors, actions, log_probs, advantages, returns,
        np.ones(actions.shape[:2], dtype=bool),
    )
    return batch, observation, {"env_steps": actual_steps, "episode_summaries": summaries}


@torch.no_grad()
def collect_teacher_self_play(
    env,
    teachers: Sequence[ComediPolicy],
    vector_steps: int,
    *,
    device,
    start_teacher: int = 0,
    should_stop: Callable[[], bool] | None = None,
):
    """Collect each frozen teacher's own three-role self-play BC samples.

    A teacher ID stays assigned to an environment until its episode ends, so
    each teacher's states and action labels come from its own self-play team.
    Environment slots and completed episodes rotate through the teacher pool.
    """
    if not teachers:
        raise ValueError("teacher self-play requires at least one frozen teacher")
    if int(vector_steps) < 0:
        raise ValueError("vector_steps cannot be negative")

    observation = env.reset()
    num_envs = int(env.num_envs)
    teacher_count = len(teachers)
    teacher_ids = (np.arange(num_envs, dtype=np.int64) + int(start_teacher)) % teacher_count
    next_teacher = (int(start_teacher) + num_envs) % teacher_count
    spatial_rows = [[] for _ in teachers]
    vector_rows = [[] for _ in teachers]
    action_rows = [[] for _ in teachers]
    episode_summaries = [[] for _ in teachers]
    teacher_env_steps = np.zeros(teacher_count, dtype=np.int64)
    teacher_episodes = np.zeros(teacher_count, dtype=np.int64)
    actual_steps = 0

    for _ in range(int(vector_steps)):
        if should_stop is not None and should_stop():
            break
        spatial, vector = _observation_tensors(observation, device)
        spatial_np = spatial.cpu().numpy()
        vector_np = vector.cpu().numpy()
        actions = np.empty((num_envs, NUM_ROLES), dtype=np.int64)
        for teacher_id, teacher in enumerate(teachers):
            env_ids = np.flatnonzero(teacher_ids == teacher_id)
            if not len(env_ids):
                continue
            teacher_observation = {
                key: np.asarray(value)[env_ids] for key, value in observation.items()
            }
            teacher_actions, _ = sample_joint_actions(
                teacher.actor, teacher_observation, device
            )
            actions[env_ids] = teacher_actions
            spatial_rows[teacher_id].append(spatial_np[env_ids])
            vector_rows[teacher_id].append(vector_np[env_ids])
            action_rows[teacher_id].append(teacher_actions)
            teacher_env_steps[teacher_id] += len(env_ids)

        next_observation, _, dones, infos = env.step(joint_action_batch(actions, num_envs))
        for teacher_id in range(teacher_count):
            env_ids = np.flatnonzero((teacher_ids == teacher_id) & np.asarray(dones, dtype=bool))
            if len(env_ids):
                teacher_episodes[teacher_id] += len(env_ids)
                episode_summaries[teacher_id].extend(
                    info["episode_summary"]
                    for info in (infos[int(index)] for index in env_ids)
                    if "episode_summary" in info
                )
        for env_id in np.flatnonzero(dones):
            teacher_ids[env_id] = next_teacher
            next_teacher = (next_teacher + 1) % teacher_count
        observation = next_observation
        actual_steps += num_envs

    batches = {}
    for teacher_id in range(teacher_count):
        if not teacher_env_steps[teacher_id]:
            continue
        spatial = np.concatenate(spatial_rows[teacher_id], axis=0)
        vector = np.concatenate(vector_rows[teacher_id], axis=0)
        actions = np.concatenate(action_rows[teacher_id], axis=0)
        flat_actions = actions.reshape(-1).astype(np.int64, copy=False)
        batches[teacher_id] = {
            "spatial": spatial.reshape(-1, *SPATIAL_SHAPE),
            "vector": vector.reshape(-1, OBS_DIM),
            "actions": flat_actions,
            "bc_actions": flat_actions.copy(),
            "valid": np.ones(flat_actions.size, dtype=bool),
        }

    return batches, {
        "env_steps": actual_steps,
        "teacher_env_steps": {
            str(teacher_id): int(teacher_env_steps[teacher_id])
            for teacher_id in range(teacher_count)
        },
        "teacher_episode_counts": {
            str(teacher_id): int(teacher_episodes[teacher_id])
            for teacher_id in range(teacher_count)
        },
        "episode_summaries_by_teacher": {
            str(teacher_id): episode_summaries[teacher_id]
            for teacher_id in range(teacher_count)
        },
        "next_teacher_cursor": int(next_teacher),
    }


def collect_xp_probe(
    env,
    current: ComediPolicy,
    conventions: Sequence[ComediPolicy],
    vector_steps: int,
    *,
    device,
    combo_cursor: int = 0,
    should_stop: Callable[[], bool] | None = None,
):
    """Measure cross-play rewards across old partners and all current roles."""
    if not conventions:
        raise ValueError("XP selection requires at least one old convention")
    observation = env.reset()
    num_envs = env.num_envs
    combinations = [
        (prior, mask_index)
        for prior in range(len(conventions))
        for mask_index in range(len(CROSSPLAY_ROLE_MASKS))
    ]
    combo_ids = (int(combo_cursor) + np.arange(num_envs)) % len(combinations)
    cursor = (int(combo_cursor) + num_envs) % len(combinations)
    scores: dict[int, list[float]] = defaultdict(list)
    role_scores: dict[str, list[float]] = defaultdict(list)
    actual_steps = 0

    for _ in range(int(vector_steps)):
        if should_stop is not None and should_stop():
            break
        mask_ids = np.asarray([combinations[index][1] for index in combo_ids], dtype=np.int64)
        current_masks = CROSSPLAY_ROLE_MASKS[mask_ids]
        priors = np.asarray([combinations[index][0] for index in combo_ids], dtype=np.int64)
        current_actions, _ = sample_joint_actions(current.actor, observation, device)
        actions = np.empty((num_envs, NUM_ROLES), dtype=np.int64)
        for prior in np.unique(priors):
            env_ids = np.flatnonzero(priors == prior)
            old_actions, _ = sample_joint_actions(
                conventions[int(prior)].actor,
                {key: value[env_ids] for key, value in observation.items()},
                device,
            )
            actions[env_ids] = old_actions
        actions[current_masks] = current_actions[current_masks]
        next_observation, rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        for env_index, reward in enumerate(rewards):
            prior, mask_index = combinations[combo_ids[env_index]]
            scores[prior].append(float(reward))
            role_scores[f"{prior}:mask_{mask_index}"].append(float(reward))
        done_ids = np.flatnonzero(dones)
        for env_index in done_ids:
            combo_ids[env_index] = cursor
            cursor = (cursor + 1) % len(combinations)
        observation = next_observation
        actual_steps += num_envs

    if actual_steps:
        # Probe chunks start from fresh episodes in the caller. Advance through
        # the six role masks even when these short samples do not reach a terminal.
        cursor = (int(combo_cursor) + actual_steps) % len(combinations)
    return {
        "scores": dict(scores),
        "role_scores": dict(role_scores),
        "combo_cursor": cursor,
        "env_steps": actual_steps,
    }


def collect_cross_play(
    env,
    current: ComediPolicy,
    prior_policy: ComediPolicy,
    prior_id: int,
    vector_steps: int,
    *,
    device,
    gamma: float,
    gae_lambda: float,
    start_config: int = 0,
    should_stop: Callable[[], bool] | None = None,
):
    """Collect current-versus-fixed-prior XP transitions with role rotation."""
    observation = env.reset()
    num_envs = env.num_envs
    mask_ids = (start_config + np.arange(num_envs)) % len(CROSSPLAY_ROLE_MASKS)
    cursor = int(start_config + num_envs) % len(CROSSPLAY_ROLE_MASKS)
    spatial_rows, vector_rows = [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    role_mask_rows, reward_rows, done_rows = [], [], []
    summaries = []
    actual_steps = 0

    for _ in range(int(vector_steps)):
        if should_stop is not None and should_stop():
            break
        spatial, vector = _observation_tensors(observation, device)
        current_actions, current_log_probs = sample_joint_actions(current.actor, observation, device)
        prior_actions, _ = sample_joint_actions(prior_policy.actor, observation, device)
        current_masks = CROSSPLAY_ROLE_MASKS[mask_ids]
        actions = prior_actions.copy()
        actions[current_masks] = current_actions[current_masks]
        values_grid = _critic_grid(current, "xp", observation, device, prior=prior_id)

        next_observation, rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vector.cpu().numpy())
        action_rows.append(current_actions)
        log_prob_rows.append(current_log_probs)
        value_rows.append(values_grid)
        role_mask_rows.append(current_masks.copy())
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        summaries.extend(_episode_rows(infos))
        for env_index in np.flatnonzero(dones):
            mask_ids[env_index] = cursor
            cursor = (cursor + 1) % len(CROSSPLAY_ROLE_MASKS)
        observation = next_observation
        actual_steps += num_envs

    if actual_steps == 0:
        return None, observation, {"env_steps": 0, "episode_summaries": []}
    spatial_array = np.asarray(spatial_rows, dtype=np.float32)
    vector_array = np.asarray(vector_rows, dtype=np.float32)
    actions_array = np.asarray(action_rows, dtype=np.int64)
    log_probs_array = np.asarray(log_prob_rows, dtype=np.float32)
    values_array = np.asarray(value_rows, dtype=np.float32)
    current_masks_array = np.asarray(role_mask_rows, dtype=bool)
    rewards_array = np.asarray(reward_rows, dtype=np.float32)
    dones_array = np.asarray(done_rows, dtype=bool)
    bootstrap_grid = _critic_grid(current, "xp", observation, device, prior=prior_id)
    bootstrap_grid[dones_array[-1]] = 0.0
    advantages = np.zeros_like(values_array, dtype=np.float32)
    returns = np.zeros_like(values_array, dtype=np.float32)
    for role in range(NUM_ROLES):
        advantages[:, :, role], returns[:, :, role] = compute_gae(
            rewards_array,
            values_array[:, :, role],
            dones_array,
            bootstrap_grid[:, role],
            gamma=gamma,
            gae_lambda=gae_lambda,
            valid=current_masks_array[:, :, role],
        )
    batch = _flatten_joint(
        spatial_array, vector_array, actions_array, log_probs_array,
        advantages, returns, current_masks_array.any(axis=-1),
    )
    # The flatten helper uses one validity bit per joint step. XP has a
    # role-specific learner mask, so restore its exact per-role mask.
    batch["valid"] = current_masks_array.reshape(-1)
    batch["priors"] = np.full(actions_array.size, int(prior_id), dtype=np.int64)
    return batch, observation, {"env_steps": actual_steps, "episode_summaries": summaries}


def collect_mixed_play(
    env,
    current: ComediPolicy,
    prior_policy: ComediPolicy,
    vector_steps: int,
    *,
    device,
    gamma: float,
    gae_lambda: float,
    rng: np.random.Generator,
    prefix_max: int,
    current_probability: float,
    should_stop: Callable[[], bool] | None = None,
):
    """Generate old/new random prefixes, then train only on all-current suffixes."""
    observation = env.reset()
    num_envs = env.num_envs
    effective_prefix_max = max(1, min(int(prefix_max), max(1, int(vector_steps) // 2)))
    prefix_left = rng.integers(1, effective_prefix_max + 1, size=num_envs, dtype=np.int64)
    spatial_rows, vector_rows = [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    reward_rows, done_rows, suffix_rows = [], [], []
    summaries = []
    actual_steps = 0

    for _ in range(int(vector_steps)):
        if should_stop is not None and should_stop():
            break
        spatial, vector = _observation_tensors(observation, device)
        current_actions, current_log_probs = sample_joint_actions(current.actor, observation, device)
        prior_actions, _ = sample_joint_actions(prior_policy.actor, observation, device)
        suffix = prefix_left <= 0
        actions = prior_actions.copy()
        for env_index in range(num_envs):
            if suffix[env_index]:
                actions[env_index] = current_actions[env_index]
            else:
                use_current = rng.random(NUM_ROLES) < current_probability
                actions[env_index, use_current] = current_actions[env_index, use_current]

        value_grid = np.zeros((num_envs, NUM_ROLES), dtype=np.float32)
        if np.any(suffix):
            for role in range(NUM_ROLES):
                selected = np.flatnonzero(suffix)
                value_grid[selected, role] = current.mp_critics[role](
                    spatial[selected, role], vector[selected, role]
                ).detach().cpu().numpy()

        next_observation, rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vector.cpu().numpy())
        action_rows.append(current_actions)
        log_prob_rows.append(current_log_probs)
        value_rows.append(value_grid)
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        suffix_rows.append(suffix.copy())
        summaries.extend(_episode_rows(infos))

        prefix_left[~suffix] -= 1
        for env_index in np.flatnonzero(dones):
            prefix_left[env_index] = int(rng.integers(1, effective_prefix_max + 1))
        observation = next_observation
        actual_steps += num_envs

    if actual_steps == 0:
        return None, observation, {
            "env_steps": 0, "prefix_env_steps": 0, "suffix_env_steps": 0,
            "episode_summaries": [],
        }
    spatials = np.asarray(spatial_rows, dtype=np.float32)
    vectors = np.asarray(vector_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.int64)
    log_probs = np.asarray(log_prob_rows, dtype=np.float32)
    values = np.asarray(value_rows, dtype=np.float32)
    rewards = np.asarray(reward_rows, dtype=np.float32)
    dones = np.asarray(done_rows, dtype=bool)
    suffix_mask = np.asarray(suffix_rows, dtype=bool)
    bootstrap_grid = _critic_grid(current, "mp", observation, device)
    bootstrap_grid[dones[-1]] = 0.0
    advantages, returns = _bootstrap_gae(
        rewards, values, dones, bootstrap_grid, gamma, gae_lambda, valid=suffix_mask
    )
    batch = _flatten_joint(
        spatials, vectors, actions, log_probs, advantages, returns,
        suffix_mask,
    )
    suffix_steps = int(suffix_mask.sum())
    return batch, observation, {
        "env_steps": actual_steps,
        "prefix_env_steps": actual_steps - suffix_steps,
        "suffix_env_steps": suffix_steps,
        "episode_summaries": summaries,
    }


def update_bundle(
    policy: ComediPolicy,
    objectives: Sequence[tuple[str, dict[str, np.ndarray], float]],
    *,
    device,
    epochs: int,
    minibatches: int,
    clip: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    prior_count: int,
    bc_coef: float = 0.0,
    bc_objectives: Sequence[tuple[int, dict[str, np.ndarray]]] = (),
    rng: np.random.Generator,
):
    """Jointly optimize on-policy objectives and optional teacher-only BC.

    All actor losses in a round are summed before one actor optimizer step,
    implementing ``L_SP - alpha*L_XP + beta*L_MP``. Optional BC objectives
    contain only teacher self-play states and labels; they update the actor but
    never enter PPO or critic regression.
    """
    prepared = []
    for phase, batch, actor_weight in objectives:
        valid = np.asarray(batch["valid"], dtype=bool)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError(f"{phase} buffer has no trainable transitions")
        for key in ("spatial", "vector", "actions", "log_probs", "advantages", "returns"):
            if not np.isfinite(np.asarray(batch[key])).all():
                raise FloatingPointError(f"{phase} rollout field {key} contains non-finite values")
        data = {
            "phase": phase,
            "batch": batch,
            "indices": indices,
            "weight": float(actor_weight),
            "spatial": torch.as_tensor(batch["spatial"], dtype=torch.float32, device=device),
            "vector": torch.as_tensor(batch["vector"], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(batch["actions"], dtype=torch.long, device=device),
            "log_probs": torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device),
            "returns": torch.as_tensor(batch["returns"], dtype=torch.float32, device=device),
            "roles": torch.as_tensor(batch["roles"], dtype=torch.long, device=device),
            "priors": (
                torch.as_tensor(batch["priors"], dtype=torch.long, device=device)
                if phase == "xp" else None
            ),
        }
        data["normalized_advantages"] = data["advantages"].clone()
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device)
        for role in torch.unique(data["roles"][valid_tensor]):
            role_mask = valid_tensor & (data["roles"] == role)
            role_adv = data["advantages"][role_mask]
            if role_adv.numel() > 1:
                data["normalized_advantages"][role_mask] = (
                    role_adv - role_adv.mean()
                ) / (role_adv.std(unbiased=False) + 1e-8)
        data["valid_tensor"] = valid_tensor
        data["shuffle"] = np.array_split(
            rng.permutation(indices), min(max(1, int(minibatches)), len(indices))
        )
        prepared.append(data)

    prepared_bc = []
    if bc_coef > 0.0:
        for teacher_id, batch in bc_objectives:
            valid = np.asarray(batch["valid"], dtype=bool)
            indices = np.flatnonzero(valid)
            if not len(indices):
                continue
            spatial_array = np.asarray(batch["spatial"])
            vector_array = np.asarray(batch["vector"])
            labels_array = np.asarray(batch["bc_actions"], dtype=np.int64).reshape(-1)
            if spatial_array.shape[0] != len(valid) or vector_array.shape[0] != len(valid):
                raise ValueError(f"teacher {teacher_id} BC state rows do not match its valid mask")
            if labels_array.size != len(valid):
                raise ValueError(f"teacher {teacher_id} BC labels do not match its state rows")
            if not np.isfinite(spatial_array[valid]).all() or not np.isfinite(vector_array[valid]).all():
                raise FloatingPointError(f"teacher {teacher_id} BC states contain non-finite values")
            prepared_bc.append({
                "teacher_id": int(teacher_id),
                "indices": indices,
                "spatial": torch.as_tensor(spatial_array, dtype=torch.float32, device=device),
                "vector": torch.as_tensor(vector_array, dtype=torch.float32, device=device),
                "labels": torch.as_tensor(labels_array, dtype=torch.long, device=device),
                "shuffle": np.array_split(
                    rng.permutation(indices), min(max(1, int(minibatches)), len(indices))
                ),
            })

    actor_losses: list[float] = []
    value_losses: dict[str, list[float]] = defaultdict(list)
    bc_losses: list[float] = []
    updates = max(len(data["shuffle"]) for data in prepared + prepared_bc)
    bc_teacher_count = len(prepared_bc)
    for epoch in range(int(epochs)):
        for minibatch_index in range(updates):
            combined_actor_loss = None
            critic_work = []
            for data in prepared:
                chunks = data["shuffle"]
                if minibatch_index >= len(chunks) or not len(chunks[minibatch_index]):
                    continue
                chunk = chunks[minibatch_index]
                idx = torch.as_tensor(chunk, dtype=torch.long, device=device)
                distribution = policy.actor.distribution(data["spatial"][idx], data["vector"][idx])
                policy_loss = ppo_actor_loss(
                    distribution.log_prob(data["actions"][idx]),
                    data["log_probs"][idx],
                    data["normalized_advantages"][idx],
                    clip=clip,
                    weight=data["weight"],
                    entropy=distribution.entropy(),
                    entropy_coef=entropy_coef,
                )
                combined_actor_loss = (
                    policy_loss if combined_actor_loss is None
                    else combined_actor_loss + policy_loss
                )
                actor_losses.append(float(policy_loss.detach().cpu()))
                critic_work.append((data, idx, chunk))

            if bc_teacher_count:
                for data in prepared_bc:
                    chunks = data["shuffle"]
                    if minibatch_index >= len(chunks) or not len(chunks[minibatch_index]):
                        continue
                    idx = torch.as_tensor(chunks[minibatch_index], dtype=torch.long, device=device)
                    distribution = policy.actor.distribution(data["spatial"][idx], data["vector"][idx])
                    bc_loss = -distribution.log_prob(data["labels"][idx]).mean()
                    weighted_bc_loss = (float(bc_coef) / bc_teacher_count) * bc_loss
                    combined_actor_loss = (
                        weighted_bc_loss if combined_actor_loss is None
                        else combined_actor_loss + weighted_bc_loss
                    )
                    actor_losses.append(float(weighted_bc_loss.detach().cpu()))
                    bc_losses.append(float(bc_loss.detach().cpu()))

            if combined_actor_loss is None:
                continue
            if not torch.isfinite(combined_actor_loss):
                raise FloatingPointError("joint SP/XP/MP actor loss is non-finite")
            policy.actor_optimizer.zero_grad(set_to_none=True)
            combined_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            policy.actor_optimizer.step()

            # Keep every phase critic independent and train it toward positive
            # environment returns, including the selected XP prior's critics.
            for data, idx, _ in critic_work:
                phase = data["phase"]
                roles = data["roles"][idx]
                priors = data["priors"][idx] if phase == "xp" else None
                for role in range(NUM_ROLES):
                    selected = roles == role
                    prior_ids = range(int(prior_count)) if phase == "xp" else (None,)
                    for prior in prior_ids:
                        group = selected if phase != "xp" else selected & (priors == prior)
                        if not torch.any(group):
                            continue
                        critic = policy.critic(phase, role, prior)
                        optimizer = policy.critic_optimizer(phase, role, prior)
                        values = critic(data["spatial"][idx][group], data["vector"][idx][group])
                        loss = torch.nn.functional.mse_loss(
                            values, data["returns"][idx][group]
                        ) * float(value_coef)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"{phase} role {role} prior {prior} critic loss is non-finite"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            critic.parameters(), max_grad_norm, error_if_nonfinite=True
                        )
                        optimizer.step()
                        value_losses[phase].append(float(loss.detach().cpu()))

    return {
        "samples_by_phase": {data["phase"]: int(len(data["indices"])) for data in prepared},
        "bc_samples_by_teacher": {
            str(data["teacher_id"]): int(len(data["indices"])) for data in prepared_bc
        },
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "value_loss_by_phase": {
            phase: float(np.mean(values)) for phase, values in value_losses.items()
        },
        "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "actor_updates": int(epochs) * updates,
    }

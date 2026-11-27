"""Joint-view rollout collection with role-wise CoMeDi PPO objectives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from .model import (
    ACTION_DIM,
    NUM_ROLES,
    OBS_DIM,
    STATE_DIM,
    SPATIAL_SHAPE,
    ComediPolicy,
    JointActor,
    compute_gae,
    joint_action_batch,
    ppo_actor_loss,
)

CROSSPLAY_ROLE_MASKS = np.asarray(
    ((1, 0, 0), (0, 1, 0), (0, 0, 1),
     (1, 1, 0), (1, 0, 1), (0, 1, 1)),
    dtype=bool,
)


def _actor_of(actor_or_policy):
    return actor_or_policy.actor if hasattr(actor_or_policy, "actor") else actor_or_policy


def _module_device(module, fallback="cpu") -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _observation_tensors(observation, device):
    spatial_array = np.asarray(observation["spatial"])
    vector_array = np.asarray(observation["vector"])
    state_array = np.asarray(observation["state"])
    if spatial_array.ndim != 5 or tuple(spatial_array.shape[1:]) != (NUM_ROLES, *SPATIAL_SHAPE):
        raise ValueError("spatial observations must have shape [env, 3, 26, 7, 7]")
    if vector_array.ndim != 3 or tuple(vector_array.shape[1:]) != (NUM_ROLES, OBS_DIM):
        raise ValueError("vector observations must have shape [env, 3, 866]")
    if state_array.shape != (len(spatial_array), STATE_DIM):
        raise ValueError("public state observations must have shape [env, 863]")
    if vector_array.shape[0] != spatial_array.shape[0]:
        raise ValueError("spatial and vector environment counts must match")
    return (
        torch.as_tensor(spatial_array, dtype=torch.float32, device=device),
        torch.as_tensor(vector_array, dtype=torch.float32, device=device),
        torch.as_tensor(state_array, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def sample_joint_actions(actor_or_policy, observation, device=None, *, deterministic=False):
    """One centralized trunk pass returns actions/log probabilities for 3 roles."""
    actor = _actor_of(actor_or_policy)
    actual_device = _module_device(actor, device or "cpu")
    spatial, _vectors, state = _observation_tensors(observation, actual_device)
    canonical = spatial[:, 0]
    logits = actor(canonical, state)
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (NUM_ROLES, ACTION_DIM):
        raise ValueError("joint actor must emit logits with shape [env, 3, 6]")
    distribution = Categorical(logits=logits)
    actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
    log_probs = distribution.log_prob(actions)
    action_rows = joint_action_batch(actions.cpu().numpy(), len(actions))
    return action_rows, log_probs.cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def _critic_grid(policy: ComediPolicy, phase: str, observation, device, *, prior=None):
    spatial, vectors, _state = _observation_tensors(observation, device)
    values = []
    for role in range(NUM_ROLES):
        values.append(policy.critic(phase, role, prior)(
            spatial[:, role], vectors[:, role]
        ).cpu().numpy())
    return np.stack(values, axis=-1).astype(np.float32, copy=False)


def _episode_rows(infos):
    return [dict(info["episode_summary"]) for info in infos if "episode_summary" in info]


def _bootstrap_gae(rewards, values, dones, bootstrap, gamma, gae_lambda, valid=None):
    advantages = np.zeros_like(values, dtype=np.float32)
    returns = np.zeros_like(values, dtype=np.float32)
    for role in range(NUM_ROLES):
        role_valid = None if valid is None else valid[:, :, role]
        advantages[:, :, role], returns[:, :, role] = compute_gae(
            rewards,
            values[:, :, role],
            dones,
            bootstrap[:, role],
            gamma=gamma,
            gae_lambda=gae_lambda,
            valid=role_valid,
        )
    return advantages, returns


def _flatten_joint(spatials, vectors, states, actions, log_probs, advantages, returns, valid):
    """Flatten only time and environment axes; preserve the three-role axis."""
    time_steps, num_envs = actions.shape[:2]
    frames = time_steps * num_envs
    return {
        "spatial": spatials[:, :, 0].reshape(frames, *SPATIAL_SHAPE),
        "state": states.reshape(frames, STATE_DIM),
        "critic_spatial": spatials.reshape(frames, NUM_ROLES, *SPATIAL_SHAPE),
        "critic_vector": vectors.reshape(frames, NUM_ROLES, OBS_DIM),
        "actions": actions.reshape(frames, NUM_ROLES),
        "log_probs": log_probs.reshape(frames, NUM_ROLES),
        "advantages": advantages.reshape(frames, NUM_ROLES),
        "returns": returns.reshape(frames, NUM_ROLES),
        "valid": valid.reshape(frames, NUM_ROLES),
    }


def _rollout_arrays(spatial_rows, vector_rows, state_rows, action_rows,
                    log_prob_rows, value_rows, reward_rows, done_rows,
                    bootstrap, gamma, gae_lambda, valid=None):
    spatials = np.asarray(spatial_rows, dtype=np.float32)
    vectors = np.asarray(vector_rows, dtype=np.float32)
    states = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.int64)
    log_probs = np.asarray(log_prob_rows, dtype=np.float32)
    values = np.asarray(value_rows, dtype=np.float32)
    rewards = np.asarray(reward_rows, dtype=np.float32)
    dones = np.asarray(done_rows, dtype=bool)
    advantages, returns = _bootstrap_gae(
        rewards, values, dones, bootstrap, gamma, gae_lambda, valid=valid
    )
    frame_valid = np.ones_like(actions, dtype=bool) if valid is None else valid
    return _flatten_joint(
        spatials, vectors, states, actions, log_probs, advantages, returns, frame_valid
    )


def collect_self_play(env, policy: ComediPolicy, vector_steps: int, *, device,
                      gamma: float, gae_lambda: float):
    observation = env.reset()
    num_envs = int(env.num_envs)
    spatial_rows, vector_rows, state_rows = [], [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    reward_rows, done_rows, summaries = [], [], []
    for _ in range(int(vector_steps)):
        spatial, vectors, state = _observation_tensors(observation, device)
        with torch.no_grad():
            logits = policy.actor(spatial[:, 0], state)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
        values = _critic_grid(policy, "sp", observation, device)
        action_array = joint_action_batch(actions.cpu().numpy(), num_envs)
        next_observation, rewards, dones, infos = env.step(action_array)
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vectors.cpu().numpy())
        state_rows.append(state.cpu().numpy())
        action_rows.append(action_array)
        log_prob_rows.append(log_probs.cpu().numpy())
        value_rows.append(values)
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        summaries.extend(_episode_rows(infos))
        observation = next_observation
    if int(vector_steps) <= 0:
        return None, observation, {"env_steps": 0, "episode_summaries": []}
    bootstrap = _critic_grid(policy, "sp", observation, device)
    dones = np.asarray(done_rows, dtype=bool)
    bootstrap[dones[-1]] = 0.0
    batch = _rollout_arrays(
        spatial_rows, vector_rows, state_rows, action_rows, log_prob_rows,
        value_rows, reward_rows, done_rows, bootstrap, gamma, gae_lambda,
    )
    return batch, observation, {
        "env_steps": num_envs * int(vector_steps),
        "episode_summaries": summaries,
    }


def collect_xp_probe(env, current: ComediPolicy, conventions: Sequence[JointActor],
                     vector_steps: int, *, device, combo_cursor: int = 0):
    if not conventions:
        raise ValueError("XP probe requires at least one completed convention")
    observation = env.reset()
    num_envs = int(env.num_envs)
    combos = [(prior, mask_id) for prior in range(len(conventions)) for mask_id in range(6)]
    combo_ids = (int(combo_cursor) + np.arange(num_envs)) % len(combos)
    cursor = (int(combo_cursor) + num_envs) % len(combos)
    scores: dict[int, list[float]] = defaultdict(list)
    role_scores: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(vector_steps)):
        mask_ids = np.asarray([combos[index][1] for index in combo_ids], dtype=np.int64)
        masks = CROSSPLAY_ROLE_MASKS[mask_ids]
        priors = np.asarray([combos[index][0] for index in combo_ids], dtype=np.int64)
        current_actions, _ = sample_joint_actions(current.actor, observation, device)
        actions = np.empty((num_envs, NUM_ROLES), dtype=np.int64)
        for prior_id in np.unique(priors):
            env_ids = np.flatnonzero(priors == prior_id)
            subset = {key: np.asarray(value)[env_ids] for key, value in observation.items()}
            prior_actions, _ = sample_joint_actions(conventions[int(prior_id)], subset, device)
            actions[env_ids] = prior_actions
        actions[masks] = current_actions[masks]
        next_observation, rewards, dones, _infos = env.step(joint_action_batch(actions, num_envs))
        for env_id, reward in enumerate(rewards):
            prior_id, mask_id = combos[combo_ids[env_id]]
            scores[prior_id].append(float(reward))
            role_scores[f"{prior_id}:mask_{mask_id}"].append(float(reward))
        for env_id in np.flatnonzero(dones):
            combo_ids[env_id] = cursor
            cursor = (cursor + 1) % len(combos)
        observation = next_observation
    actual_steps = int(vector_steps) * num_envs
    if actual_steps:
        cursor = (int(combo_cursor) + actual_steps) % len(combos)
    return {
        "scores": dict(scores),
        "role_scores": dict(role_scores),
        "combo_cursor": cursor,
        "env_steps": actual_steps,
    }


def collect_cross_play(env, current: ComediPolicy, prior_actor: JointActor,
                       prior_id: int, vector_steps: int, *, device,
                       gamma: float, gae_lambda: float, start_config: int = 0):
    observation = env.reset()
    num_envs = int(env.num_envs)
    mask_ids = (int(start_config) + np.arange(num_envs)) % len(CROSSPLAY_ROLE_MASKS)
    cursor = (int(start_config) + num_envs) % len(CROSSPLAY_ROLE_MASKS)
    spatial_rows, vector_rows, state_rows = [], [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    mask_rows, reward_rows, done_rows, summaries = [], [], [], []
    role_scores: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(vector_steps)):
        spatial, vectors, state = _observation_tensors(observation, device)
        with torch.no_grad():
            logits = current.actor(spatial[:, 0], state)
            distribution = Categorical(logits=logits)
            current_actions = distribution.sample()
            current_log_probs = distribution.log_prob(current_actions)
        current_actions_np = current_actions.cpu().numpy().astype(np.int64)
        old_actions, _ = sample_joint_actions(prior_actor, observation, device)
        masks = CROSSPLAY_ROLE_MASKS[mask_ids]
        actions = old_actions.copy()
        actions[masks] = current_actions_np[masks]
        values = _critic_grid(current, "xp", observation, device, prior=prior_id)
        next_observation, rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vectors.cpu().numpy())
        state_rows.append(state.cpu().numpy())
        action_rows.append(current_actions_np)
        log_prob_rows.append(current_log_probs.cpu().numpy())
        value_rows.append(values)
        mask_rows.append(masks.copy())
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        summaries.extend(_episode_rows(infos))
        for env_id, reward in enumerate(rewards):
            role_scores[f"mask_{mask_ids[env_id]}"] .append(float(reward))
        for env_id in np.flatnonzero(dones):
            mask_ids[env_id] = cursor
            cursor = (cursor + 1) % len(CROSSPLAY_ROLE_MASKS)
        observation = next_observation
    dones_array = np.asarray(done_rows, dtype=bool)
    bootstrap = _critic_grid(current, "xp", observation, device, prior=prior_id)
    if len(dones_array):
        bootstrap[dones_array[-1]] = 0.0
    role_masks = np.asarray(mask_rows, dtype=bool)
    batch = _rollout_arrays(
        spatial_rows, vector_rows, state_rows, action_rows, log_prob_rows,
        value_rows, reward_rows, done_rows, bootstrap, gamma, gae_lambda,
        valid=role_masks,
    )
    batch["prior_id"] = int(prior_id)
    return batch, observation, {
        "env_steps": int(vector_steps) * num_envs,
        "role_scores": dict(role_scores),
        "next_role_cursor": int(cursor),
        "episode_summaries": summaries,
    }


def _mp_vector_chunk(remaining_units: int, rollout: int) -> int:
    remaining_units = int(remaining_units)
    if remaining_units < 2:
        raise ValueError("MP phases require at least two vector steps")
    chunk = min(remaining_units, int(rollout))
    if remaining_units - chunk == 1:
        chunk += 1
    return chunk


def collect_mixed_play(env, current: ComediPolicy, prior_actor: JointActor,
                       vector_steps: int, *, device, gamma: float, gae_lambda: float,
                       rng: np.random.Generator, prefix_max: int,
                       current_probability: float):
    observation = env.reset()
    num_envs = int(env.num_envs)
    effective_prefix_max = max(1, min(int(prefix_max), max(1, int(vector_steps) // 2)))
    prefix_left = rng.integers(1, effective_prefix_max + 1, size=num_envs, dtype=np.int64)
    spatial_rows, vector_rows, state_rows = [], [], []
    action_rows, log_prob_rows, value_rows = [], [], []
    reward_rows, done_rows, suffix_rows, summaries = [], [], [], []
    for _ in range(int(vector_steps)):
        spatial, vectors, state = _observation_tensors(observation, device)
        with torch.no_grad():
            logits = current.actor(spatial[:, 0], state)
            distribution = Categorical(logits=logits)
            current_actions = distribution.sample()
            current_log_probs = distribution.log_prob(current_actions)
        current_actions_np = current_actions.cpu().numpy().astype(np.int64)
        old_actions, _ = sample_joint_actions(prior_actor, observation, device)
        suffix = prefix_left <= 0
        actions = old_actions.copy()
        for env_id in range(num_envs):
            if suffix[env_id]:
                actions[env_id] = current_actions_np[env_id]
            else:
                use_current = rng.random(NUM_ROLES) < current_probability
                actions[env_id, use_current] = current_actions_np[env_id, use_current]

        values = np.zeros((num_envs, NUM_ROLES), dtype=np.float32)
        suffix_ids = np.flatnonzero(suffix)
        if len(suffix_ids):
            for role in range(NUM_ROLES):
                values[suffix_ids, role] = current.mp_critics[role](
                    spatial[suffix_ids, role], vectors[suffix_ids, role]
                ).detach().cpu().numpy()
        next_observation, rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        spatial_rows.append(spatial.cpu().numpy())
        vector_rows.append(vectors.cpu().numpy())
        state_rows.append(state.cpu().numpy())
        action_rows.append(current_actions_np)
        log_prob_rows.append(current_log_probs.cpu().numpy())
        value_rows.append(values)
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        done_rows.append(np.asarray(dones, dtype=bool))
        suffix_rows.append(suffix.copy())
        summaries.extend(_episode_rows(infos))
        prefix_left[~suffix] -= 1
        for env_id in np.flatnonzero(dones):
            prefix_left[env_id] = int(rng.integers(1, effective_prefix_max + 1))
        observation = next_observation
    suffix_array = np.asarray(suffix_rows, dtype=bool)
    dones_array = np.asarray(done_rows, dtype=bool)
    bootstrap = _critic_grid(current, "mp", observation, device)
    if len(dones_array):
        bootstrap[dones_array[-1]] = 0.0
    valid = np.repeat(suffix_array[..., None], NUM_ROLES, axis=-1)
    batch = _rollout_arrays(
        spatial_rows, vector_rows, state_rows, action_rows, log_prob_rows,
        value_rows, reward_rows, done_rows, bootstrap, gamma, gae_lambda,
        valid=valid,
    )
    suffix_frames = int(suffix_array.sum())
    total_steps = int(vector_steps) * num_envs
    return batch, observation, {
        "env_steps": total_steps,
        "prefix_env_steps": total_steps - suffix_frames,
        "suffix_env_steps": suffix_frames,
        "episode_summaries": summaries,
    }


@torch.no_grad()
def collect_teacher_self_play(env, teachers: Sequence[JointActor], vector_steps: int,
                              *, device, start_teacher: int = 0):
    if not teachers:
        raise ValueError("teacher self-play requires completed population actors")
    observation = env.reset()
    num_envs = int(env.num_envs)
    teacher_ids = (np.arange(num_envs, dtype=np.int64) + int(start_teacher)) % len(teachers)
    next_teacher = (int(start_teacher) + num_envs) % len(teachers)
    spatial_chunks = [[] for _ in teachers]
    state_chunks = [[] for _ in teachers]
    action_chunks = [[] for _ in teachers]
    summaries = [[] for _ in teachers]
    teacher_steps = np.zeros(len(teachers), dtype=np.int64)
    teacher_episodes = np.zeros(len(teachers), dtype=np.int64)
    for _ in range(int(vector_steps)):
        actions = np.empty((num_envs, NUM_ROLES), dtype=np.int64)
        for teacher_id, actor in enumerate(teachers):
            env_ids = np.flatnonzero(teacher_ids == teacher_id)
            if not len(env_ids):
                continue
            subset = {key: np.asarray(value)[env_ids] for key, value in observation.items()}
            teacher_actions, _ = sample_joint_actions(actor, subset, device)
            actions[env_ids] = teacher_actions
            sub_spatial, _sub_vectors, sub_state = _observation_tensors(subset, "cpu")
            spatial_chunks[teacher_id].append(sub_spatial[:, 0].numpy().copy())
            state_chunks[teacher_id].append(sub_state.numpy().copy())
            action_chunks[teacher_id].append(teacher_actions.copy())
            teacher_steps[teacher_id] += len(env_ids)
        next_observation, _rewards, dones, infos = env.step(joint_action_batch(actions, num_envs))
        for teacher_id in range(len(teachers)):
            env_ids = np.flatnonzero((teacher_ids == teacher_id) & np.asarray(dones, dtype=bool))
            teacher_episodes[teacher_id] += len(env_ids)
            summaries[teacher_id].extend(
                dict(info["episode_summary"])
                for info in (infos[int(index)] for index in env_ids)
                if "episode_summary" in info
            )
        for env_id in np.flatnonzero(dones):
            teacher_ids[env_id] = next_teacher
            next_teacher = (next_teacher + 1) % len(teachers)
        observation = next_observation

    batches = {}
    for teacher_id in range(len(teachers)):
        if not teacher_steps[teacher_id]:
            continue
        states = np.concatenate(state_chunks[teacher_id], axis=0)
        spatial = np.concatenate(spatial_chunks[teacher_id], axis=0)
        actions = np.concatenate(action_chunks[teacher_id], axis=0)
        batches[teacher_id] = {
            "spatial": spatial,
            "state": states,
            "actions": actions,
            "bc_actions": actions.copy(),
            "valid": np.ones(len(actions), dtype=bool),
        }
    actual_steps = int(vector_steps) * num_envs
    return batches, {
        "env_steps": actual_steps,
        "teacher_env_steps": {str(i): int(value) for i, value in enumerate(teacher_steps)},
        "teacher_episode_counts": {str(i): int(value) for i, value in enumerate(teacher_episodes)},
        "episode_summaries_by_teacher": {str(i): summaries[i] for i in range(len(teachers))},
        "next_teacher_cursor": int(next_teacher),
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
    """Accumulate SP/XP/MP plus teacher BC, then update actor once per minibatch."""
    if not objectives and not bc_objectives:
        raise ValueError("at least one PPO or teacher-BC objective is required")
    prepared = []
    for phase, batch, weight in objectives:
        required = ("spatial", "state", "critic_spatial", "critic_vector", "actions",
                    "log_probs", "advantages", "returns", "valid")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"{phase} batch missing fields: {missing}")
        data = {key: np.asarray(batch[key]) for key in required}
        count = len(data["actions"])
        if data["spatial"].shape != (count, *SPATIAL_SHAPE):
            raise ValueError("actor minibatch spatial shape must be [frames, 26, 7, 7]")
        if data["state"].shape != (count, STATE_DIM):
            raise ValueError("actor minibatch state shape must be [frames, 863]")
        if data["critic_spatial"].shape != (count, NUM_ROLES, *SPATIAL_SHAPE):
            raise ValueError("critic spatial shape must be [frames, 3, 26, 7, 7]")
        if data["critic_vector"].shape != (count, NUM_ROLES, OBS_DIM):
            raise ValueError("critic vector shape must be [frames, 3, 866]")
        for key in ("actions", "log_probs", "advantages", "returns", "valid"):
            if data[key].shape != (count, NUM_ROLES):
                raise ValueError(f"{phase} {key} shape must be [frames, 3]")
        if data["valid"].dtype != np.bool_:
            data["valid"] = data["valid"].astype(bool)
        if not np.issubdtype(data["actions"].dtype, np.integer):
            raise TypeError("role actions must use an integer dtype")
        if not data["valid"].any():
            raise ValueError(f"{phase} batch has no trainable role samples")
        if not all(np.isfinite(value).all() for key, value in data.items() if key != "valid"):
            raise FloatingPointError(f"{phase} batch contains non-finite values")
        normalized = data["advantages"].astype(np.float32, copy=True)
        for role in range(NUM_ROLES):
            mask = data["valid"][:, role]
            role_adv = normalized[mask, role]
            if len(role_adv) > 1:
                normalized[mask, role] = (role_adv - role_adv.mean()) / (role_adv.std() + 1e-8)
        data["normalized_advantages"] = normalized
        data["weight"] = float(weight)
        data["phase"] = phase
        data["prior_id"] = int(batch["prior_id"]) if phase == "xp" else None
        data["active_frames"] = np.flatnonzero(data["valid"].any(axis=1))
        if not len(data["active_frames"]):
            raise ValueError(f"{phase} has no valid frames")
        prepared.append(data)

    prepared_bc = []
    if bc_coef > 0.0:
        for teacher_id, batch in bc_objectives:
            data = {key: np.asarray(batch[key]) for key in ("spatial", "state", "bc_actions", "valid")}
            count = len(data["actions"] if "actions" in data else data["bc_actions"])
            if data["spatial"].shape != (count, *SPATIAL_SHAPE):
                raise ValueError(f"teacher {teacher_id} spatial shape mismatch")
            if data["state"].shape != (count, STATE_DIM):
                raise ValueError(f"teacher {teacher_id} state shape mismatch")
            if data["bc_actions"].shape != (count, NUM_ROLES):
                raise ValueError(f"teacher {teacher_id} labels must have shape [frames, 3]")
            if data["valid"].shape not in ((count,), (count, NUM_ROLES)):
                raise ValueError(f"teacher {teacher_id} validity mask shape mismatch")
            if data["valid"].ndim == 1:
                data["valid"] = np.repeat(data["valid"][:, None], NUM_ROLES, axis=1)
            data["valid"] = data["valid"].astype(bool)
            if not data["valid"].any():
                continue
            if not np.isfinite(data["spatial"][data["valid"].any(axis=1)]).all() or not np.isfinite(
                data["state"][data["valid"].any(axis=1)]
            ).all():
                raise FloatingPointError(f"teacher {teacher_id} states contain non-finite values")
            data["teacher_id"] = int(teacher_id)
            data["active_frames"] = np.flatnonzero(data["valid"].any(axis=1))
            prepared_bc.append(data)

    if epochs <= 0 or minibatches <= 0:
        raise ValueError("epochs and minibatches must be positive")
    batch_count = max(
        [min(int(minibatches), len(item["active_frames"])) for item in prepared]
        + [min(int(minibatches), len(item["active_frames"])) for item in prepared_bc]
    )
    actor_losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    bc_losses: list[float] = []
    value_losses: dict[str, list[float]] = defaultdict(list)
    actor_updates = 0

    for _epoch in range(int(epochs)):
        objective_chunks = [
            np.array_split(rng.permutation(data["active_frames"]),
                           min(int(minibatches), len(data["active_frames"])))
            for data in prepared
        ]
        bc_chunks = [
            np.array_split(rng.permutation(data["active_frames"]),
                           min(int(minibatches), len(data["active_frames"])))
            for data in prepared_bc
        ]
        updates = max([len(chunks) for chunks in objective_chunks + bc_chunks], default=0)
        for minibatch_index in range(updates):
            combined_actor_loss = None
            critic_work = []
            for data, chunks in zip(prepared, objective_chunks):
                if minibatch_index >= len(chunks) or not len(chunks[minibatch_index]):
                    continue
                indices = chunks[minibatch_index]
                spatial = torch.as_tensor(data["spatial"][indices], dtype=torch.float32, device=device)
                state = torch.as_tensor(data["state"][indices], dtype=torch.float32, device=device)
                actions = torch.as_tensor(data["actions"][indices], dtype=torch.long, device=device)
                old_log_probs = torch.as_tensor(data["log_probs"][indices], dtype=torch.float32, device=device)
                advantages = torch.as_tensor(
                    data["normalized_advantages"][indices], dtype=torch.float32, device=device
                )
                valid = torch.as_tensor(data["valid"][indices], dtype=torch.bool, device=device)
                distribution = policy.actor.distribution(spatial, state)
                log_probs = distribution.log_prob(actions)
                entropies_role = distribution.entropy()
                for role in range(NUM_ROLES):
                    mask = valid[:, role]
                    if not torch.any(mask):
                        continue
                    role_loss = ppo_actor_loss(
                        log_probs[mask, role],
                        old_log_probs[mask, role],
                        advantages[mask, role],
                        clip=clip,
                        weight=data["weight"],
                        entropy=entropies_role[mask, role],
                        entropy_coef=entropy_coef,
                    )
                    combined_actor_loss = (
                        role_loss if combined_actor_loss is None
                        else combined_actor_loss + role_loss
                    )
                    actor_losses.append(float(role_loss.detach().cpu()))
                    policy_losses.append(float((role_loss + entropy_coef * entropies_role[mask, role].mean()).detach().cpu()))
                    entropies.append(float(entropies_role[mask, role].mean().detach().cpu()))
                critic_work.append((data, indices))

            if prepared_bc:
                for data, chunks in zip(prepared_bc, bc_chunks):
                    if minibatch_index >= len(chunks) or not len(chunks[minibatch_index]):
                        continue
                    indices = chunks[minibatch_index]
                    spatial = torch.as_tensor(data["spatial"][indices], dtype=torch.float32, device=device)
                    state = torch.as_tensor(data["state"][indices], dtype=torch.float32, device=device)
                    labels = torch.as_tensor(data["bc_actions"][indices], dtype=torch.long, device=device)
                    valid = torch.as_tensor(data["valid"][indices], dtype=torch.bool, device=device)
                    logits = policy.actor(spatial, state)
                    label_losses = F.cross_entropy(
                        logits.reshape(-1, ACTION_DIM), labels.reshape(-1), reduction="none"
                    ).reshape(-1, NUM_ROLES)
                    role_means = [label_losses[valid[:, role], role].mean()
                                  for role in range(NUM_ROLES) if torch.any(valid[:, role])]
                    if not role_means:
                        continue
                    bc_loss = torch.stack(role_means).mean()
                    weighted = (float(bc_coef) / len(prepared_bc)) * bc_loss
                    combined_actor_loss = weighted if combined_actor_loss is None else combined_actor_loss + weighted
                    bc_losses.append(float(bc_loss.detach().cpu()))

            if combined_actor_loss is None:
                continue
            if not torch.isfinite(combined_actor_loss):
                raise FloatingPointError("combined SP/XP/MP/BC actor loss is non-finite")
            policy.actor_optimizer.zero_grad(set_to_none=True)
            combined_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.actor.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            policy.actor_optimizer.step()
            actor_updates += 1

            for data, indices in critic_work:
                phase = data["phase"]
                for role in range(NUM_ROLES):
                    role_mask = data["valid"][indices, role]
                    selected_ids = indices[role_mask]
                    if not len(selected_ids):
                        continue
                    spatial = torch.as_tensor(
                        data["critic_spatial"][selected_ids, role], dtype=torch.float32, device=device
                    )
                    vector = torch.as_tensor(
                        data["critic_vector"][selected_ids, role], dtype=torch.float32, device=device
                    )
                    targets = torch.as_tensor(
                        data["returns"][selected_ids, role], dtype=torch.float32, device=device
                    )
                    critic = policy.critic(phase, role, data["prior_id"])
                    optimizer = policy.critic_optimizer(phase, role, data["prior_id"])
                    value = critic(spatial, vector)
                    loss = F.mse_loss(value, targets) * float(value_coef)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{phase} role {role} value loss is non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    value_losses[phase].append(float(loss.detach().cpu()))

    for module_name, module in (("actor", policy.actor), *(
        (f"critic_{phase}", policy.critic(phase, role, prior))
        for phase, role, prior in _active_critic_groups(prepared)
    )):
        for parameter in module.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"updated {module_name} contains non-finite parameters")
    return {
        "samples_by_phase": {
            data["phase"]: int(data["valid"].sum()) for data in prepared
        },
        "joint_frames_by_phase": {
            data["phase"]: int(data["valid"].any(axis=1).sum()) for data in prepared
        },
        "bc_samples_by_teacher": {
            str(data["teacher_id"]): int(data["valid"].sum()) for data in prepared_bc
        },
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "value_loss_by_phase": {
            phase: float(np.mean(values)) for phase, values in value_losses.items()
        },
        "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "actor_updates": actor_updates,
    }


def _active_critic_groups(prepared):
    groups = set()
    for data in prepared:
        for role in range(NUM_ROLES):
            if data["valid"][:, role].any():
                groups.add((data["phase"], role, data["prior_id"]))
    return sorted(groups, key=repr)

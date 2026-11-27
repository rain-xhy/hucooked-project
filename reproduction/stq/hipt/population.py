"""Self-play and JSD training for the diverse HiPT partner population."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .common import gae, ppo_components
from .model import (
    FlatActorCritic,
    SPATIAL_CHANNELS,
    SPATIAL_HEIGHT,
    SPATIAL_WIDTH,
    VECTOR_DIM,
)


class PopulationTrainer:
    """Train a fixed-size population under three-player self-play cross-play.

    Each environment uses one population policy in all three chef roles for its
    whole episode, matching self-play population training. On each training
    sample, the JSD is computed over every population policy evaluated on the
    same public observation; minimizing ``-jsd_coef * JSD`` encourages distinct
    response styles while each policy maximizes its own shared-team PPO return.
    """

    def __init__(self, env, *, num_envs: int, population_size: int,
                 device: torch.device, seed: int, epochs: int = 4,
                 minibatches: int = 4, learning_rate: float = 3e-4,
                 gamma: float = 0.9985, gae_lambda: float = 0.95,
                 clip: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, jsd_coef: float = 0.05,
                 max_grad_norm: float = 1.0, demonstrations_path=None,
                 bc_epochs: int = 8, initialize_from_demonstrations: bool = True):
        if population_size < 2:
            raise ValueError("The diversity objective requires at least two policies")
        self.env = env
        self.num_envs = int(num_envs)
        self.population_size = int(population_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.epochs, self.minibatches = int(epochs), int(minibatches)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.clip, self.value_coef = float(clip), float(value_coef)
        self.entropy_coef, self.jsd_coef = float(entropy_coef), float(jsd_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.models = [FlatActorCritic().to(device) for _ in range(population_size)]
        self.uncloned_initial_models = [
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for model in self.models
        ]
        self.behavior_cloning_metrics = []
        if demonstrations_path is not None and initialize_from_demonstrations:
            from baseline_common.demonstrations import warm_start
            for policy_id, model in enumerate(self.models):
                self.behavior_cloning_metrics.append(warm_start(
                    model,
                    lambda spatial, vector, actor=model: actor(spatial, vector)[0],
                    dataset_path=demonstrations_path,
                    device=device,
                    seed=int(seed) + 1000 + policy_id,
                    epochs=int(bc_epochs),
                ))
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate, eps=1e-5)
                           for m in self.models]
        self.agent_steps = np.zeros(population_size, dtype=np.int64)
        self._id_cursor = 0
        self.total_env_steps = 0
        self.update = 0
        self.obs = None
        self.policy_ids = None
        self.recent_team_scores = deque(maxlen=512)

    def _balanced_ids(self, count: int) -> np.ndarray:
        values = (self._id_cursor + np.arange(count, dtype=np.int64)) % self.population_size
        self._id_cursor = (self._id_cursor + count) % self.population_size
        self.rng.shuffle(values)
        return values

    def _reset_ids(self, env_indices):
        if len(env_indices):
            env_ids = self._balanced_ids(len(env_indices))
            self.policy_ids[env_indices] = env_ids[:, None]

    def _act(self, obs, deterministic=False):
        spatial = torch.as_tensor(obs["spatial"], device=self.device).reshape(
            self.num_envs * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        vector = torch.as_tensor(obs["vector"], device=self.device).reshape(
            self.num_envs * 3, VECTOR_DIM
        )
        flat_ids = self.policy_ids.reshape(-1)
        actions = np.empty(flat_ids.shape, dtype=np.int64)
        logp = np.empty(flat_ids.shape, dtype=np.float32)
        values = np.empty(flat_ids.shape, dtype=np.float32)
        with torch.no_grad():
            for policy_id in np.unique(flat_ids):
                rows = np.flatnonzero(flat_ids == policy_id)
                index = torch.as_tensor(rows, device=self.device)
                logits, value = self.models[int(policy_id)](spatial[index], vector[index])
                distribution = Categorical(logits=logits)
                action = logits.argmax(-1) if deterministic else distribution.sample()
                actions[rows] = action.cpu().numpy()
                logp[rows] = distribution.log_prob(action).cpu().numpy()
                values[rows] = value.cpu().numpy()
        counts = np.bincount(flat_ids, minlength=self.population_size)
        self.agent_steps += counts
        return actions.reshape(self.num_envs, 3), logp.reshape(self.num_envs, 3), values.reshape(
            self.num_envs, 3
        )

    def _bootstrap(self, obs):
        spatial = torch.as_tensor(obs["spatial"], device=self.device).reshape(
            self.num_envs * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        vector = torch.as_tensor(obs["vector"], device=self.device).reshape(
            self.num_envs * 3, VECTOR_DIM
        )
        flat_ids = self.policy_ids.reshape(-1)
        values = np.zeros(flat_ids.shape, dtype=np.float32)
        with torch.no_grad():
            for policy_id in np.unique(flat_ids):
                rows = np.flatnonzero(flat_ids == policy_id)
                index = torch.as_tensor(rows, device=self.device)
                _, value = self.models[int(policy_id)](spatial[index], vector[index])
                values[rows] = value.cpu().numpy()
        return values.reshape(self.num_envs, 3)

    def collect_and_update(self, rollout_steps: int):
        if self.obs is None:
            self.obs = self.env.reset()
            env_ids = self._balanced_ids(self.num_envs)
            self.policy_ids = np.repeat(env_ids[:, None], 3, axis=1)

        spatial_rows, vector_rows = [], []
        actions_rows, logp_rows, values_rows = [], [], []
        rewards_rows, dones_rows, ids_rows = [], [], []
        summaries = []
        started = time.monotonic()
        for _ in range(rollout_steps):
            actions, logp, values = self._act(self.obs)
            nxt, reward, done, infos = self.env.step(actions)
            spatial_rows.append(self.obs["spatial"].copy())
            vector_rows.append(self.obs["vector"].copy())
            actions_rows.append(actions.copy())
            logp_rows.append(logp)
            values_rows.append(values)
            rewards_rows.append(reward.copy())
            dones_rows.append(done.copy())
            ids_rows.append(self.policy_ids.copy())
            for i, info in enumerate(infos):
                summary = info.get("episode_summary")
                if summary is not None:
                    self.recent_team_scores.append(float(summary["team_score"]))
                    summaries.append(summary)
            finished = np.flatnonzero(done)
            self._reset_ids(finished)
            self.obs = nxt

        bootstrap = self._bootstrap(self.obs)
        spatial = np.asarray(spatial_rows, dtype=np.float32)
        vector = np.asarray(vector_rows, dtype=np.float32)
        actions = np.asarray(actions_rows, dtype=np.int64)
        old_logp = np.asarray(logp_rows, dtype=np.float32)
        values = np.asarray(values_rows, dtype=np.float32)
        rewards = np.repeat(np.asarray(rewards_rows, dtype=np.float32)[..., None], 3, axis=-1)
        dones = np.asarray(dones_rows, dtype=np.float32)
        advantages, returns = gae(rewards, values, dones, bootstrap,
                                  self.gamma, self.gae_lambda)
        ids = np.asarray(ids_rows, dtype=np.int64)

        # Flatten time, environment, and role. Cross-entropy diversity is
        # evaluated on the same public observation for all population members.
        flat_spatial = spatial.reshape(
            -1, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        flat_vector = vector.reshape(-1, VECTOR_DIM)
        flat_actions = actions.reshape(-1)
        flat_logp = old_logp.reshape(-1)
        flat_values = values.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_ids = ids.reshape(-1)
        batch_size = len(flat_ids)
        metric_rows = []
        for epoch in range(self.epochs):
            permutation = self.rng.permutation(batch_size)
            for mb_indices in np.array_split(permutation, self.minibatches):
                if not len(mb_indices):
                    continue
                x = torch.as_tensor(flat_spatial[mb_indices], device=self.device)
                v = torch.as_tensor(flat_vector[mb_indices], device=self.device)
                act = torch.as_tensor(flat_actions[mb_indices], device=self.device)
                lp_old = torch.as_tensor(flat_logp[mb_indices], device=self.device)
                val_old = torch.as_tensor(flat_values[mb_indices], device=self.device)
                adv = torch.as_tensor(flat_adv[mb_indices], device=self.device)
                ret = torch.as_tensor(flat_returns[mb_indices], device=self.device)
                sample_ids = torch.as_tensor(flat_ids[mb_indices], device=self.device)

                logits_all, values_all, entropies_all = [], [], []
                for model in self.models:
                    logits, value = model(x, v)
                    distribution = Categorical(logits=logits)
                    logits_all.append(logits)
                    values_all.append(value)
                    entropies_all.append(distribution.entropy())
                probabilities = torch.stack([logits.softmax(-1) for logits in logits_all], dim=0)
                mixture = probabilities.mean(0)
                entropy_mixture = -(mixture * mixture.clamp_min(1e-8).log()).sum(-1)
                mean_entropy = torch.stack(entropies_all).mean(0)
                jsd = (entropy_mixture - mean_entropy).mean()

                losses = []
                policy_metrics, value_metrics, entropy_metrics = [], [], []
                for agent_id, (logits, value) in enumerate(zip(logits_all, values_all)):
                    mask = sample_ids == agent_id
                    if not bool(mask.any()):
                        continue
                    distribution = Categorical(logits=logits[mask])
                    policy_loss, value_loss, entropy, approx_kl = ppo_components(
                        distribution.log_prob(act[mask]), lp_old[mask], adv[mask],
                        value[mask], val_old[mask], ret[mask],
                        distribution.entropy(), self.clip,
                    )
                    losses.append(policy_loss + self.value_coef * value_loss -
                                  self.entropy_coef * entropy)
                    policy_metrics.append(policy_loss.detach())
                    value_metrics.append(value_loss.detach())
                    entropy_metrics.append(entropy.detach())

                if not losses:
                    continue
                loss = torch.stack(losses).sum() - self.jsd_coef * jsd
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite partner-population loss")
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norms = []
                for model, optimizer in zip(self.models, self.optimizers):
                    grad = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm, error_if_nonfinite=True
                    )
                    grad_norms.append(float(grad.detach().cpu()))
                    optimizer.step()
                metric_rows.append((float(torch.stack(policy_metrics).mean().cpu()),
                                    float(torch.stack(value_metrics).mean().cpu()),
                                    float(torch.stack(entropy_metrics).mean().cpu()),
                                    float(jsd.detach().cpu()), float(np.mean(grad_norms))))

        self.total_env_steps += self.num_envs * rollout_steps
        self.update += 1
        metrics = {
            "stage": "population",
            "update": self.update,
            "stage_steps": self.total_env_steps,
            "policy_action_steps_min": int(self.agent_steps.min()),
            "policy_action_steps_max": int(self.agent_steps.max()),
            "active_population_members": int(np.count_nonzero(self.agent_steps)),
            "policy_loss": float(np.mean([m[0] for m in metric_rows])) if metric_rows else None,
            "value_loss": float(np.mean([m[1] for m in metric_rows])) if metric_rows else None,
            "entropy": float(np.mean([m[2] for m in metric_rows])) if metric_rows else None,
            "jsd": float(np.mean([m[3] for m in metric_rows])) if metric_rows else None,
            "grad_norm": float(np.mean([m[4] for m in metric_rows])) if metric_rows else None,
            "mean_recent_team_score": (
                float(np.mean(self.recent_team_scores)) if self.recent_team_scores else None
            ),
            "completed_episodes": len(summaries),
            "update_seconds": time.monotonic() - started,
        }
        return metrics

    def state_dict(self, include_optimizers: bool = True):
        result = {
            "models": [model.state_dict() for model in self.models],
            "agent_steps": self.agent_steps.copy(),
            "id_cursor": self._id_cursor,
            "total_env_steps": self.total_env_steps,
            "update": self.update,
        }
        if include_optimizers:
            result["optimizers"] = [optimizer.state_dict() for optimizer in self.optimizers]
        return result

    def load_state_dict(self, payload, include_optimizers: bool = True):
        if len(payload["models"]) != self.population_size:
            raise ValueError("Checkpoint population size differs from current configuration")
        for model, saved in zip(self.models, payload["models"]):
            model.load_state_dict(saved)
        if include_optimizers:
            if len(payload.get("optimizers", [])) != self.population_size:
                raise ValueError("Checkpoint is missing population optimizer states")
            for optimizer, saved in zip(self.optimizers, payload["optimizers"]):
                optimizer.load_state_dict(saved)
        self.agent_steps = np.asarray(payload["agent_steps"], dtype=np.int64)
        self._id_cursor = int(payload.get("id_cursor", 0)) % self.population_size
        self.total_env_steps = int(payload["total_env_steps"])
        self.update = int(payload["update"])

    def close(self):
        self.env.close()

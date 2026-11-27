"""Central self-play population and exact one-GPU JSD gradient staging."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .common import gae, ppo_components
from .model import CentralFlatActorCritic, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH


def _population_objective(logits_all, values_all, actions, old_logp, old_value,
                          advantage, returns, sample_ids, *, clip, value_coef,
                          entropy_coef, jsd_coef):
    """Original sum-of-member PPO plus one JSD over 16 policies × 3 roles."""
    probabilities = torch.stack([logits.softmax(-1) for logits in logits_all], dim=0)
    mixture = probabilities.mean(0)
    entropy_mixture = -(mixture * mixture.clamp_min(1e-8).log()).sum(-1)
    entropies = torch.stack(
        [Categorical(logits=logits).entropy() for logits in logits_all], dim=0
    )
    jsd = (entropy_mixture - entropies.mean(0)).mean()

    losses, pg_rows, value_rows, entropy_rows = [], [], [], []
    for member, (logits, value) in enumerate(zip(logits_all, values_all)):
        mask = sample_ids == member
        if not bool(mask.any()):
            continue
        dist = Categorical(logits=logits[mask].reshape(-1, logits.shape[-1]))
        pg, value_loss, entropy, _ = ppo_components(
            dist.log_prob(actions[mask].reshape(-1)),
            old_logp[mask].reshape(-1), advantage[mask].reshape(-1),
            value[mask].reshape(-1), old_value[mask].reshape(-1),
            returns[mask].reshape(-1), dist.entropy(), clip,
        )
        losses.append(pg + value_coef * value_loss - entropy_coef * entropy)
        pg_rows.append(pg.detach())
        value_rows.append(value_loss.detach())
        entropy_rows.append(entropy.detach())
    if not losses:
        return None, None
    loss = torch.stack(losses).sum() - jsd_coef * jsd
    metrics = {
        "policy_loss": torch.stack(pg_rows).mean().detach(),
        "value_loss": torch.stack(value_rows).mean().detach(),
        "entropy": torch.stack(entropy_rows).mean().detach(),
        "jsd": jsd.detach(),
    }
    return loss, metrics


def staged_population_step(models, optimizers, workspace, *, spatial, state,
                          actions, old_logp, old_value, advantage, returns,
                          sample_ids, device, clip=0.2, value_coef=0.5,
                          entropy_coef=0.01, jsd_coef=0.05,
                          max_grad_norm=1.0):
    """Apply the exact joint objective while one trainable model is on device.

    First evaluate every member at the same parameter point and retain only
    logits/values on CPU. Differentiate the original joint loss with respect to
    those outputs, then replay each member once and apply its cached output
    gradients. Models and Adam states remain on CPU; ``workspace`` is the only
    device-resident trainable network.
    """
    if len(models) != len(optimizers) or not models:
        raise ValueError("models and optimizers must have the same nonzero length")
    if any(next(model.parameters()).device.type != "cpu" for model in models):
        raise ValueError("Staged population models must remain on CPU")
    device = torch.device(device)
    x = torch.as_tensor(spatial, dtype=torch.float32, device=device)
    s = torch.as_tensor(state, dtype=torch.float32, device=device)
    cached_logits, cached_values = [], []
    workspace.train()
    with torch.no_grad():
        for model in models:
            workspace.load_state_dict(model.state_dict())
            logits, values = workspace(x, s)
            cached_logits.append(logits.detach().cpu())
            cached_values.append(values.detach().cpu())

    # The joint JSD graph is small; keep it on CPU and preserve gradients from
    # the single global JSD to every member's three marginal action heads.
    logits_leaf = [value.detach().requires_grad_(True) for value in cached_logits]
    values_leaf = [value.detach().requires_grad_(True) for value in cached_values]
    action_t = torch.as_tensor(actions, dtype=torch.long)
    old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32)
    old_value_t = torch.as_tensor(old_value, dtype=torch.float32)
    advantage_t = torch.as_tensor(advantage, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    ids_t = torch.as_tensor(sample_ids, dtype=torch.long)
    loss, metrics = _population_objective(
        logits_leaf, values_leaf, action_t, old_logp_t, old_value_t,
        advantage_t, returns_t, ids_t, clip=clip, value_coef=value_coef,
        entropy_coef=entropy_coef, jsd_coef=jsd_coef,
    )
    if loss is None:
        return None
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite central population loss")
    output_grads = torch.autograd.grad(
        loss, tuple(logits_leaf + values_leaf), allow_unused=True
    )
    logits_grad = output_grads[:len(models)]
    values_grad = output_grads[len(models):]

    grad_norms = []
    for member, (model, optimizer) in enumerate(zip(models, optimizers)):
        workspace.load_state_dict(model.state_dict())
        workspace.train(model.training)
        workspace.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        replay_logits, replay_values = workspace(x, s)
        if (not torch.allclose(replay_logits.detach().cpu(), cached_logits[member],
                               atol=1e-6, rtol=1e-5)
                or not torch.allclose(replay_values.detach().cpu(), cached_values[member],
                                      atol=1e-6, rtol=1e-5)):
            raise RuntimeError("Population replay outputs changed since the JSD cache pass")
        surrogate = (replay_logits * logits_grad[member].to(device)).sum()
        if values_grad[member] is not None:
            surrogate = surrogate + (replay_values * values_grad[member].to(device)).sum()
        surrogate.backward()
        source = dict(workspace.named_parameters())
        for name, parameter in model.named_parameters():
            grad = source[name].grad
            parameter.grad = None if grad is None else grad.detach().cpu().clone()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
        grad_norms.append(float(norm.detach()))
        optimizer.step()
    workspace.zero_grad(set_to_none=True)
    return {
        **{key: float(value) for key, value in metrics.items()},
        "grad_norm": float(np.mean(grad_norms)),
        "loss": float(loss.detach()),
    }


class PopulationTrainer:
    """Train 16 central policies, retaining JSD and CPU-resident Adam states."""

    def __init__(self, env, *, num_envs=8, population_size=16, device="cuda:0",
                 seed=1, epochs=1, minibatches=4, learning_rate=1e-5,
                 gamma=0.9985, gae_lambda=0.95, clip=0.2, value_coef=0.5,
                 entropy_coef=0.01, jsd_coef=0.05, max_grad_norm=1.0):
        if population_size < 2:
            raise ValueError("JSD population requires at least two policies")
        self.env, self.num_envs, self.population_size = env, int(num_envs), int(population_size)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.epochs, self.minibatches = int(epochs), int(minibatches)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.clip, self.value_coef = float(clip), float(value_coef)
        self.entropy_coef, self.jsd_coef = float(entropy_coef), float(jsd_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.models = [CentralFlatActorCritic().cpu() for _ in range(self.population_size)]
        self.initial_models = [self._cpu_state(model) for model in self.models]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
                           for model in self.models]
        self.workspace = CentralFlatActorCritic().to(self.device)
        self.agent_steps = np.zeros(self.population_size, dtype=np.int64)
        self._id_cursor = 0
        self.total_env_steps = 0
        self.update = 0
        self.obs = None
        self.policy_ids = None
        self.recent_team_scores = deque(maxlen=512)

    @staticmethod
    def _cpu_state(model):
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def behavior_clone(self, dataset_path, *, seed, epochs=64, batch_size=512):
        from joint_common.demonstrations import warm_start_joint
        metrics = []
        for member, model in enumerate(self.models):
            self.workspace.load_state_dict(model.state_dict())
            rows = warm_start_joint(
                self.workspace,
                lambda spatial, state, actor=self.workspace: actor(spatial, state)[0],
                dataset_path=dataset_path, device=self.device, seed=int(seed) + member,
                epochs=int(epochs), batch_size=int(batch_size),
            )
            model.load_state_dict(self._cpu_state(self.workspace))
            metrics.append(rows)
        self.workspace.zero_grad(set_to_none=True)
        return metrics

    def _balanced_ids(self, count):
        values = (self._id_cursor + np.arange(count, dtype=np.int64)) % self.population_size
        self._id_cursor = (self._id_cursor + count) % self.population_size
        self.rng.shuffle(values)
        return values

    def _reset_ids(self, env_indices):
        if len(env_indices):
            self.policy_ids[env_indices] = self._balanced_ids(len(env_indices))

    def _policy_forward(self, model, spatial, state):
        self.workspace.load_state_dict(model.state_dict())
        logits, values = self.workspace(
            torch.as_tensor(spatial, device=self.device, dtype=torch.float32),
            torch.as_tensor(state, device=self.device, dtype=torch.float32),
        )
        return logits, values

    @torch.no_grad()
    def _act(self, obs, deterministic=False):
        spatial, state = obs["spatial"][:, 0], obs["state"]
        actions = np.empty((self.num_envs, 3), dtype=np.int64)
        logp = np.empty((self.num_envs, 3), dtype=np.float32)
        values = np.empty((self.num_envs, 3), dtype=np.float32)
        for member in np.unique(self.policy_ids):
            rows = np.flatnonzero(self.policy_ids == member)
            logits, v = self._policy_forward(self.models[int(member)], spatial[rows], state[rows])
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            actions[rows] = action.cpu().numpy()
            logp[rows] = dist.log_prob(action).cpu().numpy()
            values[rows] = v.cpu().numpy()
        counts = np.bincount(self.policy_ids, minlength=self.population_size)
        self.agent_steps += counts * 3
        return actions, logp, values

    @torch.no_grad()
    def _bootstrap(self, obs):
        values = np.zeros((self.num_envs, 3), dtype=np.float32)
        spatial, state = obs["spatial"][:, 0], obs["state"]
        for member in np.unique(self.policy_ids):
            rows = np.flatnonzero(self.policy_ids == member)
            _, v = self._policy_forward(self.models[int(member)], spatial[rows], state[rows])
            values[rows] = v.cpu().numpy()
        return values

    def collect_and_update(self, rollout_steps):
        if self.obs is None:
            self.obs = self.env.reset()
            self.policy_ids = self._balanced_ids(self.num_envs)
        spatial_rows, state_rows, action_rows, logp_rows, value_rows = [], [], [], [], []
        reward_rows, done_rows, id_rows, summaries = [], [], [], []
        started = time.monotonic()
        for _ in range(int(rollout_steps)):
            actions, logp, values = self._act(self.obs)
            next_obs, reward, done, infos = self.env.step(actions)
            spatial_rows.append(self.obs["spatial"][:, 0].copy())
            state_rows.append(self.obs["state"].copy())
            action_rows.append(actions.copy())
            logp_rows.append(logp)
            value_rows.append(values)
            reward_rows.append(np.repeat(reward[:, None], 3, axis=1))
            done_rows.append(done.copy())
            id_rows.append(self.policy_ids.copy())
            for index, info in enumerate(infos):
                summary = info.get("episode_summary")
                if summary is not None:
                    self.recent_team_scores.append(float(summary["team_score"]))
                    summaries.append(summary)
            ended = np.flatnonzero(done)
            self._reset_ids(ended)
            self.obs = next_obs

        bootstrap = self._bootstrap(self.obs)
        spatial = np.asarray(spatial_rows, dtype=np.float32).reshape(
            -1, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        state = np.asarray(state_rows, dtype=np.float32).reshape(-1, 863)
        actions = np.asarray(action_rows, dtype=np.int64).reshape(-1, 3)
        old_logp = np.asarray(logp_rows, dtype=np.float32).reshape(-1, 3)
        values = np.asarray(value_rows, dtype=np.float32)
        rewards = np.asarray(reward_rows, dtype=np.float32)
        dones = np.asarray(done_rows, dtype=np.float32)
        advantages, returns = gae(
            rewards, values, dones, bootstrap, self.gamma, self.gae_lambda
        )
        values = values.reshape(-1, 3)
        advantages, returns = advantages.reshape(-1, 3), returns.reshape(-1, 3)
        ids = np.asarray(id_rows, dtype=np.int64).reshape(-1)
        metric_rows = []
        for _ in range(self.epochs):
            permutation = self.rng.permutation(len(ids))
            for indices in np.array_split(permutation, self.minibatches):
                if not len(indices):
                    continue
                metric = staged_population_step(
                    self.models, self.optimizers, self.workspace,
                    spatial=spatial[indices], state=state[indices],
                    actions=actions[indices], old_logp=old_logp[indices],
                    old_value=values[indices], advantage=advantages[indices],
                    returns=returns[indices], sample_ids=ids[indices], device=self.device,
                    clip=self.clip, value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef, jsd_coef=self.jsd_coef,
                    max_grad_norm=self.max_grad_norm,
                )
                if metric is not None:
                    metric_rows.append(metric)
        self.total_env_steps += self.num_envs * int(rollout_steps)
        self.update += 1
        return {
            "stage": "population", "update": self.update,
            "stage_steps": self.total_env_steps,
            "policy_action_steps_min": int(self.agent_steps.min()),
            "policy_action_steps_max": int(self.agent_steps.max()),
            "active_population_members": int(np.count_nonzero(self.agent_steps)),
            "policy_loss": float(np.mean([row["policy_loss"] for row in metric_rows])) if metric_rows else None,
            "value_loss": float(np.mean([row["value_loss"] for row in metric_rows])) if metric_rows else None,
            "entropy": float(np.mean([row["entropy"] for row in metric_rows])) if metric_rows else None,
            "jsd": float(np.mean([row["jsd"] for row in metric_rows])) if metric_rows else None,
            "grad_norm": float(np.mean([row["grad_norm"] for row in metric_rows])) if metric_rows else None,
            "mean_recent_team_score": float(np.mean(self.recent_team_scores)) if self.recent_team_scores else None,
            "completed_episodes": len(summaries),
            "update_seconds": time.monotonic() - started,
        }

    def state_dict(self):
        return {
            "models": [self._cpu_state(model) for model in self.models],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "agent_steps": self.agent_steps.copy(), "id_cursor": self._id_cursor,
            "total_env_steps": self.total_env_steps, "update": self.update,
        }

    def load_state_dict(self, payload):
        if len(payload.get("models", [])) != self.population_size:
            raise ValueError("Population checkpoint size mismatch")
        if len(payload.get("optimizers", [])) != self.population_size:
            raise ValueError("Population checkpoint optimizer count mismatch")
        for model, state, optimizer, opt_state in zip(
            self.models, payload["models"], self.optimizers, payload["optimizers"]
        ):
            model.load_state_dict(state)
            optimizer.load_state_dict(opt_state)
        self.agent_steps = np.asarray(payload["agent_steps"], dtype=np.int64)
        self._id_cursor = int(payload.get("id_cursor", 0)) % self.population_size
        self.total_env_steps = int(payload["total_env_steps"])
        self.update = int(payload["update"])

    def close(self):
        self.env.close()
        self.release_workspace()

    def release_workspace(self):
        """Drop the population learner replica before the main-stage learner loads."""
        self.workspace.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

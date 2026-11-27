"""CUDA rollout storage for the pinned SB3 PPO trainers."""

from typing import NamedTuple

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from torch.nn import functional as F

from .env import NoExecutableSubtask


def actor_logits(policy, observations):
    features = policy.extract_features(observations)
    if not policy.share_features_extractor:
        features = features[0]
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def actor_critic(policy, observations):
    features = policy.extract_features(observations)
    actor, critic = (features, features) if policy.share_features_extractor else features
    logits = policy.action_net(policy.mlp_extractor.forward_actor(actor))
    values = policy.value_net(policy.mlp_extractor.forward_critic(critic))
    return logits, values


def action_log_probs(logits, mask):
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    if mask is not None:
        log_probs = log_probs.masked_fill(~mask, -1e8)
        log_probs = log_probs - log_probs.logsumexp(dim=-1, keepdim=True)
    return log_probs


def sample_actions(logits, deterministic=False, generator=None):
    if deterministic:
        return logits.argmax(dim=-1)
    noise = torch.empty_like(logits).exponential_(generator=generator)
    return (logits - noise.log()).argmax(dim=-1)


def sample_policy(policy, observations, mask=None, *, deterministic=False, generator=None):
    # Categorical validation and probability materialization are unnecessary
    # for these fixed discrete heads and masks with an always-available action.
    logits, values = actor_critic(policy, observations)
    log_probs = action_log_probs(logits, mask)
    actions = sample_actions(log_probs, deterministic, generator)
    return actions, values, log_probs.gather(1, actions[:, None]).squeeze(1)


def configure_adam(optimizer, *, fused):
    """Change Adam's backend while retaining the update state and hyperparameters."""
    options = {"fused": True, "foreach": False} if fused else {"fused": None, "foreach": None}
    groups = [{**group, **options} for group in optimizer.param_groups]
    defaults = {**optimizer.defaults, **options}
    configured = torch.optim.Adam(groups, **defaults)
    state = optimizer.state_dict()
    for group in state["param_groups"]:
        group.update(options)
        if not group["fused"] and not group["capturable"]:
            for parameter in group["params"]:
                saved = state["state"].get(parameter)
                if saved and "step" in saved:
                    state["state"][parameter] = {**saved, "step": saved["step"].cpu()}
    configured.load_state_dict(state)
    return configured


def ppo_terms(values, log_prob, entropy, old_log_prob, advantages, returns,
              clip, ent_coef, vf_coef, normalize):
    if normalize and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(advantages * ratio, advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
    value_loss = F.mse_loss(values.flatten(), returns)
    entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
    loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
    with torch.no_grad():
        fraction = ((ratio - 1).abs() > clip).float().mean()
        approx_kl = (ratio - 1 - log_ratio).mean()
        metrics = torch.stack((entropy_loss, policy_loss, value_loss, fraction, approx_kl))
    return loss, metrics


def tensor_ppo_terms(values, logits, actions, mask, old_log_prob, advantages, returns,
                     clip, ent_coef, vf_coef, normalize):
    all_log_probs = action_log_probs(logits, mask)
    log_prob = all_log_probs.gather(1, actions).squeeze(1)
    terms = all_log_probs * all_log_probs.softmax(dim=-1)
    if mask is not None:
        terms = terms.masked_fill(~mask, 0.)
    entropy = -terms.sum(dim=-1)
    return ppo_terms(values, log_prob, entropy, old_log_prob, advantages, returns,
                     clip, ent_coef, vf_coef, normalize)


class TensorSamples(NamedTuple):
    observations: dict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor | None


class TensorRolloutBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, *,
                 gamma, gae_lambda, n_envs):
        self.shape = (buffer_size, n_envs)
        self.device, self.gamma, self.gae_lambda = device, gamma, gae_lambda
        # The last row is a sink for asynchronous Worker lanes that did not act.
        storage = (buffer_size + 1, n_envs)
        self.observations = {
            key: torch.empty((*storage, *space.shape), device=device,
                             dtype=(torch.bool if key == "subtask_mask" else
                                    torch.uint8 if space.shape == (27, 7, 7) else torch.float32))
            for key, space in observation_space.spaces.items()
        }
        # MixedKitchen has fractional features and an endogenous closing time.
        # Allocate terminal observations only for environments with time limits.
        self.terminal = None
        self.actions = torch.empty(storage, device=device, dtype=torch.long)
        self.rewards = torch.empty(storage, device=device)
        self.dones = torch.empty(storage, device=device, dtype=torch.bool)
        self.truncated = torch.empty_like(self.dones)
        self.value_tensor = torch.empty_like(self.rewards)
        self.log_probs = torch.empty_like(self.rewards)
        self.use_masking = False

    def add(self, step, observations, actions, values, log_probs, terminal, metadata):
        if terminal is not None and self.terminal is None:
            self.terminal = {key: torch.empty_like(value) for key, value in self.observations.items()}
        for key, storage in self.observations.items():
            storage[step] = observations[key].to(storage.dtype)
            if terminal is not None:
                self.terminal[key][step] = terminal[key].to(storage.dtype)
        self.actions[step] = actions.flatten()
        self.rewards[step] = metadata[:, 0]
        self.dones[step] = metadata[:, 1].bool()
        self.truncated[step] = metadata[:, 2].bool()
        self.value_tensor[step] = values.flatten()
        self.log_probs[step] = log_probs.flatten()

    def finish(self, policy, last_observation, batch_size):
        steps = self.shape[0]
        if self.terminal is not None:
            truncated = self.truncated[:steps].flatten().nonzero().flatten()
            for indices in truncated.split(batch_size):
                if not indices.numel():
                    continue
                terminal_obs = {key: value.flatten(0, 1)[indices] for key, value in self.terminal.items()}
                self.rewards.view(-1)[indices] += self.gamma * policy.predict_values(terminal_obs).flatten()
        last_values = policy.predict_values(last_observation).reshape(1, -1)
        next_values = torch.cat((self.value_tensor[1:steps], last_values))
        continue_episode = ~self.dones[:steps]
        self.advantages = self.rewards[:steps] + self.gamma * next_values * continue_episode - self.value_tensor[:steps]
        discount = continue_episode * (self.gamma * self.gae_lambda)
        # Combine adjacent suffix segments in parallel to evaluate the GAE recurrence.
        span = 1
        while span < self.shape[0]:
            prefix = self.advantages[:-span] + discount[:-span] * self.advantages[span:]
            self.advantages = torch.cat((prefix, self.advantages[-span:]))
            discount = torch.cat((discount[:-span] * discount[span:], discount[-span:]))
            span *= 2
        self.return_tensor = self.advantages + self.value_tensor[:steps]

    def get(self, batch_size):
        order = torch.randperm(self.advantages.numel(), device=self.device)
        flat_obs = {key: value.flatten(0, 1) for key, value in self.observations.items()}
        fields = (self.actions.view(-1, 1), self.value_tensor.flatten(), self.log_probs.flatten(),
                  self.advantages.flatten(), self.return_tensor.flatten())
        # Gather several minibatches together, then consume contiguous slices.
        # Preserve the shuffle order and bound the extra gathered storage.
        row_bytes = sum(value[0].numel() * value.element_size() for value in (*flat_obs.values(), *fields))
        block_size = batch_size * max(1, (32 * 1024**2) // (row_bytes * batch_size))
        for indices in order.split(block_size):
            observations = {key: value.index_select(0, indices) for key, value in flat_obs.items()}
            samples = tuple(field.index_select(0, indices) for field in fields)
            for start in range(0, len(indices), batch_size):
                part = slice(start, start + batch_size)
                obs = {key: value[part] for key, value in observations.items()}
                yield TensorSamples(obs, *(value[part] for value in samples),
                                    obs.get("subtask_mask") if self.use_masking else None)

    # SB3 uses these NumPy views once per PPO update for explained-variance logging.
    @property
    def values(self):
        return self.value_tensor[:self.shape[0]].cpu().numpy()

    @property
    def returns(self):
        return self.return_tensor.cpu().numpy()


class TensorCollection:
    def _setup_model(self):
        self.rollout_buffer_class = TensorRolloutBuffer
        super()._setup_model()
        self._tensor_loss_terms = (torch.compile(tensor_ppo_terms, fullgraph=True)
                                   if getattr(self.env, "compile_training", False) else tensor_ppo_terms)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "rollout_buffer_class", "_last_obs",
                "_last_episode_starts", "_tensor_loss_terms"]

    def train(self):
        if self.target_kl is not None or self.clip_range_vf is not None or self.use_sde:
            return super().train()

        # Use the same discrete PPO losses as SB3. Transfer the training metrics
        # once per update instead of synchronizing CUDA for each minibatch.
        self.policy.set_training_mode(True)
        optimizer = self.policy.optimizer
        self._update_learning_rate(optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        buffer = self.rollout_buffer
        tensor_policy = getattr(self.env, "tensor_policy", False)
        coefficients = (clip, self.ent_coef, self.vf_coef, self.normalize_advantage)
        epoch_metrics = []
        for _ in range(self.n_epochs):
            metrics = []
            for batch in buffer.get(self.batch_size):
                if tensor_policy:
                    logits, values = actor_critic(self.policy, batch.observations)
                    loss, stats = self._tensor_loss_terms(
                        values, logits, batch.actions, batch.action_masks, batch.old_log_prob,
                        batch.advantages, batch.returns, *coefficients,
                    )
                else:
                    options = {"action_masks": batch.action_masks} if self.masked else {}
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        batch.observations, batch.actions.flatten(), **options,
                    )
                    loss, stats = ppo_terms(values, log_prob, entropy, batch.old_log_prob,
                                            batch.advantages, batch.returns, *coefficients)
                metrics.append(stats)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                optimizer.step()
            epoch_metrics.append(torch.stack(metrics).mean(0))

        self._n_updates += self.n_epochs
        returns = buffer.return_tensor
        variance = returns.var(correction=0)
        errors = returns - buffer.value_tensor[:buffer.shape[0]]
        explained = torch.where(variance == 0, torch.nan, 1 - errors.var(correction=0) / variance)
        means = torch.stack(epoch_metrics).mean(0)
        stats = torch.cat((means[:4], torch.stack((epoch_metrics[-1][4], loss.detach(), explained))))
        names = ("entropy_loss", "policy_gradient_loss", "value_loss", "clip_fraction",
                 "approx_kl", "loss", "explained_variance")
        for name, value in zip(names, stats.tolist()):
            self.logger.record(f"train/{name}", value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip)

    @torch.no_grad()
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        self.policy.set_training_mode(False)
        rollout_buffer.use_masking = use_masking
        observations = env.tensor_observations
        tensor_policy = getattr(env, "tensor_policy", False)
        callback.on_rollout_start()
        metadata_batch = []
        first_timestep = self.num_timesteps
        for step in range(n_rollout_steps):
            mask = observations["subtask_mask"] if use_masking else None
            if tensor_policy:
                actions, values, log_probs = sample_policy(self.policy, observations, mask)
            else:
                actions, values, log_probs = self.policy(observations, action_masks=mask)
            next_observations, terminal, metadata = env.step_tensors(actions.flatten())
            rollout_buffer.add(step, observations, actions, values, log_probs, terminal, metadata)
            metadata_batch.append(metadata)
            observations = next_observations
            self.num_timesteps += env.num_envs
            if not callback.on_step():
                self._flush_events(env, callback, metadata_batch, first_timestep)
                return False
        rollout_buffer.finish(self.policy, observations, self.batch_size)
        self._flush_events(env, callback, metadata_batch, first_timestep)
        callback.on_rollout_end()
        return True

    def _flush_events(self, env, callback, metadata, first_timestep):
        # CUDA nonzero would synchronize to obtain the output length before
        # transferring the records. Filter the single bulk copy on the host.
        events = env.rollout_events(torch.stack(metadata).cpu().numpy())
        self._write_events(callback, [(first_timestep + (step + 1) * env.num_envs, info)
                                      for step, _, info in events])

    def _write_events(self, callback, records):
        if hasattr(callback, "write_events"):
            callback.write_events(records)
        else:
            for timestep, info in records:
                callback.write_event(timestep, info)
        self._update_info_buffer([info for _, info in records])


class TensorPPO(TensorCollection, PPO):
    masked = False

    @torch.no_grad()
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        observations = env.tensor_observations
        tensor_policy = getattr(env, "tensor_policy", False)
        first_timestep = self.num_timesteps
        cursors = torch.zeros_like(env.kitchen.rows)
        samples = torch.zeros((), device=self.device, dtype=torch.long)
        timestamps = torch.zeros((n_rollout_steps + 1, env.num_envs), device=self.device, dtype=torch.long)
        events = None
        interval = getattr(env, "worker_sync_interval", 8)
        ticks = 0
        while True:
            # Check host deadlines after several kitchen ticks. Each lane advances
            # its own cursor only when it has a skill and takes a policy action.
            block = min(interval, n_rollout_steps - ticks) if ticks < n_rollout_steps else interval
            for _ in range(block):
                if tensor_policy:
                    actions, values, log_probs = sample_policy(self.policy, observations)
                else:
                    actions, values, log_probs = self.policy(observations)
                next_obs, terminal, metadata, collected = env.step_worker_tick(
                    actions.flatten(), cursors < n_rollout_steps,
                )
                positions = (torch.where(collected, cursors, n_rollout_steps), env.kitchen.rows)
                rollout_buffer.add(positions, observations, actions, values, log_probs, terminal, metadata)
                if events is None:
                    events = metadata.new_zeros((n_rollout_steps + 1, *metadata.shape))
                samples += collected.sum()
                timestamps[positions] = first_timestep + samples
                events[positions] = metadata
                cursors += collected
                observations = next_obs
            ticks += block
            count, exhausted = torch.stack((samples, env.exhausted.any())).tolist()
            self.num_timesteps = first_timestep + count
            if exhausted:
                raise NoExecutableSubtask("No executable Worker skill after eight kitchen horizons")
            if not callback.on_step():
                self._flush_worker_events(env, callback, events, timestamps)
                return False
            if count == n_rollout_steps * env.num_envs:
                break
        rollout_buffer.finish(self.policy, observations, self.batch_size)
        self._flush_worker_events(env, callback, events, timestamps)
        callback.on_rollout_end()
        return True

    def _flush_worker_events(self, env, callback, events, timestamps):
        times = timestamps[:-1].cpu().numpy()
        records = list(env.rollout_events(events[:-1].cpu().numpy()))
        records.sort(key=lambda record: (times[record[0], record[1]], record[1]))
        self._write_events(callback, [(int(times[step, index]), info) for step, index, info in records])


class TensorMaskablePPO(TensorCollection, MaskablePPO):
    masked = True

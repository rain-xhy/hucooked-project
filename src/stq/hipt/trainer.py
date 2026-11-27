"""Shared-parameter three-chef HiPT best-response trainer."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .common import OptionClock, PendingOption, gae, high_level_gae, ppo_components, sample_durations
from .model import (
    FlatActorCritic,
    HIDDEN_SIZE,
    HiPTPolicy,
    influence_kl,
    SPATIAL_CHANNELS,
    SPATIAL_HEIGHT,
    SPATIAL_WIDTH,
    VECTOR_DIM,
)


class HierarchicalTrainer:
    """Train a single hierarchical actor shared by all three chef roles.

    Every env/role stream has a distinct LSTM state and option clock. Half the
    episodes default to three-agent self-play. Other episodes use two current
    HiPT actors and one fixed historical population snapshot; the fixed slot is
    rotated between chef roles over episode resets.
    """

    def __init__(self, env, population_snapshots, *, num_envs: int,
                 population_size: int, z_dim: int, device: torch.device,
                 seed: int, epochs: int = 4, minibatches: int = 4,
                 learning_rate: float = 3e-4, gamma: float = 0.9985,
                 gae_lambda: float = 0.95, clip: float = 0.2,
                 value_coef: float = 0.5, low_entropy_coef: float = 0.01,
                 high_entropy_coef: float = 0.01, max_grad_norm: float = 1.0,
                 p_min: int = 20, p_max: int = 40,
                 selfplay_fraction: float = 0.5, target_env_steps: int = 1,
                 model=None, optimizer=None,
                 initial_update: int = 0, bptt_context: int = 128):
        if not 0.0 <= selfplay_fraction <= 1.0:
            raise ValueError("selfplay_fraction must be in [0, 1]")
        if not 20 <= p_min <= p_max <= 40:
            raise ValueError("HiPT option duration must remain within [20, 40]")
        self.env = env
        self.num_envs = int(num_envs)
        self.population_size = int(population_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.epochs, self.minibatches = int(epochs), int(minibatches)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.clip, self.value_coef = float(clip), float(value_coef)
        self.low_entropy_coef, self.high_entropy_coef = float(low_entropy_coef), float(high_entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.bptt_context = max(1, int(bptt_context))
        self.p_min, self.p_max = int(p_min), int(p_max)
        self.selfplay_fraction = float(selfplay_fraction)
        self.target_env_steps = max(1, int(target_env_steps))
        self.model = (model or HiPTPolicy(z_dim=z_dim)).to(device)
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, eps=1e-5
        )
        self.update = int(initial_update)
        self.total_env_steps = 0
        self.role_steps = np.zeros(3, dtype=np.int64)
        self.obs = None
        self.hidden = None
        self.previous_done = np.zeros(self.num_envs, dtype=bool)
        self.clock = OptionClock((self.num_envs, 3))
        self.pending: list[list[dict | None]] = [
            [None for _ in range(3)] for _ in range(self.num_envs)
        ]
        self.episode_histories: list[list[list]] = [
            [[] for _ in range(3)] for _ in range(self.num_envs)
        ]
        self.episode_index = np.zeros(self.num_envs, dtype=np.int64)
        self.selfplay = np.ones(self.num_envs, dtype=bool)
        self.partner_role = np.zeros(self.num_envs, dtype=np.int64)
        self.partner_profile = np.zeros(self.num_envs, dtype=np.int64)
        self.learner_mask = np.ones((self.num_envs, 3), dtype=bool)
        self.recent_team_scores = deque(maxlen=512)
        self.pool_models = self._make_pool_models(population_snapshots)
        self.pool_profiles = len(self.pool_models)
        if self.pool_profiles != 3 * self.population_size:
            raise ValueError("Expected initial/mid/final pool snapshots for every population actor")

    def _make_pool_models(self, snapshots):
        expected = {"initial", "mid", "final"}
        if set(snapshots) != expected:
            raise ValueError("Population snapshots must include initial, mid, and final levels")
        models = []
        for level in ("initial", "mid", "final"):
            entries = snapshots[level]
            if len(entries) != self.population_size:
                raise ValueError(f"Pool level {level!r} has the wrong population size")
            for state in entries:
                model = FlatActorCritic().to(self.device)
                model.load_state_dict(state)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                models.append(model)
        return models

    def _assign_episode(self, env_index: int):
        self.selfplay[env_index] = self.rng.random() < self.selfplay_fraction
        if self.selfplay[env_index]:
            self.learner_mask[env_index] = True
            self.partner_role[env_index] = -1
            self.partner_profile[env_index] = -1
            return
        # Round-robin role by episode and vector slot so all chefs take their
        # turn as the fixed population partner.
        role = int((self.episode_index[env_index] + env_index) % 3)
        self.partner_role[env_index] = role
        self.learner_mask[env_index] = True
        self.learner_mask[env_index, role] = False
        self.partner_profile[env_index] = int(self.rng.integers(self.pool_profiles))

    def _pool_actions(self, obs, actions):
        mixed_envs = np.flatnonzero(~self.selfplay)
        if not len(mixed_envs):
            return actions
        spatial = obs["spatial"]
        vector = obs["vector"]
        profile_ids = self.partner_profile[mixed_envs]
        with torch.no_grad():
            for profile in np.unique(profile_ids):
                rows = mixed_envs[profile_ids == profile]
                roles = self.partner_role[rows]
                x = torch.as_tensor(spatial[rows, roles], device=self.device)
                v = torch.as_tensor(vector[rows, roles], device=self.device)
                logits, _ = self.pool_models[int(profile)](x, v)
                selected = Categorical(logits=logits).sample().cpu().numpy()
                actions[rows, roles] = selected
        return actions

    def _finish_pending(self, env_index, role, bootstrap, terminal, ready, update_index):
        wrapper = self.pending[env_index][role]
        if wrapper is None:
            return
        record = wrapper["option"].finish(
            bootstrap_value=bootstrap, terminal=terminal
        )
        record["cross_rollout"] = wrapper["started_update"] < update_index
        record["policy_update_lag"] = max(0, update_index - wrapper["started_update"])
        ready[env_index * 3 + role].append(record)
        self.pending[env_index][role] = None

    def _high_policy_outputs(self, records):
        """Recompute decision-state distributions with recurrent BPTT context.

        Each macro keeps the episode observation prefix through its z decision.
        Replay is capped to one rollout's worth of context; the detached LSTM
        state saved at the start of that context is the standard truncated-BPTT
        boundary. Multiple decisions from one episode share one recurrent pass.
        """
        logps = [None] * len(records)
        values = [None] * len(records)
        entropies = [None] * len(records)
        by_history = {}
        for index, record in enumerate(records):
            context_start = max(0, record["decision_index"] - self._bptt_context + 1)
            key = (id(record["history"]), context_start)
            if key not in by_history:
                by_history[key] = (record["history"], [])
            by_history[key][1].append((index, record))
        windows = []
        for history, entries in by_history.values():
            max_index = max(record["decision_index"] for _, record in entries)
            start_index = max(0, max_index - self._bptt_context + 1)
            rows = history[start_index:max_index + 1]
            windows.append((history, entries, start_index, rows))

        # Pack a bounded number of separate episode/context windows into each
        # recurrent pass. Padding is masked by terminal-reset flags and outputs
        # are gathered only at real high-level decision indices.
        for offset in range(0, len(windows), 16):
            packed = windows[offset:offset + 16]
            max_time = max(len(window[3]) for window in packed)
            batch_size = len(packed)
            spatial_np = np.zeros((
                max_time, batch_size, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
            ), dtype=np.float32)
            vector_np = np.zeros((max_time, batch_size, VECTOR_DIM), dtype=np.float32)
            done_np = np.ones((max_time, batch_size), dtype=np.float32)
            h0_np = np.zeros((batch_size, HIDDEN_SIZE), dtype=np.float32)
            c0_np = np.zeros((batch_size, HIDDEN_SIZE), dtype=np.float32)
            for batch_index, (_, _, _, rows) in enumerate(packed):
                h0_np[batch_index] = rows[0]["h"]
                c0_np[batch_index] = rows[0]["c"]
                for t, row in enumerate(rows):
                    spatial_np[t, batch_index] = row["spatial"]
                    vector_np[t, batch_index] = row["vector"]
                    done_np[t, batch_index] = float(row["done"])
            spatial = torch.as_tensor(spatial_np, device=self.device)
            vector = torch.as_tensor(vector_np, device=self.device)
            done = torch.as_tensor(done_np, device=self.device)
            h0 = torch.as_tensor(h0_np, device=self.device).unsqueeze(0)
            c0 = torch.as_tensor(c0_np, device=self.device).unsqueeze(0)
            high_logits, high_values, _, _, _, _ = self.model.unroll(
                spatial, vector, (h0, c0), done
            )
            for batch_index, (_, entries, start_index, _) in enumerate(packed):
                for original_index, record in entries:
                    row_index = record["decision_index"] - start_index
                    distribution = Categorical(logits=high_logits[row_index, batch_index])
                    action = torch.as_tensor(record["z"], device=self.device)
                    logps[original_index] = distribution.log_prob(action)
                    values[original_index] = high_values[row_index, batch_index]
                    entropies[original_index] = distribution.entropy()
        return torch.stack(logps), torch.stack(values), torch.stack(entropies)

    def _joint_update(self, batch, initial_state, ready_records,
                      low_entropy_coef, high_entropy_coef):
        """Update low and high objectives together on each shared-encoder step."""
        high_level_gae(ready_records, self.gamma, self.gae_lambda)
        high_records = [record for stream in ready_records for record in stream]
        spatial = batch["spatial"]
        vector = batch["vector"]
        time_steps = spatial.shape[0]
        slot_count = min(self.minibatches, self.num_envs)
        low_rows, high_rows = [], []

        for _ in range(self.epochs):
            low_order = self.rng.permutation(self.num_envs)
            low_batches = np.array_split(low_order, slot_count)
            if high_records:
                high_order = self.rng.permutation(len(high_records))
                high_batches = np.array_split(high_order, slot_count)
            else:
                high_batches = [np.empty(0, dtype=np.int64) for _ in range(slot_count)]

            for env_indices, record_indices in zip(low_batches, high_batches):
                joint_loss = None
                if len(env_indices):
                    x = torch.as_tensor(spatial[:, env_indices], device=self.device).reshape(
                        time_steps, -1, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
                    )
                    v = torch.as_tensor(vector[:, env_indices], device=self.device).reshape(
                        time_steps, -1, VECTOR_DIM
                    )
                    done = torch.as_tensor(
                        np.repeat(batch["done_before"][:, env_indices, None], 3, axis=2),
                        device=self.device,
                    ).reshape(time_steps, -1)
                    h0 = initial_state[0].reshape(
                        1, self.num_envs, 3, HIDDEN_SIZE
                    )[:, env_indices]
                    c0 = initial_state[1].reshape(
                        1, self.num_envs, 3, HIDDEN_SIZE
                    )[:, env_indices]
                    h0, c0 = h0.reshape(1, -1, HIDDEN_SIZE), c0.reshape(1, -1, HIDDEN_SIZE)
                    _, _, low_logits, low_values, _, _ = self.model.unroll(
                        x, v, (h0, c0), done
                    )
                    z = torch.as_tensor(batch["z"][:, env_indices], device=self.device).reshape(
                        time_steps, -1
                    )
                    gather_z = z.unsqueeze(-1).unsqueeze(-1).expand(
                        -1, -1, 1, self.model.action_dim
                    )
                    selected_logits = low_logits.gather(2, gather_z).squeeze(2)
                    selected_value = low_values.gather(2, z.unsqueeze(-1)).squeeze(-1)
                    action = torch.as_tensor(
                        batch["action"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    old_logp = torch.as_tensor(
                        batch["logp"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    old_value = torch.as_tensor(
                        batch["value"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    advantage = torch.as_tensor(
                        batch["advantage"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    returns = torch.as_tensor(
                        batch["return"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    learner = torch.as_tensor(
                        batch["learner"][:, env_indices], device=self.device
                    ).reshape(time_steps, -1)
                    if bool(learner.any()):
                        distribution = Categorical(logits=selected_logits[learner])
                        low_pg, low_value_loss, low_entropy, low_kl = ppo_components(
                            distribution.log_prob(action[learner]), old_logp[learner],
                            advantage[learner], selected_value[learner], old_value[learner],
                            returns[learner], distribution.entropy(), self.clip,
                        )
                        low_loss = (low_pg + self.value_coef * low_value_loss -
                                    low_entropy_coef * low_entropy)
                        joint_loss = low_loss if joint_loss is None else joint_loss + low_loss
                        low_rows.append((float(low_loss.detach().cpu()),
                                         float(low_pg.detach().cpu()),
                                         float(low_value_loss.detach().cpu()),
                                         float(low_entropy.detach().cpu()),
                                         float(low_kl.detach().cpu())))

                if len(record_indices):
                    minibatch = [high_records[int(index)] for index in record_indices]
                    new_logp, new_value, entropy = self._high_policy_outputs(minibatch)
                    old_logp = torch.as_tensor(
                        [record["old_logp"] for record in minibatch], device=self.device
                    )
                    old_value = torch.as_tensor(
                        [record["old_value"] for record in minibatch], device=self.device
                    )
                    advantage = torch.as_tensor(
                        [record["advantage"] for record in minibatch], device=self.device
                    )
                    returns = torch.as_tensor(
                        [record["return"] for record in minibatch], device=self.device
                    )
                    high_pg, high_value_loss, high_entropy, high_kl = ppo_components(
                        new_logp, old_logp, advantage, new_value, old_value, returns,
                        entropy, self.clip,
                    )
                    high_loss = (high_pg + self.value_coef * high_value_loss -
                                 high_entropy_coef * high_entropy)
                    joint_loss = high_loss if joint_loss is None else joint_loss + high_loss
                    high_rows.append((float(high_pg.detach().cpu()),
                                      float(high_value_loss.detach().cpu()),
                                      float(high_entropy.detach().cpu()),
                                      float(high_kl.detach().cpu())))

                if joint_loss is None:
                    continue
                if not torch.isfinite(joint_loss):
                    raise FloatingPointError("Nonfinite joint HiPT loss")
                self.optimizer.zero_grad(set_to_none=True)
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()

        low_metrics = {
            "loss": float(np.mean([row[0] for row in low_rows])) if low_rows else None,
            "policy_loss": float(np.mean([row[1] for row in low_rows])) if low_rows else None,
            "value_loss": float(np.mean([row[2] for row in low_rows])) if low_rows else None,
            "entropy": float(np.mean([row[3] for row in low_rows])) if low_rows else None,
            "approx_kl": float(np.mean([row[4] for row in low_rows])) if low_rows else None,
        }
        high_metrics = {
            "policy_loss": float(np.mean([row[0] for row in high_rows])) if high_rows else None,
            "value_loss": float(np.mean([row[1] for row in high_rows])) if high_rows else None,
            "entropy": float(np.mean([row[2] for row in high_rows])) if high_rows else None,
            "approx_kl": float(np.mean([row[3] for row in high_rows])) if high_rows else None,
            "transitions": len(high_records),
            "cross_rollout_transitions": sum(bool(record["cross_rollout"]) for record in high_records),
            "max_policy_update_lag": max(
                (int(record["policy_update_lag"]) for record in high_records), default=0
            ),
        }
        return low_metrics, high_metrics

    def collect_and_update(self, rollout_steps: int, influence_coef: float):
        if self.obs is None:
            self.obs = self.env.reset()
            for env_index in range(self.num_envs):
                self._assign_episode(env_index)
            self.hidden = self.model.initial_state(self.num_envs * 3, device=self.device)

        started = time.monotonic()
        state_start = tuple(value.detach().clone() for value in self.hidden)
        ready = [[] for _ in range(self.num_envs * 3)]
        spatial_rows, vector_rows = [], []
        actions_rows, z_rows, logp_rows, value_rows = [], [], [], []
        reward_rows, done_rows, done_before_rows, learner_rows = [], [], [], []
        high_kl_rows, task_reward_sum = [], 0.0
        rollout_start_update = self.update

        for _ in range(rollout_steps):
            done_before = self.previous_done.copy()
            state_before = tuple(value.detach().clone() for value in self.hidden)
            spatial_np = self.obs["spatial"]
            vector_np = self.obs["vector"]
            done_roles_np = np.repeat(done_before[:, None], 3, axis=1).reshape(-1)
            spatial = torch.as_tensor(spatial_np, device=self.device).reshape(
                self.num_envs * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
            )
            vector = torch.as_tensor(vector_np, device=self.device).reshape(
                self.num_envs * 3, VECTOR_DIM
            )
            with torch.no_grad():
                high_logits, high_value, low_logits, low_values, next_state = self.model.step(
                    spatial, vector, self.hidden,
                    torch.as_tensor(done_roles_np, device=self.device, dtype=torch.float32),
                )
            self.hidden = next_state
            high_logits = high_logits.reshape(self.num_envs, 3, -1)
            high_value = high_value.reshape(self.num_envs, 3)
            low_logits = low_logits.reshape(self.num_envs, 3, self.model.z_dim,
                                            self.model.action_dim)
            low_values = low_values.reshape(self.num_envs, 3, self.model.z_dim)

            for env_index in range(self.num_envs):
                for role in range(3):
                    history = self.episode_histories[env_index][role]
                    flat_index = env_index * 3 + role
                    history.append({
                        "spatial": spatial_np[env_index, role].copy(),
                        "vector": vector_np[env_index, role].copy(),
                        "done": bool(done_before[env_index]),
                        "h": state_before[0][0, flat_index].detach().cpu().numpy().copy(),
                        "c": state_before[1][0, flat_index].detach().cpu().numpy().copy(),
                    })
                    if self.clock.remaining[env_index, role] == 0 and \
                            self.pending[env_index][role] is not None:
                        self._finish_pending(env_index, role, high_value[env_index, role],
                                             False, ready, self.update)

            with torch.no_grad():
                high_dist = Categorical(logits=high_logits)
                proposed_z = high_dist.sample().cpu().numpy()
                high_logp = high_dist.log_prob(torch.as_tensor(proposed_z, device=self.device)).cpu().numpy()
                due = self.clock.due.copy()
                chosen_z = np.where(due, proposed_z, self.clock.z)
                durations = sample_durations(self.rng, (self.num_envs, 3), self.p_min, self.p_max)
                self.clock.start(due, chosen_z, durations)
                learner = self.learner_mask.copy()
                for env_index, role in zip(*np.nonzero(due & learner)):
                    history = self.episode_histories[env_index][role]
                    self.pending[env_index][role] = {
                        "option": PendingOption(
                            history=history,
                            decision_index=len(history) - 1,
                            z=int(chosen_z[env_index, role]),
                            old_logp=float(high_logp[env_index, role]),
                            old_value=float(high_value[env_index, role].detach().cpu()),
                        ),
                        "started_update": self.update,
                    }
                chosen_z_t = torch.as_tensor(chosen_z.reshape(-1), device=self.device)
                low_logits_flat = low_logits.reshape(-1, self.model.z_dim, self.model.action_dim)
                low_values_flat = low_values.reshape(-1, self.model.z_dim)
                low_indices = chosen_z_t[:, None, None].expand(-1, 1, self.model.action_dim)
                selected_low_logits = low_logits_flat.gather(1, low_indices).squeeze(1)
                low_dist = Categorical(logits=selected_low_logits)
                sampled_actions = low_dist.sample()
                low_logp = low_dist.log_prob(sampled_actions).cpu().numpy().reshape(
                    self.num_envs, 3
                )
                selected_value = low_values_flat.gather(1, chosen_z_t[:, None]).squeeze(1)
                kl = influence_kl(
                    low_logits_flat, high_logits.reshape(-1, self.model.z_dim), chosen_z_t
                ).reshape(self.num_envs, 3).cpu().numpy()
                model_actions = sampled_actions.cpu().numpy().reshape(self.num_envs, 3)

            actions = self._pool_actions(self.obs, model_actions.copy())
            next_obs, rewards, done, infos = self.env.step(actions)
            for env_index, role in zip(*np.nonzero(learner)):
                pending = self.pending[env_index][role]
                if pending is not None:
                    pending["option"].add_step(float(rewards[env_index]) +
                                                float(influence_coef) * float(kl[env_index, role]))
            task_reward_sum += float(np.sum(rewards))
            self.role_steps += learner.sum(axis=0)

            spatial_rows.append(spatial_np.copy())
            vector_rows.append(vector_np.copy())
            actions_rows.append(model_actions.copy())
            z_rows.append(chosen_z.copy())
            logp_rows.append(low_logp.copy())
            value_rows.append(selected_value.detach().cpu().numpy().reshape(self.num_envs, 3))
            reward_rows.append(rewards.copy())
            done_rows.append(done.copy())
            done_before_rows.append(done_before.copy())
            learner_rows.append(learner)
            high_kl_rows.append(kl.copy())

            self.clock.advance(done)
            for env_index, ended in enumerate(done):
                if not ended:
                    continue
                for role in range(3):
                    self._finish_pending(env_index, role, 0.0, True, ready, self.update)
                    self.episode_histories[env_index][role] = []
                self.episode_index[env_index] += 1
                self._assign_episode(env_index)
                summary = infos[env_index].get("episode_summary")
                if summary is not None:
                    self.recent_team_scores.append(float(summary["team_score"]))
            self.obs = next_obs
            self.previous_done = done.copy()

        # The low policy's rollout bootstrap uses the actual next observation
        # and the current z, with true terminals masked by the last done flag.
        spatial_next = torch.as_tensor(self.obs["spatial"], device=self.device).reshape(
            self.num_envs * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        vector_next = torch.as_tensor(self.obs["vector"], device=self.device).reshape(
            self.num_envs * 3, VECTOR_DIM
        )
        done_next = torch.as_tensor(
            np.repeat(self.previous_done[:, None], 3, axis=1).reshape(-1),
            device=self.device, dtype=torch.float32,
        )
        with torch.no_grad():
            next_high_logits, _, _, next_low_values, _ = self.model.step(
                spatial_next, vector_next, self.hidden, done_next
            )
            next_high_probs = next_high_logits.softmax(-1).reshape(
                self.num_envs, 3, self.model.z_dim
            )
            next_low_values = next_low_values.reshape(self.num_envs, 3, self.model.z_dim)
            z_now = torch.as_tensor(self.clock.z, device=self.device).reshape(-1)
            selected_bootstrap = next_low_values.reshape(-1, self.model.z_dim).gather(
                1, z_now[:, None]
            ).squeeze(-1)
            expected_bootstrap = (next_high_probs * next_low_values).sum(-1).reshape(-1)
            due_bootstrap = torch.as_tensor(self.clock.due.reshape(-1), device=self.device)
            selected_bootstrap = torch.where(due_bootstrap, expected_bootstrap, selected_bootstrap)
            low_bootstrap = selected_bootstrap.cpu().numpy().reshape(self.num_envs, 3)

        batch = {
            "spatial": np.asarray(spatial_rows, dtype=np.float32),
            "vector": np.asarray(vector_rows, dtype=np.float32),
            "action": np.asarray(actions_rows, dtype=np.int64),
            "z": np.asarray(z_rows, dtype=np.int64),
            "logp": np.asarray(logp_rows, dtype=np.float32),
            "value": np.asarray(value_rows, dtype=np.float32),
            "reward": np.repeat(np.asarray(reward_rows, dtype=np.float32)[:, :, None], 3, axis=2),
            "done": np.asarray(done_rows, dtype=np.float32),
            "done_before": np.asarray(done_before_rows, dtype=np.float32),
            "learner": np.asarray(learner_rows, dtype=bool),
        }
        low_adv, low_return = gae(batch["reward"], batch["value"], batch["done"],
                                  low_bootstrap, self.gamma, self.gae_lambda)
        batch["advantage"], batch["return"] = low_adv, low_return
        progress = min(1.0, self.total_env_steps / self.target_env_steps)
        low_entropy_coef = self.low_entropy_coef * (1.0 - progress)
        high_entropy_coef = self.high_entropy_coef * (1.0 - progress)
        low_metrics, high_metrics = self._joint_update(
            batch, state_start, ready, low_entropy_coef, high_entropy_coef
        )
        self.total_env_steps += self.num_envs * rollout_steps
        self.update += 1
        pending_count = sum(item is not None for row in self.pending for item in row)
        metrics = {
            "stage": "hierarchical",
            "update": self.update,
            "stage_steps": self.total_env_steps,
            "low": low_metrics,
            "high": high_metrics,
            "influence_coef": float(influence_coef),
            "low_entropy_coef": float(low_entropy_coef),
            "high_entropy_coef": float(high_entropy_coef),
            "influence_per_learner_action": float(
                np.sum(np.asarray(high_kl_rows) * np.asarray(learner_rows)) /
                max(1, int(np.sum(learner_rows)))
            ),
            "task_reward_per_env_step": task_reward_sum /
            max(1, self.num_envs * rollout_steps),
            "selfplay_episodes_fraction_current": float(self.selfplay.mean()),
            "learner_actions_by_role": self.role_steps.tolist(),
            "mean_recent_team_score": (
                float(np.mean(self.recent_team_scores)) if self.recent_team_scores else None
            ),
            "pending_macro_options": pending_count,
            "pending_options_crossing_rollout": sum(
                item is not None and item["started_update"] < rollout_start_update
                for row in self.pending for item in row
            ),
            "update_seconds": time.monotonic() - started,
        }
        return metrics

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_env_steps": self.total_env_steps,
            "update": self.update,
            "role_steps": self.role_steps.copy(),
        }

    def load_state_dict(self, payload):
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.total_env_steps = int(payload.get("total_env_steps", 0))
        self.update = int(payload.get("update", 0))
        self.role_steps = np.asarray(payload.get("role_steps", [0, 0, 0]), dtype=np.int64)

    @property
    def _bptt_context(self):
        return self.bptt_context

    def close(self):
        self.env.close()

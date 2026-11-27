"""Three-chef central-state HiPT trainer with independent options per role."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .common import OptionClock, PendingOption, gae, high_level_gae, ppo_components, sample_durations
from .model import (
    CentralFlatActorCritic, HIDDEN_SIZE, JointHiPTPolicy, SPATIAL_CHANNELS, SPATIAL_HEIGHT,
    SPATIAL_WIDTH, influence_kl,
)


class HierarchicalTrainer:
    """One central LSTM state feeds three independent HiPT option streams."""

    def __init__(self, env, population_snapshots, *, num_envs, population_size=16,
                 z_dim=4, device="cuda:0", seed=1, epochs=1, minibatches=4,
                 learning_rate=1e-5, gamma=0.9985, gae_lambda=0.95,
                 clip=0.2, value_coef=0.5, low_entropy_coef=0.01,
                 high_entropy_coef=0.01, max_grad_norm=1.0, p_min=20, p_max=40,
                 selfplay_fraction=0.5, target_env_steps=1, model=None,
                 optimizer=None, initial_update=0, bptt_context=300):
        if not 0. <= selfplay_fraction <= 1.:
            raise ValueError("selfplay_fraction must be in [0,1]")
        if not 20 <= p_min <= p_max <= 40:
            raise ValueError("HiPT option duration must satisfy 20 <= p_min <= p_max <= 40")
        self.env, self.num_envs = env, int(num_envs)
        self.population_size, self.device = int(population_size), torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.epochs, self.minibatches = int(epochs), int(minibatches)
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.clip, self.value_coef = float(clip), float(value_coef)
        self.low_entropy_coef, self.high_entropy_coef = float(low_entropy_coef), float(high_entropy_coef)
        self.max_grad_norm, self.p_min, self.p_max = float(max_grad_norm), int(p_min), int(p_max)
        self.selfplay_fraction = float(selfplay_fraction)
        self.target_env_steps = max(1, int(target_env_steps))
        self.bptt_context = max(1, int(bptt_context))
        self.model = (model or JointHiPTPolicy(z_dim=z_dim)).to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.update, self.total_env_steps = int(initial_update), 0
        self.role_steps = np.zeros(3, dtype=np.int64)
        self.obs, self.hidden = None, None
        self.previous_done = np.zeros(self.num_envs, dtype=bool)
        self.clock = OptionClock(self.num_envs)
        self.pending = [[None for _ in range(3)] for _ in range(self.num_envs)]
        self.episode_histories = [[] for _ in range(self.num_envs)]
        self.episode_index = np.zeros(self.num_envs, dtype=np.int64)
        self.selfplay = np.ones(self.num_envs, dtype=bool)
        self.partner_role = np.zeros(self.num_envs, dtype=np.int64)
        self.partner_profile = np.zeros(self.num_envs, dtype=np.int64)
        self.learner_mask = np.ones((self.num_envs, 3), dtype=bool)
        self.recent_team_scores = deque(maxlen=512)
        self.pool_models = self._make_pool_models(population_snapshots)

    def _make_pool_models(self, snapshots):
        if set(snapshots) != {"initial", "mid", "final"}:
            raise ValueError("Partner snapshots must include initial, mid, and final")
        models = []
        for level in ("initial", "mid", "final"):
            entries = snapshots[level]
            if len(entries) != self.population_size:
                raise ValueError(f"Pool level {level!r} has wrong population size")
            for state in entries:
                model = CentralFlatActorCritic().cpu()
                model.load_state_dict(state)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                models.append(model)
        return models

    def _assign_episode(self, env_index):
        self.selfplay[env_index] = self.rng.random() < self.selfplay_fraction
        if self.selfplay[env_index]:
            self.learner_mask[env_index] = True
            self.partner_role[env_index] = -1
            self.partner_profile[env_index] = -1
            return
        role = int((self.episode_index[env_index] + env_index) % 3)
        self.partner_role[env_index] = role
        self.learner_mask[env_index] = True
        self.learner_mask[env_index, role] = False
        self.partner_profile[env_index] = int(self.rng.integers(len(self.pool_models)))

    @torch.no_grad()
    def _pool_actions(self, obs, actions):
        mixed = np.flatnonzero(~self.selfplay)
        if not len(mixed):
            return actions
        spatial, state = obs["spatial"][mixed, 0], obs["state"][mixed]
        profiles = self.partner_profile[mixed]
        for profile in np.unique(profiles):
            rows = mixed[profiles == profile]
            roles = self.partner_role[rows]
            model = self.pool_models[int(profile)]
            logits, _ = model(
                torch.as_tensor(obs["spatial"][rows, 0], dtype=torch.float32),
                torch.as_tensor(obs["state"][rows], dtype=torch.float32),
            )
            selected = Categorical(logits=logits[torch.arange(len(rows)), roles]).sample()
            actions[rows, roles] = selected.numpy()
        return actions

    def _finish_pending(self, env_index, role, bootstrap, terminal, ready, update_index):
        pending = self.pending[env_index][role]
        if pending is None:
            return
        record = pending.finish(bootstrap_value=bootstrap, terminal=terminal)
        record["cross_rollout"] = pending.started_update < update_index
        record["policy_update_lag"] = max(0, update_index - pending.started_update)
        ready[env_index * 3 + role].append(record)
        self.pending[env_index][role] = None

    def _high_policy_outputs(self, records):
        logps = [None] * len(records)
        values = [None] * len(records)
        entropies = [None] * len(records)
        groups = {}
        for index, record in enumerate(records):
            context_start = max(0, record["decision_index"] - self.bptt_context + 1)
            key = (id(record["history"]), context_start)
            groups.setdefault(key, (record["history"], []))[1].append((index, record))
        windows = []
        for history, entries in groups.values():
            max_index = max(record["decision_index"] for _, record in entries)
            start_index = max(0, max_index - self.bptt_context + 1)
            windows.append((history, entries, start_index,
                            history[start_index:max_index + 1]))

        for offset in range(0, len(windows), self.minibatches):
            packed = windows[offset:offset + self.minibatches]
            max_time = max(len(rows) for _, _, _, rows in packed)
            batch_size = len(packed)
            spatial_np = np.zeros((max_time, batch_size, SPATIAL_CHANNELS,
                                   SPATIAL_HEIGHT, SPATIAL_WIDTH), np.float32)
            state_np = np.zeros((max_time, batch_size, 863), np.float32)
            done_np = np.ones((max_time, batch_size), np.float32)
            h0_np = np.zeros((batch_size, HIDDEN_SIZE), np.float32)
            c0_np = np.zeros((batch_size, HIDDEN_SIZE), np.float32)
            for col, (_, _, _, rows) in enumerate(packed):
                h0_np[col], c0_np[col] = rows[0]["h"], rows[0]["c"]
                for t, row in enumerate(rows):
                    spatial_np[t, col] = row["spatial"]
                    state_np[t, col] = row["state"]
                    done_np[t, col] = float(row["done"])
            high_logits, high_values, _, _, _, _ = self.model.unroll(
                torch.as_tensor(spatial_np, device=self.device),
                torch.as_tensor(state_np, device=self.device),
                (torch.as_tensor(h0_np, device=self.device).unsqueeze(0),
                 torch.as_tensor(c0_np, device=self.device).unsqueeze(0)),
                torch.as_tensor(done_np, device=self.device),
            )
            for col, (_, entries, start_index, _) in enumerate(packed):
                for original_index, record in entries:
                    row = record["decision_index"] - start_index
                    role = record["role"]
                    dist = Categorical(logits=high_logits[row, col, role])
                    z = torch.as_tensor(record["z"], dtype=torch.long, device=self.device)
                    logps[original_index] = dist.log_prob(z)
                    values[original_index] = high_values[row, col, role]
                    entropies[original_index] = dist.entropy()
        return torch.stack(logps), torch.stack(values), torch.stack(entropies)

    def _joint_update(self, batch, initial_state, ready, low_entropy_coef, high_entropy_coef):
        high_level_gae(ready, self.gamma, self.gae_lambda)
        high_records = [record for stream in ready for record in stream]
        spatial, state = batch["spatial"], batch["state"]
        time_steps = len(spatial)
        slot_count = min(self.minibatches, self.num_envs)
        low_rows, high_rows = [], []
        for _ in range(self.epochs):
            low_batches = np.array_split(self.rng.permutation(self.num_envs), slot_count)
            high_batches = (np.array_split(self.rng.permutation(len(high_records)), slot_count)
                           if high_records else [np.empty(0, dtype=np.int64) for _ in range(slot_count)])
            for env_indices, record_indices in zip(low_batches, high_batches):
                joint_loss = None
                if len(env_indices):
                    x = torch.as_tensor(spatial[:, env_indices], device=self.device)
                    s = torch.as_tensor(state[:, env_indices], device=self.device)
                    done = torch.as_tensor(batch["done_before"][:, env_indices], device=self.device)
                    h0 = initial_state[0][:, env_indices]
                    c0 = initial_state[1][:, env_indices]
                    _, _, low_logits, low_values, _, _ = self.model.unroll(
                        x, s, (h0, c0), done
                    )
                    z = torch.as_tensor(batch["z"][:, env_indices], device=self.device)
                    gather_z = z[..., None, None].expand(*z.shape, 1, self.model.action_dim)
                    selected_logits = low_logits.gather(3, gather_z).squeeze(3)
                    selected_value = low_values.gather(3, z.unsqueeze(-1)).squeeze(-1)
                    learner = torch.as_tensor(batch["learner"][:, env_indices], device=self.device)
                    if bool(learner.any()):
                        dist = Categorical(logits=selected_logits[learner])
                        actions = torch.as_tensor(batch["action"][:, env_indices], device=self.device)
                        old_logp = torch.as_tensor(batch["logp"][:, env_indices], device=self.device)
                        old_value = torch.as_tensor(batch["value"][:, env_indices], device=self.device)
                        advantage = torch.as_tensor(batch["advantage"][:, env_indices], device=self.device)
                        returns = torch.as_tensor(batch["return"][:, env_indices], device=self.device)
                        low_pg, low_v, low_ent, low_kl = ppo_components(
                            dist.log_prob(actions[learner]), old_logp[learner],
                            advantage[learner], selected_value[learner], old_value[learner],
                            returns[learner], dist.entropy(), self.clip,
                        )
                        low_loss = low_pg + self.value_coef * low_v - low_entropy_coef * low_ent
                        joint_loss = low_loss
                        low_rows.append((float(low_loss.detach().cpu()), float(low_pg.detach().cpu()),
                                         float(low_v.detach().cpu()), float(low_ent.detach().cpu()),
                                         float(low_kl.detach().cpu())))

                if len(record_indices):
                    records = [high_records[int(i)] for i in record_indices]
                    new_logp, new_value, entropy = self._high_policy_outputs(records)
                    old_logp = torch.as_tensor([r["old_logp"] for r in records], device=self.device)
                    old_value = torch.as_tensor([r["old_value"] for r in records], device=self.device)
                    advantage = torch.as_tensor([r["advantage"] for r in records], device=self.device)
                    returns = torch.as_tensor([r["return"] for r in records], device=self.device)
                    high_pg, high_v, high_ent, high_kl = ppo_components(
                        new_logp, old_logp, advantage, new_value, old_value, returns,
                        entropy, self.clip,
                    )
                    high_loss = high_pg + self.value_coef * high_v - high_entropy_coef * high_ent
                    joint_loss = high_loss if joint_loss is None else joint_loss + high_loss
                    high_rows.append((float(high_pg.detach().cpu()), float(high_v.detach().cpu()),
                                      float(high_ent.detach().cpu()), float(high_kl.detach().cpu())))

                if joint_loss is None:
                    continue
                if not torch.isfinite(joint_loss):
                    raise FloatingPointError("Nonfinite central HiPT joint loss")
                self.optimizer.zero_grad(set_to_none=True)
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm,
                                               error_if_nonfinite=True)
                self.optimizer.step()
        low = {
            "loss": float(np.mean([r[0] for r in low_rows])) if low_rows else None,
            "policy_loss": float(np.mean([r[1] for r in low_rows])) if low_rows else None,
            "value_loss": float(np.mean([r[2] for r in low_rows])) if low_rows else None,
            "entropy": float(np.mean([r[3] for r in low_rows])) if low_rows else None,
            "approx_kl": float(np.mean([r[4] for r in low_rows])) if low_rows else None,
        }
        high = {
            "policy_loss": float(np.mean([r[0] for r in high_rows])) if high_rows else None,
            "value_loss": float(np.mean([r[1] for r in high_rows])) if high_rows else None,
            "entropy": float(np.mean([r[2] for r in high_rows])) if high_rows else None,
            "approx_kl": float(np.mean([r[3] for r in high_rows])) if high_rows else None,
            "transitions": len(high_records),
            "cross_rollout_transitions": sum(bool(r["cross_rollout"]) for r in high_records),
            "max_policy_update_lag": max((int(r["policy_update_lag"]) for r in high_records), default=0),
        }
        return low, high

    def collect_and_update(self, rollout_steps, influence_coef):
        if self.obs is None:
            self.obs = self.env.reset()
            self.hidden = self.model.initial_state(self.num_envs, device=self.device)
            for env_index in range(self.num_envs):
                self._assign_episode(env_index)
        state_start = tuple(value.detach().clone() for value in self.hidden)
        ready = [[] for _ in range(self.num_envs * 3)]
        spatial_rows, state_rows = [], []
        action_rows, z_rows, logp_rows, value_rows = [], [], [], []
        reward_rows, done_rows, done_before_rows, learner_rows = [], [], [], []
        rollout_start_update = self.update
        task_reward_sum = 0.0
        started = time.monotonic()

        for _ in range(int(rollout_steps)):
            done_before = self.previous_done.copy()
            hidden_before = tuple(value.detach().clone() for value in self.hidden)
            spatial_np, state_np = self.obs["spatial"][:, 0], self.obs["state"]
            with torch.no_grad():
                high_logits, high_value, low_logits, low_values, next_hidden = self.model.step(
                    torch.as_tensor(spatial_np, device=self.device, dtype=torch.float32),
                    torch.as_tensor(state_np, device=self.device, dtype=torch.float32),
                    self.hidden,
                    torch.as_tensor(done_before, device=self.device, dtype=torch.float32),
                )
            self.hidden = next_hidden
            for env_index in range(self.num_envs):
                self.episode_histories[env_index].append({
                    "spatial": spatial_np[env_index].copy(), "state": state_np[env_index].copy(),
                    "done": bool(done_before[env_index]),
                    "h": hidden_before[0][0, env_index].detach().cpu().numpy().copy(),
                    "c": hidden_before[1][0, env_index].detach().cpu().numpy().copy(),
                })
                for role in range(3):
                    if self.clock.remaining[env_index, role] == 0 and self.pending[env_index][role] is not None:
                        self._finish_pending(env_index, role, high_value[env_index, role], False,
                                             ready, self.update)

            with torch.no_grad():
                high_dist = Categorical(logits=high_logits)
                proposed_z = high_dist.sample().cpu().numpy()
                high_logp = high_dist.log_prob(torch.as_tensor(proposed_z, device=self.device)).cpu().numpy()
                due = self.clock.due.copy()
                chosen_z = np.where(due, proposed_z, self.clock.z)
                durations = sample_durations(self.rng, (self.num_envs, 3), self.p_min, self.p_max)
                self.clock.start(due, chosen_z, durations)
                for env_index, role in zip(*np.nonzero(due & self.learner_mask)):
                    history = self.episode_histories[env_index]
                    self.pending[env_index][role] = PendingOption(
                        history=history, decision_index=len(history) - 1, role=int(role),
                        z=int(chosen_z[env_index, role]),
                        old_logp=float(high_logp[env_index, role]),
                        old_value=float(high_value[env_index, role].detach().cpu()),
                        started_update=self.update,
                    )
                z_t = torch.as_tensor(chosen_z, device=self.device, dtype=torch.long)
                gather_z = z_t[..., None, None].expand(
                    self.num_envs, 3, 1, self.model.action_dim
                )
                selected_low = low_logits.gather(2, gather_z).squeeze(2)
                low_dist = Categorical(logits=selected_low)
                sampled = low_dist.sample()
                low_logp = low_dist.log_prob(sampled).cpu().numpy()
                selected_value = low_values.gather(2, z_t.unsqueeze(-1)).squeeze(-1)
                kl = influence_kl(low_logits, high_logits, z_t).cpu().numpy()
                model_actions = sampled.cpu().numpy()

            actions = self._pool_actions(self.obs, model_actions.copy())
            next_obs, rewards, done, infos = self.env.step(actions)
            for env_index, role in zip(*np.nonzero(self.learner_mask)):
                pending = self.pending[env_index][role]
                if pending is not None:
                    pending.add_step(float(rewards[env_index]) + float(influence_coef) * float(kl[env_index, role]))
            task_reward_sum += float(np.sum(rewards))
            self.role_steps += self.learner_mask.sum(axis=0)
            spatial_rows.append(spatial_np.copy())
            state_rows.append(state_np.copy())
            action_rows.append(model_actions.copy())
            z_rows.append(chosen_z.copy())
            logp_rows.append(low_logp.copy())
            value_rows.append(selected_value.detach().cpu().numpy())
            reward_rows.append(rewards.copy())
            done_rows.append(done.copy())
            done_before_rows.append(done_before.copy())
            learner_rows.append(self.learner_mask.copy())
            self.clock.advance(done)
            for env_index, ended in enumerate(done):
                if not ended:
                    continue
                for role in range(3):
                    self._finish_pending(env_index, role, 0.0, True, ready, self.update)
                self.episode_histories[env_index] = []
                self.episode_index[env_index] += 1
                self._assign_episode(env_index)
                summary = infos[env_index].get("episode_summary")
                if summary is not None:
                    self.recent_team_scores.append(float(summary["team_score"]))
            self.obs, self.previous_done = next_obs, done.copy()

        next_spatial = torch.as_tensor(self.obs["spatial"][:, 0], device=self.device, dtype=torch.float32)
        next_state = torch.as_tensor(self.obs["state"], device=self.device, dtype=torch.float32)
        done_next = torch.as_tensor(self.previous_done, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            next_high, _, _, next_low_values, _ = self.model.step(
                next_spatial, next_state, self.hidden, done_next
            )
            probs = next_high.softmax(-1)
            selected = next_low_values.gather(2, torch.as_tensor(
                self.clock.z, device=self.device, dtype=torch.long
            ).unsqueeze(-1)).squeeze(-1)
            expected = (probs * next_low_values).sum(-1)
            selected = torch.where(torch.as_tensor(self.clock.due, device=self.device), expected, selected)
            bootstrap = selected.cpu().numpy()

        batch = {
            "spatial": np.asarray(spatial_rows, dtype=np.float32),
            "state": np.asarray(state_rows, dtype=np.float32),
            "action": np.asarray(action_rows, dtype=np.int64),
            "z": np.asarray(z_rows, dtype=np.int64),
            "logp": np.asarray(logp_rows, dtype=np.float32),
            "value": np.asarray(value_rows, dtype=np.float32),
            "reward": np.repeat(np.asarray(reward_rows, dtype=np.float32)[..., None], 3, axis=2),
            "done": np.asarray(done_rows, dtype=np.float32),
            "done_before": np.asarray(done_before_rows, dtype=np.float32),
            "learner": np.asarray(learner_rows, dtype=bool),
        }
        low_adv, low_return = gae(batch["reward"], batch["value"], batch["done"],
                                  bootstrap, self.gamma, self.gae_lambda)
        batch["advantage"], batch["return"] = low_adv, low_return
        progress = min(1.0, self.total_env_steps / self.target_env_steps)
        low_coef = self.low_entropy_coef * (1.0 - progress)
        high_coef = self.high_entropy_coef * (1.0 - progress)
        low_metrics, high_metrics = self._joint_update(batch, state_start, ready, low_coef, high_coef)
        self.total_env_steps += self.num_envs * int(rollout_steps)
        self.update += 1
        return {
            "stage": "hierarchical", "update": self.update, "stage_steps": self.total_env_steps,
            "low": low_metrics, "high": high_metrics,
            "influence_coef": float(influence_coef),
            "selfplay_episodes_active": int(self.selfplay.sum()),
            "partner_episodes_active": int((~self.selfplay).sum()),
            "partner_roles_active": {
                str(role): int(np.sum((~self.selfplay) & (self.partner_role == role)))
                for role in range(3)
            },
            "learner_role_steps": self.role_steps.tolist(),
            "pending_options": sum(value is not None for row in self.pending for value in row),
            "task_reward_sum": task_reward_sum,
            "mean_recent_team_score": float(np.mean(self.recent_team_scores)) if self.recent_team_scores else None,
            "update_seconds": time.monotonic() - started,
            "rollout_start_update": rollout_start_update,
        }

    def state_dict(self):
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "total_env_steps": self.total_env_steps, "update": self.update,
            "role_steps": self.role_steps.copy(),
        }

    def load_state_dict(self, payload):
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.total_env_steps = int(payload.get("total_env_steps", 0))
        self.update = int(payload.get("update", 0))
        self.role_steps = np.asarray(payload.get("role_steps", [0, 0, 0]), dtype=np.int64)

    def close(self):
        self.env.close()

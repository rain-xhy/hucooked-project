"""Online PACE history and rollout replay with shared observation storage."""
from collections import deque

import torch

from .features import OBS_DIM
from .policy import HISTORY_EPISODES as E, HISTORY_STEPS as S, HIDDEN_DIM


def heads(model, current, pooled):
    latent = model.period.transform(pooled)
    return model.actor(current, latent), model.critic(current, latent), model.partner_head(latent)


class OnlineHistory:
    """Synchronized or phase-staggered lanes; rings contain only executed transitions."""

    def __init__(self, num_envs, device, stagger=False):
        self.obs = torch.zeros(E, S, num_envs, OBS_DIM, device=device)
        self.ids = torch.full((E, S, num_envs), -1, device=device, dtype=torch.long)
        self.encoded = self.obs.new_zeros(E, S, num_envs, HIDDEN_DIM)
        self.stagger = bool(stagger)
        self.groups = E if self.stagger else 1
        self.lengths = [0] * E
        self.episode = 0
        self.episodes = list(range(E)) if self.stagger else [0]
        self.group_lengths = [[0] * E for _ in range(self.groups)]
        self.cached = False

    def state_dict(self):
        state = dict(obs=self.obs, ids=self.ids, lengths=self.lengths.copy(), episode=self.episode)
        if self.stagger:
            state.update(stagger=True, episodes=self.episodes.copy(),
                         group_lengths=[row.copy() for row in self.group_lengths])
        return state

    def load_state_dict(self, state):
        self.obs.copy_(state["obs"])
        self.ids.copy_(state["ids"])
        self.lengths = list(state["lengths"])
        self.episode = state["episode"]
        if self.stagger:
            if not state.get("stagger"):
                raise ValueError("Checkpoint history is not staggered")
            self.episodes = list(state["episodes"])
            self.group_lengths = [list(row) for row in state["group_lengths"]]
        else:
            if state.get("stagger"):
                raise ValueError("Checkpoint history is staggered")
            self.episodes = [self.episode]
            self.group_lengths = [self.lengths]
        self.cached = False

    @torch.no_grad()
    def refresh(self, model):
        if not self.stagger:
            for i, length in enumerate(self.lengths):
                n = min(length, S)
                if n:
                    self.encoded[i, :n] = model.episode.steps(model.history(self.obs[i, :n], self.ids[i, :n]))
        else:
            for g in range(E):
                lanes = slice(g, None, E)
                for i, length in enumerate(self.group_lengths[g]):
                    n = min(length, S)
                    if n:
                        self.encoded[i, :n, lanes] = model.episode.steps(
                            model.history(self.obs[i, :n, lanes], self.ids[i, :n, lanes]))
        self.cached = True

    @torch.no_grad()
    def forward(self, model, current):
        if not self.cached:
            self.refresh(model)
        if not self.stagger:
            means = [self.encoded[i, :min(length, S)].mean(0)
                     for i, length in enumerate(self.lengths) if length]
            pooled = torch.stack(means).mean(0) if means else self.encoded[0, 0] * 0
            return heads(model, current, pooled)
        pooled = current.new_zeros(len(current), HIDDEN_DIM)
        for g in range(E):
            lanes = slice(g, None, E)
            means = [self.encoded[i, :min(length, S), lanes].mean(0)
                     for i, length in enumerate(self.group_lengths[g]) if length]
            if means:
                pooled[lanes] = torch.stack(means).mean(0)
        return heads(model, current, pooled)

    @torch.no_grad()
    def append(self, current, partner_joint_ids, model=None):
        if not self.stagger:
            ep = self.episode
            slot = self.lengths[ep] % S
            self.obs[ep, slot].copy_(current)
            self.ids[ep, slot].copy_(partner_joint_ids)
            if model is None:
                self.cached = False
            elif self.cached:
                self.encoded[ep, slot] = model.episode.steps(model.history(current, partner_joint_ids))
            self.lengths[ep] += 1
            return
        enc = None
        if model is None:
            self.cached = False
        elif self.cached:
            enc = model.episode.steps(model.history(current, partner_joint_ids))
        for g in range(E):
            lanes = slice(g, None, E)
            ep = self.episodes[g]
            slot = self.group_lengths[g][ep] % S
            self.obs[ep, slot, lanes].copy_(current[lanes])
            self.ids[ep, slot, lanes].copy_(partner_joint_ids[lanes])
            if enc is not None:
                self.encoded[ep, slot, lanes].copy_(enc[lanes])
            self.group_lengths[g][ep] += 1

    def finish_episode(self):
        if not self.stagger:
            self.episode = (self.episode + 1) % E
            if self.episode == 0:
                self.lengths = [0] * E
            return
        for g in range(E):
            self.episodes[g] = (self.episodes[g] + 1) % E
            if self.episodes[g] == 0:
                self.group_lengths[g] = [0] * E


class RolloutHistory:
    """Stores compact time-offset tables for all lanes (shared or per phase group)."""

    def __init__(self, live, steps):
        self.stagger = live.stagger
        self.groups = live.groups
        lanes = live.obs.shape[2]
        if not self.stagger:
            prefix = sum(min(n, S) for n in live.lengths)
            self.obs = live.obs.new_empty(prefix + steps, lanes, OBS_DIM)
            self.ids = live.ids.new_empty(prefix + steps, lanes)
            self.tables = torch.full((steps, E, S), -1, dtype=torch.long)
            self.windows = [deque(maxlen=S) for _ in range(E)]
            self.episode = live.episode
            self.prefix = prefix
            self.t = 0
            k = 0
            for ep, length in enumerate(live.lengths):
                for tick in range(max(0, length - S), length):
                    self.obs[k].copy_(live.obs[ep, tick % S])
                    self.ids[k].copy_(live.ids[ep, tick % S])
                    self.windows[ep].append(k)
                    k += 1
            return
        prefix = max(sum(min(n, S) for n in row) for row in live.group_lengths)
        self.obs = live.obs.new_zeros(prefix + steps, lanes, OBS_DIM)
        self.ids = live.ids.new_full((prefix + steps, lanes), -1)
        self.tables = torch.full((steps, E, E, S), -1, dtype=torch.long)
        self.windows = [[deque(maxlen=S) for _ in range(E)] for _ in range(E)]
        self.episodes = list(live.episodes)
        self.prefix = prefix
        self.t = 0
        for g in range(E):
            lane_slice = slice(g, None, E)
            k = 0
            for ep, length in enumerate(live.group_lengths[g]):
                for tick in range(max(0, length - S), length):
                    self.obs[k, lane_slice].copy_(live.obs[ep, tick % S, lane_slice])
                    self.ids[k, lane_slice].copy_(live.ids[ep, tick % S, lane_slice])
                    self.windows[g][ep].append(k)
                    k += 1

    def record(self, current, partner_ids):
        k = self.prefix + self.t
        if not self.stagger:
            for ep, window in enumerate(self.windows):
                if window:
                    self.tables[self.t, ep, :len(window)] = torch.tensor(list(window))
            self.obs[k].copy_(current)
            self.ids[k].copy_(partner_ids)
            self.windows[self.episode].append(k)
            self.t += 1
            return
        for g in range(E):
            for ep, window in enumerate(self.windows[g]):
                if window:
                    self.tables[self.t, g, ep, :len(window)] = torch.tensor(list(window))
        self.obs[k].copy_(current)
        self.ids[k].copy_(partner_ids)
        for g in range(E):
            self.windows[g][self.episodes[g]].append(k)
        self.t += 1

    def finish_episode(self):
        if not self.stagger:
            self.episode = (self.episode + 1) % E
            if self.episode == 0:
                for window in self.windows:
                    window.clear()
            return
        for g in range(E):
            self.episodes[g] = (self.episodes[g] + 1) % E
            if self.episodes[g] == 0:
                for window in self.windows[g]:
                    window.clear()

    def seal(self):
        self.tables = self.tables.to(self.obs.device)

    def replay(self, model, indices):
        lanes = self.obs.size(1)
        ticks, lane = indices // lanes, indices % lanes
        current = self.obs[self.prefix + ticks, lane]
        if self.tables.ndim == 4:
            times = self.tables[ticks, lane % self.tables.size(1)]
        else:
            times = self.tables[ticks]
        valid = times >= 0
        counts = valid.sum(-1)
        present = counts > 0
        weights = (counts.clamp_min(1) * present.sum(-1, keepdim=True).clamp_min(1)).float().reciprocal()
        sample, episode, slot = valid.nonzero(as_tuple=True)
        keys = times[sample, episode, slot] * lanes + lane[sample]
        unique, inverse = keys.unique(return_inverse=True)
        encoded = model.episode.steps(model.history(self.obs.flatten(0, 1)[unique],
                                                    self.ids.flatten()[unique]))
        offsets = torch.cat((counts.new_zeros(1), counts.sum(-1).cumsum(0)))
        pooled = torch.nn.functional.embedding_bag(
            inverse, encoded, offsets, mode="sum", per_sample_weights=weights[sample, episode],
            include_last_offset=True) if unique.numel() else encoded.new_zeros(len(indices), HIDDEN_DIM)
        return (*heads(model, current, pooled), present.any(-1))

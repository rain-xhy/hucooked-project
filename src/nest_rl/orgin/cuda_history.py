"""CUDA trajectory storage and batched mean-history encoding for L2 PPO."""

from functools import wraps

import numpy as np
import torch


class CudaHistory:
    def __init__(self, num_envs, rollout_steps, obs_shape, horizon, clear_period, device):
        self.num_envs = num_envs
        self.clear_period = clear_period
        # A rollout can start in the middle of a period. Keep that entire period,
        # plus the new rollout with one possible terminal observation per step.
        self.frame_capacity = clear_period * (horizon + 1) + 2 * rollout_steps + 1
        self.episode_capacity = clear_period + rollout_steps + 1
        self.obs = torch.zeros(num_envs, self.frame_capacity, *obs_shape, device=device)
        self.device = self.obs.device
        self.actions = torch.zeros(num_envs, self.frame_capacity, dtype=torch.long, device=device)
        self.starts = torch.zeros(num_envs, self.episode_capacity, dtype=torch.long, device=device)
        self.lengths = torch.zeros_like(self.starts)
        self.rows = torch.arange(num_envs, device=device)
        self.steps = torch.arange(horizon + 1, device=device)
        self.episodes = torch.arange(clear_period, device=device)
        self.cursor = torch.zeros_like(self.rows)
        self.episode_id = torch.zeros_like(self.rows)
        self.first_period = torch.zeros_like(self.rows)
        # The pinned trainer checks the episode number on the host for done rows.
        self.current_episode = np.zeros(num_envs, dtype=np.int64)

    def get_all_current_indices(self):
        period = self.episode_id // self.clear_period
        episode = self.episode_id % self.clear_period
        length = self.lengths[self.rows, self.episode_id % self.episode_capacity]
        return period, episode, length

    def append(self, obs, actions, terminal_obs, done, finished=None):
        slot = self.episode_id % self.episode_capacity
        frame = self.cursor % self.frame_capacity
        self.obs[self.rows, frame] = obs
        self.actions[self.rows, frame] = actions
        # Reserve the next cell. Only terminal rows advance over this second frame.
        terminal_frame = (self.cursor + 1) % self.frame_capacity
        self.obs[self.rows, terminal_frame] = terminal_obs
        self.actions[self.rows, terminal_frame] = -1
        self.lengths[self.rows, slot] += 1 + done
        self.cursor += 1 + done
        self.episode_id += done
        next_slot = self.episode_id % self.episode_capacity
        self.starts[self.rows, next_slot] = torch.where(
            done, self.cursor, self.starts[self.rows, next_slot],
        )
        self.lengths[self.rows, next_slot] = torch.where(
            done, 0, self.lengths[self.rows, next_slot],
        )
        if finished is not None:
            self.current_episode[finished] = (self.current_episode[finished] + 1) % self.clear_period
        return done & (self.episode_id % self.clear_period == 0)

    def trim(self):
        # Match the reference's retained period count for the logger. Absolute
        # frame indices let expired data be overwritten by the next rollout.
        self.first_period = (self.episode_id - 1).clamp_min(0) // self.clear_period

    @property
    def history_sizes(self):
        """Return per-episode frame counts for the pinned trainer's logging."""
        counts = (self.episode_id // self.clear_period - self.first_period + 1).tolist()
        periods = self.first_period[:, None] + torch.arange(max(counts), device=self.device)
        episode_ids = periods[..., None] * self.clear_period + self.episodes
        slots = episode_ids % self.episode_capacity
        sizes = self.lengths[self.rows[:, None, None], slots]
        sizes = sizes.masked_fill(episode_ids > self.episode_id[:, None, None], 0).cpu()
        return [row[:count] for row, count in zip(sizes, counts, strict=True)]

    def get_periods(self, processes, periods):
        slots = (periods[:, None] * self.clear_period + self.episodes) % self.episode_capacity
        starts = self.starts[processes[:, None], slots]
        lengths = self.lengths[processes[:, None], slots]
        frames = (starts[..., None] + self.steps) % self.frame_capacity
        return self.obs[processes[:, None, None], frames], self.actions[processes[:, None, None], frames], lengths


def encode_cuda_history(encoder, history):
    storage, (process, period, episode, length) = history
    process = process.to(storage.device)
    # Share each period's step MLP and prefix sums across samples in this batch.
    keys, inverse = torch.unique(period * storage.num_envs + process, return_inverse=True)
    obs, actions, lengths = storage.get_periods(keys % storage.num_envs, keys // storage.num_envs)
    action_ids = torch.arange(encoder.action_dim, device=storage.device)
    if encoder.cnn is not None:
        obs = encoder.cnn(obs)
    inputs = torch.cat((obs, actions[..., None] == action_ids), dim=-1)
    base = encoder.base
    cumulative = base.pre_agg_mlp(inputs).cumsum(dim=-2)

    def mean_at_prefix(periods, episodes, counts):
        totals = cumulative[periods, episodes, (counts - 1).clamp_min(0)]
        return (totals / counts[..., None].clamp_min(1)).masked_fill(counts[..., None] == 0, 0)

    periods = torch.arange(len(keys), device=storage.device)
    means = mean_at_prefix(periods[:, None], storage.episodes, lengths)[inverse]
    rows = torch.arange(len(process), device=storage.device)
    means[rows, episode] = mean_at_prefix(inverse, episode, length)
    episode_mask = storage.episodes > episode[:, None]
    return base.period_forward(means, episode_mask).relu()


def install_cuda_history(storage_class, encoder_class):
    from learning.model import AggregatedMLPEncoder

    original_to = storage_class.to
    original_forward = encoder_class.forward

    @wraps(original_to)
    def to(storage, device):
        if isinstance(storage.history, CudaHistory):
            if torch.device(device) != storage.history.device and str(device) != "cuda":
                raise ValueError("Cannot move an active CUDA history to a different device")
            return
        original_to(storage, device)
        reference = storage.history
        assert reference.include_current_episode and reference.merge_encoder_computation
        assert reference.max_size_per_process == reference.clear_period
        assert not reference.has_rew_done and not reference.has_meta_time_step and not reference.last_episode_only
        assert not reference.pop_oldest_episode and not storage.leave_on_cpu
        storage.history = CudaHistory(
            reference.num_processes, storage.num_steps, reference.obs_shape,
            reference.max_episode_length - 1, reference.clear_period, device,
        )
        storage.period_idx = storage.period_idx.to(device)
        storage.episode_idx = storage.episode_idx.to(device)
        storage.length_idx = storage.length_idx.to(device)

    @wraps(original_forward)
    def forward(encoder, history, agent_indices):
        if history is not None and isinstance(history[0], CudaHistory):
            assert isinstance(encoder.base, AggregatedMLPEncoder) and encoder.base.agg_func == "mean"
            assert encoder.deterministic_latent and encoder.discrete_action
            return encode_cuda_history(encoder, history), ()
        return original_forward(encoder, history, agent_indices)

    storage_class.to = to
    encoder_class.forward = forward

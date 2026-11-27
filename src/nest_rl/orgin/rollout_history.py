"""Batch the pinned mean-MLP history encoder during CUDA action selection."""

from functools import wraps

import torch
from torch.nn import functional as F

from learning.model import AggregatedMLPEncoder, aggregate_with_mask
from learning.storage_ import PeriodicHistoryStorage


def encode_batched_history(encoder, history, device):
    storage, (proc_idx, period_idx, episode_idx, length_idx) = history
    # History buffers and their indices live on CPU. Gather each period once.
    periods = list(zip(proc_idx.tolist(), period_idx.tolist()))
    obs = torch.stack([storage.history_obs[proc][period] for proc, period in periods])
    act = torch.stack([storage.history_act[proc][period] for proc, period in periods])
    step_mask = torch.stack([storage.history_step_mask[proc][period] for proc, period in periods])
    # Upstream replaces the current episode's mask with the sampled prefix.
    step_mask[torch.arange(len(periods)), episode_idx] = (
        torch.arange(step_mask.shape[-1]) >= length_idx[:, None]
    )
    episode_mask = torch.arange(step_mask.shape[1]) > episode_idx[:, None]

    # Reserve class zero for the terminal observation's dummy action (-1).
    one_hot = F.one_hot(act + 1, num_classes=encoder.action_dim + 1)[..., 1:]
    inputs = torch.cat((obs, one_hot.to(obs.dtype)), dim=-1).to(device, non_blocking=True)
    step_mask = step_mask.to(device, non_blocking=True)
    episode_mask = episode_mask.to(device, non_blocking=True)

    # Dense shapes avoid CUDA boolean gathers, which must learn their output size.
    base = encoder.base
    episodes = aggregate_with_mask("mean", base.pre_agg_mlp(inputs), step_mask, None)
    return base.period_forward(episodes, episode_mask).relu()


def install_batched_history(policy_class):
    original_act = policy_class.act

    @wraps(original_act)
    def act(policy, inputs, rnn_hxs, masks, agent_indices, history, deterministic=False,
            latents=None, params=None, query_ind=False):
        encoder = policy.encoder
        base = getattr(encoder, "base", None)
        if (
            inputs.is_cuda and not torch.is_grad_enabled()
            and policy.latent_training_mode and not query_ind and latents is None and params is None
            and isinstance(base, AggregatedMLPEncoder) and base.agg_func == "mean"
            and encoder.deterministic_latent and encoder.discrete_action and encoder.cnn is None
            and history is not None and isinstance(history[0], PeriodicHistoryStorage)
            and history[0].merge_encoder_computation and not history[0].last_episode_only
            and all(isinstance(idx, torch.Tensor) and idx.device.type == "cpu" for idx in history[1])
        ):
            latents = encode_batched_history(encoder, history, inputs.device)
        return original_act(policy, inputs, rnn_hxs, masks, agent_indices, history,
                            deterministic, latents, params, query_ind)

    policy_class.act = act

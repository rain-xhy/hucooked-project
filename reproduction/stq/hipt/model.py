"""Recurrent high/low HiPT policy for SharedKitchen observations.

The visual encoder follows the HiPT layout (three same-padded convolutions,
two 128-wide layers, then a 128-wide LSTM). SharedKitchen supplies an 866-value
public vector and a 26-channel 7x7 spatial tensor; the vector is concatenated
with the spatial features before the first fully connected layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

VECTOR_DIM = 866
SPATIAL_CHANNELS = 26
SPATIAL_HEIGHT = 7
SPATIAL_WIDTH = 7
SPATIAL_FEATURE_CHANNELS = 16
HIDDEN_SIZE = 128
HEAD_SIZE = 64


def _init(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


def influence_kl(low_logits: torch.Tensor, high_logits: torch.Tensor,
                 selected_z: torch.Tensor | None = None) -> torch.Tensor:
    """Return HiPT's directed KL influence reward.

    ``low_logits`` has shape ``[..., z, action]`` and ``high_logits`` has shape
    ``[..., z]``.  The marginal action policy is the high-policy-weighted
    mixture of all low-level action distributions.  If ``selected_z`` is
    supplied, compute KL(current selected low policy || mixture); otherwise
    return the value for each z, shaped ``[..., z]``.

    The implementation keeps the mixture in probability space and evaluates
    the analytic categorical KL, so identical low policies produce exact zero
    up to floating-point roundoff.
    """
    if low_logits.ndim < 2 or high_logits.shape != low_logits.shape[:-1]:
        raise ValueError("Expected low logits [..., z, action] and high logits [..., z]")
    low_prob = torch.softmax(low_logits, dim=-1)
    high_prob = torch.softmax(high_logits, dim=-1)
    marginal = torch.sum(high_prob.unsqueeze(-1) * low_prob, dim=-2)
    log_low = torch.log_softmax(low_logits, dim=-1)
    log_marginal = torch.log(marginal.clamp_min(torch.finfo(marginal.dtype).tiny))
    per_z = torch.sum(low_prob * (log_low - log_marginal.unsqueeze(-2)), dim=-1)
    if selected_z is None:
        return per_z.clamp_min(0.0)
    if selected_z.shape != low_logits.shape[:-2]:
        raise ValueError("selected_z must match the leading low-logit dimensions")
    selected = per_z.gather(-1, selected_z.long().unsqueeze(-1)).squeeze(-1)
    return selected.clamp_min(0.0)


class HiPTPolicy(nn.Module):
    """Two-level categorical actor-critic with explicit recurrent unrolling.

    ``step`` is used for sampling. ``unroll`` recomputes recurrent features
    across a complete rollout and deliberately accepts initial hidden/cell
    states, preserving BPTT across rollout time instead of treating stored
    hidden activations as independent samples.
    """

    def __init__(self, z_dim: int = 4, action_dim: int = 6,
                 vector_dim: int = VECTOR_DIM, channels: int = SPATIAL_CHANNELS,
                 height: int = SPATIAL_HEIGHT, width: int = SPATIAL_WIDTH) -> None:
        super().__init__()
        if z_dim < 2 or action_dim < 2 or min(vector_dim, channels, height, width) <= 0:
            raise ValueError("Invalid policy dimensions")
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.vector_dim = int(vector_dim)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

        self.spatial = nn.Sequential(
            _init(nn.Conv2d(channels, SPATIAL_FEATURE_CHANNELS, 5, padding="same")),
            nn.LeakyReLU(),
            _init(nn.Conv2d(SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS,
                            3, padding="same")),
            nn.LeakyReLU(),
            _init(nn.Conv2d(SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS,
                            3, padding="same")),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=-3),
        )
        flat_dim = SPATIAL_FEATURE_CHANNELS * height * width + vector_dim
        self.encoder = nn.Sequential(
            _init(nn.Linear(flat_dim, HIDDEN_SIZE)),
            nn.LeakyReLU(),
            _init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.LeakyReLU(),
        )
        self.lstm = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)

        self.high_feature = nn.Sequential(_init(nn.Linear(HIDDEN_SIZE, HEAD_SIZE)), nn.LeakyReLU())
        self.high_actor = _init(nn.Linear(HEAD_SIZE, z_dim), gain=0.01)
        self.high_critic = _init(nn.Linear(HEAD_SIZE, 1), gain=1.0)
        self.low_feature = nn.Sequential(
            _init(nn.Linear(HIDDEN_SIZE + z_dim, HEAD_SIZE)), nn.LeakyReLU()
        )
        self.low_actor = _init(nn.Linear(HEAD_SIZE, action_dim), gain=0.01)
        self.low_critic = _init(nn.Linear(HEAD_SIZE, 1), gain=1.0)

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (1, batch_size, self.lstm.hidden_size)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    def _encode(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or spatial.shape[1:] != (
            self.channels, self.height, self.width
        ):
            raise ValueError(
                f"spatial must have shape [B, {self.channels}, {self.height}, {self.width}]"
            )
        if vector.ndim != 2 or vector.shape != (spatial.shape[0], self.vector_dim):
            raise ValueError(f"vector must have shape [B, {self.vector_dim}]")
        return self.encoder(torch.cat((self.spatial(spatial), vector), dim=-1))

    def _heads(self, recurrent: torch.Tensor):
        """Compute all z-specific logits/values from [..., HIDDEN_SIZE] features."""
        leading = recurrent.shape[:-1]
        high = self.high_feature(recurrent)
        high_logits = self.high_actor(high)
        high_value = self.high_critic(high).squeeze(-1)

        z_eye = torch.eye(self.z_dim, device=recurrent.device, dtype=recurrent.dtype)
        z_onehot = z_eye.expand(*leading, self.z_dim, self.z_dim)
        expanded = recurrent.unsqueeze(-2).expand(*leading, self.z_dim, HIDDEN_SIZE)
        low_input = torch.cat((expanded, z_onehot), dim=-1)
        low = self.low_feature(low_input)
        low_logits = self.low_actor(low)
        low_values = self.low_critic(low).squeeze(-1)
        return high_logits, high_value, low_logits, low_values

    def step(self, spatial: torch.Tensor, vector: torch.Tensor, state,
             episode_done: torch.Tensor | None = None):
        """Process one observation per recurrent stream and return all heads."""
        encoded = self._encode(spatial, vector)
        h, c = state
        if h.ndim != 3 or h.shape != c.shape or h.shape != (
            1, encoded.shape[0], self.lstm.hidden_size
        ):
            raise ValueError(f"LSTM state must have shape [1, B, {HIDDEN_SIZE}]")
        if episode_done is not None:
            if episode_done.shape != (encoded.shape[0],):
                raise ValueError("episode_done must have shape [B]")
            keep = (1.0 - episode_done.to(encoded.dtype)).view(1, -1, 1)
            h, c = h * keep, c * keep
        recurrent, next_state = self.lstm(encoded.unsqueeze(0), (h, c))
        heads = self._heads(recurrent.squeeze(0))
        return (*heads, next_state)

    def unroll(self, spatial: torch.Tensor, vector: torch.Tensor, state,
               episode_done: torch.Tensor):
        """Run a time-major sequence with gradients through every LSTM step.

        ``spatial`` is ``[T, B, C, H, W]``; ``vector`` is ``[T, B, D]`` and
        ``episode_done[t,b]`` marks that the observation at ``t`` follows a
        terminal step, so the incoming recurrent state is cleared before
        processing that observation.
        """
        if spatial.ndim != 5 or vector.ndim != 3:
            raise ValueError("Expected time-major spatial [T,B,C,H,W] and vector [T,B,D]")
        time, batch = spatial.shape[:2]
        if vector.shape[:2] != (time, batch) or episode_done.shape != (time, batch):
            raise ValueError("All sequence inputs must agree on [T, B]")
        if time == 0:
            raise ValueError("Cannot unroll an empty sequence")
        flat_spatial = spatial.flatten(0, 1)
        flat_vector = vector.flatten(0, 1)
        encoded = self._encode(flat_spatial, flat_vector).reshape(time, batch, HIDDEN_SIZE)
        h, c = state
        outputs = []
        for t in range(time):
            keep = (1.0 - episode_done[t].to(encoded.dtype)).view(1, batch, 1)
            output, (h, c) = self.lstm(
                encoded[t:t + 1], (h * keep, c * keep)
            )
            outputs.append(output.squeeze(0))
        recurrent = torch.stack(outputs, dim=0)
        high_logits, high_value, low_logits, low_values = self._heads(recurrent)
        return high_logits, high_value, low_logits, low_values, (h, c), recurrent

    def parameter_counts(self) -> dict[str, int]:
        parts = {
            "spatial_encoder": self.spatial,
            "vector_encoder": self.encoder,
            "lstm": self.lstm,
            "high_actor_critic": nn.ModuleList(
                [self.high_feature, self.high_actor, self.high_critic]
            ),
            "low_actor_critic": nn.ModuleList(
                [self.low_feature, self.low_actor, self.low_critic]
            ),
        }
        result = {name: sum(p.numel() for p in module.parameters())
                  for name, module in parts.items()}
        result["total"] = sum(p.numel() for p in self.parameters())
        result["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return result


class FlatActorCritic(nn.Module):
    """Feed-forward self-play population policy with the same public inputs."""

    def __init__(self, action_dim: int = 6, vector_dim: int = VECTOR_DIM,
                 channels: int = SPATIAL_CHANNELS, height: int = SPATIAL_HEIGHT,
                 width: int = SPATIAL_WIDTH) -> None:
        super().__init__()
        self.action_dim, self.vector_dim = action_dim, vector_dim
        self.channels, self.height, self.width = channels, height, width
        self.spatial = nn.Sequential(
            _init(nn.Conv2d(channels, SPATIAL_FEATURE_CHANNELS, 5, padding="same")), nn.LeakyReLU(),
            _init(nn.Conv2d(SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS,
                            3, padding="same")), nn.LeakyReLU(),
            _init(nn.Conv2d(SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS,
                            3, padding="same")), nn.LeakyReLU(),
            nn.Flatten(start_dim=-3),
        )
        self.trunk = nn.Sequential(
            _init(nn.Linear(SPATIAL_FEATURE_CHANNELS * height * width + vector_dim,
                            HIDDEN_SIZE)), nn.LeakyReLU(),
            _init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)), nn.LeakyReLU(),
        )
        self.actor = _init(nn.Linear(HIDDEN_SIZE, action_dim), gain=0.01)
        self.critic = _init(nn.Linear(HIDDEN_SIZE, 1), gain=1.0)

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor):
        if spatial.ndim != 4 or spatial.shape[1:] != (
            self.channels, self.height, self.width
        ):
            raise ValueError("Invalid spatial observation shape")
        if vector.shape != (spatial.shape[0], self.vector_dim):
            raise ValueError("Invalid vector observation shape")
        hidden = self.trunk(torch.cat((self.spatial(spatial), vector), dim=-1))
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def parameter_counts(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

"""Central-state HiPT networks for the three-chef SharedKitchen task."""

from __future__ import annotations

import math

import torch
from torch import nn

SPATIAL_CHANNELS = 26
SPATIAL_HEIGHT = 7
SPATIAL_WIDTH = 7
STATE_DIM = 863
ACTION_DIM = 6
NUM_ROLES = 3
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
    """Per-chef KL(low policy at z || that chef's high-weighted low mixture)."""
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
    return per_z.gather(-1, selected_z.long().unsqueeze(-1)).squeeze(-1).clamp_min(0.0)


class CentralEncoder(nn.Module):
    """One CNN/MLP encoding of canonical spatial and public team state."""

    def __init__(self, *, state_dim=STATE_DIM, channels=SPATIAL_CHANNELS,
                 height=SPATIAL_HEIGHT, width=SPATIAL_WIDTH,
                 hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.state_dim, self.channels = int(state_dim), int(channels)
        self.height, self.width, self.hidden_size = int(height), int(width), int(hidden_size)
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
        self.encoder = nn.Sequential(
            _init(nn.Linear(SPATIAL_FEATURE_CHANNELS * height * width + state_dim,
                            hidden_size)),
            nn.LeakyReLU(),
            _init(nn.Linear(hidden_size, hidden_size)),
            nn.LeakyReLU(),
        )

    def forward(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or spatial.shape[1:] != (
            self.channels, self.height, self.width
        ):
            raise ValueError(
                f"spatial must be [B,{self.channels},{self.height},{self.width}]"
            )
        if state.ndim != 2 or state.shape != (spatial.shape[0], self.state_dim):
            raise ValueError(f"state must be [B,{self.state_dim}]")
        return self.encoder(torch.cat((self.spatial(spatial), state), dim=-1))


class _RoleHeads(nn.Module):
    """Shared feature necks with separate action/value projections per role."""

    def __init__(self, hidden_size=HIDDEN_SIZE, z_dim=4, action_dim=ACTION_DIM,
                 head_size=HEAD_SIZE):
        super().__init__()
        self.z_dim, self.action_dim = int(z_dim), int(action_dim)
        self.high_feature = nn.Sequential(
            _init(nn.Linear(hidden_size, head_size)), nn.LeakyReLU()
        )
        self.high_actor = _init(nn.Linear(head_size, NUM_ROLES * z_dim), gain=0.01)
        self.high_critic = _init(nn.Linear(head_size, NUM_ROLES), gain=1.0)
        self.low_feature = nn.Sequential(
            _init(nn.Linear(hidden_size + z_dim, head_size)), nn.LeakyReLU()
        )
        self.low_actor = _init(nn.Linear(head_size, NUM_ROLES * action_dim), gain=0.01)
        self.low_critic = _init(nn.Linear(head_size, NUM_ROLES), gain=1.0)

    def forward(self, recurrent: torch.Tensor):
        leading = recurrent.shape[:-1]
        high = self.high_feature(recurrent)
        high_logits = self.high_actor(high).reshape(*leading, NUM_ROLES, self.z_dim)
        high_values = self.high_critic(high).reshape(*leading, NUM_ROLES)

        z_eye = torch.eye(self.z_dim, device=recurrent.device, dtype=recurrent.dtype)
        per_z = recurrent.unsqueeze(-2).expand(*leading, self.z_dim, recurrent.shape[-1])
        z_code = z_eye.expand(*leading, self.z_dim, self.z_dim)
        low = self.low_feature(torch.cat((per_z, z_code), dim=-1))
        low_logits = self.low_actor(low).reshape(
            *leading, self.z_dim, NUM_ROLES, self.action_dim
        ).transpose(-3, -2)
        low_values = self.low_critic(low).reshape(
            *leading, self.z_dim, NUM_ROLES
        ).transpose(-2, -1)
        return high_logits, high_values, low_logits, low_values


class JointHiPTPolicy(nn.Module):
    """One central recurrent state per team, with independent role options.

    Inputs are the canonical spatial observation for chef 0 and the shared
    863-dimensional public state. The same encoder and LSTM run once per
    environment; role-conditioned heads produce three distinct option and
    primitive-action distributions.
    """

    def __init__(self, z_dim=4, action_dim=ACTION_DIM, state_dim=STATE_DIM,
                 channels=SPATIAL_CHANNELS, height=SPATIAL_HEIGHT,
                 width=SPATIAL_WIDTH, hidden_size=HIDDEN_SIZE, head_size=HEAD_SIZE):
        super().__init__()
        if z_dim < 2 or action_dim < 2 or min(state_dim, channels, height, width,
                                               hidden_size, head_size) <= 0:
            raise ValueError("Invalid policy dimensions")
        self.z_dim, self.action_dim = int(z_dim), int(action_dim)
        self.state_dim, self.channels = int(state_dim), int(channels)
        self.height, self.width = int(height), int(width)
        self.hidden_size = int(hidden_size)
        self.central = CentralEncoder(
            state_dim=state_dim, channels=channels, height=height, width=width,
            hidden_size=hidden_size,
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
        self.heads = _RoleHeads(hidden_size, z_dim, action_dim, head_size)

    def initial_state(self, batch_size: int, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (1, int(batch_size), self.hidden_size)
        return (torch.zeros(shape, device=device, dtype=dtype),
                torch.zeros(shape, device=device, dtype=dtype))

    def _encode(self, spatial, state):
        return self.central(spatial, state)

    def step(self, spatial: torch.Tensor, public_state: torch.Tensor, recurrent_state,
             episode_done: torch.Tensor | None = None):
        encoded = self._encode(spatial, public_state)
        h, c = recurrent_state
        expected = (1, encoded.shape[0], self.hidden_size)
        if h.ndim != 3 or h.shape != c.shape or h.shape != expected:
            raise ValueError(f"LSTM state must have shape {expected}")
        if episode_done is not None:
            if episode_done.shape != (encoded.shape[0],):
                raise ValueError("episode_done must have shape [B]")
            keep = (1.0 - episode_done.to(encoded.dtype)).view(1, -1, 1)
            h, c = h * keep, c * keep
        output, next_state = self.lstm(encoded.unsqueeze(0), (h, c))
        return (*self.heads(output.squeeze(0)), next_state)

    def unroll(self, spatial: torch.Tensor, public_state: torch.Tensor,
               recurrent_state, episode_done: torch.Tensor):
        """Differentiably process [T,B] team sequences with reset boundaries."""
        if spatial.ndim != 5 or public_state.ndim != 3:
            raise ValueError("Expected spatial [T,B,C,H,W] and state [T,B,D]")
        time, batch = spatial.shape[:2]
        if public_state.shape[:2] != (time, batch) or episode_done.shape != (time, batch):
            raise ValueError("Sequence inputs must agree on [T,B]")
        if time == 0:
            raise ValueError("Cannot unroll an empty sequence")
        # Exactly one CNN/encoder evaluation per team frame, never per chef.
        encoded = self._encode(
            spatial.flatten(0, 1), public_state.flatten(0, 1)
        ).reshape(time, batch, self.hidden_size)
        h, c = recurrent_state
        outputs = []
        for t in range(time):
            keep = (1.0 - episode_done[t].to(encoded.dtype)).view(1, batch, 1)
            out, (h, c) = self.lstm(encoded[t:t + 1], (h * keep, c * keep))
            outputs.append(out.squeeze(0))
        recurrent = torch.stack(outputs, dim=0)
        return (*self.heads(recurrent), (h, c), recurrent)

    def unroll_complete_episodes(self, spatial: torch.Tensor, public_state: torch.Tensor,
                                 recurrent_state):
        """Fused BPTT for complete, independent episode columns.

        This fast path assumes every batch column starts at episode time zero
        with its own initial state and contains no mid-sequence terminal reset.
        The LSTM consumes the whole time-major tensor in one call, preserving
        gradients across every valid frame; callers may pad episode tails and
        exclude those outputs from their loss.
        """
        if spatial.ndim != 5 or public_state.ndim != 3:
            raise ValueError("Expected spatial [T,B,C,H,W] and state [T,B,D]")
        time, batch = spatial.shape[:2]
        if time == 0 or public_state.shape[:2] != (time, batch):
            raise ValueError("Complete episode inputs must agree on nonempty [T,B]")
        h, c = recurrent_state
        if h.shape != c.shape or h.shape != (1, batch, self.hidden_size):
            raise ValueError("Complete episode LSTM state must have shape [1,B,H]")
        encoded = self._encode(
            spatial.flatten(0, 1), public_state.flatten(0, 1)
        ).reshape(time, batch, self.hidden_size)
        recurrent, (h, c) = self.lstm(encoded, (h, c))
        return (*self.heads(recurrent), (h, c), recurrent)

    def parameter_counts(self):
        parts = {
            "spatial_encoder": self.central.spatial,
            "state_encoder": self.central.encoder,
            "lstm": self.lstm,
            "role_conditioned_heads": self.heads,
        }
        result = {name: sum(p.numel() for p in module.parameters())
                  for name, module in parts.items()}
        result["total"] = sum(p.numel() for p in self.parameters())
        result["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return result


class CentralFlatActorCritic(nn.Module):
    """Central one-forward population actor, producing one action per chef."""

    def __init__(self, action_dim=ACTION_DIM, state_dim=STATE_DIM,
                 channels=SPATIAL_CHANNELS, height=SPATIAL_HEIGHT,
                 width=SPATIAL_WIDTH, hidden_size=HIDDEN_SIZE, head_size=HEAD_SIZE):
        super().__init__()
        self.action_dim, self.state_dim = int(action_dim), int(state_dim)
        self.central = CentralEncoder(
            state_dim=state_dim, channels=channels, height=height, width=width,
            hidden_size=hidden_size,
        )
        role = torch.eye(NUM_ROLES)
        self.register_buffer("role_codes", role, persistent=False)
        self.actor_feature = nn.Sequential(
            _init(nn.Linear(hidden_size + NUM_ROLES, head_size)), nn.LeakyReLU()
        )
        self.actor = _init(nn.Linear(head_size, action_dim), gain=0.01)
        self.critic = _init(nn.Linear(head_size, 1), gain=1.0)

    def forward(self, spatial: torch.Tensor, state: torch.Tensor):
        encoded = self.central(spatial, state)
        role = self.role_codes.to(device=encoded.device, dtype=encoded.dtype)
        role = role.expand(encoded.shape[0], NUM_ROLES, NUM_ROLES)
        repeated = encoded[:, None, :].expand(-1, NUM_ROLES, -1)
        features = self.actor_feature(torch.cat((repeated, role), dim=-1))
        return self.actor(features), self.critic(features).squeeze(-1)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

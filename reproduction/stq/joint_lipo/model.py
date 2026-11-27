"""Centralized, joint-action LIPO actor and unchanged value/MI modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

STATE_DIM = 863
OBS_DIM = 866
ACTION_DIM = 6
NUM_ROLES = 3
LATENT_DIM = 8
SHARED_KITCHEN_ENVIRONMENT = "SharedKitchen"
SHARED_KITCHEN_COMMIT = "de37e0818c2e76311e105ab8689e5162bd84133f"


@dataclass(frozen=True)
class JointLipoConfig:
    state_dim: int = STATE_DIM
    obs_dim: int = OBS_DIM
    action_dim: int = ACTION_DIM
    num_roles: int = NUM_ROLES
    latent_dim: int = LATENT_DIM
    hidden_dim: int = 64
    hidden_layers: int = 2
    population: int = 8


def _mlp(input_dim: int, output_dim: int, config: JointLipoConfig, output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(config.hidden_layers):
        linear = nn.Linear(width, config.hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ELU()))
        width = config.hidden_dim
    head = nn.Linear(width, output_dim)
    nn.init.orthogonal_(head.weight, gain=output_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class JointLipoActor(nn.Module):
    """Encode one team state and emit logits for all three independent actions.

    Three independent episode latents are concatenated in fixed chef order;
    the output's three contiguous six-logit slices are the fixed role heads.
    The state/latent trunk is evaluated once per team row.
    """

    def __init__(self, config: JointLipoConfig):
        super().__init__()
        self.config = config
        input_dim = config.state_dim + config.num_roles * config.latent_dim
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(config.hidden_layers):
            layer = nn.Linear(width, config.hidden_dim)
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ELU()))
            width = config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        # The contiguous 18 logits are the fixed role heads 0, 1, and 2.
        self.head = nn.Linear(config.hidden_dim, config.num_roles * config.action_dim)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(f"state must have shape [B, {self.config.state_dim}]")
        expected = (state.shape[0], self.config.num_roles)
        if tuple(latent_ids.shape) != expected:
            raise ValueError(f"latent_ids must have shape {expected}")
        if torch.any((latent_ids < 0) | (latent_ids >= self.config.latent_dim)):
            raise ValueError(f"latent_ids must be in [0, {self.config.latent_dim})")
        latents = torch.nn.functional.one_hot(
            latent_ids.long(), self.config.latent_dim
        ).to(dtype=state.dtype).flatten(start_dim=1)
        encoded = self.trunk(torch.cat((state, latents), dim=-1))
        return self.head(encoded).reshape(
            state.shape[0], self.config.num_roles, self.config.action_dim
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class JointLipoMember(nn.Module):
    """One centralized three-action policy, critics, and the original MI model."""

    def __init__(self, config: JointLipoConfig):
        super().__init__()
        self.config = config
        self.actor = JointLipoActor(config)
        self.sp_critic = _mlp(config.state_dim, 1, config, output_gain=1.0)
        self.xp_critics = nn.ModuleList(
            [_mlp(config.state_dim, 1, config, output_gain=1.0)
             for _ in range(config.population)]
        )
        # q(z_i | native vector_i, role_i, action probabilities_i), unchanged.
        self.discriminator = _mlp(
            config.obs_dim + config.num_roles + config.action_dim,
            config.latent_dim,
            config,
            output_gain=1.0,
        )

    def value(self, state: torch.Tensor, opponent: int | None = None) -> torch.Tensor:
        critic = self.sp_critic if opponent is None else self.xp_critics[opponent]
        return critic(state).squeeze(-1)

    def infer_latent(
        self,
        obs: torch.Tensor,
        role_ids: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        roles = torch.nn.functional.one_hot(
            role_ids.long(), self.config.num_roles
        ).to(obs.dtype)
        return self.discriminator(torch.cat((obs, roles, action_probs), dim=-1))


def actor_parameter_count(config: JointLipoConfig | None = None) -> int:
    """Return the analytic count used in the model manifest."""
    config = config or JointLipoConfig()
    input_dim = config.state_dim + config.num_roles * config.latent_dim
    hidden = config.hidden_dim
    trunk = input_dim * hidden + hidden
    for _ in range(config.hidden_layers - 1):
        trunk += hidden * hidden + hidden
    heads = config.num_roles * (hidden * config.action_dim + config.action_dim)
    return trunk + heads


def model_config_dict(config: JointLipoConfig) -> dict[str, int]:
    return asdict(config)

"""Small LIPO policy, critic, and latent-inference networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from baseline_common import ENVIRONMENT_COMMIT as SHARED_KITCHEN_COMMIT
from baseline_common import LEGACY_ENVIRONMENT_COMMIT

SHARED_KITCHEN_ENVIRONMENT = "SharedKitchen"


@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int = 866
    state_dim: int = 863
    action_dim: int = 6
    num_roles: int = 3
    latent_dim: int = 8
    hidden_dim: int = 64
    hidden_layers: int = 2
    population: int = 8


def _mlp(input_dim: int, output_dim: int, cfg: ModelConfig, output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(cfg.hidden_layers):
        linear = nn.Linear(width, cfg.hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ELU()))
        width = cfg.hidden_dim
    head = nn.Linear(width, output_dim)
    nn.init.orthogonal_(head.weight, gain=output_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class LipoMember(nn.Module):
    """One joint policy and its SP/partner-specific value estimators.

    A member's actor is shared across all chefs. Role and latent one-hots make
    the shared actor distinguish the three chef slots and the independent
    per-chef latent choices.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        actor_input = config.obs_dim + config.num_roles + config.latent_dim
        self.actor = _mlp(actor_input, config.action_dim, config, output_gain=0.01)
        self.sp_critic = _mlp(config.state_dim, 1, config, output_gain=1.0)
        # One XP critic for each possible partner index. Self-play never uses
        # the diagonal critic, but retaining a fixed K-wide bank simplifies
        # checkpoints and partner-indexed value targets.
        self.xp_critics = nn.ModuleList(
            [_mlp(config.state_dim, 1, config, output_gain=1.0) for _ in range(config.population)]
        )
        # q(z | local observation, action probabilities), shared across roles.
        self.discriminator = _mlp(
            config.obs_dim + config.num_roles + config.action_dim,
            config.latent_dim,
            config,
            output_gain=1.0,
        )

    def actor_logits(self, obs: torch.Tensor, role_ids: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        roles = torch.nn.functional.one_hot(role_ids.long(), self.config.num_roles).to(obs.dtype)
        latents = torch.nn.functional.one_hot(latent_ids.long(), self.config.latent_dim).to(obs.dtype)
        return self.actor(torch.cat((obs, roles, latents), dim=-1))

    def value(self, state: torch.Tensor, opponent: int | None = None) -> torch.Tensor:
        network = self.sp_critic if opponent is None else self.xp_critics[opponent]
        return network(state).squeeze(-1)

    def infer_latent(
        self,
        obs: torch.Tensor,
        role_ids: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        roles = torch.nn.functional.one_hot(role_ids.long(), self.config.num_roles).to(obs.dtype)
        return self.discriminator(torch.cat((obs, roles, action_probs), dim=-1))


class LipoPopulation:
    """Inference wrapper for a set of complete three-chef joint policies."""

    def __init__(self, members: list[LipoMember], device: str | torch.device = "cpu"):
        self.members = members
        self.device = torch.device(device)
        for member in members:
            member.to(self.device).eval()

    @torch.no_grad()
    def act(
        self,
        member_id: int,
        observations: np.ndarray,
        latent_ids: np.ndarray | None = None,
        *,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return simultaneous actions for observations shaped ``[E,3,866]``.

        If ``latent_ids`` is omitted, each environment/chef independently
        samples a latent at the caller's episode boundary. Pass the same IDs
        on subsequent steps to keep a latent fixed for each episode.
        """
        if not 0 <= member_id < len(self.members):
            raise IndexError(f"member_id must be in [0,{len(self.members) - 1}]")
        obs = np.asarray(observations, dtype=np.float32)
        cfg = self.members[member_id].config
        if obs.ndim != 3 or obs.shape[1:] != (cfg.num_roles, cfg.obs_dim):
            raise ValueError(f"Expected observations [E,{cfg.num_roles},{cfg.obs_dim}], got {obs.shape}")
        if latent_ids is None:
            rng = rng or np.random.default_rng()
            latent_ids = rng.integers(cfg.latent_dim, size=obs.shape[:2], dtype=np.int64)
        latent_ids = np.asarray(latent_ids, dtype=np.int64)
        if latent_ids.shape != obs.shape[:2] or np.any((latent_ids < 0) | (latent_ids >= cfg.latent_dim)):
            raise ValueError(f"Expected latent IDs [E,{cfg.num_roles}] in [0,{cfg.latent_dim})")
        member = self.members[member_id]
        obs_t = torch.as_tensor(obs, device=self.device).reshape(-1, cfg.obs_dim)
        role_t = torch.arange(cfg.num_roles, device=self.device).repeat(obs.shape[0])
        z_t = torch.as_tensor(latent_ids, device=self.device).reshape(-1)
        dist = Categorical(logits=member.actor_logits(obs_t, role_t, z_t))
        actions = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
        return actions.reshape(obs.shape[0], cfg.num_roles).cpu().numpy().astype(np.int64)


def _checkpoint_load(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before the weights_only parameter was added.
        return torch.load(path, map_location=map_location)


def load_population(path: str | Path, device: str | torch.device = "cpu") -> LipoPopulation:
    """Load a SharedKitchen LIPO v2 population for evaluation/use."""
    checkpoint = _checkpoint_load(path, device)
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("environment") != SHARED_KITCHEN_ENVIRONMENT
        or checkpoint.get("environment_commit") not in (SHARED_KITCHEN_COMMIT, LEGACY_ENVIRONMENT_COMMIT)
    ):
        raise ValueError(f"{path} is not a SharedKitchen LIPO v2 checkpoint")
    config_data = checkpoint.get("model_config")
    if config_data is None:
        raise ValueError(f"{path} is not a LIPO population checkpoint")
    config = ModelConfig(**config_data)
    if config.obs_dim != 866 or config.state_dim != 863:
        raise ValueError("Checkpoint dimensions do not match the SharedKitchen observation contract")
    members = [LipoMember(config) for _ in range(config.population)]
    state_list = checkpoint.get("member_states")
    if state_list is None or len(state_list) != config.population:
        raise ValueError("Checkpoint does not contain every population member")
    for member, state in zip(members, state_list):
        member.load_state_dict(state["model"] if "model" in state else state)
    return LipoPopulation(members, device)

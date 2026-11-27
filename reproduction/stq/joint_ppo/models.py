"""Small centralized actors and an independent state-value critic."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

STATE_DIM = 863
NUM_AGENTS = 3
ACTION_DIM = 6
SPATIAL_SHAPE = (26, 7, 7)
HEAD_SHAPE = (NUM_AGENTS, ACTION_DIM)


def _init_linear(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _validate_actor_inputs(spatial: torch.Tensor, state: torch.Tensor) -> int:
    if not torch.is_tensor(spatial) or not torch.is_tensor(state):
        raise TypeError("actor inputs must be torch tensors")
    if spatial.ndim != 4 or tuple(spatial.shape[1:]) != SPATIAL_SHAPE:
        raise ValueError("canonical spatial input must have shape [batch, 26, 7, 7]")
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError("centralized state input must have shape [batch, 863]")
    if spatial.shape[0] != state.shape[0]:
        raise ValueError("spatial and state batch sizes must match")
    if spatial.device != state.device:
        raise ValueError("spatial and state inputs must be on the same device")
    return int(state.shape[0])


class JointActor(nn.Module):
    """Actor contract: encode each team state once and emit three action heads."""

    backbone: str

    def __init__(self, model_size: str = "small") -> None:
        super().__init__()
        if model_size not in ("small", "large", "million"):
            raise ValueError(f"unknown model_size: {model_size}")
        self.model_size = model_size
        self.head = _init_linear(nn.Linear(self.head_size, NUM_AGENTS * ACTION_DIM), 0.01)

    @property
    def head_size(self) -> int:
        raise NotImplementedError

    def encode(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = _validate_actor_inputs(spatial, state)
        hidden = self.encode(spatial, state)
        if hidden.ndim != 2 or hidden.shape != (batch, self.head_size):
            raise RuntimeError("actor trunk must return one feature vector per team state")
        return self.head(hidden).reshape(batch, NUM_AGENTS, ACTION_DIM)

    def distribution(self, spatial: torch.Tensor, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self(spatial, state))


class MLPActor(JointActor):
    """Two-layer state MLP with a single joint 18-logit output projection."""

    backbone = "mlp"

    @property
    def head_size(self) -> int:
        return {"small": 128, "large": 256, "million": 640}[self.model_size]

    def __init__(self, model_size: str = "small") -> None:
        super().__init__(model_size)
        hidden = {"small": 128, "large": 256, "million": 640}[model_size]
        self.trunk = nn.Sequential(
            _init_linear(nn.Linear(STATE_DIM, hidden)),
            nn.Tanh(),
            _init_linear(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )

    def encode(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.trunk(state)


class CNNActor(JointActor):
    """Spatial convolution plus full state, fused before a joint action head."""

    backbone = "cnn"

    @property
    def head_size(self) -> int:
        return {"small": 64, "large": 128, "million": 256}[self.model_size]

    def __init__(self, model_size: str = "small") -> None:
        super().__init__(model_size)
        channels, fusion_hidden, output_hidden = {
            "small": (16, 96, 64),
            "large": (32, 192, 128),
            "million": (64, 384, 256),
        }[model_size]
        self.conv = nn.Conv2d(26, channels, kernel_size=3, padding=0)
        self.fusion = nn.Sequential(
            _init_linear(nn.Linear(channels * 5 * 5 + STATE_DIM, fusion_hidden)),
            nn.Tanh(),
            _init_linear(nn.Linear(fusion_hidden, output_hidden)),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.conv.weight, math.sqrt(2.0))
        nn.init.zeros_(self.conv.bias)

    def encode(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(self.conv(spatial)).flatten(start_dim=1)
        return self.fusion(torch.cat((features, state), dim=-1))


class TransformerActor(JointActor):
    """49 board tokens plus one public-state token, encoded as one sequence."""

    backbone = "transformer"
    @property
    def head_size(self) -> int:
        return 256 if self.model_size == "million" else 128

    def __init__(self, model_size: str = "small") -> None:
        super().__init__(model_size)
        d_model = {"small": 48, "large": 96, "million": 192}[model_size]
        self.d_model = d_model
        self.spatial_projection = _init_linear(nn.Linear(26, d_model))
        self.state_projection = _init_linear(nn.Linear(STATE_DIM, d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, 50, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward={"small": 192, "large": 256, "million": 512}[model_size],
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(d_model),
        )
        if model_size == "million":
            self.fusion = nn.Sequential(
                _init_linear(nn.Linear(192, 256)),
                nn.Tanh(),
                _init_linear(nn.Linear(256, 256)),
                nn.Tanh(),
            )
        elif model_size == "large":
            self.fusion = nn.Sequential(
                _init_linear(nn.Linear(96, 192)),
                nn.Tanh(),
                _init_linear(nn.Linear(192, 128)),
                nn.Tanh(),
            )
        else:
            self.fusion = nn.Sequential(
                _init_linear(nn.Linear(48, 128)),
                nn.Tanh(),
                _init_linear(nn.Linear(128, 128)),
                nn.Tanh(),
            )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self._initialize_transformer()

    def _initialize_transformer(self) -> None:
        for layer in self.encoder.layers:
            attention = layer.self_attn
            nn.init.xavier_uniform_(attention.in_proj_weight)
            if attention.in_proj_bias is not None:
                nn.init.zeros_(attention.in_proj_bias)
            nn.init.xavier_uniform_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)
            for linear in (layer.linear1, layer.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
            for norm in (layer.norm1, layer.norm2):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)
        nn.init.ones_(self.encoder.norm.weight)
        nn.init.zeros_(self.encoder.norm.bias)

    def encode(self, spatial: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # The centralized state is token zero; board positions occupy tokens 1:50.
        spatial_tokens = spatial.flatten(start_dim=2).transpose(1, 2)
        spatial_tokens = self.spatial_projection(spatial_tokens)
        state_token = self.state_projection(state).unsqueeze(1)
        tokens = torch.cat((state_token, spatial_tokens), dim=1)
        encoded = self.encoder(tokens + self.position_embedding)
        return self.fusion(encoded[:, 0])


class StateCritic(nn.Module):
    """Independent centralized value network; it shares no actor parameters."""

    def __init__(self, model_size: str = "small") -> None:
        super().__init__()
        if model_size not in ("small", "large", "million"):
            raise ValueError(f"unknown model_size: {model_size}")
        hidden = {"small": 128, "large": 256, "million": 512}[model_size]
        self.network = nn.Sequential(
            _init_linear(nn.Linear(STATE_DIM, hidden)),
            nn.Tanh(),
            _init_linear(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _init_linear(nn.Linear(hidden, 1), 1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(state) or state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError("critic state input must have shape [batch, 863]")
        return self.network(state).squeeze(-1)


class JointPolicy(nn.Module):
    """Checkpointable actor/critic bundle with deliberately independent trunks."""

    def __init__(self, backbone: str, model_size: str = "small") -> None:
        super().__init__()
        if model_size not in ("small", "large", "million"):
            raise ValueError(f"unknown model_size: {model_size}")
        if backbone == "mlp":
            self.actor = MLPActor(model_size)
        elif backbone == "cnn":
            self.actor = CNNActor(model_size)
        elif backbone == "transformer":
            self.actor = TransformerActor(model_size)
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.critic = StateCritic(model_size)
        self.backbone = backbone
        self.model_size = model_size

    def parameter_counts(self) -> dict[str, int]:
        actor = sum(parameter.numel() for parameter in self.actor.parameters())
        critic = sum(parameter.numel() for parameter in self.critic.parameters())
        return {"actor": actor, "critic": critic, "total": actor + critic}


def build_policy(backbone: str, model_size: str = "small") -> JointPolicy:
    """Construct one supported actor size and its independent critic."""
    return JointPolicy(backbone, model_size)

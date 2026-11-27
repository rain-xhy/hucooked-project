"""MAT's joint observation encoder and causal action decoder."""

import math

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def feed_forward(width, outputs):
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs))


class DecoderBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feed_forward = feed_forward(width, width)
        self.prefix_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, prefix, encoded, mask):
        prefix_attention, _ = self.self_attention(prefix, prefix, prefix, attn_mask=mask, need_weights=False)
        prefix = self.prefix_norm(prefix + prefix_attention)
        cross_attention, _ = self.cross_attention(encoded, prefix, prefix, attn_mask=mask, need_weights=False)
        combined = self.cross_norm(encoded + cross_attention)
        return self.output_norm(combined + self.feed_forward(combined))


class MultiAgentTransformer(nn.Module):
    def __init__(self, observation_size, agents, actions, width=64, blocks=1, heads=1):
        super().__init__()
        self.agents = agents
        self.actions = actions
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(observation_size), nn.Linear(observation_size, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.encoder = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(width, heads, width, dropout=0, activation="gelu", batch_first=True)
                for _ in range(blocks)
            ]
        )
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, 1))
        self.action_projection = nn.Sequential(nn.Linear(actions + 1, width, bias=False), nn.GELU(), nn.LayerNorm(width))
        self.decoder = nn.ModuleList(DecoderBlock(width, heads) for _ in range(blocks))
        self.policy_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, actions)
        )
        self.register_buffer("causal_mask", torch.triu(torch.ones(agents, agents, dtype=torch.bool), diagonal=1))
        self.apply(self._initialize)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            with torch.no_grad():
                for projection in module.in_proj_weight.chunk(3, dim=0):
                    nn.init.orthogonal_(projection)
                nn.init.zeros_(module.in_proj_bias)

    def _encode(self, observations):
        return self.encoder(self.observation_projection(observations))

    def _tokens(self, encoded):
        tokens = encoded.new_zeros((encoded.shape[0], self.agents, self.actions + 1))
        tokens[:, 0, 0] = 1
        return tokens

    def _decode(self, tokens, encoded):
        prefix = self.action_projection(tokens)
        for block in self.decoder:
            prefix = block(prefix, encoded, self.causal_mask)
        return self.policy_head(prefix)

    def value(self, observations):
        return self.value_head(self._encode(observations)).squeeze(-1)

    def forward(self, observations, actions, temperature=1.0, detach_value_encoder=False):
        encoded = self._encode(observations)
        tokens = self._tokens(encoded)
        tokens[:, 1:, 1:] = F.one_hot(actions[:, :-1], self.actions).to(tokens.dtype)
        distribution = Categorical(logits=self._decode(tokens, encoded) / temperature)
        value_input = encoded.detach() if detach_value_encoder else encoded
        return distribution.log_prob(actions), self.value_head(value_input).squeeze(-1), distribution.entropy()

    @torch.no_grad()
    def act(self, observations, greedy=False, temperature=1.0):
        encoded = self._encode(observations)
        tokens = self._tokens(encoded)
        actions = []
        log_probs = []
        for agent in range(self.agents):
            logits = self._decode(tokens, encoded)[:, agent]
            distribution = Categorical(logits=logits / temperature)
            action = logits.argmax(dim=-1) if greedy else distribution.sample()
            actions.append(action)
            log_probs.append(distribution.log_prob(action))
            if agent + 1 < self.agents:
                tokens[:, agent + 1, 1:] = F.one_hot(action, self.actions).to(tokens.dtype)
        return torch.stack(actions, 1), torch.stack(log_probs, 1), self.value_head(encoded).squeeze(-1)

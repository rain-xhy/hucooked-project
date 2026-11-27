"""Inference-only normal-order fallback before picking the joint ingredient."""

import torch

from .mixed_kitchen import ACTIVE, HANDOFF_TILE
from .mixed_tasks import (
    DISH, EMPTY, GET_DISH, GET_ONION, GET_TOMATO, LOAD_POT,
    PLATE_SOUP, PREPARE_HANDOFF, SERVE, TOMATO, TOMATO_SOUP, UNKNOWN,
)

INITIAL_DELAY = 10
HISTORY_DECAY = .9
SUPPORT_THRESHOLD = .5
REJECTION_THRESHOLD = -2.
MIN_OBSERVED_MOVES = 4
GATE_METRICS = ("early_handoff_vetoes", "supported_partner_ticks", "observable_partner_moves",
                "supported_teammate_0_ticks", "supported_teammate_1_ticks")


class EarlyHandoffGate:
    def __init__(self, mode, env):
        self.mode = mode
        self.history = mode in ("history", "reject")
        self.env = env
        self.teammates = env.kitchen.player_order[env.players, 1:]
        self.evidence = env.returns.new_zeros((env.num_envs, 2))
        self.moves_seen = torch.zeros_like(self.teammates)
        self.batch_start = env.state.tick.clone()
        self.metrics = env.kitchen.rows.new_zeros((env.num_envs, len(GATE_METRICS)))
        if self.history:
            self.policy = env.partner_policies.get("bc") or env.load_partner_policy("bc")

    def choose(self, actions, active):
        env = self.env
        state, teammates = env.state, self.teammates
        held = state.held[env.kitchen.rows, env.players]
        partner_items = state.held.gather(1, teammates)
        positions = state.position.gather(1, teammates)
        orientation = state.orientation.gather(1, teammates)
        front = env.kitchen.front[positions, orientation]
        ready = ((partner_items == EMPTY) & (front == env.kitchen.facilities[:, HANDOFF_TILE, None])).any(1)

        delayed = state.tick - self.batch_start < INITIAL_DELAY
        if self.history:
            supported = (self.evidence >= SUPPORT_THRESHOLD) & (partner_items == EMPTY)
            supported_any = supported.any(1)
            self.metrics[:, 1] += active & supported_any
            self.metrics[:, 3:5] += active[:, None] & supported
            if self.mode == "reject":
                rejected = (self.evidence <= REJECTION_THRESHOLD) & (self.moves_seen >= MIN_OBSERVED_MOVES)
                delayed = rejected.all(1)
            else:
                delayed = delayed | ~supported_any
            self.move_probabilities = self.env.partner_logits(env.teammate_views(teammates), self.policy)[..., :4].softmax(-1)

        normal_pot = (state.pots == 0).any(1)
        joint_pot = (state.pots == 1).any(1)
        tomato_available = state.stock[:, 0] & (state.pots < 0).any(1)
        alternative = torch.full_like(actions, UNKNOWN)
        alternative = torch.where((held == EMPTY) & normal_pot, GET_DISH, alternative)
        alternative = torch.where((held == EMPTY) & tomato_available, GET_TOMATO, alternative)
        alternative = torch.where(held == TOMATO, LOAD_POT, alternative)
        alternative = torch.where((held == DISH) & normal_pot & ~joint_pot, PLATE_SOUP, alternative)
        alternative = torch.where(held == TOMATO_SOUP, SERVE, alternative)
        legal = env.tensor_observations["subtask_mask"].gather(1, alternative[:, None]).squeeze(1)

        unopened = (state.status[:, 1] == ACTIVE) & state.stock[:, 1]
        normal_open = (state.status[:, 0] == ACTIVE) & (state.tick < state.due[:, 0])
        pursuing_joint = (actions == GET_ONION) | (actions == PREPARE_HANDOFF)
        blocked = active & unopened & normal_open & pursuing_joint & delayed & ~ready & legal & (alternative != UNKNOWN)
        self.metrics[:, 0] += blocked
        return torch.where(blocked, alternative, actions)

    def update(self, before, active, done):
        after, teammates = self.env.state, self.teammates
        self.batch_start = torch.where(active & ((before.batch != after.batch) | done), after.tick, self.batch_start)
        if not self.history:
            return

        next_orientation = after.orientation.gather(1, teammates)
        observable = active[:, None] & ~done[:, None] & (
            (before.position.gather(1, teammates) != after.position.gather(1, teammates)) |
            (before.orientation.gather(1, teammates) != next_orientation)
        )
        self.metrics[:, 2] += observable.sum(1)
        self.moves_seen = torch.where(done[:, None], 0, self.moves_seen + observable)
        # A changed pose reveals a direction. Staying and interacting can be
        # observationally identical, so neither is imputed from a fixed pose.
        probability = self.move_probabilities.gather(2, next_orientation[:, :, None]).squeeze(2)
        likelihood_ratio = (.1 + 3.6 * probability).log()
        updated = HISTORY_DECAY * self.evidence + torch.where(observable, likelihood_ratio, 0.)
        self.evidence = torch.where(active[:, None], torch.where(done[:, None], 0., updated), self.evidence)

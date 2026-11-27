"""MixedKitchen-v1 adapter with one shared learner rotating across the three roles."""

from collections import deque
from functools import lru_cache
from itertools import combinations

import numpy as np
import torch
from overcooked_ai_py.cooperative.kitchen import DIRECTIONS, INTERACT, STAY
from overcooked_ai_py.cooperative.model import KitchenConfig
from overcooked_ai_py.cooperative.torch_kitchen import TorchMixedKitchen
from overcooked_ai_py.cooperative.torch_state import (
    ACTIVE, CERTIFIED, DISH, EMPTY, INGREDIENT, ITEM_CERTIFIED, ITEM_KIND, ITEM_ORDER,
    JOINT, METRIC_INDEX, NORMAL, POT_ORDER, REMAINING, SOUP, STATUS, STOCK,
)

from torch_kitchen_vec import TorchKitchenVecEnv


ROLES = ("joint_cook", "normal_cook", "helper")
CONFIG = KitchenConfig(shuffle_players=False, shaping_scale=2.0)
GOALS = (
    ((1, 1), (-1, 0)), ((9, 1), (1, 0)), ((2, 1), (0, -1)), ((8, 1), (0, -1)),
    ((1, 5), (-1, 0)), ((4, 2), (1, 0)), ((6, 2), (-1, 0)),
    ((4, 3), (1, 0)), ((6, 3), (-1, 0)), ((9, 5), (1, 0)),
    ((3, 3), None), ((9, 3), None), ((7, 5), None),
)
ONION, TOMATO, LEFT_POT, RIGHT_POT, PLATE, DROP, TAKE, SEND, RECEIVE, SERVE, PARK_LEFT, PARK_RIGHT, PARK_HELPER = range(len(GOALS))


@lru_cache(maxsize=1)
def routes(floor, width, height):
    size = width * height
    table = np.full((size, size, len(GOALS), size), STAY, dtype=np.uint8)
    positions = {y * width + x for x, y in floor}
    offsets = tuple(y * width + x for x, y in DIRECTIONS)
    neighbors = {cell: tuple((a, cell + d) for a, d in enumerate(offsets)
                            if cell + d in positions) for cell in positions}
    for a, b in combinations(sorted(positions), 2):
        available = positions - {a, b}
        for goal, ((x, y), _) in enumerate(GOALS):
            target = y * width + x
            if target not in available:
                continue
            queue = deque([target])
            distance = {target: 0}
            while queue:
                cell = queue.popleft()
                for _, other in neighbors[cell]:
                    if other in available and other not in distance:
                        distance[other] = distance[cell] + 1
                        queue.append(other)
            for cell, steps in distance.items():
                if steps:
                    table[a, b, goal, cell] = next(
                        action for action, other in neighbors[cell] if distance.get(other) == steps - 1
                    )
    return torch.from_numpy(table)


class MixedRules:
    """Batched compact-layout roles following the native reference workflow."""

    def __init__(self, kitchen):
        self.env = kitchen.env
        env = self.env
        self.table = routes(frozenset(env._native.floor), env.width, env.height).to(env.device)
        self.cells = torch.tensor([x + y * env.width for (x, y), _ in GOALS], device=env.device)
        self.facing = torch.tensor([DIRECTIONS.index(d) if d else -1 for _, d in GOALS], device=env.device)
        self.peers = torch.tensor([[1, 2], [0, 2], [0, 1]], device=env.device)
        self.helper_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.batch = torch.full_like(self.helper_stage, -1)

    def act(self):
        state = self.env.state
        kind, slot, certified = state.held.unbind(-1)
        active = state.orders[..., STATUS] == ACTIVE
        stock = state.orders[..., STOCK] > 0
        joint_pending = active[:, JOINT] & (state.orders[:, JOINT, CERTIFIED] == 0)
        plate_available = state.buffers[:, 0, ITEM_KIND] == DISH
        self.helper_stage.masked_fill_((state.tick == 0) | (self.batch != state.batch), 0)
        self.batch.copy_(state.batch)
        stage = self.helper_stage
        helper_kind = kind[:, 2]
        stage.masked_fill_((stage == 1) & (helper_kind == EMPTY), 2)
        stage.masked_fill_((stage == 0) & (plate_available | ~active[:, NORMAL]), 2)
        stage.masked_fill_((stage == 0) & (helper_kind == DISH), 1)

        goals, interact = [], []
        for player, order in enumerate((JOINT, NORMAL)):
            k, s = kind[:, player], slot[:, player]
            pot = state.pots[:, player]
            own_pot = LEFT_POT + player
            needs_ingredient = active[:, order] & stock[:, order]
            needs_plate = active[:, order] & (pot[:, POT_ORDER] == order)
            plate_goal = PLATE if player == 0 else TAKE
            goal = torch.full_like(k, PARK_LEFT + player)
            goal = torch.where(needs_plate, torch.where(k == DISH, own_pot, plate_goal), goal)
            goal = torch.where(needs_ingredient, torch.where(k == EMPTY, ONION + player, PLATE), goal)
            goal = torch.where(k == INGREDIENT, torch.where(s == JOINT, LEFT_POT, RIGHT_POT), goal)
            goal = torch.where(k == SOUP, torch.where((s == NORMAL) | certified[:, player].bool(), SERVE, SEND), goal)
            goals.append(goal)
            interact.append((goal < PARK_LEFT) & ((goal != TAKE) | plate_available)
                            & ((goal != own_pot) | (k != DISH) | (pot[:, REMAINING] == 0)))
        helper_goal = torch.where(joint_pending, RECEIVE, PARK_HELPER)
        helper_goal = torch.where(stage == 0, PLATE, torch.where(stage == 1, DROP, helper_goal))
        helper_goal = torch.where(helper_kind == SOUP, SERVE, helper_goal)
        goals.append(helper_goal)
        interact.append((helper_goal != PARK_HELPER) & ((helper_goal != DROP) | ~plate_available))

        goals = torch.stack(goals, dim=1)
        press = torch.stack(interact, dim=1)
        positions = state.positions
        a, b = positions[:, self.peers].unbind(-1)
        action = self.table[a.minimum(b), a.maximum(b), goals, positions].long()
        facing = self.facing[goals]
        at_goal = positions == self.cells[goals]
        on_goal = torch.where((facing >= 0) & (facing != state.orientations), facing,
                              torch.where(press, INTERACT, STAY))
        return torch.where(state.pair, INTERACT, torch.where(at_goal, on_goal, action))


class MixedKitchenBatch:
    def __init__(self, num_envs, device, seed):
        self.env = TorchMixedKitchen(CONFIG, device=device, num_envs=num_envs)
        self.seed = seed

    @property
    def ticks(self):
        return self.env.state.tick

    def reset(self, mask=None):
        obs, _ = self.env.reset(self.seed if mask is None else None, mask=mask)
        return obs

    def step(self, actions):
        obs, reward, terminated, truncated, _ = self.env.step(actions)
        state = self.env.state
        normal = state.metrics[:, METRIC_INDEX["normal_completed"]]
        joint = state.metrics[:, METRIC_INDEX["joint_completed"]]
        self.delivery_metrics = torch.stack((normal + joint, state.latencies.sum(dim=1)), dim=1)
        done = terminated | truncated
        return obs, reward, done, truncated, torch.where(done & (normal > 0) & (joint > 0), 0, -1)


class MixedVecEnv(TorchKitchenVecEnv):
    layout_name = "MixedKitchen-v1/compact"
    kitchen_type = MixedKitchenBatch
    horizon = CONFIG.horizon
    observation_size = 190
    recipe_names = ("MixedKitchen orders",)
    reports_deliveries = True

    def reset(self):
        self.roles = torch.tensor([ROLES.index(spec) for spec in self.opponent_specs], device=self.device)
        self.rows = torch.arange(self.num_envs, device=self.device)
        self.peer_slots = torch.tensor([[1, 2], [0, 2], [0, 1]], device=self.device)[self.roles]
        self.role_features = torch.eye(3, device=self.device).expand(self.num_envs, -1, -1)
        return super().reset()

    def observe(self, dense, joint_actions=None):
        return torch.cat((dense, self.role_features), dim=-1)

    def learner_view(self, obs):
        return obs[self.rows, self.roles]

    def make_rule_partner(self):
        return MixedRules(self.kitchen)

    def combine_actions(self, learner, team):
        first, second = team.gather(1, self.peer_slots).unbind(1)
        team[self.rows, self.roles] = learner
        return team, first * 6 + second

    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        if infos.finished.size:
            metrics = self.kitchen.delivery_metrics.cpu().tolist()
            for lane in infos.finished:
                infos[lane]["delivery_count"], infos[lane]["delivery_steps"] = metrics[lane]
        return obs, reward, done, infos

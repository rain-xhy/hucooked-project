"""Random single-ticket service on the native two-chef soup_coordination map."""

from functools import lru_cache

import numpy as np
import torch
from torch.nn.functional import one_hot
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.mdp.torch_env import (
    COOK_TICK, COUNTER, DISH, DISH_SOURCE, EMPTY, FLOOR, KIND, ONION,
    ONION_SOURCE, POT, SERVING, SOUP, STAY_INDEX, TOMATO, TOMATO_SOURCE, TorchOvercookedEnv,
)

from rule_chefs import RuleChef
from torch_kitchen_vec import TorchKitchenVecEnv


LAYOUT = "soup_coordination"
HORIZON = 400
ROLES = ("prep_cook", "plate_serve")


@lru_cache(maxsize=1)
def route_table():
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    planner = RuleChef()
    planner.set_mdp(mdp)
    size = mdp.width * mdp.height
    costs = np.full((size, 4, size, size), 255, dtype=np.uint8)
    actions = np.full_like(costs, STAY_INDEX)

    def cell(pos):
        return pos[1] * mdp.width + pos[0]

    for pos in planner.floor:
        for blocker in planner.floor - {pos}:
            for heading, direction in enumerate(Direction.ALL_DIRECTIONS):
                _, targets = planner._routes(pos, direction, {blocker})
                for target, (cost, action) in targets.items():
                    key = (cell(pos), heading, cell(blocker), cell(target))
                    costs[key] = cost
                    actions[key] = Action.ACTION_TO_INDEX[action]
    return torch.from_numpy(costs), torch.from_numpy(actions)


class ComplexKitchen:
    def __init__(self, num_envs, device, seed):
        mdp = OvercookedGridworld.from_layout_name(LAYOUT)
        self.env = TorchOvercookedEnv(mdp, HORIZON, device=device, num_envs=num_envs)
        self.device = self.env.device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.rows = torch.arange(num_envs, device=device)
        self.floor = self.env._cells[self.env._tiles == FLOOR]
        self.pot = self.env._flat(mdp.get_pot_locations()[0])
        self.orders = torch.tensor([[r.ingredients.count("onion"), r.ingredients.count("tomato")]
                                    for r in mdp.get_standard_start_state().all_orders], device=device)
        self.ticket = torch.zeros_like(self.rows)
        self.issued = torch.zeros_like(self.rows)
        self.delivery_count = torch.zeros_like(self.rows)
        self.delivery_steps = torch.zeros_like(self.rows)
        rewards = self.env._initial_rewards
        axis = torch.arange(rewards.shape[0], device=device)
        valid = (axis[None, :, None] == self.orders[:, 0, None, None]) & (
            axis[None, None, :] == self.orders[:, 1, None, None])
        self.rewards = valid * rewards
        self.viable = ((axis[None, :, None] <= self.orders[:, 0, None, None]) &
                       (axis[None, None, :] <= self.orders[:, 1, None, None])) * self.rewards.amax(dim=(1, 2))[:, None, None]
        cells = self.env._cells
        self.coords = torch.stack((cells % mdp.width / (mdp.width - 1),
                                   cells // mdp.width / (mdp.height - 1)), dim=-1)

    @property
    def ticks(self):
        return self.env.state.timestep

    def checkpoint(self):
        return {
            "state": self.env.state,
            "sparse": self.env._sparse_total,
            "shaped": self.env._shaped_total,
            "rng": self.generator.get_state(),
            "ticket": self.ticket,
            "issued": self.issued,
            "delivery_count": self.delivery_count,
            "delivery_steps": self.delivery_steps,
        }

    def restore_checkpoint(self, snapshot):
        state = snapshot["state"]
        self.env.state = type(state)(*(field.to(self.device) for field in state))
        self.env._sparse_total.copy_(snapshot["sparse"])
        self.env._shaped_total.copy_(snapshot["shaped"])
        self.generator.set_state(snapshot["rng"])
        self.ticket.copy_(snapshot["ticket"])
        self.issued.copy_(snapshot["issued"])
        self.delivery_count.copy_(snapshot["delivery_count"])
        self.delivery_steps.copy_(snapshot["delivery_steps"])
        self.env._rewards = self.rewards[self.ticket]
        self.env._optimal = self.viable[self.ticket]

    def issue(self, mask):
        chosen = torch.randint(len(self.orders), self.ticket.shape, device=self.device, generator=self.generator)
        self.ticket.copy_(torch.where(mask, chosen, self.ticket))
        self.issued.copy_(torch.where(mask, self.ticks, self.issued))
        self.env._rewards = self.rewards[self.ticket]
        self.env._optimal = self.viable[self.ticket]

    def reset(self, mask=None):
        self.env.reset(mask)
        if mask is None:
            mask = torch.ones_like(self.rows, dtype=torch.bool)
        a = torch.randint(len(self.floor), self.rows.shape, device=self.device, generator=self.generator)
        b = torch.randint(len(self.floor) - 1, self.rows.shape, device=self.device, generator=self.generator)
        positions = self.floor[torch.stack((a, b + (b >= a)), dim=1)]
        facing = torch.randint(4, positions.shape, device=self.device, generator=self.generator)
        state = self.env.state
        self.env.state = state._replace(positions=torch.where(mask[:, None], positions, state.positions),
                                        orientations=torch.where(mask[:, None], facing, state.orientations))
        self.delivery_count.masked_fill_(mask, 0)
        self.delivery_steps.masked_fill_(mask, 0)
        self.issue(mask)
        return self.observe()

    def contents(self, items):
        return torch.stack(self.env._ingredient_counts(items), dim=-1)

    def observe(self):
        state = self.env.state
        positions, facing, objects, held, _ = state
        players = torch.cat((self.coords[positions], one_hot(facing, 4), one_hot(held[..., KIND].long(), 5),
                             self.contents(held) / 3), dim=-1)
        pot = objects[:, self.pot]
        common = torch.cat((self.contents(pot) / 3, pot[:, COOK_TICK, None].float() / 100,
                            self.orders[self.ticket] / 3, self.ticks[:, None] / HORIZON,
                            (self.ticks - self.issued)[:, None] / HORIZON), dim=1)
        front = self.env._neighbors[positions, facing]
        front_kind = objects[self.rows[:, None], front, KIND].long()
        return torch.cat((players[:, self.env._view_order].flatten(2),
                          common[:, None].expand(-1, 2, -1),
                          one_hot(self.env._tiles[front], 8), one_hot(front_kind, 5)), dim=-1)

    def step(self, actions):
        _, sparse, done, info = self.env.step(actions)
        served = (info["sparse_r_by_agent"] > 0).sum(dim=1)
        self.delivery_count += served
        self.delivery_steps += served * (self.ticks - self.issued)
        self.metrics = torch.stack((self.delivery_count, self.delivery_steps), dim=1)
        self.issue(served > 0)
        return (self.observe(), sparse + info["shaped_r_by_agent"].sum(dim=1), done, done,
                torch.where(done & (self.delivery_count > 0), 0, -1))


class ComplexRules:
    def __init__(self, kitchen, learner_roles, teammate):
        self.kitchen = kitchen
        self.prep = learner_roles == 1
        self.teammate = teammate
        self.costs, self.actions = [table.to(kitchen.device) for table in route_table()]

    def act(self):
        k = self.kitchen
        rows, me = k.rows, self.teammate
        positions, facing, objects, held, _ = k.env.state
        key = (positions[rows, me], facing[rows, me], positions[rows, 1 - me])
        costs, actions = self.costs[key], self.actions[key]
        tiles = k.env._tiles
        item = held[rows, me, KIND]
        pot = objects[:, k.pot]
        counts = k.contents(pot)
        missing = k.orders[k.ticket] - counts
        tick = pot[:, COOK_TICK]
        cooking = tick >= 0
        exact = (missing == 0).all(dim=1)
        wrong = (missing < 0).any(dim=1) | (counts.sum(dim=1) >= 3)
        start = ~cooking & (exact | wrong) & (counts.sum(dim=1) > 0)
        wants_plate = cooking & (held[rows, 1 - me, KIND] != DISH)
        ingredient = (item == ONION) | (item == TOMATO)
        can_add = ~cooking & ~wrong & (
            ((item == ONION) & (missing[:, 0] > 0)) | ((item == TOMATO) & (missing[:, 1] > 0)))
        target = torch.full_like(rows, -1)
        target = torch.where((item == EMPTY) & self.prep & ~cooking & ~start,
                             torch.where(missing[:, 0] > 0, ONION_SOURCE, TOMATO_SOURCE), target)
        target = torch.where((item == EMPTY) & self.prep & start, POT, target)
        target = torch.where((item == EMPTY) & ~self.prep & wants_plate, DISH_SOURCE, target)
        target = torch.where(ingredient, torch.where(can_add, POT, COUNTER), target)
        ready = cooking & (tick >= k.env._cook_time(pot))
        target = torch.where((item == DISH) & ready, POT, target)
        target = torch.where(item == SOUP, SERVING, target)
        valid = (tiles == target[:, None]) & ((tiles != COUNTER) | (objects[..., KIND] == EMPTY))
        nearest = costs.masked_fill(~valid, 255).argmin(dim=1)
        return torch.where(valid[rows, nearest] & (costs[rows, nearest] < 255),
                           actions[rows, nearest].long(), STAY_INDEX)


class ComplexVecEnv(TorchKitchenVecEnv):
    layout_name = LAYOUT
    horizon = HORIZON
    observation_size = 48
    kitchen_type = ComplexKitchen
    recipe_names = ("ordered soup",)
    reports_deliveries = True

    def reset(self):
        self.roles = torch.tensor([ROLES.index(spec) for spec in self.opponent_specs], device=self.device)
        self.rows = torch.arange(self.num_envs, device=self.device)
        self.slots = self.roles
        self.role_features = one_hot(self.roles, 2).float()[:, None].expand(-1, 2, -1)
        return super().reset()

    def make_rule_partner(self):
        return ComplexRules(self.kitchen, self.roles, 1 - self.slots)

    def learner_view(self, obs):
        return obs[self.rows, self.slots]

    def observe(self, dense, joint_actions=None):
        return torch.cat((dense, self.role_features), dim=-1)

    def combine_actions(self, learner, partner):
        first = self.slots == 0
        return torch.stack((torch.where(first, learner, partner), torch.where(first, partner, learner)), dim=1), partner

    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        if infos.finished.size:
            metrics = self.kitchen.metrics.cpu().tolist()
            for lane in infos.finished:
                infos[lane]["delivery_count"], infos[lane]["delivery_steps"] = metrics[lane]
        return obs, reward, done, infos

"""Native CUDA kitchen and rule chefs for the ordinary three-player handoff."""

from functools import lru_cache

import torch
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.mdp.torch_env import (
    COOK_TICK, COUNTER, DISH, DISH_SOURCE, EMPTY, FLOOR, INGREDIENTS, KIND,
    ONION, ONION_SOURCE, POT, SERVING, SOUP, TorchOvercookedEnv,
)

from hucooked_adapter import hucooked_action, nest_action_index
from rule_chefs import RuleChef


LAYOUT = "three_player_handoff"
HORIZON = 400
OBSERVATION_SIZE = 80
SUPPLY, COOK, SERVE = range(3)
HANDOFF = (5, 3)
PARKING = (9, 4)


class HandoffKitchen:
    def __init__(self, num_envs, device, seed=0):
        mdp = OvercookedGridworld.from_layout_name(LAYOUT)
        self.env = TorchOvercookedEnv(mdp, HORIZON, device=device, num_envs=num_envs)
        self.device = self.env.device
        self.fixtures = torch.tensor([
            self.cell(mdp.get_onion_dispenser_locations()[0]),
            self.cell(mdp.get_dish_dispenser_locations()[0]),
            self.cell(mdp.get_pot_locations()[0]),
            self.cell(mdp.get_serving_locations()[0]),
            self.cell(HANDOFF),
        ], device=device)
        self.pot = self.cell(mdp.get_pot_locations()[0])
        self.handoff = self.cell(HANDOFF)
        self.cook_time = Recipe(["onion"] * 3).time
        self.native_actions = torch.tensor([
            Action.ACTION_TO_INDEX[hucooked_action(i)] for i in range(Action.NUM_ACTIONS)
        ], device=device)
        self.view_order = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 0, 1]], device=device)
        self.tile_types = torch.tensor([FLOOR, COUNTER, ONION_SOURCE, DISH_SOURCE, POT, SERVING], device=device)
        self.item_types = torch.arange(6, device=device)
        self.loose_types = torch.tensor([ONION, DISH, SOUP], device=device)
        cells = self.env._cells
        xy = torch.stack((cells % mdp.width, cells // mdp.width), dim=-1)
        dimensions = torch.tensor(mdp.shape, device=device)
        delta = xy[None, :, :] - xy[:, None, :]
        self.xy = xy / dimensions
        self.relative = delta / dimensions
        self.distance = delta.abs().sum(dim=-1)
        self.rows = torch.arange(num_envs, device=device)

    def cell(self, position):
        x, y = position
        return y * self.env.width + x

    @property
    def ticks(self):
        return self.env.state.timestep

    def reset(self, mask=None):
        self.env.reset(mask)
        return self.observe()

    def step(self, actions):
        _, sparse, done, info = self.env.step(self.native_actions[actions])
        reward = sparse + info["shaped_r_by_agent"].sum(dim=1)
        success = done & (info["episode"]["ep_sparse_r"] > 0)
        return self.observe(), reward, done, done, torch.where(success, 0, -1)

    def checkpoint(self):
        return {
            "state": self.env.state,
            "sparse_total": self.env._sparse_total,
            "shaped_total": self.env._shaped_total,
        }

    def restore_checkpoint(self, checkpoint):
        state = checkpoint["state"]
        self.env.state = type(state)(*(field.to(self.device) for field in state))
        self.env._sparse_total.copy_(checkpoint["sparse_total"])
        self.env._shaped_total.copy_(checkpoint["shaped_total"])

    def _item_ids(self, items):
        kind = items[..., KIND]
        ingredients = (items[..., INGREDIENTS:] != EMPTY).sum(dim=-1)
        return torch.where(kind == SOUP, 2 + ingredients, torch.where(kind == DISH, 2, kind))

    def observe(self):
        positions, orientations, objects, held, _ = self.env.state
        peer_positions = positions[:, self.view_order[:, 1:]]
        teammate = self.relative[positions[..., None], peer_positions].flatten(2)
        facing = orientations[..., None] == self.native_actions[1:5]
        items = self._item_ids(held)[..., None] == self.item_types
        fixtures = self.relative[positions[..., None], self.fixtures].flatten(2)
        counters = self.env._tiles == COUNTER
        kinds = objects[..., KIND]
        loose_mask = counters & (kinds[:, None, :] == self.loose_types[:, None])
        distances = self.distance[positions][:, :, None, :].masked_fill(
            ~loose_mask[:, None], self.env.width * self.env.height,
        )
        closest = distances.argmin(dim=-1)
        present = loose_mask.any(dim=-1)[:, None, :, None].expand(-1, 3, -1, -1)
        loose = torch.cat((self.relative[positions[..., None], closest] * present, present), dim=-1)

        front = self.env._neighbors[positions, orientations]
        front_tiles = self.env._tiles[front]
        front_items = self._item_ids(objects[self.rows[:, None], front]).masked_fill(front_tiles != COUNTER, EMPTY)
        blocked = self.env._tiles[self.env._neighbors[positions][..., self.native_actions[1:5]]] != FLOOR
        pot = objects[:, self.pot]
        progress = torch.stack(((pot[:, INGREDIENTS:] != EMPTY).sum(dim=-1) / 3,
                                pot[:, COOK_TICK].float() / self.cook_time, self.ticks / HORIZON), dim=-1)
        handoff = self._item_ids(objects[:, self.handoff])[:, None] == self.item_types
        return torch.cat((
            self.xy[positions], teammate, facing[:, self.view_order].flatten(2), items[:, self.view_order].flatten(2),
            fixtures, loose.flatten(2), front_tiles[..., None] == self.tile_types,
            front_items[..., None] == self.item_types, blocked,
            progress[:, None].expand(-1, 3, -1), handoff[:, None].expand(-1, 3, -1),
        ), dim=-1)


ROUTE_ONION, ROUTE_HANDOFF, ROUTE_DISH, ROUTE_POT, ROUTE_SERVE, ROUTE_PARK = range(6)


@lru_cache(maxsize=1)
def route_table():
    """Build next-action lookups for each pose and middle chef position."""
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    planner = RuleChef()
    planner.set_mdp(mdp)
    fixtures = (mdp.get_onion_dispenser_locations()[0], HANDOFF, mdp.get_dish_dispenser_locations()[0],
                mdp.get_pot_locations()[0], mdp.get_serving_locations()[0])
    size = mdp.width * mdp.height
    table = torch.zeros((size, len(Direction.ALL_DIRECTIONS), size, 6), dtype=torch.long)

    def cell(position):
        return position[0] + mdp.width * position[1]

    for position in planner.floor:
        for blocker in planner.floor - {position}:
            for facing, direction in enumerate(Direction.ALL_DIRECTIONS):
                locations, targets = planner._routes(position, direction, {blocker})
                park = locations.get(PARKING, (0, Action.STAY))[1]
                actions = [targets.get(target, (0, park if route == ROUTE_POT else Action.STAY))[1]
                           for route, target in enumerate(fixtures)]
                table[cell(position), facing, cell(blocker)] = torch.tensor([
                    nest_action_index(action) for action in (*actions, park)
                ])
    return table


class HandoffRuleTeam:
    """Stage onions on the shared counter and plate and deliver the cooked soup."""

    def __init__(self, kitchen):
        self.kitchen = kitchen
        self.routes = route_table().to(kitchen.device)

    def act(self):
        k = self.kitchen
        positions, facing, objects, held, _ = k.env.state
        held_kind = held[..., KIND]
        counter_free = objects[:, k.handoff, KIND] == EMPTY
        supply_routes = self.routes[positions[:, SUPPLY], facing[:, SUPPLY], positions[:, COOK]]
        supply = torch.where(held_kind[:, SUPPLY] == EMPTY, supply_routes[:, ROUTE_ONION],
                             torch.where(counter_free, supply_routes[:, ROUTE_HANDOFF], 0))

        serve_routes = self.routes[positions[:, SERVE], facing[:, SERVE], positions[:, COOK]]
        ready = objects[:, k.pot, COOK_TICK] >= k.cook_time
        serve_item = held_kind[:, SERVE]
        serve = torch.where(ready & (serve_item == DISH), serve_routes[:, ROUTE_POT], serve_routes[:, ROUTE_PARK])
        serve = torch.where(serve_item == EMPTY, serve_routes[:, ROUTE_DISH], serve)
        serve = torch.where(serve_item == SOUP, serve_routes[:, ROUTE_SERVE], serve)
        return torch.stack((supply, serve), dim=1)

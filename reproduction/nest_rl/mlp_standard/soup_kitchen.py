"""Batched Torch port of the native cramped_room onion-soup kitchen."""

from functools import lru_cache
from types import SimpleNamespace

import torch
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe

from hucooked_adapter import hucooked_action, nest_action_index
from rule_chefs import RuleChef


HORIZON = 400
OBSERVATION_SIZE = 60
FLOOR, COUNTER, ONION_SOURCE, DISH_SOURCE, POT, SERVE = range(6)
EMPTY, ONION, DISH, SOUP_1, SOUP_2, SOUP_3 = range(6)


def native_mdp():
    return OvercookedGridworld.from_layout_name("cramped_room")


class SoupKitchen:
    def __init__(self, num_envs, device, seed=0):
        mdp = native_mdp()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.width, self.height = mdp.shape
        self.dimensions = torch.tensor(mdp.shape, device=device)
        self.tiles = torch.tensor([" XODPS".index(t) for row in mdp.terrain_mtx for t in row], device=device)
        self.cells = torch.arange(len(self.tiles), device=device)
        self.rows = torch.arange(num_envs, device=device)
        self.xy = torch.stack((self.cells % self.width, self.cells // self.width), dim=-1)
        self.start = torch.tensor([x + y * self.width for x, y in mdp.start_player_positions], device=device)
        directions = torch.tensor([hucooked_action(i) for i in range(5)], device=device)
        neighbors = (self.xy[:, None, :] + directions).clamp_min(0).minimum(self.dimensions - 1)
        self.neighbors = neighbors[..., 0] + neighbors[..., 1] * self.width
        self.sources = {
            tile: self.cells[self.tiles == tile] for tile in (ONION_SOURCE, DISH_SOURCE, POT, SERVE)
        }
        order = Recipe(["onion"] * 3)
        self.cook_time, self.soup_reward = order.time, order.value
        self.add_reward = mdp.reward_shaping_params["PLACEMENT_IN_POT_REW"]
        self.dish_reward = mdp.reward_shaping_params["DISH_PICKUP_REWARD"]
        self.plate_reward = mdp.reward_shaping_params["SOUP_PICKUP_REWARD"]
        self.agent_pos = self.start.expand(num_envs, -1).clone()
        self.orientation = torch.full_like(self.agent_pos, 4)
        self.held = torch.zeros_like(self.agent_pos)
        self.counter_items = torch.zeros((num_envs, len(self.tiles)), dtype=torch.long, device=device)
        self.onions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cook_tick = torch.full_like(self.onions, -1)
        self.ticks = torch.zeros_like(self.onions)
        self.delivered = torch.zeros_like(self.onions, dtype=torch.bool)
        self.last_sparse = torch.zeros_like(self.onions, dtype=torch.float32)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones_like(self.delivered)
        self.agent_pos.copy_(torch.where(mask[:, None], self.start, self.agent_pos))
        self.orientation.masked_fill_(mask[:, None], 4)
        self.held.masked_fill_(mask[:, None], EMPTY)
        self.counter_items.masked_fill_(mask[:, None], EMPTY)
        self.onions.masked_fill_(mask, 0)
        self.cook_tick.masked_fill_(mask, -1)
        self.ticks.masked_fill_(mask, 0)
        self.delivered.masked_fill_(mask, False)
        self.last_sparse.masked_fill_(mask, 0)
        return self.observe()

    def step(self, actions):
        sparse = torch.zeros_like(self.last_sparse)
        shaped = torch.zeros_like(sparse)
        # The original MDP takes one pot-state snapshot for dish shaping, then
        # resolves interactions in player order before either player moves.
        nonempty_for_dish = (self.onions > 0) & ((self.onions < 3) | (self.cook_tick >= 0))
        for player in range(2):
            target = self.neighbors[self.agent_pos[:, player], self.orientation[:, player]]
            tile = self.tiles[target]
            interact = actions[:, player] == 5
            item = self.held[:, player]
            empty = item == EMPTY
            counter = self.counter_items[self.rows, target]
            counter_exchange = interact & (tile == COUNTER) & (empty != (counter == EMPTY))
            pickup_onion = interact & empty & (tile == ONION_SOURCE)
            pickup_dish = interact & empty & (tile == DISH_SOURCE)
            useful_dish = pickup_dish & nonempty_for_dish & ~(self.held == DISH).any(dim=1)
            useful_dish &= ~(self.counter_items == DISH).any(dim=1)
            at_pot = interact & (tile == POT)
            add = at_pot & (item == ONION) & (self.onions < 3) & (self.cook_tick < 0)
            start = at_pot & empty & (self.onions > 0) & (self.cook_tick < 0)
            plate = at_pot & (item == DISH) & (self.cook_tick == self.cook_time)
            deliver = interact & (tile == SERVE) & (item >= SOUP_1)

            sparse += self.soup_reward * (deliver & (item == SOUP_3))
            shaped += self.add_reward * add + self.dish_reward * useful_dish + self.plate_reward * plate
            self.counter_items[self.rows, target] = torch.where(counter_exchange, item, counter)
            self.held[:, player] = torch.where(counter_exchange, counter, item)
            self.held[:, player].masked_fill_(pickup_onion, ONION)
            self.held[:, player].masked_fill_(pickup_dish, DISH)
            self.held[:, player].masked_fill_(add | deliver, EMPTY)
            self.held[:, player].copy_(torch.where(plate, DISH + self.onions, self.held[:, player]))
            self.onions.add_(add).masked_fill_(plate, 0)
            self.cook_tick.masked_fill_(start, 0).masked_fill_(plate, -1)

        movement = (actions >= 1) & (actions <= 4)
        self.orientation.copy_(torch.where(movement, actions, self.orientation))
        targets = self.neighbors[self.agent_pos, actions.clamp_max(4)]
        next_pos = torch.where(movement & (self.tiles[targets] == FLOOR), targets, self.agent_pos)
        collision = (next_pos[:, 0] == next_pos[:, 1]) | (next_pos == self.agent_pos.flip(1)).all(dim=1)
        self.agent_pos.copy_(torch.where(collision[:, None], self.agent_pos, next_pos))
        self.cook_tick.add_((self.cook_tick >= 0) & (self.cook_tick < self.cook_time))
        self.ticks += 1
        self.last_sparse.copy_(sparse)
        self.delivered |= sparse > 0
        done = self.ticks >= HORIZON
        return self.observe(), sparse + shaped, done, done, torch.where(done & self.delivered, 0, -1)

    def observe(self):
        xy = self.xy[self.agent_pos]
        pos = xy / self.dimensions
        teammate = (xy.flip(1) - xy) / self.dimensions
        orientation = self.orientation[..., None] == self.cells[:4] + 1
        held = self.held[..., None] == self.cells[:6]

        def relative(candidates, valid=None):
            delta = self.xy[candidates] - xy[:, :, None]
            distance = delta.abs().sum(dim=-1)
            if valid is not None:
                distance = distance.masked_fill(~valid[:, None], len(self.tiles))
            index = distance.argmin(dim=-1, keepdim=True)
            result = delta.gather(2, index[..., None].expand(-1, -1, -1, 2)).squeeze(2) / self.dimensions
            if valid is None:
                return result
            present = valid.any(dim=1)[:, None, None].expand(-1, 2, -1)
            return torch.cat((result * present, present), dim=-1)

        fixtures = [relative(positions) for positions in self.sources.values()]
        loose = [relative(self.cells, mask) for mask in (self.counter_items == ONION,
                                                        self.counter_items == DISH,
                                                        self.counter_items >= SOUP_1)]
        front = self.neighbors[self.agent_pos, self.orientation]
        front_tile = self.tiles[front][..., None] == self.cells[:6]
        front_item = self.counter_items.gather(1, front)[..., None] == self.cells[:6]
        blocked = self.tiles[self.neighbors[self.agent_pos][..., 1:]] != FLOOR
        progress = torch.stack((self.onions / 3, self.cook_tick / self.cook_time, self.ticks / HORIZON), dim=-1)
        return torch.cat((pos, teammate, orientation, orientation.flip(1), held, held.flip(1),
                          *fixtures, *loose, front_tile, front_item, blocked,
                          progress[:, None].expand(-1, 2, -1)), dim=-1)


@lru_cache(maxsize=1)
def prep_tables():
    """Precompute the native prep chef's routes and yielding decisions."""
    mdp = native_mdp()
    chef = RuleChef("prep")
    chef.set_mdp(mdp)
    width, height = mdp.shape
    size = width * height

    def cell(xy):
        return xy[0] + xy[1] * width

    costs = torch.full((size, 5, size, size), 10_000, dtype=torch.long)
    actions = torch.zeros_like(costs)
    yields = torch.zeros((size, 5, size, 5, 3), dtype=torch.long)
    pots = set(mdp.get_pot_locations())
    sources = set(mdp.get_onion_dispenser_locations())
    dish_and_serve = set(mdp.get_dish_dispenser_locations() + mdp.get_serving_locations())
    for pos in chef.floor:
        for other in chef.floor - {pos}:
            for facing in range(1, 5):
                me = SimpleNamespace(position=pos, orientation=hucooked_action(facing))
                _, routes = chef._routes(pos, me.orientation, {other})
                for tile, (steps, action) in routes.items():
                    key = (cell(pos), facing, cell(other), cell(tile))
                    costs[key] = steps * (size + 1) + tile[0] * height + tile[1]
                    actions[key] = nest_action_index(action)
                for peer_facing in range(1, 5):
                    peer = SimpleNamespace(position=other, orientation=hucooked_action(peer_facing))
                    for task, targets in enumerate((pots | sources, pots, pots | dish_and_serve)):
                        action = chef._yield_action(me, [peer], targets)
                        yields[cell(pos), facing, cell(other), peer_facing, task] = nest_action_index(action)
    return costs, actions, yields


class SoupPrepPartner:
    def __init__(self, kitchen, player):
        self.kitchen, self.player = kitchen, player
        self.costs, self.actions, self.yields = [table.to(kitchen.device) for table in prep_tables()]

    def act(self):
        k = self.kitchen
        p, other = self.player, 1 - self.player
        key = (k.agent_pos[:, p], k.orientation[:, p], k.agent_pos[:, other])
        costs, actions = self.costs[key], self.actions[key]
        yields = self.yields[(*key, k.orientation[:, other])]
        staged_onion = k.counter_items == ONION
        reachable_onion = staged_onion & (costs < 10_000)
        target_mask = torch.where(reachable_onion.any(dim=1, keepdim=True), reachable_onion,
                                  k.tiles[None] == ONION_SOURCE)
        target = costs.masked_fill(~target_mask, 10_000).argmin(dim=1)
        can_fetch = reachable_onion.any(dim=1) | (3 - k.onions > (k.held[:, other] == ONION) + staged_onion.sum(dim=1))
        staged_soup = (k.counter_items >= SOUP_1).any(dim=1) | (k.held[:, other] >= SOUP_1)
        pot_action = actions[:, k.sources[POT]].squeeze(1)
        pot_open = (k.cook_tick < 0) & (k.onions < 3)
        idle = torch.where(k.cook_tick >= 0, yields[:, 2], torch.where(pot_open, yields[:, 0], yields[:, 1]))
        fetch = torch.where((costs[k.rows, target] < 10_000) & can_fetch, actions[k.rows, target], idle)
        empty_action = torch.where(staged_soup, 0, torch.where(pot_open, fetch, idle))
        free_counter = (k.tiles == COUNTER) & (k.counter_items == EMPTY)
        counter = costs.masked_fill(~free_counter, 10_000).argmin(dim=1)
        drop_action = torch.where(costs[k.rows, counter] < 10_000, actions[k.rows, counter], 0)
        item = k.held[:, p]
        deliver_action = actions[:, k.sources[SERVE]].squeeze(1)
        result = torch.where(item == EMPTY, empty_action,
                             torch.where(item == ONION, torch.where(pot_open, pot_action, drop_action), 0))
        result = torch.where((item == DISH) & (k.cook_tick == k.cook_time), pot_action, result)
        result = torch.where(item >= SOUP_1, deliver_action, result)
        return torch.where((k.cook_tick >= 0) & (k.onions != 3), 0, result)

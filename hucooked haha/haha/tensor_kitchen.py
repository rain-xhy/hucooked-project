"""Batched dynamics for the five pinned onion-soup layouts."""

from typing import NamedTuple

import numpy as np
import torch

from overcooked_ai_py.mdp.overcooked_mdp import Direction, OvercookedGridworld

from .subtasks import LAYOUTS, UNKNOWN, completed

VOID, FLOOR, COUNTER, POT, ONIONS, DISHES, SERVE = range(7)
TERRAIN = {" ": FLOOR, "X": COUNTER, "P": POT, "O": ONIONS, "D": DISHES, "S": SERVE}
EMPTY, ONION, DISH, SOUP = range(4)
OBJECT_NAMES = (None, "onion", "dish", "soup")
STAY, INTERACT = 4, 5


class KitchenState(NamedTuple):
    position: torch.Tensor      # [batch, player], row-major cell index
    orientation: torch.Tensor   # [batch, player], Direction index
    held: torch.Tensor          # [batch, player, (kind, onions, cooking tick)]
    objects: torch.Tensor       # [batch, cell, (kind, onions, cooking tick)]
    object_order: torch.Tensor  # [batch, cell], native object insertion order
    timestep: torch.Tensor      # [batch]


class KitchenStep(NamedTuple):
    state: KitchenState
    reward: torch.Tensor        # [batch], sparse team reward
    done: torch.Tensor          # [batch], native kitchen horizon
    completed: torch.Tensor     # [batch, player], -1 for no completed subtask


class TensorKitchen:
    def __init__(self, layouts, *, device="cuda", horizon=400):
        if not layouts or any(layout not in LAYOUTS for layout in layouts):
            raise ValueError("Expected a batch of original HA² layouts")
        if horizon <= 0:
            raise ValueError("Kitchen horizon must be positive")
        self.device = torch.device(device)
        self.horizon = horizon
        self.layouts = tuple(layouts)
        names = tuple(dict.fromkeys(layouts))
        self.mdps = {name: OvercookedGridworld.from_layout_name(name) for name in names}
        self.width = max(mdp.shape[0] for mdp in self.mdps.values())
        self.height = max(mdp.shape[1] for mdp in self.mdps.values())
        self.num_cells = self.width * self.height
        self.batch_size = len(layouts)
        self.rows = torch.arange(self.batch_size, device=self.device)
        self.cells = torch.arange(self.num_cells, device=self.device)
        self.directions = torch.arange(4, device=self.device)
        self.layout_ids = self._tensor([names.index(name) for name in layouts])

        terrain = np.full((len(names), self.num_cells), VOID, dtype=np.int64)
        cook_times, soup_values = [], []
        for index, name in enumerate(names):
            mdp = self.mdps[name]
            initial = mdp.get_standard_start_state()
            if (len(initial.all_orders) != 1 or
                    tuple(initial.all_orders[0].ingredients) != ("onion",) * 3 or initial.bonus_orders):
                raise ValueError(f"Unsupported recipe configuration: {name}")
            cook_times.append(initial.all_orders[0].time)
            soup_values.append(initial.all_orders[0].value)
            for y, row in enumerate(mdp.terrain_mtx):
                for x, tile in enumerate(row):
                    terrain[index, self._cell((x, y))] = TERRAIN[tile]

        front = np.tile(np.arange(self.num_cells)[:, None], (1, 4))
        view = np.full((self.num_cells, 4, 49), self.num_cells, dtype=np.int64)
        turns = (2, 0, -1, 1)  # North, south, east, west.
        for cell in range(self.num_cells):
            x, y = self._position(cell)
            for direction, (dx, dy) in enumerate(Direction.ALL_DIRECTIONS):
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                    front[cell, direction] = self._cell((x + dx, y + dy))
            crop = np.full((7, 7), self.num_cells, dtype=np.int64)
            for i in range(7):
                for j in range(7):
                    cx, cy = x + i - 3, y + j - 3
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        crop[i, j] = self._cell((cx, cy))
            for direction, rotation in enumerate(turns):
                view[cell, direction] = np.rot90(crop, rotation).reshape(-1)
        self.front = self._tensor(front)
        self.view = self._tensor(view)
        self.terrain = self._tensor(terrain)[self.layout_ids]
        self.cook_time = self._tensor(cook_times)[self.layout_ids]
        self.soup_value = self._tensor(soup_values)[self.layout_ids]
        self.empty = self._tensor([EMPTY, 0, -1])
        self.onion = self._tensor([ONION, 0, -1])
        self.dish = self._tensor([DISH, 0, -1])
        self.initial = self.pack([self.mdps[name].get_standard_start_state() for name in layouts])

        events = np.full((len(TERRAIN) + 1, len(OBJECT_NAMES), len(OBJECT_NAMES)), -1, dtype=np.int64)
        for tile, terrain_id in TERRAIN.items():
            for before, before_name in enumerate(OBJECT_NAMES):
                for after, after_name in enumerate(OBJECT_NAMES):
                    try:
                        event = completed(before_name, after_name, tile)
                    except ValueError:
                        continue
                    if event is not None:
                        events[terrain_id, before, after] = event
        self.events = self._tensor(events)

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def _cell(self, position):
        x, y = position
        return y * self.width + x

    def _position(self, cell):
        return cell % self.width, cell // self.width

    def pack(self, states):
        """Copy canonical native states into a batch for initialization or verification."""
        if len(states) != self.batch_size:
            raise ValueError("Expected one native state per environment")

        def object_fields(obj):
            if obj is None:
                return EMPTY, 0, -1
            kind = OBJECT_NAMES.index(obj.name)
            if kind != SOUP:
                return kind, 0, -1
            if any(name != "onion" for name in obj.ingredients) or obj._cook_time is not None:
                raise ValueError("Expected a native onion soup with the layout's cook time")
            return kind, len(obj.ingredients), obj._cooking_tick

        objects = np.tile([EMPTY, 0, -1], (self.batch_size, self.num_cells, 1))
        order = np.full((self.batch_size, self.num_cells), -1, dtype=np.int64)
        for row, state in enumerate(states):
            for rank, (position, obj) in enumerate(state.objects.items()):
                cell = self._cell(position)
                objects[row, cell] = object_fields(obj)
                order[row, cell] = rank - self.num_cells
        return KitchenState(
            self._tensor([[self._cell(p.position) for p in s.players] for s in states]),
            self._tensor([[Direction.DIRECTION_TO_INDEX[p.orientation] for p in s.players] for s in states]),
            self._tensor([[object_fields(p.held_object) for p in s.players] for s in states]),
            self._tensor(objects),
            self._tensor(order),
            self._tensor([s.timestep for s in states]),
        )

    def reset_where(self, state, reset):
        return KitchenState(*(torch.where(reset.view(-1, *([1] * (field.ndim - 1))), initial, field)
                              for initial, field in zip(self.initial, state)))

    def step(self, state, actions):
        """Advance valid rows with native action indices; reset done rows before reuse."""
        position, orientation, held, objects, order, timestep = state
        objects, order = objects.clone(), order.clone()
        new_held, rewards, events = [], [], []
        # The pinned environment resolves the two players in order, then moves them.
        for player in (0, 1):
            action = actions[:, player]
            target = self.front[position[:, player], orientation[:, player]]
            tile = self.terrain[self.rows, target]
            held_obj = held[:, player]
            target_obj = objects[self.rows, target]
            carried, present = held_obj[:, 0], target_obj[:, 0]
            interacts = action == INTERACT
            counter = interacts & (tile == COUNTER)
            drop = counter & (carried != EMPTY) & (present == EMPTY)
            pickup = counter & (carried == EMPTY) & (present != EMPTY)
            pot = interacts & (tile == POT)
            pot_onion = pot & (carried == ONION) & (target_obj[:, 2] < 0) & (target_obj[:, 1] < 3)
            pot_ready = pot & (carried == DISH) & (present == SOUP) & (target_obj[:, 2] >= self.cook_time)
            deliver = interacts & (tile == SERVE) & (carried == SOUP)
            onions = target_obj[:, 1] + 1
            soup = torch.stack((torch.full_like(onions, SOUP), onions,
                                torch.where(onions == 3, 0, -1)), dim=-1)
            taken = pickup | pot_ready
            new_target = torch.where(drop[:, None], held_obj, target_obj)
            new_target = torch.where(taken[:, None], self.empty, new_target)
            new_target = torch.where(pot_onion[:, None], soup, new_target)
            objects[self.rows, target] = new_target
            created = drop | (pot_onion & (present == EMPTY))
            order[self.rows, target] = torch.where(
                created, 2 * timestep + player,
                torch.where(taken, -1, order[self.rows, target]),
            )
            result = torch.where(taken[:, None], target_obj, held_obj)
            result = torch.where((drop | pot_onion | deliver)[:, None], self.empty, result)
            result = torch.where((interacts & (carried == EMPTY) & (tile == ONIONS))[:, None],
                                 self.onion, result)
            result = torch.where((interacts & (carried == EMPTY) & (tile == DISHES))[:, None],
                                 self.dish, result)
            new_held.append(result)
            rewards.append(torch.where(deliver, self.soup_value, 0))
            events.append(self.events[tile, carried, result[:, 0]])

        motion = actions < 4
        directions = torch.where(motion, actions, orientation)
        next_position = self.front[position, directions]
        next_position = torch.where(motion & (self.terrain[self.rows[:, None], next_position] == FLOOR),
                                    next_position, position)
        p0, p1 = next_position.unbind(1)
        collision = (p0 == p1) | ((p0 == position[:, 1]) & (p1 == position[:, 0]))
        next_position = torch.where(collision[:, None], position, next_position)
        ticks = objects[:, :, 2]
        cooking = (objects[:, :, 0] == SOUP) & (ticks >= 0) & (ticks < self.cook_time[:, None])
        objects[:, :, 2] = ticks + cooking
        timestep = timestep + 1
        state = KitchenState(next_position, directions, torch.stack(new_held, 1), objects, order, timestep)
        return KitchenStep(state, rewards[0] + rewards[1], timestep >= self.horizon, torch.stack(events, 1))

    def observation(self, state, players, tasks=None):
        """Return the original cropped [batch, 27, 7, 7] observations."""
        actors = torch.stack((players, 1 - players), 1)
        position = state.position.gather(1, actors)
        direction = state.orientation.gather(1, actors)
        player_cells = position[:, :, None] == self.cells
        orientations = player_cells[:, :, None, :] & (
            direction[:, :, None, None] == self.directions[None, None, :, None]
        )
        objects = state.objects
        for actor in (0, 1):
            carrying = (state.held[:, actor, 0] != EMPTY)[:, None]
            at_cell = state.position[:, actor, None] == self.cells
            objects = torch.where((carrying & at_cell)[:, :, None], state.held[:, actor, None, :], objects)
        kind, count, ticks = objects.unbind(-1)
        pot = self.terrain == POT
        counter = self.terrain == COUNTER
        soup = kind == SOUP
        idle = pot & soup & (ticks < 0)
        cooking = pot & soup & (ticks >= 0)
        soup_outside = soup & ~pot
        ready = soup_outside | (cooking & (ticks >= self.cook_time[:, None]))
        onion_dispenser = self.terrain == ONIONS
        dish_dispenser = self.terrain == DISHES
        serving = self.terrain == SERVE
        onion, dish = kind == ONION, kind == DISH
        empty_counter = counter & (kind == EMPTY)
        zeros = torch.zeros_like(pot)
        if tasks is None:
            tasks = torch.full_like(players, UNKNOWN)
        goals = torch.stack((onion_dispenser, onion, pot & ~cooking & (count < 3), empty_counter,
                             dish_dispenser, dish, empty_counter, cooking, soup_outside, empty_counter,
                             serving, zeros), dim=1)
        urgency = ((self.horizon - state.timestep) < 40)[:, None] & (self.terrain != VOID)
        features = torch.stack((pot, counter, onion_dispenser, zeros, dish_dispenser, serving,
                                count * idle, zeros, count * (cooking | soup_outside), zeros,
                                torch.where(cooking, self.cook_time[:, None] - ticks, 0),
                                ready, dish, onion, zeros, urgency, goals[self.rows, tasks]), dim=1)
        planes = torch.cat((player_cells, orientations.flatten(1, 2), features), dim=1)
        planes = torch.nn.functional.pad(planes, (0, 1))
        crop = self.view[position[:, 0], direction[:, 0]]
        return planes.gather(2, crop[:, None, :].expand(-1, 27, -1)).reshape(-1, 27, 7, 7).float()

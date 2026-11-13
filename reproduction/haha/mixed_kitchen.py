"""Batched MixedKitchen-v1 transitions with device-resident state."""

import importlib
from collections import deque
from functools import cached_property, lru_cache
from itertools import permutations
from typing import NamedTuple

import numpy as np
import torch
from torch.nn.functional import one_hot

from .env import counter_transfer_bonus
from .mixed_tasks import (
    BUFFER_PUT, BUFFER_TAKE, CELLS, CERTIFIED_SOUP, DISH, EMPTY, GET_DISH, GET_ONION, GET_TOMATO,
    HANDOFF, HEIGHT, INTERACT, LOAD_POT, ONION, ONION_SOUP, PLATE_SOUP, RETURN_ITEM, SERVE, TOMATO,
    TOMATO_SOUP, WIDTH, item_order, mixed_config, mixed_source,
)

DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
ACTIVE, COMPLETED, EXPIRED = 0, 1, 2
ONION_SOURCE, TOMATO_SOURCE, DISH_SOURCE, SERVING, HANDOFF_TILE = range(5)
GAME_METRICS = ("normal_completed", "joint_completed", "joint_starts", "joint_aborts",
                "batches_completed", "normal_missed", "joint_missed", "collisions")


class MixedState(NamedTuple):
    position: torch.Tensor
    orientation: torch.Tensor
    held: torch.Tensor
    buffers: torch.Tensor
    pots: torch.Tensor
    remaining: torch.Tensor
    stock: torch.Tensor
    status: torch.Tensor
    due: torch.Tensor
    sync_due: torch.Tensor
    cook: torch.Tensor
    sync_steps: torch.Tensor
    certified: torch.Tensor
    pair: torch.Tensor
    progress: torch.Tensor
    tick: torch.Tensor
    batch: torch.Tensor


class MixedStep(NamedTuple):
    state: MixedState
    reward: torch.Tensor
    raw: torch.Tensor
    done: torch.Tensor
    events: torch.Tensor
    continuing: torch.Tensor
    aborted: torch.Tensor
    metrics: torch.Tensor | None = None


def update_state(state, **changes):
    # Build the tuple explicitly so Dynamo can trace it on PyTorch 2.3.
    return MixedState(*(changes.get(name, value) for name, value in zip(MixedState._fields, state)))


def select_state(mask, selected, other):
    return MixedState(*(torch.where(mask.view(-1, *([1] * (a.ndim - 1))), a, b)
                        for a, b in zip(selected, other)))


def cell(position):
    return position[1] * WIDTH + position[0]


@lru_cache(maxsize=1)
def egocentric_crop():
    """Index the same centered and oriented 7 × 7 crop as the original HA²."""
    views = []
    for index in range(CELLS):
        x, y = index % WIDTH, index // WIDTH
        crop = np.full((7, 7), CELLS, dtype=np.int64)
        for i in range(7):
            for j in range(7):
                cx, cy = x + i - 3, y + j - 3
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    crop[i, j] = cell((cx, cy))
        views.append([np.rot90(crop, turns).ravel() for turns in (2, 0, -1, 1)])
    return np.asarray(views)


@lru_cache(maxsize=None)
def counter_bonus_table(layout):
    """Precompute the author's counter reward for pots and serving facilities."""
    source = mixed_source()
    layouts = importlib.import_module(f"{source.__name__}.layouts")
    _, facilities, floor = layouts.load_layout(layout)

    def adjacent(position, direction):
        return tuple(a + b for a, b in zip(position, direction))

    poses = {(position, direction): [] for position in floor for direction in DIRECTIONS}
    for position, orientation in poses:
        for direction in DIRECTIONS:
            destination = adjacent(position, direction)
            target = (destination if destination in floor else position, direction)
            poses[target].append((position, orientation))
    table = torch.zeros((CELLS, 4, 2, 2), dtype=torch.float32)
    for is_soup, name in enumerate(("P", "S")):
        targets = facilities[name]
        distance = {pose: 1 for pose in poses if adjacent(*pose) in targets}
        frontier = deque(distance)
        while frontier:
            pose = frontier.popleft()
            for previous in poses[pose]:
                if previous not in distance:
                    distance[previous] = distance[pose] + 1
                    frontier.append(previous)

        def cost(pose, _features):
            return distance.get(pose, float("inf"))

        for position, orientation in poses:
            if adjacent(position, orientation) not in facilities["B"]:
                continue
            for pickup in (False, True):
                table[cell(position), DIRECTIONS.index(orientation), is_soup, int(pickup)] = (
                    counter_transfer_bonus(floor, cost, position, orientation, targets, pickup)
                )
    return table


def pack_native(states, device):
    """Project native snapshots onto the CUDA state contract."""
    def item_code(item, order_slot):
        if item is None:
            return EMPTY
        if item.kind == "dish":
            return DISH
        if item.certified:
            return CERTIFIED_SOUP
        return (TOMATO if item.kind == "ingredient" else TOMATO_SOUP) + order_slot[item.order_id]

    records = []
    for state in states:
        slots = {order.id: i for i, order in enumerate(state.orders)}
        records.append(MixedState(
            [cell(chef.position) for chef in state.chefs],
            [DIRECTIONS.index(chef.orientation) for chef in state.chefs],
            [item_code(chef.held, slots) for chef in state.chefs],
            [item_code(item, slots) for item in state.buffers],
            [slots.get(pot.order_id, -1) for pot in state.pots],
            [pot.remaining for pot in state.pots],
            [bool(order.stock) for order in state.orders],
            [("active", "completed", "expired").index(order.status) for order in state.orders],
            [order.due_at for order in state.orders], state.orders[1].sync_due_at,
            [order.cook_steps for order in state.orders], state.orders[1].sync_steps,
            state.orders[1].certified, [i in state.joint.pair for i in range(3)],
            state.joint.progress, state.tick, state.batch,
        ))
    return MixedState(*(torch.as_tensor(field, device=device) for field in zip(*records)))


class TensorMixedKitchen:
    def __init__(self, layouts, horizon, *, device="cuda", track_metrics=False):
        source = mixed_source()
        layout_module = importlib.import_module(f"{source.__name__}.layouts")
        self.layouts = tuple(layouts)
        self.config = mixed_config(layouts[0], horizon)
        self.device = torch.device(device)
        self.track_metrics = track_metrics
        self.rows = torch.arange(len(layouts), device=device)
        self.slots = torch.arange(2, device=device)
        self.eye = torch.eye(3, dtype=torch.bool, device=device)
        self.player_order = self._tensor(((0, 1, 2), (1, 0, 2), (2, 0, 1)))
        self.spawn_permutations = self._tensor(tuple(permutations(range(3))))
        self.categories = self._tensor((0, 1, 2, 3, 4, 5, 5))
        self.serve_items = self._tensor((TOMATO_SOUP, CERTIFIED_SOUP))
        self.serve_rewards = torch.tensor((self.config.normal_reward, self.config.joint_reward),
                                          device=device)
        self.order_kinds = torch.eye(2, device=device).expand(len(layouts), -1, -1)
        # An out-of-bounds neighbor is a non-walkable sentinel.
        self.front = self._tensor([
            [cell((x + dx, y + dy)) if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT else CELLS
             for dx, dy in DIRECTIONS]
            for y in range(HEIGHT) for x in range(WIDTH)
        ])
        starts, floors, facilities = [], [], []
        geometries = {name: layout_module.load_layout(name) for name in dict.fromkeys(layouts)}
        for name in layouts:
            _, loc, floor = geometries[name]
            starts.append([cell(loc[str(i)][0]) for i in (1, 2, 3)])
            floors.append([(x, y) in floor for y in range(HEIGHT) for x in range(WIDTH)] + [False])
            facilities.append([cell(loc[k][0]) for k in "OTDSH"] +
                              [cell(p) for p in loc["P"] + loc["B"]])
        self.starts = self._tensor(starts)
        self.floor = torch.as_tensor(floors, device=device)
        self.facilities = self._tensor(facilities)
        self.map_obs = torch.cat((self.floor[:, :CELLS].float(), self._coords(self.facilities).flatten(1)), 1)
        c = self.config
        self.duration_scale = max(64, c.cook_steps + c.duration_jitter, c.sync_steps + c.duration_jitter)
        self.time_scale = max(c.horizon, c.normal_deadline, c.joint_deadline) + c.deadline_jitter

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.long, device=self.device)

    def goal_facilities(self, state, players, tasks, *, allow_cooking=False):
        held = state.held[self.rows, players]
        returning = tasks == RETURN_ITEM
        loading = (tasks == LOAD_POT)[:, None] & (state.pots < 0)
        plating = (tasks == PLATE_SOUP)[:, None] & (state.pots >= 0)
        if not allow_cooking:
            plating = plating & (state.remaining == 0)
        placing = (tasks == BUFFER_PUT)[:, None] & (state.buffers == EMPTY)
        taking = (tasks == BUFFER_TAKE)[:, None] & (state.buffers != EMPTY)
        fixed = torch.stack((
            (tasks == GET_ONION) | (returning & (held == ONION)),
            (tasks == GET_TOMATO) | (returning & (held == TOMATO)),
            (tasks == GET_DISH) | (returning & (held == DISH)),
            tasks == SERVE,
            tasks == HANDOFF,
        ), dim=1)
        return torch.cat((fixed, loading | plating, placing | taking), dim=1)

    def goal_plane(self, state, players, tasks, *, allow_cooking=False):
        """Mark interaction targets in the same grid as the floor observation."""
        marked = self.goal_facilities(state, players, tasks, allow_cooking=allow_cooking)
        plane = torch.zeros((len(self.layouts), CELLS), dtype=torch.bool, device=self.device)
        return plane.scatter_(1, self.facilities, marked)

    @cached_property
    def crop_indices(self):
        return self._tensor(egocentric_crop())

    @cached_property
    def terrain_planes(self):
        facilities = one_hot(self.facilities, CELLS).bool()
        return torch.cat((self.floor[:, None, :CELLS], facilities[:, :5],
                          facilities[:, 5:7].any(1, keepdim=True),
                          facilities[:, 7:9].any(1, keepdim=True)), dim=1)

    def spatial_observation(self, state, players, tasks=None, *, allow_cooking=False):
        actors = self.player_order[players]
        positions = state.position.gather(1, actors)
        directions = state.orientation.gather(1, actors)
        occupancy = one_hot(positions, CELLS).bool()
        orientations = occupancy[:, :, None] & one_hot(directions, 4).bool()[..., None]
        objects = torch.zeros((len(self.layouts), CELLS), dtype=torch.long, device=self.device)
        objects.scatter_(1, self.facilities[:, 7:9], state.buffers)
        objects.scatter_(1, state.position, state.held)
        object_planes = one_hot(objects, CERTIFIED_SOUP + 1)[..., 1:].transpose(1, 2)
        pot_cells = one_hot(self.facilities[:, 5:7], CELLS)
        pot_order = (one_hot(state.pots + 1, 3)[..., 1:, None] * pot_cells[:, :, None]).sum(1)
        pot_time = (pot_cells * state.remaining[..., None]).sum(1, keepdim=True) / self.duration_scale
        pair = (occupancy & state.pair.gather(1, actors)[..., None]).any(1, keepdim=True)
        goal = torch.zeros_like(pair) if tasks is None else self.goal_plane(
            state, players, tasks, allow_cooking=allow_cooking,
        )[:, None]
        planes = torch.cat((occupancy, orientations.flatten(1, 2), self.terrain_planes,
                            object_planes, pot_order, pot_time, pair, goal), dim=1)
        planes = torch.nn.functional.pad(planes, (0, 1))
        crop = self.crop_indices[positions[:, 0], directions[:, 0]]
        view = planes.gather(2, crop[:, None].expand(-1, planes.shape[1], -1))
        orders, clocks = self.order_observation(state)
        return {"visual_obs": view.reshape(-1, planes.shape[1], 7, 7),
                "order_obs": torch.cat((orders.flatten(1), clocks), dim=1)}

    def skill_bonus(self, state, players, tasks, counter_bonuses):
        position = state.position[self.rows, players]
        orientation = state.orientation[self.rows, players]
        taking = tasks == BUFFER_TAKE
        placing = tasks == BUFFER_PUT
        target = self.front[position, orientation]
        buffer_index = (target == self.facilities[:, -1]).long()
        item = torch.where(taking, state.buffers[self.rows, buffer_index],
                           state.held[self.rows, players])
        soup = (item >= TOMATO_SOUP).long()
        transfer = placing | taking
        bonus = counter_bonuses[self.rows, position, orientation, soup, taking.long()]
        return (tasks == SERVE) + transfer * bonus

    @staticmethod
    def _coords(positions):
        return torch.stack(((positions % WIDTH).float() / (WIDTH - 1),
                            (positions // WIDTH).float() / (HEIGHT - 1)), -1)

    def draws(self, generator):
        c = self.config
        shape = (len(self.layouts),)
        delta = torch.randint(-c.deadline_jitter, c.deadline_jitter + 1, shape,
                              device=self.device, generator=generator)
        durations = torch.randint(-c.duration_jitter, c.duration_jitter + 1, (*shape, 3),
                                  device=self.device, generator=generator)
        return torch.cat((delta[:, None], durations), 1)

    def sample_seats(self, generator):
        indices = torch.randint(6, self.rows.shape, device=self.device, generator=generator)
        return self.spawn_permutations[indices]

    def reset(self, generator):
        return self._initial_state(self.sample_seats(generator), self.draws(generator))

    def reset_where(self, state, mask, seats, draws):
        return select_state(mask, self._initial_state(seats, draws), state)

    def _initial_state(self, seats, draws):
        position = self.starts.gather(1, seats)
        zeros = torch.zeros_like(self.rows)
        pair = torch.zeros_like(position, dtype=torch.bool)
        two = zeros[:, None].expand(-1, 2)
        state = MixedState(position, torch.zeros_like(position), torch.zeros_like(position),
                           two, two - 1, two, two.bool(), two, two, zeros, two, zeros,
                           zeros.bool(), pair, zeros, zeros, zeros)
        return self._spawn(state, draws)

    def _spawn(self, state, draws):
        c = self.config
        delta = draws[:, 0] + state.tick
        due = torch.stack((delta + c.normal_deadline, delta + c.joint_deadline), 1)
        return update_state(state,
            batch=state.batch + 1, due=due, sync_due=delta + c.sync_deadline,
            cook=draws[:, 1:3] + c.cook_steps, sync_steps=draws[:, 3] + c.sync_steps,
            stock=torch.ones_like(state.stock), status=torch.zeros_like(state.status),
            certified=torch.zeros_like(state.certified), pair=torch.zeros_like(state.pair),
            progress=torch.zeros_like(state.progress),
        )

    def _joint(self, state, targets, pressed, tick):
        held = state.held
        pair = pressed & (targets == self.facilities[:, HANDOFF_TILE, None])
        donor = pair & (held == ONION_SOUP)
        receiver = pair & (held == EMPTY)
        valid = (pair.sum(1) == 2) & donor.any(1) & receiver.any(1) & (
            (state.status[:, 1] == ACTIVE) & (tick <= state.sync_due)
        )
        continuing = valid & (pair == state.pair).all(1)
        progress = torch.where(valid, torch.where(continuing, state.progress, 0) + 1, 0)
        finished = valid & (progress == state.sync_steps)
        held = torch.where(finished[:, None] & donor, EMPTY, held)
        held = torch.where(finished[:, None] & receiver, CERTIFIED_SOUP, held)
        events = torch.where(finished[:, None] & pair, HANDOFF, -1)
        pending = valid & ~finished
        aborted = (state.progress > 0) & ~continuing
        state = update_state(state, held=held, certified=state.certified | finished,
                             pair=pair & pending[:, None], progress=torch.where(pending, progress, 0))
        return state, events, valid & ~continuing, aborted

    def _ordinary(self, state, targets, pressed, tick, events):
        same = targets[:, :, None] == targets[:, None, :]
        contested = (same & pressed[:, None, :] & ~self.eye).any(2)
        interact = pressed & ~contested
        held = state.held
        stock = []
        # Observation coordinates follow O/T, while order state follows normal/joint.
        sources = ((TOMATO_SOURCE, TOMATO, GET_TOMATO), (ONION_SOURCE, ONION, GET_ONION))
        for slot, (facility, code, task) in enumerate(sources):
            at = interact & (targets == self.facilities[:, facility, None])
            take = at & (held == EMPTY) & state.stock[:, slot, None] & (
                (state.status[:, slot, None] == ACTIVE) & (tick <= state.due[:, slot])[:, None]
            )
            back = at & (held == code)
            held = torch.where(take, code, torch.where(back, EMPTY, held))
            events = torch.where(take, task, torch.where(back, RETURN_ITEM, events))
            stock.append((state.stock[:, slot] & ~take.any(1)) | back.any(1))

        at_dishes = interact & (targets == self.facilities[:, DISH_SOURCE, None])
        take = at_dishes & (held == EMPTY)
        back = at_dishes & (held == DISH)
        held = torch.where(take, DISH, torch.where(back, EMPTY, held))
        events = torch.where(take, GET_DISH, torch.where(back, RETURN_ITEM, events))

        buffers, pots, remaining = [], [], []
        for index in range(2):
            old = state.buffers[:, index]
            at = interact & (targets == self.facilities[:, 7 + index, None])
            put = at & (held != EMPTY) & (old[:, None] == EMPTY)
            take = at & (held == EMPTY) & (old[:, None] != EMPTY)
            buffers.append(torch.where(put.any(1), torch.where(put, held, 0).sum(1),
                                       torch.where(take.any(1), EMPTY, old)))
            held = torch.where(put, EMPTY, torch.where(take, old[:, None], held))
            events = torch.where(put, BUFFER_PUT, torch.where(take, BUFFER_TAKE, events))

            old_pot, old_time = state.pots[:, index], state.remaining[:, index]
            at = interact & (targets == self.facilities[:, 5 + index, None])
            load = at & ((held == TOMATO) | (held == ONION)) & (old_pot[:, None] < 0)
            plate = at & (held == DISH) & (old_pot[:, None] >= 0) & (old_time[:, None] == 0)
            slot = torch.where(load, item_order(held), 0).sum(1)
            pots.append(torch.where(load.any(1), slot, torch.where(plate.any(1), -1, old_pot)))
            cook = state.cook.gather(1, slot[:, None]).squeeze(1)
            remaining.append(torch.where(load.any(1), cook + 1, old_time))
            held = torch.where(load, EMPTY, torch.where(plate, TOMATO_SOUP + old_pot[:, None], held))
            events = torch.where(load, LOAD_POT, torch.where(plate, PLATE_SOUP, events))

        at_serve = interact & (targets == self.facilities[:, SERVING, None])
        delivered = at_serve[:, :, None] & (
            held[:, :, None] == self.serve_items
        ) & (tick[:, None, None] <= state.due[:, None, :])
        events = torch.where(delivered.any(2), SERVE, events)
        status = torch.where(delivered.any(1), COMPLETED, state.status)
        raw = (delivered.any(1) * self.serve_rewards).sum(1)
        state = update_state(state,
            held=torch.where(delivered.any(2), EMPTY, held), buffers=torch.stack(buffers, 1),
            pots=torch.stack(pots, 1), remaining=torch.stack(remaining, 1),
            stock=torch.stack(stock, 1), status=status,
        )
        return state, events, raw, delivered.any(1)

    def _move(self, state, actions):
        moving = actions < 4
        orientation = torch.where(moving, actions, state.orientation)
        target = self.front[state.position, orientation]
        intended = torch.where(moving & self.floor[self.rows[:, None], target], target, state.position)
        overlap = intended[:, :, None] == intended[:, None, :]
        enters = intended[:, :, None] == state.position[:, None, :]
        blocked = ((overlap | (enters & enters.transpose(1, 2))) & ~self.eye).any(2)
        for _ in range(2):
            blocked = blocked | (enters & blocked[:, None, :]).any(2)
        collisions = (blocked & (intended != state.position)).sum(1) if self.track_metrics else None
        return (update_state(state, position=torch.where(blocked, state.position, intended),
                             orientation=orientation), collisions)

    def _expire(self, state):
        late_joint = ~state.certified & (state.tick > state.sync_due)
        late = (state.tick[:, None] > state.due) | torch.stack((torch.zeros_like(late_joint), late_joint), 1)
        expired = (state.status == ACTIVE) & late

        def purge(items):
            return torch.where((items >= TOMATO) & expired.gather(1, item_order(items)), EMPTY, items)

        clear_pot = (state.pots >= 0) & expired.gather(1, state.pots.clamp_min(0))
        return update_state(state,
            held=purge(state.held), buffers=purge(state.buffers),
            pots=torch.where(clear_pot, -1, state.pots),
            remaining=torch.where(clear_pot, 0, state.remaining),
            stock=state.stock & ~expired, status=torch.where(expired, EXPIRED, state.status),
            pair=state.pair & ~expired[:, 1, None], progress=torch.where(expired[:, 1], 0, state.progress),
        ), expired

    def potential(self, state):
        items = torch.cat((state.held, state.buffers), 1)
        stage = torch.where(items == CERTIFIED_SOUP, 5, torch.where(items >= TOMATO_SOUP, 4, 1))
        item_stage = torch.where((items[:, :, None] >= TOMATO) &
                                 (item_order(items)[:, :, None] == self.slots), stage[:, :, None], 0).amax(1)
        pot_stage = torch.where(state.pots[:, :, None] == self.slots,
                                torch.where(state.remaining[:, :, None] > 0, 2, 3), 0).amax(1)
        stages = torch.maximum(item_stage, pot_stage) * (state.status == ACTIVE)
        return stages.sum(1).double() + state.progress.double() / state.sync_steps

    def step(self, state, actions, draws):
        c = self.config
        targets = self.front[state.position, state.orientation]
        pressed = actions == INTERACT
        tick = state.tick + 1
        before = self.potential(state) if c.shaping_scale else None
        old_pair = state.pair
        state, events, started, aborted = self._joint(state, targets, pressed, tick)
        progressing = state.pair
        state, events, raw, delivered = self._ordinary(state, targets, pressed, tick, events)
        state, collisions = self._move(state, actions)
        state = update_state(state, tick=tick, remaining=(state.remaining - 1).clamp_min(0))
        state, expired = self._expire(state)
        resolved = (state.status != ACTIVE).all(1)
        batch_completed = (state.status == COMPLETED).all(1)
        raw = raw + expired.sum(1) * c.expiry_penalty + batch_completed * c.batch_bonus
        close = tick >= c.horizon
        spawn = resolved & ~close & (state.batch < c.max_batches)
        done = close | (resolved & ~spawn)
        state = select_state(spawn, self._spawn(state, draws), state)
        reward = raw
        if c.shaping_scale:
            next_potential = torch.where(done, 0., self.potential(state))
            reward = raw + c.shaping_scale * (c.shaping_gamma * next_potential - before)
        metrics = None
        if self.track_metrics:
            metrics = torch.cat((delivered, torch.stack((started, aborted, batch_completed), 1),
                                 expired, collisions[:, None]), 1)
        return MixedStep(state, reward.float(), raw.float(), done, events,
                         progressing & state.pair, old_pair & aborted[:, None], metrics)

    def observations(self, state):
        def encode_item(item):
            return torch.cat((one_hot(self.categories[item], 6), (item == CERTIFIED_SOUP)[..., None]), -1)

        chefs = torch.cat((torch.ones_like(state.position)[..., None], self._coords(state.position),
                           one_hot(state.orientation, 4), encode_item(state.held)), -1)
        chefs = chefs[:, self.player_order].flatten(2)
        pots = torch.cat((one_hot(state.pots + 1, 3), state.remaining[..., None] / self.duration_scale), -1)
        orders, clocks = self.order_observation(state)
        shared = torch.cat((pots.flatten(1), encode_item(state.buffers).flatten(1), orders.flatten(1), clocks), 1)
        return torch.cat((
            self.map_obs[:, None].expand(-1, 3, -1), chefs,
            shared[:, None].expand(-1, 3, -1), state.pair[:, self.player_order],
        ), -1)

    def order_observation(self, state):
        sync_due = torch.stack((torch.zeros_like(state.sync_due), state.sync_due), 1)
        sync_steps = torch.stack((torch.zeros_like(state.sync_steps), state.sync_steps), 1)
        certified = torch.stack((torch.zeros_like(state.certified), state.certified), 1)
        orders = torch.cat((
            self.order_kinds, one_hot(state.status, 3),
            torch.stack((state.stock, (state.due - state.tick[:, None]).clamp_min(0) / self.time_scale,
                         (sync_due - state.tick[:, None]).clamp_min(0) / self.time_scale,
                         state.cook / self.duration_scale, sync_steps / self.duration_scale, certified), -1),
        ), -1)
        clocks = torch.stack(((self.config.horizon - state.tick).clamp_min(0) / self.config.horizon,
                              state.batch / self.config.max_batches, state.progress / state.sync_steps), 1)
        return orders, clocks

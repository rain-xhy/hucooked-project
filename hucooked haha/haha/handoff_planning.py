"""Batched preparation actions and arrival estimates for MixedKitchen."""

from collections import deque
from functools import lru_cache
from importlib import import_module

import torch

from .mixed_kitchen import ACTIVE, DIRECTIONS, HANDOFF_TILE, cell
from .mixed_tasks import (
    BUFFER_TAKE, CELLS, CERTIFIED_SOUP, DISH, EMPTY, GET_DISH, GET_ONION, HANDOFF, INTERACT,
    LOAD_POT, ONION, ONION_SOUP, PLATE_SOUP, STAY, UNKNOWN, mixed_source,
)

UNREACHABLE = CELLS * len(DIRECTIONS) + 1
HANDOFF_ROUTES = slice(6, 8)
DELIVERY_ROUTE = 8
HANDOFF_PATIENCE = 6
HANDOFF_RETRY_DELAY = 12


@lru_cache(maxsize=None)
def route_tables(layout):
    """Shortest movement/turn distances to facilities and each handoff stance."""
    source = mixed_source()
    _, facilities, floor = import_module(f"{source.__name__}.layouts").load_layout(layout)
    predecessors = [[] for _ in range(CELLS * 4)]
    targets = facilities["O"] + facilities["D"] + facilities["P"] + facilities["B"]
    station = facilities["H"][0]
    serving = facilities["S"][0]
    interaction_poses = {target: [] for target in (*targets, station, serving)}
    for x, y in sorted(floor):
        start = cell((x, y)) * 4
        for action, (dx, dy) in enumerate(DIRECTIONS):
            front = (x + dx, y + dy)
            moved = front if front in floor else (x, y)
            predecessors[cell(moved) * 4 + action].extend(range(start, start + 4))
            if front in interaction_poses:
                interaction_poses[front].append(start + action)
    stances = interaction_poses[station]
    if len(stances) != 2:
        raise ValueError("Handoff preparation requires two handoff stances")
    groups = [interaction_poses[target] for target in targets] + [[pose] for pose in stances]
    groups.append(interaction_poses[serving])
    distances = []
    for goals in groups:
        route = [UNREACHABLE] * len(predecessors)
        for pose in goals:
            route[pose] = 0
        frontier = deque(goals)
        while frontier:
            pose = frontier.popleft()
            for previous in predecessors[pose]:
                if route[previous] == UNREACHABLE:
                    route[previous] = route[pose] + 1
                    frontier.append(previous)
        distances.append(route)
    return distances, [pose // 4 for pose in stances]


class HandoffPreparation:
    def __init__(self, kitchen, *, include_eta, receive_early=False):
        self.kitchen = kitchen
        self.include_eta = include_eta
        self.receive_early = receive_early
        layouts = tuple(dict.fromkeys(kitchen.layouts))
        distances, stances = zip(*(route_tables(name) for name in layouts))
        self.layout_ids = kitchen._tensor([layouts.index(name) for name in kitchen.layouts])
        self.distances = kitchen._tensor(distances).transpose(1, 2).contiguous()
        self.stances = kitchen._tensor(stances)[self.layout_ids]
        costs = self.distances[:, :, HANDOFF_ROUTES]
        stance_poses = costs.argmin(1)
        self.serving_stance = self.distances[:, :, DELIVERY_ROUTE].gather(1, stance_poses).argmin(1)[self.layout_ids]
        self.directions = torch.arange(4, device=kitchen.device)

    def _teammates(self, players):
        return self.kitchen.player_order[players, 1:]

    @staticmethod
    def _compatible(held, teammate_items):
        return ((held[:, None] == EMPTY) & (teammate_items == ONION_SOUP)) | (
            (held[:, None] == ONION_SOUP) & (teammate_items == EMPTY)
        )

    def partner_ready(self, state, players):
        kitchen = self.kitchen
        teammates = self._teammates(players)
        positions = state.position.gather(1, teammates)
        facing = kitchen.front[positions, state.orientation.gather(1, teammates)]
        compatible = self._compatible(state.held[kitchen.rows, players], state.held.gather(1, teammates))
        return compatible & (facing == kitchen.facilities[:, HANDOFF_TILE, None])

    def available_after_backoff(self, state, players, retry_at, ready_at_backoff):
        rows = self.kitchen.rows
        newly_ready = (self.partner_ready(state, players) & ~ready_at_backoff).any(1)
        in_pair = state.pair[rows, players]
        pose = state.position[rows, players] * 4 + state.orientation[rows, players]
        arrival = self.distances[self.layout_ids, pose, HANDOFF_ROUTES].amin(1)
        finish = state.tick + arrival + state.sync_steps - torch.where(in_pair, state.progress, 0)
        feasible = finish <= state.sync_due.clamp_max(self.kitchen.config.horizon)
        return feasible & ((state.tick >= retry_at) | newly_ready | in_pair)

    def _task(self, state, players):
        held = state.held[self.kitchen.rows, players]
        empty = held == EMPTY
        free_pot = (state.pots < 0).any(1)
        joint_pot = (state.pots == 1).any(1)
        receiving = empty & (state.held == ONION_SOUP).any(1)
        if self.receive_early:
            teammate_items = state.held.gather(1, self._teammates(players))
            potential_donor = ((teammate_items == ONION) & free_pot[:, None]) | (
                (teammate_items == DISH) & joint_pot[:, None]
            )
            receiving = receiving | (empty & potential_donor.any(1))
        taking = empty & (state.buffers == ONION_SOUP).any(1)
        task = torch.full_like(players, UNKNOWN)
        task = torch.where(empty & state.stock[:, 1] & free_pot, GET_ONION, task)
        task = torch.where((held == ONION) & free_pot, LOAD_POT, task)
        task = torch.where(empty & joint_pot, GET_DISH, task)
        task = torch.where((held == DISH) & joint_pot, PLATE_SOUP, task)
        task = torch.where(taking, BUFFER_TAKE, task)
        return torch.where((held == ONION_SOUP) | receiving, HANDOFF, task)

    def observe(self, state, players):
        kitchen = self.kitchen
        open_order = (state.status[:, 1] == ACTIVE) & ~state.certified
        time_left = (state.sync_due.clamp_max(kitchen.config.horizon) - state.tick).clamp_min(0)
        time_left = torch.where(open_order, time_left, 0)
        in_pair = state.pair[kitchen.rows, players]
        required = state.sync_steps - torch.where(in_pair, state.progress, 0)
        available = (time_left >= required) & (self._task(state, players) != UNKNOWN)
        if not self.include_eta:
            return available, None

        teammates = self._teammates(players)
        held = state.held[kitchen.rows, players]
        compatible = self._compatible(held, state.held.gather(1, teammates))
        pose = state.position * 4 + state.orientation
        distance = self.distances[self.layout_ids[:, None], pose, HANDOFF_ROUTES]
        own = distance[kitchen.rows, players]
        others = distance[kitchen.rows[:, None], teammates]
        # Check both assignments to the two opposite sides. Arrival distances
        # are optimistic and exclude collisions, waiting and teammate intent.
        arrival = torch.maximum(own[:, None], others.flip(-1)).amin(-1)
        same_pair = in_pair[:, None] & state.pair.gather(1, teammates)
        progress = torch.where(same_pair, state.progress[:, None], 0)
        finish = arrival + state.sync_steps[:, None] - progress
        estimate = torch.where((time_left > 0)[:, None] & compatible, finish, kitchen.time_scale)
        eta = torch.cat((estimate, time_left[:, None]), dim=1).float()
        return available, (eta / kitchen.time_scale).clamp(0., 1.)

    def action(self, state, players):
        kitchen = self.kitchen
        rows = kitchen.rows
        teammates = self._teammates(players)
        their_positions = state.position.gather(1, teammates)
        position = state.position[rows, players]
        orientation = state.orientation[rows, players]
        task = self._task(state, players)
        handoff = task == HANDOFF
        pot = (task == LOAD_POT)[:, None] & (state.pots < 0)
        plate = (task == PLATE_SOUP)[:, None] & (state.pots == 1)
        buffer = (task == BUFFER_TAKE)[:, None] & (state.buffers == ONION_SOUP)
        occupied = (self.stances[:, :, None] == their_positions[:, None, :]).any(-1)
        stance = handoff[:, None] & ~occupied
        if self.receive_early:
            receiver = state.held[rows, players] == EMPTY
            preferred = torch.nn.functional.one_hot(self.serving_stance, 2).bool()
            open_preferred = (stance & preferred).any(1)
            stance = stance & (~(receiver & open_preferred)[:, None] | preferred)
        fixed = torch.stack((task == GET_ONION, task == GET_DISH), dim=1)
        goals = torch.cat((fixed, pot | plate, buffer, stance, torch.zeros_like(handoff[:, None])), dim=1)
        current, distance, move = self._route(position, orientation, their_positions, goals)
        ready_pot = (plate & (state.remaining == 0) & (current[:, 2:4] == 0)).any(1)
        ready = torch.where(handoff, self.partner_ready(state, players).any(1), (task != PLATE_SOUP) | ready_pot)
        return torch.where((distance == 0) & ready, INTERACT, move), task

    def _route(self, position, orientation, their_positions, goals):
        kitchen = self.kitchen
        rows = kitchen.rows
        current = self.distances[self.layout_ids, position * 4 + orientation]
        distance = current.masked_fill(~goals, UNREACHABLE).amin(1)
        front = kitchen.front[position]
        walkable = kitchen.floor[rows[:, None], front]
        blocked = walkable & (front[:, :, None] == their_positions[:, None, :]).any(-1)
        next_position = torch.where(walkable, front, position[:, None])
        after_move = self.distances[self.layout_ids[:, None], next_position * 4 + self.directions]
        costs = after_move.masked_fill(~goals.unsqueeze(-2), UNREACHABLE).amin(-1)
        cost, move = costs.masked_fill(blocked, UNREACHABLE).min(1)
        return current, distance, torch.where(cost < distance, move, STAY)

    def serve_distance(self, state):
        pose = state.position * 4 + state.orientation
        return self.distances[self.layout_ids[:, None], pose, DELIVERY_ROUTE] + 1

    def serve_certified(self, state, players):
        rows = self.kitchen.rows
        targets = self.directions.new_zeros(self.distances.shape[-1], dtype=torch.bool)
        targets[DELIVERY_ROUTE] = True
        _, distance, move = self._route(state.position[rows, players], state.orientation[rows, players],
                                        state.position.gather(1, self._teammates(players)), targets)
        action = torch.where(distance == 0, INTERACT, move)
        available = (state.held[rows, players] == CERTIFIED_SOUP) & (action != STAY) & (
            state.tick + distance + 1 <= state.due[:, 1].clamp_max(self.kitchen.config.horizon)
        )
        return action, available

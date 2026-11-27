"""Subtask navigation and receding-horizon role allocation for MixedKitchen."""

from collections import deque
from copy import copy
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from overcooked_ai_py.cooperative.kitchen import DIRECTIONS, INTERACT, STAY, adjacent

from haha.routing import advancing_moves


ASSIGNMENTS = tuple(permutations(range(3)))
GET_ONION, GET_TOMATO, GET_PLATE, LEFT_POT, RIGHT_POT, DONATE, RECEIVE, SERVE, PUT_PLATE, TAKE_PLATE, PARK_JOINT, PARK_NORMAL, PARK_HELPER = range(13)
GOAL_COUNT = 13


@dataclass(frozen=True)
class Goal:
    subtask: int
    cell: tuple[int, int]
    facing: tuple[int, int]
    interact: bool = True


def goals(env, assignment):
    state = env.state
    normal, joint = state.orders
    joint_chef, normal_chef, helper = assignment
    left, right = env.pot_positions
    donate, receive = env.sync_positions
    meetup = (receive, (-1, 0))
    rendezvous = (donate, (1, 0))

    def station(subtask, coordinate, facing, interact=True):
        cell = adjacent(coordinate, (-facing[0], -facing[1]))
        return Goal(subtask, cell, facing, interact)

    dishes = station(GET_PLATE, env.locations["D"][0], (-1, 0))
    service = station(SERVE, env.locations["S"][0], (1, 0))
    old_pair = (state.chefs[joint_chef], state.chefs[helper])
    sync_ready = (
        old_pair[0].held is not None and old_pair[0].held.kind == "soup"
        and not old_pair[0].held.certified and old_pair[1].held is None
        and (old_pair[0].position, old_pair[0].orientation) == rendezvous
        and (old_pair[1].position, old_pair[1].orientation) == meetup
    )
    planned = []
    for role, i, ticket, pot_coord in (
        (0, joint_chef, joint, left), (1, normal_chef, normal, right)
    ):
        held = state.chefs[i].held
        held_order = env.order(held.order_id) if held else None
        if held and held.kind == "soup":
            if held_order.kind == "normal" or held.certified:
                goal = service
            else:
                goal = Goal(DONATE, *rendezvous, sync_ready)
        elif held and held.kind == "ingredient":
            both = held_order.kind == "joint"
            goal = station(LEFT_POT if both else RIGHT_POT, left if both else right, (0, -1))
        elif ticket.status == "active" and ticket.stock:
            source = "O" if role == 0 else "T"
            face = (-1, 0) if role == 0 else (1, 0)
            goal = dishes if held else station(GET_ONION if role == 0 else GET_TOMATO,
                                                env.locations[source][0], face)
        elif ticket.status == "active" and any(p.order_id == ticket.id for p in state.pots):
            pot = next(p for p in state.pots if p.order_id == ticket.id)
            if held and held.kind == "dish":
                goal = station(LEFT_POT if role == 0 else RIGHT_POT, pot_coord,
                               (0, -1), pot.remaining == 0)
            elif role == 1:
                goal = station(TAKE_PLATE, env.buffer_positions[0], (-1, 0), state.buffers[0] is not None)
            else:
                goal = dishes
        else:
            # Reserve the ingredient approach while the remaining dish is
            # delivered. Inventory stays untouched until the next ticket exists.
            source, facing = ("O", (-1, 0)) if role == 0 else ("T", (1, 0))
            subtask = PARK_JOINT if role == 0 else PARK_NORMAL
            goal = station(subtask, env.locations[source][0], facing, False)
        planned.append(goal)

    chef = state.chefs[helper]
    if chef.held and chef.held.kind == "soup":
        support = service
    elif chef.held and chef.held.kind == "dish":
        support = station(PUT_PLATE, env.buffer_positions[0], (1, 0), state.buffers[0] is None)
    else:
        normal_held = state.chefs[normal_chef].held
        normal_plated = normal_held is not None and normal_held.kind in ("dish", "soup")
        if normal.status == "active" and not normal_plated and state.buffers[0] is None:
            support = dishes
        elif joint.status == "active" and not joint.certified:
            support = Goal(RECEIVE, *meetup, sync_ready)
        else:
            support = Goal(PARK_HELPER, (7, 5), (0, 1), False)
    planned.append(support)
    if state.joint.pair:
        for role, i in enumerate(assignment):
            if i in state.joint.pair:
                if i == state.joint.donor:
                    planned[role] = Goal(DONATE, *rendezvous)
                else:
                    planned[role] = Goal(RECEIVE, *meetup)
    return tuple(planned)


def navigation(env, chef_id, goal, reserved=()):
    chef = env.state.chefs[chef_id]
    if chef.position == goal.cell:
        if chef.orientation == goal.facing:
            return INTERACT if goal.interact else STAY
        if adjacent(goal.cell, goal.facing) in env.floor:
            return STAY
        return DIRECTIONS.index(goal.facing)
    occupied = {c.position for i, c in enumerate(env.state.chefs) if i != chef_id}
    occupied.update(reserved)
    queue = deque([(chef.position, None)])
    visited = {chef.position}
    while queue:
        cell, initial = queue.popleft()
        for action, direction in enumerate(DIRECTIONS):
            neighbor = adjacent(cell, direction)
            if neighbor not in env.floor or neighbor in occupied or neighbor in visited:
                continue
            first = action if initial is None else initial
            if neighbor == goal.cell:
                return first
            queue.append((neighbor, first))
            visited.add(neighbor)
    return STAY


def reserve(env, chef, action, reserved):
    if action < STAY:
        target = adjacent(env.state.chefs[chef].position, DIRECTIONS[action])
        if target in env.floor:
            reserved.add(target)


def actions_for(env, assignment, plans):
    actions = [STAY] * 3
    reserved = set()
    # The cook nearest a deadline already has a frozen joint goal in plans.
    for role in (2, 1, 0):
        chef = assignment[role]
        action = navigation(env, chef, plans[role], reserved)
        actions[chef] = action
        reserve(env, chef, action, reserved)
    return actions


def default_assignment(env):
    slots = ((1, 1), (9, 1), (1, 5))
    return min(ASSIGNMENTS, key=lambda team: sum(
        abs(env.state.chefs[i].position[0] - target[0]) + abs(env.state.chefs[i].position[1] - target[1])
        for i, target in zip(team, slots)
    ))


def project_batch(env, team, limit):
    simulation = copy(env)
    # Step copies the initial state. The planning horizon ends at the batch
    # boundary, so no future order or transition RNG is read from the live game.
    simulation.rng = np.random.default_rng(0)
    initial = env.state
    for _ in range(limit):
        _, _, terminated, truncated, _ = simulation.step(actions_for(simulation, team, goals(simulation, team)))
        if terminated or truncated or simulation.state.batch != initial.batch:
            break
    state = simulation.state
    completed = state.metrics["normal_completed"] + state.metrics["joint_completed"]
    new_bonus = state.metrics["batches_completed"] - initial.metrics["batches_completed"]
    return state.score - initial.score, new_bonus, completed, -(state.tick - initial.tick)


class ScheduledTeam:
    def __init__(self, planning=False):
        self.batch = None
        self.team = None
        self.planning = planning

    def assign(self, env):
        if self.batch == env.state.batch:
            return self.team
        self.batch = env.state.batch
        greedy = default_assignment(env)
        if self.planning:
            limit = min(62, env.config.horizon - env.state.tick)
            self.team = max(ASSIGNMENTS, key=lambda team: (project_batch(env, team, limit), team == greedy))
        else:
            self.team = greedy
        return self.team

    def __call__(self, env):
        team = self.assign(env)
        return actions_for(env, team, goals(env, team))


def choices(env, assignment, role, preferred):
    state = env.state
    chef = state.chefs[assignment[role]]
    choices_by_id = {
        preferred.subtask: preferred,
        GOAL_COUNT: Goal(GOAL_COUNT, chef.position, chef.orientation, False),
    }
    if assignment[role] in state.joint.pair:
        return {preferred.subtask: preferred}
    if role == 1 and preferred.subtask == TAKE_PLATE and state.buffers[0] is None:
        target = env.locations["D"][0]
        choices_by_id[GET_PLATE] = Goal(GET_PLATE, adjacent(target, (1, 0)), (-1, 0))
    if role == 2 and preferred.subtask == GET_PLATE and chef.held is None:
        joint = state.orders[1]
        if joint.status == "active" and not joint.certified:
            choices_by_id[RECEIVE] = Goal(RECEIVE, env.sync_positions[1], (-1, 0), False)
    return choices_by_id


def worker_mask(env, chef_id, goal, reserved, *, progress=False):
    chef = env.state.chefs[chef_id]
    mask = np.zeros(6, dtype=bool)
    if chef.position == goal.cell:
        action = navigation(env, chef_id, goal)
        mask[action] = True
    else:
        blocked = {c.position for i, c in enumerate(env.state.chefs) if i != chef_id}
        blocked.update(reserved)
        neighbors = [adjacent(chef.position, direction) for direction in DIRECTIONS]
        if not progress:
            mask[:STAY] = [cell in env.floor and cell not in blocked for cell in neighbors]
        else:
            mask[:STAY] = advancing_moves(env.floor, blocked, chef.position, goal.cell, DIRECTIONS)
        mask[STAY] = not progress or not mask.any()
    return mask

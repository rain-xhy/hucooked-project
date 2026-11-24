"""Subtask selection and execution for the shared three-chef kitchen."""

import numpy as np
import torch
from torch import nn

from overcooked_ai_py.cooperative.shared.controller import ScriptedController, navigation_action
from overcooked_ai_py.cooperative.shared.kitchen import DIRECTIONS, INTERACT, STAY
from overcooked_ai_py.cooperative.shared.layouts import FLOOR, STATIONS

from haha.goal_training import mlp
from haha.routing import advancing_moves


# Each goal names a work position and, when needed, the station to face.
GOALS = (
    ((1, 3), "S"),
    ((2, 3), "A"),
    ((2, 2), "G"),
    ((1, 5), "D"),
    ((2, 3), None),
    ((4, 2), "G"),
    ((4, 1), "M"),
    ((4, 2), None),
    ((5, 2), "K"),
    ((4, 3), "A"),
    ((5, 3), None),
    ((5, 1), "L"),
)
SERVE, ASSEMBLE, UNLOAD_WEST, GET_PLATE, WAIT_WEST, USE_COOKER, GET_MAIN, WAIT_EAST, CUT, DROP_SIDE, WAIT_SIDE, GET_SIDE = range(len(GOALS))
IDENTITY = np.eye(3, dtype=np.float32)
GOAL_FEATURES = np.eye(len(GOALS), dtype=np.float32)
ITEMS = (None, "raw_main", "raw_side", "chopped_side", "cooked_main", "plate", "meal")
TILES = {tile: index for index, tile in enumerate(sorted(FLOOR))}
TILE_FEATURES = np.eye(len(TILES), dtype=np.float32)
DIRECTION_FEATURES = np.eye(len(DIRECTIONS), dtype=np.float32)
ITEM_FEATURES = np.eye(len(ITEMS), dtype=np.float32)
FEATURE_DIM = 3 * (len(TILES) + len(DIRECTIONS) + len(ITEMS)) + 13


def features(env, controller):
    state = env.state
    encoded = []
    for index in (controller.server, controller.main, controller.side):
        chef = state.chefs[index]
        encoded.extend(TILE_FEATURES[TILES[chef.position]])
        encoded.extend(DIRECTION_FEATURES[DIRECTIONS.index(chef.orientation)])
        encoded.extend(ITEM_FEATURES[ITEMS.index(chef.held.kind if chef.held else None)])
    table, cooker = state.assembly, state.cooker
    order = env.order(cooker.item.order_id) if cooker.item else None
    work = next((ticket for ticket in state.orders if ticket.status == "active"), None)
    remaining = sum(ticket.status != "served" for ticket in state.orders)
    encoded.extend((
        cooker.item is not None,
        order is not None and order.kind == "joint",
        cooker.remaining / env.config.cook_steps,
        state.cutting.item is not None,
        state.cutting.progress / env.config.chop_steps,
        table.main is not None,
        table.side is not None,
        table.plate is not None,
        table.plate is not None and table.plate.kind == "meal",
        work is not None,
        work is not None and work.kind == "joint",
        remaining / len(state.orders),
        state.tick / env.config.horizon,
    ))
    return np.asarray(encoded, dtype=np.float32)


def actor_inputs(common):
    shared = np.broadcast_to(common[:, None, :], (len(common), 3, FEATURE_DIM))
    roles = np.broadcast_to(IDENTITY, (len(common), 3, 3))
    return np.concatenate((shared, roles), axis=-1)


def targets(env, controller, early=0, *, pipeline=False, lock_commitment=False, queue_prefetch=False):
    state = env.state
    cooker, table, cutting = state.cooker, state.assembly, state.cutting
    server, main, side = (state.chefs[i] for i in (controller.server, controller.main, controller.side))
    joint = (
        cooker.item is not None and cooker.remaining <= early
        and env.order(cooker.item.order_id).kind == "joint" and table.main is None
    )
    if server.held is not None:
        west_goal = SERVE if server.held.kind == "meal" else ASSEMBLE
    elif table.plate is not None and table.plate.kind == "meal":
        west_goal = ASSEMBLE
    elif table.main is not None and table.side is not None and table.plate is not None:
        west_goal = ASSEMBLE
    elif joint:
        west_goal = UNLOAD_WEST
    else:
        west_goal = GET_PLATE if table.plate is None else WAIT_WEST

    if main.held is not None:
        east_goal = USE_COOKER if cooker.item is None else GET_MAIN
    elif cooker.item is None and any(order.status == "active" for order in state.orders):
        east_goal = GET_MAIN
    elif cooker.item is not None and cooker.remaining <= early and table.main is None:
        east_goal = USE_COOKER
    else:
        east_goal = WAIT_EAST

    side_statuses = ("active", "started", "queued") if queue_prefetch else ("active", "started")
    needs_side = sum(order.status in side_statuses and order.assembled_at < 0 for order in state.orders)
    queued_side = int(table.side is not None)
    if side.held is not None and side.held.kind == "raw_side":
        side_goal = CUT
    elif side.held is not None and side.held.kind == "chopped_side":
        side_goal = DROP_SIDE if table.side is None else WAIT_SIDE
    elif cutting.item is not None:
        side_goal = CUT
    elif needs_side > queued_side and (pipeline or not queued_side):
        side_goal = GET_SIDE
    else:
        side_goal = WAIT_SIDE
    if lock_commitment and state.metrics.get("joint_run", 0):
        west_goal, east_goal = UNLOAD_WEST, USE_COOKER
    if lock_commitment and cutting.item is not None and cutting.item.kind == "raw_side" and cutting.progress:
        side_goal = CUT
    return np.asarray((west_goal, east_goal, side_goal), dtype=np.int64)


def goal_mask(env, controller, *, lock_commitment=False, queue_prefetch=False):
    """Keep the role-specific tasks reachable, including both early and late arrival."""
    mask = np.zeros((3, len(GOALS)), dtype=bool)
    for early in (0, env.config.cook_steps):
        for pipeline in (False, True):
            mask[np.arange(3), targets(env, controller, early, pipeline=pipeline, lock_commitment=lock_commitment,
                                       queue_prefetch=queue_prefetch)] = True
    table = env.state.assembly
    west = env.state.chefs[controller.server]
    main = env.state.chefs[controller.main]
    side = env.state.chefs[controller.side]
    # During cooking, the west chef can finish staging the next plate or wait
    # on a different side of the station. The policy learns the tradeoff.
    if west.held is None and table.plate is None:
        mask[0, GET_PLATE] = True
    if west.held is None:
        mask[0, WAIT_WEST] = True
    if main.held is None:
        mask[1, WAIT_EAST] = True
    if side.held is None or side.held.kind == "chopped_side":
        mask[2, WAIT_SIDE] = True
    if lock_commitment and env.state.metrics.get("joint_run", 0):
        mask[:2] = False
        mask[0, UNLOAD_WEST] = mask[1, USE_COOKER] = True
    cutting = env.state.cutting
    if lock_commitment and cutting.item is not None and cutting.item.kind == "raw_side" and cutting.progress:
        mask[2] = False
        mask[2, CUT] = True
    return mask


def valid_actions(env, chef_id, goal_id, *, synchronize=False, goal_lock=False,
                  route_progress=False, reserved=()):
    goal, station = GOALS[goal_id]
    chef = env.state.chefs[chef_id]
    mask = np.ones(6, dtype=bool)
    mask[INTERACT] = False
    if chef.position == goal and station is not None:
        target = STATIONS[station]
        facing = (target[0] - goal[0], target[1] - goal[1])
        mask[INTERACT] = chef.orientation == facing
        cooker = env.state.cooker
        if station == "G" and cooker.item is not None:
            if cooker.remaining > 0:
                mask[INTERACT] = False
            elif synchronize and env.order(cooker.item.order_id).kind == "joint":
                other_port = GOALS[USE_COOKER if goal_id == UNLOAD_WEST else UNLOAD_WEST][0]
                partner_facing = (target[0] - other_port[0], target[1] - other_port[1])
                mask[INTERACT] &= any(
                    other.held is None and other.position == other_port and other.orientation == partner_facing
                    for i, other in enumerate(env.state.chefs) if i != chef_id
                )
    if route_progress and chef.position != goal:
        blocked = {c.position for i, c in enumerate(env.state.chefs) if i != chef_id} | set(reserved)
        mask[:STAY] = advancing_moves(FLOOR, blocked, chef.position, goal, DIRECTIONS)
        mask[STAY] = not mask[:STAY].any()
    elif goal_lock:
        if chef.position != goal:
            x, y = chef.position
            mask[:STAY] = [(x + dx, y + dy) in FLOOR for dx, dy in DIRECTIONS]
        else:
            interact = mask[INTERACT]
            mask[:] = False
            if station is None:
                mask[STAY] = True
            elif chef.orientation != facing:
                mask[DIRECTIONS.index(facing)] = True
            else:
                mask[INTERACT if interact else STAY] = True
    return mask


def scripted_action(env, chef, goal, reserved, *, synchronize=False):
    tile, station = GOALS[goal]
    action = navigation_action(env, chef, tile, STATIONS[station] if station else None, reserved)
    return action if valid_actions(env, chef, goal, synchronize=synchronize)[action] else STAY


def ordered_chefs(controller):
    return controller.server, controller.main, controller.side


def reserve(env, chef, action, reserved):
    if action < STAY:
        pos = env.state.chefs[chef].position
        delta = DIRECTIONS[action]
        destination = (pos[0] + delta[0], pos[1] + delta[1])
        if destination in FLOOR:
            reserved.add(destination)


def worker_inputs(obs, goals, reserved):
    return np.concatenate((obs, GOAL_FEATURES[goals], reserved), axis=-1)


def reserved_features(reserved):
    flags = np.zeros(len(TILES), dtype=np.float32)
    for tile in reserved:
        flags[TILES[tile]] = 1.0
    return flags


class SharedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.manager = mlp(FEATURE_DIM + 3, len(GOALS))
        self.worker = mlp(FEATURE_DIM + 3 + len(GOALS) + len(TILES), 6)
        self.critic = mlp(FEATURE_DIM, 1)

    def logits(self, module, obs, mask):
        return module(obs).masked_fill(~mask, -1e9)

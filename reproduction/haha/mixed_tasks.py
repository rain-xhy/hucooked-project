"""Task and observation contract for the three-chef HA² adaptation."""

import importlib.util
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

MIXED_COMMIT = "671c17bf4ba48ecb11e056dbf1e077b0bd434b2d"
MIXED_LAYOUTS = ("compact", "offset", "bottleneck")
OBS_SIZE = 187
WIDTH, HEIGHT = 11, 7
CELLS = WIDTH * HEIGHT
LEGACY_GOAL = "category"
EGOCENTRIC_GOAL = "egocentric"
GOAL_ENCODINGS = (LEGACY_GOAL, EGOCENTRIC_GOAL)
PHYSICAL_TASKS = "physical"
AUTHOR_TASKS = "author"
TASK_RULES = (PHYSICAL_TASKS, AUTHOR_TASKS)
SPATIAL_CHANNELS = 34
ORDER_FEATURES = 25
NAMES = (
    "get_tomato", "get_onion", "load_pot", "get_dish", "plate_soup",
    "buffer_put", "buffer_take", "handoff", "serve", "return_item", "unknown",
)
(GET_TOMATO, GET_ONION, LOAD_POT, GET_DISH, PLATE_SOUP,
 BUFFER_PUT, BUFFER_TAKE, HANDOFF, SERVE, RETURN_ITEM, UNKNOWN) = range(len(NAMES))
CORE_SKILLS = (GET_TOMATO, GET_ONION, LOAD_POT, GET_DISH, PLATE_SOUP, HANDOFF, SERVE)
EMPTY, DISH, TOMATO, ONION, TOMATO_SOUP, ONION_SOUP, CERTIFIED_SOUP = range(7)
STAY, INTERACT = 4, 5


@lru_cache(maxsize=1)
def mixed_source():
    checkout = Path(__file__).resolve().parents[2] / "hucooked"
    package = checkout / "src" / "overcooked_ai_py" / "cooperative"
    if not (package / "__init__.py").is_file():
        raise RuntimeError("Initialize the hucooked submodule before using --environment mixed")
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != MIXED_COMMIT:
        raise RuntimeError(f"MixedKitchen requires hucooked commit {MIXED_COMMIT}; got {head}")
    name = f"{__package__}._hucooked_cooperative"
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mixed_config(layout, horizon):
    return mixed_source().KitchenConfig(layout=layout, horizon=horizon)


def observation_space(manager, worker_goal=LEGACY_GOAL):
    if worker_goal == EGOCENTRIC_GOAL:
        obs = {"visual_obs": spaces.Box(0., 1., (SPATIAL_CHANNELS, 7, 7), dtype=np.float32),
               "order_obs": spaces.Box(0., 1., (ORDER_FEATURES,), dtype=np.float32)}
    else:
        obs = {"visual_obs": spaces.Box(0., 1., (OBS_SIZE + (0 if manager else len(NAMES)),),
                                        dtype=np.float32)}
    if manager:
        obs["subtask_mask"] = spaces.MultiBinary(len(NAMES))
    return spaces.Dict(obs)


def item_order(items):
    return torch.where(items == CERTIFIED_SOUP, 1, (items - TOMATO).remainder(2))


def skill_mask(state, previous=None, handoff_targets=None):
    """Return [kitchen, chef, skill] availability from the physical state."""
    held = state.held
    empty = held == EMPTY
    ingredient = (held == TOMATO) | (held == ONION)
    dish = held == DISH
    active = state.status == 0
    source = active & state.stock & (state.tick[:, None] + 1 <= state.due)
    donor = held == ONION_SOUP
    if handoff_targets is None:
        receiver = empty & donor.any(1, keepdim=True)
    else:
        front, station = handoff_targets
        facing_station = front[state.position, state.orientation] == station[:, None]
        receiver = empty & (donor & facing_station).any(1, keepdim=True)
        donor = donor & (empty & facing_station).any(1, keepdim=True)
    handoff = (donor | receiver) & active[:, 1, None] & (
        state.tick[:, None] + 1 <= state.sync_due[:, None]
    )
    load = ingredient & (state.pots < 0).any(1, keepdim=True)
    put = ~empty & (state.buffers == EMPTY).any(1, keepdim=True)
    take = empty & (state.buffers != EMPTY).any(1, keepdim=True)
    platable = state.pots >= 0
    if previous is None:
        platable = platable & (state.remaining == 0)
    plate = dish & platable.any(1, keepdim=True)
    returning = ingredient | dish
    if previous is not None:
        # The author's mask prevents immediately undoing a transfer. MixedKitchen
        # also permits returning to a source, so keep that escape only when needed.
        picked_from_source = ((previous == GET_TOMATO) & (held == TOMATO)) | (
            (previous == GET_ONION) & (held == ONION)
        ) | ((previous == GET_DISH) & dish)
        returning = returning & ~(picked_from_source & (load | put | plate))
    legal = (
        empty & source[:, 0, None],
        empty & source[:, 1, None],
        load,
        empty,
        plate,
        put,
        take,
        handoff,
        (held == TOMATO_SOUP) | (held == CERTIFIED_SOUP),
        returning,
        torch.ones_like(empty),
    )
    return torch.stack(legal, dim=-1)


def worker_outcome(actions, tasks, events, continuing, aborted, legal, task_steps, timeout):
    success = events == tasks
    handoff = tasks == HANDOFF
    progress = handoff & continuing
    interacted = actions == INTERACT
    attempt = (interacted & ~progress) | (handoff & aborted)
    invalid = ~legal & ~attempt & ~success
    expired = (task_steps >= timeout) & ~success & ~attempt
    done = success | attempt | invalid | expired
    reward = torch.where(success, 1., torch.where(attempt | expired, -1., -.01))
    return reward, done, success, invalid, expired

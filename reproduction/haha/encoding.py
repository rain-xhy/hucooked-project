"""Encoding adapted from StephAO/HAHA a206f11 (MIT; see LICENSE.upstream)."""

import fcntl
from pathlib import Path

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import Direction
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.data.planners import PLANNERS_DIR


def make_planner(mdp):
    counters = mdp.get_counter_locations()
    params = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": counters,
        "counter_drop": counters,
        "counter_pickup": counters,
        "same_motion_goals": True,
    }
    # Spawned environments may create the same upstream planner cache concurrently.
    with (Path(PLANNERS_DIR) / f".{mdp.layout_name}.haha.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return MediumLevelActionManager.from_pickle_or_compute(mdp, params, force_compute=False)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def bc_features(mdp, state, planner, player):
    # Request both players explicitly: indexing is unambiguous across fork versions.
    features = mdp.featurize_state(state, planner, num_pots=2)
    return np.asarray(features[player], dtype=np.float32)


def egocentric(mdp, state, player, horizon, goal=None):
    try:
        encoded = mdp.lossless_state_encoding(
            state, goal_objects=goal, horizon=horizon, p_idx=player
        )
    except TypeError as exc:
        raise RuntimeError(
            "HA² needs StephAO/overcooked_ai's goal_objects/p_idx encoding API; "
            "install the pinned environment from this algorithm's requirements."
        ) from exc
    grid = np.asarray(encoded, dtype=np.float32).transpose(2, 0, 1)
    if grid.shape[0] != 27:
        raise ValueError(f"Expected 27 encoding channels, got {grid.shape}")
    padded = np.pad(grid, ((0, 0), (3, 3), (3, 3)))
    x, y = state.players[player].position
    view = padded[:, x:x + 7, y:y + 7]
    turns = {
        Direction.SOUTH: 0, Direction.NORTH: 2,
        Direction.EAST: -1, Direction.WEST: 1,
    }
    view = np.rot90(view, k=turns[state.players[player].orientation], axes=(1, 2))
    if view.shape != (27, 7, 7):
        raise ValueError(f"Unexpected egocentric observation: {view.shape}")
    return np.ascontiguousarray(view, dtype=np.float32)

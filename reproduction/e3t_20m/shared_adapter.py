"""E3T station choices executed one tick at a time in native SharedKitchen."""

from collections import deque
from functools import lru_cache

import numpy as np

from .vendor.shared import SharedKitchen
from .vendor.shared.kitchen import DIRECTIONS, INTERACT, STAY
from .vendor.shared.layouts import (
    ASSEMBLY_PORTS,
    COOKER_PORTS,
    FLOOR,
    STATIONS,
)

ENGINE_COMMIT = "de37e0818c2e76311e105ab8689e5162bd84133f"
ENGINE_FINGERPRINT = "675c139d2478d5ea790931989baba22e0bc36758a037a4a5f482caadbd4ccbc0"


def adjacent_floor(station):
    x, y = station
    return frozenset((x + dx, y + dy) for dx, dy in DIRECTIONS if (x + dx, y + dy) in FLOOR)


# Separate east/west choices let the neural policy choose where to cooperate.
# Every entry is an interaction location, never an assignment of a chef to a job.
COOK_W, COOK_E = sorted(COOKER_PORTS)
ASSEMBLE_W, ASSEMBLE_E = sorted(ASSEMBLY_PORTS)
TARGETS = {
    "get_main": ("M", adjacent_floor(STATIONS["M"])),
    "get_side": ("L", adjacent_floor(STATIONS["L"])),
    "get_plate": ("D", adjacent_floor(STATIONS["D"])),
    "cooker_west": ("G", frozenset((COOK_W,))),
    "cooker_east": ("G", frozenset((COOK_E,))),
    "cutting": ("K", adjacent_floor(STATIONS["K"])),
    "assemble_west": ("A", frozenset((ASSEMBLE_W,))),
    "assemble_east": ("A", frozenset((ASSEMBLE_E,))),
    "serve": ("S", adjacent_floor(STATIONS["S"])),
}
SERVICE_PORTS = frozenset().union(*(goals for _, goals in TARGETS.values()))
TARGETS["clear_lane"] = (None, FLOOR - SERVICE_PORTS)


@lru_cache(maxsize=65536)
def route(start, orientation, name, blocked):
    """BFS to the chosen port, avoiding the other chefs and reserved destinations."""
    symbol, goals = TARGETS[name]
    if start in goals:
        if symbol is None:
            return STAY
        sx, sy = STATIONS[symbol]
        facing = (sx - start[0], sy - start[1])
        return INTERACT if orientation == facing else DIRECTIONS.index(facing)
    queue = deque([(start, None)])
    seen = set(blocked) | {start}
    while queue:
        pos, first = queue.popleft()
        for action, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in seen or nxt not in FLOOR:
                continue
            initial = action if first is None else first
            if nxt in goals:
                return initial
            seen.add(nxt)
            queue.append((nxt, initial))
    return STAY


class SharedE3TEnv:
    """Native state, default continuous orders and configurable-by-policy task allocation."""

    names = (
        "stay",
        "clear_lane",
        "get_main",
        "get_side",
        "get_plate",
        "cooker_west",
        "cooker_east",
        "cutting",
        "assemble_west",
        "assemble_east",
        "serve",
    )
    action_count = len(names)

    def __init__(self):
        self.core = SharedKitchen()
        if self.core.fingerprint() != ENGINE_FINGERPRINT:
            raise ValueError(f"SharedKitchen source fingerprint mismatches {ENGINE_COMMIT}")

    def reset(self, seed):
        observations, _ = self.core.reset(seed=seed)
        return np.stack(observations), self._masks()

    def _masks(self):
        state = self.core.state
        table, cooker, cutting = state.assembly, state.cooker, state.cutting
        held = tuple(c.held.kind if c.held else None for c in state.chefs)
        positions = tuple(c.position for c in state.chefs)
        pending = any(ticket.status == "active" for ticket in state.orders)
        main_needed = pending and cooker.item is None
        side_needed = cutting.item is None and table.side is None
        plate_needed = table.plate is None
        # Limit redundant source pickup while a chef already carries the needed item.
        # This station-only variant does not use the optional buffer or discard actions.
        get_main = main_needed and "raw_main" not in held
        get_side = side_needed and "raw_side" not in held and "chopped_side" not in held
        get_plate = plate_needed and "plate" not in held
        use_table_empty = table.plate is not None and (
            table.plate.kind == "meal" or (table.main is not None and table.side is not None)
        )
        use_cooker_empty = cooker.item is not None and table.main is None
        masks = []
        for i, item in enumerate(held):
            occupied = set(positions) - {positions[i]}
            empty = item is None
            cook = (empty and use_cooker_empty) or (item == "raw_main" and main_needed)
            cut = (empty and cutting.item is not None) or (
                item == "raw_side" and cutting.item is None
            )
            assemble = (
                (empty and use_table_empty)
                or (item == "chopped_side" and table.side is None)
                or (item == "plate" and plate_needed)
            )
            masks.append(
                [
                    True,
                    positions[i] in SERVICE_PORTS,
                    empty and get_main,
                    empty and get_side,
                    empty and get_plate,
                    cook and COOK_W not in occupied,
                    cook and COOK_E not in occupied,
                    cut,
                    assemble and ASSEMBLE_W not in occupied,
                    assemble and ASSEMBLE_E not in occupied,
                    item == "meal",
                ]
            )
        return np.asarray(masks, dtype=bool)

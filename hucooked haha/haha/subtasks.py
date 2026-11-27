"""Task semantics adapted from StephAO/HAHA a206f11 (MIT).

The masks intentionally retain the original five-layout assumptions. This module
does not claim to support new recipes or the group's future joint-action tasks.
"""

import numpy as np

NAMES = (
    "get_onion_from_dispenser", "get_onion_from_counter", "put_onion_in_pot",
    "put_onion_closer", "get_plate_from_dish_rack", "get_plate_from_counter",
    "put_plate_closer", "get_soup", "get_soup_from_counter", "put_soup_closer",
    "serve_soup", "unknown",
)
UNKNOWN = 11
# The pinned environment checks 'counters', not the upstream task map's 'counter'.
GOALS = (
    "onion_dispenser", "onion", "empty_pot", "counters", "dish_dispenser",
    "dish", "counters", "full_pot", "soup", "counters", "serving_station", "nothing",
)
LAYOUTS = (
    "forced_coordination", "counter_circuit_o_1order", "asymmetric_advantages",
    "cramped_room", "coordination_ring",
)


def held_name(player):
    return player.held_object.name if player.held_object is not None else None


def facing(mdp, player):
    x, y = player.position
    dx, dy = player.orientation
    return mdp.terrain_mtx[y + dy][x + dx]


def completed(before, after, terrain):
    if before == after:
        return None
    transitions = {
        (None, "onion", "O"): 0, (None, "onion", "X"): 1,
        ("onion", None, "P"): 2, ("onion", None, "X"): 3,
        (None, "dish", "D"): 4, (None, "dish", "X"): 5,
        ("dish", None, "X"): 6, ("dish", "soup", "P"): 7,
        (None, "soup", "X"): 8, ("soup", None, "X"): 9,
        ("soup", None, "S"): 10,
    }
    key = (before, after, terrain)
    if key not in transitions:
        raise ValueError(f"Transition outside the original task set: {key}")
    return transitions[key]


def doable(mdp, state, player, previous, valid_counters, max_counters=2):
    layout = mdp.layout_name
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported original-paper layout: {layout}")
    mask = np.zeros(len(NAMES), dtype=bool)
    loose = [o for o in state.objects.values()
             if mdp.terrain_mtx[o.position[1]][o.position[0]] == "X"]
    obj = held_name(state.players[player])
    source_side = not (layout == "forced_coordination" and player == 0)
    pot_side = not (layout == "forced_coordination" and player == 1)
    if obj is None:
        mask[0] = mask[4] = source_side
        for item in loose:
            if item.position not in valid_counters[player]:
                continue
            if item.name == "onion" and previous != 3:
                mask[1] = True
            elif item.name == "dish" and previous != 6:
                mask[5] = True
            elif item.name == "soup" and previous != 9:
                mask[8] = True
    elif obj == "onion":
        full = sum(o.name == "soup" and o.is_full
                   for o in state.objects.values()
                   if mdp.terrain_mtx[o.position[1]][o.position[0]] == "P")
        mask[2] = pot_side and full < len(mdp.get_pot_locations())
        mask[3] = previous != 1 and (len(loose) < max_counters or not mask.any())
    elif obj == "dish":
        mask[7] = pot_side and any(
            o.name == "soup" and not o.is_idle
            for o in state.objects.values()
            if mdp.terrain_mtx[o.position[1]][o.position[0]] == "P"
        )
        mask[6] = previous != 5 and (len(loose) < max_counters or not mask.any())
    elif obj == "soup":
        mask[10] = pot_side
        mask[9] = len(loose) < max_counters or not mask.any()
    else:
        raise ValueError(f"Object outside the original task set: {obj}")
    mask[UNKNOWN] = True
    return mask

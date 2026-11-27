"""Project a two-player hucooked soup state into NestRL's dense features."""

import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction


_DIRECTIONS = (Direction.WEST, Direction.EAST, Direction.SOUTH, Direction.NORTH)
_ACTIONS = (Action.STAY, *_DIRECTIONS, Action.INTERACT)
_ACTION_INDICES = {action: index for index, action in enumerate(_ACTIONS)}
_NEIGHBOR_DIRECTIONS = (Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH)

_POSITION_FOODS = ("lettuce", "tomato", "potato", "onion", "carrot", "broccoli")
_FOODS = ("tomato", "lettuce", "onion", "carrot", "potato", "broccoli")
_HOLDABLE = (*_FOODS[:3], "plate", *_FOODS[3:])
_FRONT = ("counter", "cutboard", "delivery", *_HOLDABLE[:4], "agent", "blender", *_HOLDABLE[4:])
_DISPENSERS = {"O": "onion", "T": "tomato", "D": "plate"}
_FIXTURES = {"X": "counter", "P": "cutboard", "S": "delivery"}


def hucooked_action(index):
    """Convert a NestRL index to a native hucooked action."""
    if not 0 <= index < len(_ACTIONS):
        raise ValueError(f"Invalid NestRL action index: {index}")
    return _ACTIONS[index]


def nest_action_index(action):
    """Encode a native hucooked action for the L1 history encoder."""
    return _ACTION_INDICES[action]


def _contents(obj, in_pot=False):
    """Return the visible object classes and whether their food is prepared."""
    if obj is None:
        return {}
    if obj.name == "soup":
        contents = dict.fromkeys(obj.ingredients, obj.is_ready)
        if not in_pot:
            contents["plate"] = False
        return contents
    return {"plate" if obj.name == "dish" else obj.name: False}


class HucookedObservationAdapter:
    """Approximate NestRL's feature semantics without modifying soup dynamics.

    Pots use CutBoard slots. Cooked ingredients use the chopped bits, and soup
    outside a pot uses the plate and ingredient bits. Missing positions are
    zero pairs. Duplicate objects or dispensers resolve to the nearest one.
    """

    def __init__(self, mdp):
        if mdp.num_players != 2:
            raise ValueError("The L1 observation schema requires two players")
        self.dimensions = np.asarray(mdp.shape, dtype=np.float32)
        self.terrain = {
            position: tile for tile, positions in mdp.terrain_pos_dict.items() for position in positions
        }
        self.pots = mdp.terrain_pos_dict["P"]
        self.delivery = mdp.terrain_pos_dict["S"]
        self.sources = {
            name: mdp.terrain_pos_dict[tile] for tile, name in _DISPENSERS.items()
        }

    def observe(self, state, player_index):
        player = state.players[player_index]
        teammate = state.players[1 - player_index]
        x, y = player.position

        def distance(position):
            return ((position[0] - x)**2 + (position[1] - y)**2, position)

        def relative(position):
            if position is None:
                return (0, 0)
            return (position[0] - x, position[1] - y)

        objects = {
            pos: _contents(obj, self.terrain[pos] == "P") for pos, obj in state.objects.items()
        }
        for chef in state.players:
            if chef.has_object():
                objects[chef.position] = _contents(chef.held_object)

        nearest_objects = {}
        for position in sorted(objects, key=distance):
            for name, prepared in objects[position].items():
                nearest_objects.setdefault(name, (position, prepared))
        for name, positions in self.sources.items():
            nearest_objects.setdefault(name, (min(positions, key=distance, default=None), False))

        pots = sorted(self.pots, key=distance)[:2]
        pots += [None] * (2 - len(pots))
        serving = min(self.delivery, key=distance, default=None)
        positions = [player.position, relative(teammate.position),
                     *(relative(pos) for pos in (*pots, serving))]
        food_prepared = {}
        for name in ("plate", *_POSITION_FOODS):
            position, prepared = nearest_objects.get(name, (None, False))
            positions.append(relative(position))
            food_prepared[name] = prepared

        adjacent = [self.terrain.get((x + dx, y + dy)) != " " for dx, dy in _NEIGHBOR_DIRECTIONS]
        holding = _contents(player.held_object)
        front_pos = Action.move_in_direction(player.position, player.orientation)
        front_tile = self.terrain.get(front_pos)
        front = objects.get(front_pos, {}).copy()
        if front_tile in _DISPENSERS:
            front["counter"] = False
            front[_DISPENSERS[front_tile]] = False
        elif front_tile in _FIXTURES:
            front[_FIXTURES[front_tile]] = False

        return np.concatenate((
            (np.asarray(positions, dtype=np.float32) / self.dimensions).ravel(),
            [direction == chef.orientation for chef in (player, teammate) for direction in _DIRECTIONS],
            [food_prepared[name] for name in _POSITION_FOODS],
            adjacent,
            [name in holding for name in _HOLDABLE],
            [holding.get(name, False) for name in _FOODS],
            [name in front for name in _FRONT],
            [front.get(name, False) for name in _FOODS],
        ))

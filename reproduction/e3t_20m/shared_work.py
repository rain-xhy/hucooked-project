"""Optional SharedKitchen work and staging actions for a learned three-chef E3T policy."""

from numbers import Integral

import numpy as np

from .shared_adapter import ASSEMBLE_W, COOK_E, COOK_W, SharedE3TEnv, route
from .vendor.shared.controller import navigation_action
from .vendor.shared.kitchen import DIRECTIONS, STAY
from .vendor.shared.layouts import FLOOR, SPAWNS

# These are public positions on the pinned layout. All three chefs get the same
# menu. A staging choice moves one tile toward a position, or waits on arrival.
STAGES = {
    "stage_assembly_west": ASSEMBLE_W,
    "stage_cooker_east": COOK_E,
    "stage_cutting_side": (5, 3),
}


class SharedWorkEnv(SharedE3TEnv):
    names = (*SharedE3TEnv.names, *STAGES)
    action_count = len(names)

    def reset(self, seed):
        obs, mask = super().reset(seed)
        # These three player-visible starting positions persist as an explicit
        # one-hot memory for each chef after all three have moved.
        spawn = [SPAWNS.index(c.position) for c in self.core.state.chefs]
        self._spawn_features = np.eye(3, dtype=np.float32)[spawn]
        return np.concatenate((obs, self._spawn_features), axis=1), mask

    def _masks(self):
        valid = super()._masks()
        state = self.core.state
        held = [chef.held.kind if chef.held else None for chef in state.chefs]
        pending = any(order.status == "active" for order in state.orders)
        # One spare unit can be fetched while its downstream station is busy.
        # The carrying chef can return excess supplies to their native dispenser.
        for i, item in enumerate(held):
            empty = item is None
            valid[i, 2] = item == "raw_main" or (empty and pending and "raw_main" not in held)
            valid[i, 3] = item in ("raw_side", "chopped_side") or (
                empty
                and state.cutting.item is None
                and "raw_side" not in held
                and "chopped_side" not in held
            )
            valid[i, 4] = item == "plate" or (empty and "plate" not in held)
            if empty and state.cooker.item is not None:
                occupied = {chef.position for j, chef in enumerate(state.chefs) if j != i}
                valid[i, 5:7] = (COOK_W not in occupied, COOK_E not in occupied)
        return np.concatenate((valid, np.ones((3, len(STAGES)), dtype=bool)), axis=1)

    def primitive_actions(self, intents):
        intents = tuple(intents)
        if len(intents) != 3 or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or not 0 <= i < self.action_count
            for i in intents
        ):
            raise ValueError("Expected three integer work/stage choices")
        chefs = self.core.state.chefs
        occupied = {c.position for c in chefs}
        reserved = set()
        actions = []
        for i, (chef, intent) in enumerate(zip(chefs, intents)):
            name = self.names[intent]
            if name == "stay":
                action = STAY
            elif name in STAGES:
                action = navigation_action(self.core, i, STAGES[name], reserved=reserved)
            else:
                action = route(
                    chef.position,
                    chef.orientation,
                    name,
                    frozenset((occupied - {chef.position}) | reserved),
                )
            actions.append(action)
            if action < STAY:
                dx, dy = DIRECTIONS[action]
                nxt = (chef.position[0] + dx, chef.position[1] + dy)
                if nxt in FLOOR:
                    reserved.add(nxt)
        return tuple(actions)

    def step(self, intents):
        actions = self.primitive_actions(intents)
        obs, reward, term, trunc, info = self.core.step(actions)
        # Reward irreversible production events during training. Serving is scored
        # exclusively by the engine, and evaluation consumes only its terminal metrics.
        weights = {
            "load": 0.75,
            "place_cutting": 0.5,
            "chop": 0.1,
            "place_assembly": 0.5,
            "assemble": 1.0,
        }
        bonus = sum(
            (2.0 if self.core.order(e["order_id"]).kind == "joint" else 1.0)
            if e["type"] == "unload"
            else weights.get(e["type"], 0.0)
            for e in info["events"]
        )
        info["primitive_actions"] = actions
        observation = np.concatenate((np.stack(obs), self._spawn_features), axis=1)
        return observation, self._masks(), reward, bonus, term, trunc, info

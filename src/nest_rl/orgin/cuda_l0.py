"""Batched full rule partners for the pinned fc_small_divider_test kitchen."""

import torch

from overcooked_setup import RECIPES
from torch_kitchen import EMPTY, FLOOR, FOOD_NAMES, HEIGHT, PLATE, WIDTH


WAIT, LEFT, RIGHT, DOWN, UP, INTERACT = range(6)
NO_EVENT = -1
NO_TARGET = -1

# Per-recipe order from RuleBasedPolicy.support_set. The two recipe ingredients
# appear in RECIPES order, independently of their IDs in TorchKitchen.
(
    PUT_FOOD0_CUT_BOARD, PUT_FOOD0_DIVIDER, PUT_FOOD0_PLATE,
    PUT_FOOD1_CUT_BOARD, PUT_FOOD1_DIVIDER, PUT_FOOD1_PLATE,
    PUT_PLATE_DELIVERY, PUT_PLATE_DIVIDER,
    TAKE_FOOD0, TAKE_FOOD1, TAKE_PLATE, CHOP_FOOD0, CHOP_FOOD1,
) = range(13)
NUM_EVENTS = CHOP_FOOD1 + 1


class CudaRulePartner:
    """Keep independent rule policies and their current events on the kitchen device."""

    def __init__(self, kitchen, recipes, player, seed=0, event_weights=None):
        if len(recipes) != kitchen.num_envs:
            raise ValueError("Expected one rule-partner recipe per kitchen")
        if player not in (0, 1):
            raise ValueError("Expected rule-partner player 0 or 1")
        self.kitchen = kitchen
        self.player = player
        self.device = kitchen.device
        self._rows = torch.arange(kitchen.num_envs, device=self.device)
        self._foods = torch.tensor(
            [[FOOD_NAMES.index(food) for food in RECIPES[recipe]] for recipe in recipes],
            dtype=torch.long, device=self.device,
        )
        self._supported = torch.zeros((kitchen.num_envs, PLATE), dtype=torch.bool, device=self.device)
        self._supported.scatter_(1, self._foods, True)
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        weight_shape = (kitchen.num_envs, NUM_EVENTS)
        if event_weights is None:
            self._weights = 1 + torch.rand(
                weight_shape, dtype=torch.float64, device=self.device, generator=self._generator,
            )
        else:
            self._weights = torch.as_tensor(event_weights, dtype=torch.float64, device=self.device)
            if self._weights.shape != weight_shape:
                raise ValueError(f"Expected rule-partner event weights with shape {weight_shape}")
        self.current_event = torch.full_like(self._rows, NO_EVENT)
        self.current_target = torch.full_like(self._rows, NO_TARGET)
        self._event_object = self._foods[:, (0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1)]
        self._event_object[:, (PUT_PLATE_DELIVERY, PUT_PLATE_DIVIDER, TAKE_PLATE)] = PLATE
        self._follow_up = torch.full((NUM_EVENTS,), NO_EVENT, device=self.device, dtype=torch.long)
        self._follow_up[PUT_FOOD0_CUT_BOARD] = CHOP_FOOD0
        self._follow_up[PUT_FOOD1_CUT_BOARD] = CHOP_FOOD1
        self._follow_up[CHOP_FOOD0] = TAKE_FOOD0
        self._follow_up[CHOP_FOOD1] = TAKE_FOOD1

        # The pinned layout keeps each chef on its own side with one cut board.
        self._board = kitchen.static_positions[player]
        # Keep this fixed index as an int so lookup never extracts a CUDA scalar.
        self._delivery = int(kitchen.static_positions[2])
        self._divider = torch.arange(1, 6, device=self.device) * WIDTH + 3

        # Table rows are agent positions, columns are target positions. The
        # rule policy prefers vertical progress and tests static obstacles only.
        positions = torch.arange(WIDTH * HEIGHT, device=self.device)
        x = positions % WIDTH
        y = positions // WIDTH
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        distance = dx.abs() + dy.abs()
        self._reachable = (x[None, :] - 3) * (x[:, None] - 3) >= 0
        self._divider_distance = distance[:, self._divider]
        horizontal = torch.where(dx > 0, RIGHT, LEFT)
        vertical = torch.where(dy > 0, DOWN, UP)
        next_x = kitchen.neighbors[positions[:, None], horizontal]
        next_y = kitchen.neighbors[positions[:, None], vertical]
        can_move_x = (dx != 0) & (kitchen.tiles[next_x] == FLOOR)
        can_move_y = (dy != 0) & (kitchen.tiles[next_y] == FLOOR)
        navigation = torch.where(can_move_y, vertical, torch.where(can_move_x, horizontal, WAIT))
        adjacent = distance <= 1
        self._navigate = navigation.masked_fill(adjacent, WAIT)
        facing = torch.where(dx == 0, vertical, horizontal)
        orientation = torch.arange(INTERACT, device=self.device)
        self._event_action = torch.where(
            adjacent[:, :, None],
            torch.where(facing[:, :, None] == orientation, INTERACT, facing[:, :, None]),
            self._navigate[:, :, None],
        )

    def reset(self):
        """Match the CPU policy, which carries its current event across resets."""

    def _available_targets(self, agent, held, objects):
        kitchen = self.kitchen
        plate = objects[:, PLATE]
        food_pos = objects.gather(1, self._foods)
        food_done = kitchen.chopped.gather(1, self._foods)
        empty_hands = held == EMPTY

        divider_free = ~(objects[:, :, None] == self._divider).any(dim=1)
        # argmin breaks equal-distance ties in ascending divider row order.
        divider_index = self._divider_distance[agent].masked_fill(~divider_free, WIDTH + HEIGHT).argmin(dim=1)
        divider = torch.where(divider_free.any(dim=1), self._divider[divider_index], NO_TARGET)
        board_free = ~(objects == self._board).any(dim=1)
        delivery_free = ~(objects == self._delivery).any(dim=1)
        plate_reachable = self._reachable[agent, plate]

        put_food = held[:, None] == self._foods
        put_food_targets = torch.stack((
            torch.where(put_food & ~food_done & board_free[:, None], self._board, NO_TARGET),
            torch.where(put_food, divider[:, None], NO_TARGET),
            torch.where(put_food & food_done & plate_reachable[:, None], plate[:, None], NO_TARGET),
        ), dim=2).reshape(-1, 6)

        put_plate = held == PLATE
        # The world keeps food contained in the plate at the plate's location.
        in_plate = objects[:, :PLATE] == plate[:, None]
        put_delivery = put_plate & in_plate.any(dim=1) & delivery_free & self._reachable[agent, self._delivery]

        food_reachable = self._reachable[agent[:, None], food_pos]
        food_alone = (food_pos[:, :, None] == objects[:, None, :]).sum(dim=2) == 1
        take_food = empty_hands[:, None] & food_reachable & food_alone
        recipe_met = (in_plate.sum(dim=1) >= 2) & ~(in_plate & ~self._supported).any(dim=1)
        take_plate = empty_hands & recipe_met & plate_reachable
        chop = empty_hands[:, None] & food_reachable & ~food_done & (food_pos == self._board)
        return torch.cat((
            put_food_targets,
            torch.where(put_delivery, self._delivery, NO_TARGET)[:, None],
            torch.where(put_plate, divider, NO_TARGET)[:, None],
            torch.where(take_food, food_pos, NO_TARGET),
            torch.where(take_plate, plate, NO_TARGET)[:, None],
            torch.where(chop, food_pos, NO_TARGET),
        ), dim=1)

    def _still_available(self, objects, event, target):
        item = self._event_object[self._rows, event.clamp_min(0)]
        item_at_target = objects[self._rows, item] == target
        item_done = self.kitchen.chopped[self._rows, item.clamp_max(PLATE - 1)]
        put_into_plate = (event == PUT_FOOD0_PLATE) | (event == PUT_FOOD1_PLATE)
        target_free = ~(objects == target[:, None]).any(dim=1)
        # A selected event checks its old target rather than re-running the
        # selection criteria, even if a better target has become available.
        return (target != NO_TARGET) & torch.where(
            event < TAKE_FOOD0,
            torch.where(put_into_plate, objects[:, PLATE] == target, target_free),
            item_at_target & ((event < CHOP_FOOD0) | ~item_done),
        )

    @torch.no_grad()
    def act(self, draw=None):
        """Return one long action per kitchen, optionally using supplied uniforms in [0, 1)."""
        kitchen = self.kitchen
        agent = kitchen.agent_pos[:, self.player]
        orientation = kitchen.orientation[:, self.player]
        held = kitchen.held[:, self.player]
        objects = kitchen.object_pos

        keep = self._still_available(objects, self.current_event, self.current_target)
        targets = self._available_targets(agent, held, objects)
        weights = self._weights.masked_fill(targets == NO_TARGET, 0)
        cdf = weights.cumsum(dim=1)
        total = cdf[:, -1]
        if draw is None:
            draw = torch.rand(
                kitchen.num_envs, dtype=cdf.dtype, device=self.device, generator=self._generator,
            )
        else:
            draw = torch.as_tensor(draw, dtype=cdf.dtype, device=self.device)
            if draw.shape != total.shape:
                raise ValueError(f"Expected one event draw per kitchen with shape {total.shape}")
        selected = (cdf <= (draw * total)[:, None]).sum(dim=1).clamp_max(NUM_EVENTS - 1)
        event = torch.where(keep, self.current_event, torch.where(total > 0, selected, NO_EVENT))
        target = torch.where(keep, self.current_target, targets[self._rows, selected])
        active = event != NO_EVENT
        action = torch.where(
            active, self._event_action[agent, target.clamp_min(0), orientation],
            self._navigate[agent, self._delivery],
        )

        # The CPU creates a targetless follow-up after a cut board drop or a
        # chop. The next call invalidates that event before sampling again.
        interacted = action == INTERACT
        self.current_event.copy_(torch.where(interacted, self._follow_up[event.clamp_min(0)], event))
        self.current_target.copy_(torch.where(active & ~interacted, target, NO_TARGET))
        return action

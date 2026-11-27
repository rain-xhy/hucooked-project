"""Dense observations and shaped rewards for the pinned NestRL kitchen."""

import torch

from torch_kitchen import COUNTER, CUT_BOARD, DELIVERY, EMPTY, FLOOR, FOOD_NAMES, HEIGHT, HORIZON, PLATE, WIDTH


# The world inserts foods in level JSON order. Holding and front features instead
# follow the class indexes in upstream world_objects.py.
_OBSERVATION_FOODS = ("Lettuce", "Tomato", "Potato", "Onion", "Carrot", "Broccoli")
_RECIPE_INGREDIENTS = (
    ("Potato", "Broccoli"),
    ("Broccoli", "Potato"),
    ("Lettuce", "Onion"),
    ("Onion", "Lettuce"),
    ("Tomato", "Carrot"),
    ("Carrot", "Tomato"),
)
RECIPE_NAMES = tuple(f"{first}{second}Salad" for first, second in _RECIPE_INGREDIENTS)

_NUM_FOODS = len(FOOD_NAMES)
_PLATE_STATE_OFFSET = 2 * _NUM_FOODS
_EMPTY_HOLD_STATE = _PLATE_STATE_OFFSET + 2 ** _NUM_FOODS
_NUM_RECIPE_NODES = 5


class KitchenFeatures:
    def __init__(self, kitchen):
        self.kitchen = kitchen
        device = kitchen.device
        num_envs = kitchen.num_envs

        locations = torch.arange(WIDTH * HEIGHT, device=device)
        self._coordinates = torch.stack((locations % WIDTH, locations // WIDTH), dim=-1).float()
        self._dimensions = torch.tensor((WIDTH, HEIGHT), dtype=torch.float32, device=device)
        self._orientation_encoding = torch.eye(4, dtype=torch.bool, device=device)

        food_order = [FOOD_NAMES.index(name) for name in _OBSERVATION_FOODS]
        self._food_order = torch.tensor(food_order, device=device)
        self._dynamic_order = torch.tensor([PLATE, *food_order], device=device)
        counter_dirs = torch.tensor([2, 3, 1, 4], device=device)  # Right, down, left, up.
        self._adjacent_counter = kitchen.tiles[kitchen.neighbors[:, counter_dirs]] != FLOOR
        self._front_static = torch.stack(
            (kitchen.tiles == COUNTER, kitchen.tiles == CUT_BOARD, kitchen.tiles == DELIVERY), dim=-1
        )
        # Front lookup scans world_objects, which excludes agents. This layout has no Blender.
        self._absent_front_classes = torch.zeros((num_envs, 2, 2), dtype=torch.bool, device=device)

        self._food_ids = torch.arange(_NUM_FOODS, device=device)
        self._food_bits = 1 << self._food_ids
        self._empty_hold_state = torch.full((num_envs, 1), _EMPTY_HOLD_STATE, device=device)
        recipe_foods = torch.tensor(
            [[FOOD_NAMES.index(name) for name in pair] for pair in _RECIPE_INGREDIENTS], device=device
        )
        self._recipe_first = recipe_foods[:, 0]
        self._recipe_second = recipe_foods[:, 1]
        self._other_foods = (self._food_ids != self._recipe_first[:, None]) & (
            self._food_ids != self._recipe_second[:, None]
        )
        self._progress = torch.zeros((num_envs, len(RECIPE_NAMES)), dtype=torch.int32, device=device)
        self._seen_held = torch.zeros((num_envs, 2, _EMPTY_HOLD_STATE + 1), dtype=torch.bool, device=device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        progress, _ = self._recipe_progress()
        if mask is None:
            self._progress.copy_(progress)
            self._seen_held.zero_()
        else:
            self._progress.copy_(torch.where(mask[:, None], progress, self._progress))
            self._seen_held.logical_and_(~mask[:, None, None])

    def observe(self) -> torch.Tensor:
        kitchen = self.kitchen
        own_xy = self._coordinates[kitchen.agent_pos]
        object_pos = torch.cat(
            (kitchen.static_positions.expand(kitchen.num_envs, -1), kitchen.object_pos[:, self._dynamic_order]),
            dim=-1,
        )
        relative_agent = own_xy.flip(1) - own_xy
        relative_objects = self._coordinates[object_pos][:, None] - own_xy[:, :, None]
        position = torch.cat((own_xy[:, :, None], relative_agent[:, :, None], relative_objects), dim=2)
        position = (position / self._dimensions).flatten(2)
        orientation = self._orientation_encoding[kitchen.orientation - 1]
        directions = torch.cat((orientation, orientation.flip(1)), dim=-1)
        food_done = kitchen.chopped[:, self._food_order][:, None].expand(-1, 2, -1)
        adjacent_counter = self._adjacent_counter[kitchen.agent_pos]

        holding_plate = kitchen.held[:, :, None] == PLATE
        holding_food = (kitchen.held[:, :, None] == self._food_ids) | (holding_plate & kitchen.plated[:, None])
        holding = torch.cat((holding_food[..., :3], holding_plate, holding_food[..., 3:]), dim=-1)
        holding_done = holding_food & kitchen.chopped[:, None]

        front = kitchen.neighbors[kitchen.agent_pos, kitchen.orientation]
        front_dynamic = front[:, :, None] == kitchen.object_pos[:, None]
        front_objects = torch.cat(
            (
                self._front_static[front],
                front_dynamic[..., :3],
                front_dynamic[..., PLATE:PLATE + 1],
                self._absent_front_classes,
                front_dynamic[..., 3:PLATE],
            ),
            dim=-1,
        )
        front_done = front_dynamic[..., :PLATE] & kitchen.chopped[:, None]

        return torch.cat(
            (position, directions, food_done, adjacent_counter, holding, holding_done, front_objects, front_done),
            dim=-1,
        )

    def _recipe_progress(self) -> tuple[torch.Tensor, torch.Tensor]:
        kitchen = self.kitchen
        plate_pos = kitchen.object_pos[:, PLATE]
        # Recipe nodes and is_completable check colocation, regardless of container membership.
        colocated = kitchen.object_pos[:, :PLATE] == plate_pos[:, None]
        prepared_at_plate = colocated & kitchen.chopped
        first_at_plate = prepared_at_plate[:, self._recipe_first]
        pair_at_plate = first_at_plate & prepared_at_plate[:, self._recipe_second]
        delivered = pair_at_plate & (plate_pos == kitchen.static_positions[-1])[:, None]
        progress = (
            kitchen.chopped[:, self._recipe_first].to(torch.int32)
            + kitchen.chopped[:, self._recipe_second]
            + first_at_plate
            + pair_at_plate
            + delivered
        )
        completable = ~(colocated[:, None] & self._other_foods).any(dim=-1)
        return progress, completable

    def _interaction_reward(self) -> torch.Tensor:
        kitchen = self.kitchen
        # Each chef earns the bonus once per food preparation state or plate content set.
        plate_contents = (kitchen.plated * self._food_bits).sum(dim=-1, keepdim=True)
        hold_states = torch.cat(
            (2 * self._food_ids + kitchen.chopped, _PLATE_STATE_OFFSET + plate_contents, self._empty_hold_state),
            dim=-1,
        )
        held_state = hold_states.gather(1, kitchen.held).unsqueeze(-1)
        first_hold = ~self._seen_held.gather(2, held_state).squeeze(-1) & (kitchen.held != EMPTY)
        self._seen_held.scatter_(2, held_state, True)
        return 0.5 * first_hold.sum(dim=-1, dtype=torch.float32)

    def reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        progress, completable = self._recipe_progress()
        recipe_reward = torch.where(progress == _NUM_RECIPE_NODES, 10, progress - self._progress)
        recipe_reward = torch.where(completable, recipe_reward, 0).amax(dim=-1)
        self._progress.copy_(torch.where(completable, progress, self._progress))

        completed = self._progress == _NUM_RECIPE_NODES
        delivered = completed.any(dim=-1)
        # Both order variants can complete together. The CPU chooses the first configured recipe.
        recipe_index = torch.where(delivered, completed.long().argmax(dim=-1), -1)
        at_horizon = self.kitchen.ticks >= HORIZON
        reward = self._interaction_reward() + recipe_reward - 0.2
        return reward, delivered | at_horizon, at_horizon & ~delivered, recipe_index

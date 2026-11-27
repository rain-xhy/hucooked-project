"""Batched dynamics for the pinned Overcooked fc_small_divider_test kitchen."""

from itertools import permutations

import torch


FOOD_NAMES = ("Tomato", "Lettuce", "Onion", "Carrot", "Potato", "Broccoli")
PLATE = len(FOOD_NAMES)
EMPTY = PLATE + 1
WIDTH, HEIGHT = 7, 8
HORIZON = 40
FLOOR, COUNTER, CUT_BOARD, DELIVERY = range(4)


class TorchKitchen:
    """Keep a batch of independent two-player kitchens on one device."""

    def __init__(self, num_envs, device, seed=0):
        from torch_kitchen_features import KitchenFeatures

        self.num_envs = num_envs
        self.agent_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.long)
        self.device = self.agent_pos.device
        self.orientation = torch.ones_like(self.agent_pos)
        self.held = torch.full_like(self.agent_pos, EMPTY)
        self.object_pos = torch.zeros((num_envs, EMPTY), device=self.device, dtype=torch.long)
        self.chopped = torch.zeros((num_envs, PLATE), device=self.device, dtype=torch.bool)
        self.plated = torch.zeros_like(self.chopped)
        self.ticks = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._object_ids = torch.arange(EMPTY, device=self.device)

        # Flatten coordinates in the pinned level's row-major order.
        static = [4 * WIDTH, 7 * WIDTH + 5, 3 * WIDTH + 6]
        tiles = []
        neighbors = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                tiles.append(FLOOR if 0 < y < HEIGHT - 1 and x in (1, 2, 4, 5) else COUNTER)
                neighbors.append([
                    y * WIDTH + x,
                    y * WIDTH + max(x - 1, 0),
                    y * WIDTH + min(x + 1, WIDTH - 1),
                    min(y + 1, HEIGHT - 1) * WIDTH + x,
                    max(y - 1, 0) * WIDTH + x,
                    y * WIDTH + x,
                ])
        for pos, tile in zip(static, (CUT_BOARD, CUT_BOARD, DELIVERY)):
            tiles[pos] = tile
        self.tiles = torch.tensor(tiles, device=self.device)
        self.neighbors = torch.tensor(neighbors, device=self.device)
        self.static_positions = torch.tensor(static, device=self.device)

        self._left_agent_spawns = torch.tensor(
            [y * WIDTH + x for y in (3, 4, 5, 6) for x in (1, 2)], device=self.device,
        )
        self._right_agent_spawns = torch.tensor(
            [y * WIDTH + x for y in (4, 5, 6) for x in (4, 5)], device=self.device,
        )
        self._left_food_spawns = torch.tensor(
            list(permutations(y * WIDTH for y in (1, 2, 3))), device=self.device,
        )
        self._right_food_spawns = torch.tensor(
            list(permutations(y * WIDTH + 6 for y in (4, 5, 6))), device=self.device,
        )
        self._plate_spawns = torch.tensor([y * WIDTH + 3 for y in (2, 3, 4)], device=self.device)
        self.features = KitchenFeatures(self)

    def _sample_layout(self):
        # Independent uniform choices match the level's collision-free spawn regions.
        random_args = {"device": self.device, "generator": self._generator}
        batch = (self.num_envs,)
        left_agent = self._left_agent_spawns[torch.randint(8, batch, **random_args)]
        right_agent = self._right_agent_spawns[torch.randint(6, batch, **random_args)]
        food_draw = torch.randint(6, (self.num_envs, 2), **random_args)
        left_food = self._left_food_spawns[food_draw[:, 0]]
        right_food = self._right_food_spawns[food_draw[:, 1]]
        plate = self._plate_spawns[torch.randint(3, batch, **random_args)]
        objects = torch.stack((
            left_food[:, 0], left_food[:, 1], right_food[:, 0], right_food[:, 1],
            left_food[:, 2], right_food[:, 2], plate,
        ), dim=1)
        return torch.stack((left_agent, right_agent), dim=1), objects

    def reset(self, mask=None, layout=None):
        """Reset the selected rows, optionally using supplied full-batch positions."""
        if mask is None:
            mask = torch.ones_like(self.ticks, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if layout is None:
            agent_pos, object_pos = self._sample_layout()
        else:
            agent_pos, object_pos = (
                torch.as_tensor(positions, device=self.device, dtype=torch.long)
                for positions in layout
            )
        rows = mask[:, None]
        self.agent_pos.copy_(torch.where(rows, agent_pos, self.agent_pos))
        self.object_pos.copy_(torch.where(rows, object_pos, self.object_pos))
        self.orientation.masked_fill_(rows, 1)
        self.held.masked_fill_(rows, EMPTY)
        self.chopped.masked_fill_(rows, False)
        self.plated.masked_fill_(rows, False)
        self.ticks.masked_fill_(mask, 0)
        self.features.reset(mask)
        return self.features.observe()

    def _act(self, player, action, destination):
        self.agent_pos[:, player] = destination
        target = self.neighbors[destination, self.orientation[:, player]]
        interact = (action == 5) & (target != self.agent_pos[:, 0]) & (target != self.agent_pos[:, 1])
        tile = self.tiles[target]
        held = self.held[:, player]
        at_target = self.object_pos == target[:, None]
        # The plate takes precedence over its contents when choosing an object.
        target_item = torch.where(at_target, self._object_ids, -1).amax(dim=1)
        target_vacant = target_item < 0
        target_plate = target_item == PLATE
        target_food = ~(target_vacant | target_plate)
        held_plate = held == PLATE
        held_food = held < PLATE
        empty_hands = held == EMPTY
        checked_food = torch.stack((held, target_item), dim=1).clamp(0, PLATE - 1)
        held_chopped, target_chopped = self.chopped.gather(1, checked_food).unbind(dim=1)

        chop = interact & empty_hands & (tile == CUT_BOARD) & target_food & ~target_chopped
        pickup = interact & empty_hands & ~target_vacant & ~chop
        tile_accepts = (tile == COUNTER) | (tile == DELIVERY) | ((tile == CUT_BOARD) & held_food)
        put_down = interact & ~empty_hands & target_vacant & tile_accepts
        merge_into_target = interact & held_food & held_chopped & target_plate
        merge_into_held = interact & held_plate & target_food & target_chopped
        release = put_down | merge_into_target

        held_item = held[:, None] == self._object_ids
        foods_at_target = at_target[:, :PLATE]
        self.chopped.logical_or_(chop[:, None] & foods_at_target)
        self.plated.logical_or_(
            (merge_into_target[:, None] & held_item[:, :PLATE])
            | (merge_into_held[:, None] & foods_at_target)
        )

        # Moving the plate moves all its contents. Pickup and either merge update
        # physical positions in the same order as the original world actions.
        move_held = held_item | torch.cat((
            held_plate[:, None] & self.plated, torch.zeros_like(held_plate[:, None]),
        ), dim=1)
        move_target = (pickup | merge_into_held)[:, None] & at_target
        new_location = torch.where(release, target, destination)[:, None]
        self.object_pos.copy_(torch.where(move_held | move_target, new_location, self.object_pos))
        self.held[:, player] = torch.where(pickup, target_item, torch.where(release, EMPTY, held))

    def step(self, actions):
        """Advance all rows by one joint action and return their transition data."""
        moving = (0 < actions) & (actions < 5)
        self.orientation.copy_(torch.where(moving, actions, self.orientation))
        targets = self.neighbors[self.agent_pos, actions]
        destinations = torch.where(moving & (self.tiles[targets] == FLOOR), targets, self.agent_pos)
        left, right = self.agent_pos.unbind(dim=1)
        left_target, right_target = destinations.unbind(dim=1)
        # Agent 0 has priority if both players try to enter the same square.
        left_destination = torch.where(left_target == right, left, left_target)
        right_destination = torch.where(
            (right_target == left) | (right_target == left_target), right, right_target,
        )
        self._act(0, actions[:, 0], left_destination)
        self._act(1, actions[:, 1], right_destination)
        self.ticks.add_(1)
        reward, done, truncated, recipe_index = self.features.reward()
        return self.features.observe(), reward, done, truncated, recipe_index

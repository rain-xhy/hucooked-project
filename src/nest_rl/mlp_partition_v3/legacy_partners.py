"""State-driven counter-layout chefs, vectorized across independent kitchens."""
from collections import deque
from functools import lru_cache

import numpy as np
import torch

from overcooked_ai_py.cooperative.partition.model import (
    BEEF, BITS, BOARDS, CUT_POTATO, CUT_TOMATO, DELIVERY_SIDE, EGG, EMPTY,
    FIRST_MEAL, FLOOR, FLOORS, HEIGHT, INTERACT, OFFSETS, POTATO, POTS, RECIPES,
    SERVERS, SOURCES, STAY, TOMATO, WALL, WIDTH, cell,
)


@lru_cache(maxsize=1)
def _routing():
    """First move for each destination, occupied teammate cell and start.

    A station's interaction ports seed a reverse breadth-first search. A value
    of 6..9 means turn toward the station or interact if already facing it.
    The two parking destinations are floor cells and require no interaction.
    """
    stations = (*SOURCES.values(), *BOARDS, *POTS, *SERVERS, *WALL,
                cell(11, 1), cell(11, 5))
    size = WIDTH * HEIGHT
    paths = np.full((size, size, size), STAY, dtype=np.uint8)
    for station in stations:
        for side in FLOORS:
            floor = set(side)
            for blocked in (0, *side):
                row = paths[station, blocked]
                queue = deque()
                if station in floor:
                    queue.append(station)
                elif station not in FLOOR:
                    for direction, offset in enumerate(OFFSETS):
                        port = station - offset
                        if port in floor and port != blocked:
                            queue.append(port)
                            row[port] = 6 + direction
                seen = set(queue) | {blocked}
                while queue:
                    pos = queue.popleft()
                    for direction, offset in enumerate(OFFSETS):
                        prev = pos - offset
                        if prev in floor and prev not in seen:
                            seen.add(prev)
                            queue.append(prev)
                            row[prev] = direction
    return paths


class LegacyPartners:
    """Bind role 0 (left), 1 (upper right), 2 (lower right) at reset.

    A lane commits to its selected ticket until delivery or expiry. Ties use
    the first queue slot. All actions are suggestions, including the learner's.
    Consult observed hands and stations anew after each environment step.
    """

    version = "serial-v1"

    def __init__(self, env, roles, styles):
        if env.config.layout != "counter":
            raise ValueError("Scripted partners require the counter layout")
        self.env = env
        self.roles = self._labels(roles)
        self.styles = self._labels(styles)
        self.rows = np.arange(env.num_envs)
        self.paths = _routing()
        self.reset()

    def _labels(self, value):
        a = value.detach().cpu().numpy().copy()
        if (a.shape != (self.env.num_envs,) or a.dtype.kind not in "iu"
                or np.any((a < 0) | (a > 2))):
            raise ValueError("Expected one integer label 0..2 per kitchen")
        return a

    def reset(self):
        pos = self.env.state["pos"].cpu().numpy()
        self.chef_ids = np.argsort(np.where(pos % WIDTH < 6, -1, pos), axis=1)
        self._refresh_learner_ids()
        self.ticket_ids = np.full(len(self.rows), -1, dtype=np.int64)

    def _refresh_learner_ids(self):
        self.learner_ids = torch.as_tensor(
            self.chef_ids[self.rows, self.roles], device=self.env.device)

    def state_dict(self):
        return {k: torch.from_numpy(getattr(self, k).copy())
                for k in ("roles", "styles", "chef_ids", "ticket_ids")}

    def load_state_dict(self, state):
        for k in ("roles", "styles", "chef_ids", "ticket_ids"):
            setattr(self, k, state[k].detach().cpu().numpy().copy())
        self._refresh_learner_ids()

    def _choose_recipe(self, ids, due, recipes):
        matching = ids == self.ticket_ids[:, None]
        earliest = due.argmin(1)
        smallest = np.array([2, 1, 3])[recipes].argmin(1)
        preferred = np.choose(self.styles, (0, earliest, smallest))
        slot = np.where(matching.any(1), matching.argmax(1), preferred)
        self.ticket_ids = ids[self.rows, slot].copy()
        return recipes[self.rows, slot]

    def actions(self):
        keys = ("pos", "facing", "held", "ids", "due", "recipe", "board",
                "counter", "mix", "locked")
        # One device-to-host transfer, independent of the number of lanes.
        state = self.env.state
        packed = torch.cat([state[k] for k in keys], dim=1).cpu().numpy()
        widths = [state[k].shape[1] for k in keys]
        pos, facing, hand, ids, due, recipes, board, counter, mix, locked = (
            np.split(packed, np.cumsum(widths)[:-1], axis=1))
        pos, facing, hand = [a[self.rows[:, None], self.chef_ids] for a in (pos, facing, hand)]
        target = self._choose_recipe(ids, due, recipes)
        # Finish a partially filled left pot if it can only make the beef meal.
        # A locked pot must finish its current meal even if its ticket expired.
        target = np.where((target == 1) & ((mix[:, 0] & 9) != 0), 2, target)
        side = np.asarray(DELIVERY_SIDE)[target]
        target = np.where(locked[self.rows, side] >= 0, locked[self.rows, side], target)
        ingredients = np.asarray(RECIPES)[target]
        missing = ingredients & ~mix[self.rows, side]
        wall = np.array(WALL)
        # Transfer at the upper opening for the egg meal and lower for beef.
        order = np.where((target == 0)[:, None], [1, 0, 2, 3, 4], [3, 4, 2, 1, 0])
        arranged = counter[self.rows[:, None], order]

        def stored(item):
            hits = arranged == np.asarray(item).reshape(-1, 1)
            return np.where(hits.any(1), wall[order[self.rows, hits.argmax(1)]], 0)

        empty_slot = stored(EMPTY)
        dest = np.zeros_like(hand)
        meal = FIRST_MEAL + target
        # Either right chef can fix a premature pickup. Otherwise one works
        # each ticket and the other clears its ports.
        default = np.where(target == 0, 1, 2)
        partner = 3 - default
        useful = ((hand[self.rows, partner] == meal)
                  | ((target == 0) & np.isin(hand[self.rows, partner], [EGG, CUT_TOMATO]))
                  | ((target == 2) & (hand[self.rows, partner] == BEEF)))
        active = np.where(useful & (hand[self.rows, default] == EMPTY), partner, default)
        other = 3 - active
        dest[self.rows, other] = np.where(other == 1, cell(11, 1), cell(11, 5))

        left_hand = hand[:, 0]
        right_hand = hand[self.rows, active]
        # Start vegetables only while the required portion is missing from
        # both the pot and the transfer chain.
        tomato = ((missing & BITS[CUT_TOMATO]) != 0)
        tomato &= (target != 0) | ((stored(CUT_TOMATO) == 0)
                                  & ~(hand[:, 1:] == CUT_TOMATO).any(1))
        potato = (missing & BITS[CUT_POTATO]) != 0
        veg = np.where(potato, POTATO, np.where(tomato, TOMATO, EMPTY))
        board_slot = np.where(veg == POTATO, 1, 0)
        # Use an existing board even if the learner used the other left board.
        already = board[:, :2] == veg[:, None]
        board_slot = np.where(already.any(1), already.argmax(1), board_slot)
        veg_board = np.array(BOARDS)[board_slot]
        veg_source = np.where(veg == POTATO, SOURCES[POTATO], SOURCES[TOMATO])
        cut = np.where(veg == TOMATO, CUT_TOMATO, CUT_POTATO)
        veg_work = np.where(board[self.rows, board_slot] == veg, veg_board, veg_source)
        raw_slot = stored(veg)
        cut_slot = stored(cut)
        veg_work = np.where(raw_slot != 0, raw_slot, veg_work)
        veg_work = np.where(cut_slot != 0, cut_slot, veg_work)
        left = np.where(veg != EMPTY, veg_work, 0)
        need_beef = (missing & BITS[BEEF]) != 0
        left = np.where((target == 2) & need_beef & (stored(BEEF) != 0), stored(BEEF), left)
        left = np.where((target != 0) & (missing == 0), POTS[0], left)
        need_egg = (missing & BITS[EGG]) != 0
        right = np.where((target == 0) & need_egg, SOURCES[EGG], 0)
        right = np.where((target == 0) & ((missing & 1) != 0) & (stored(CUT_TOMATO) != 0),
                         stored(CUT_TOMATO), right)
        right = np.where((target == 0) & (missing == 0), POTS[1], right)
        right = np.where((target == 2) & need_beef & (stored(BEEF) == 0)
                         & (left_hand != BEEF), SOURCES[BEEF], right)
        for chef_side, held, work, role in ((0, left_hand, left, 0), (1, right_hand, right, active)):
            work = np.where((side == chef_side) & (stored(meal) != 0), stored(meal), work)
            # Stash surplus or expired work until it becomes useful again.
            carrying = empty_slot.copy()
            for raw in (TOMATO, POTATO):
                boards = np.asarray(BOARDS[2 * chef_side:2 * chef_side + 2])
                board_contents = board[:, 2 * chef_side:2 * chef_side + 2]
                first = int(raw == POTATO)
                slot = np.where(board_contents[:, first] == EMPTY, first, 1 - first)
                carrying = np.where((held == raw) & (board_contents[self.rows, slot] == EMPTY),
                                    boards[slot], carrying)
            for item, bit in BITS.items():
                carrying = np.where((held == item) & (side == chef_side) & ((missing & bit) != 0),
                                    POTS[chef_side], carrying)
            for cooked, delivery_side in enumerate(DELIVERY_SIDE):
                if delivery_side == chef_side:
                    carrying = np.where((held == FIRST_MEAL + cooked) & (recipes == cooked).any(1),
                                        SERVERS[chef_side], carrying)
            dest[self.rows, role] = np.where(held == EMPTY, work, carrying)

        # If two carriers need the same empty slot, reserve it for the left
        # chef and use the next free slot on the right. Counter interactions
        # by two chefs at once would otherwise cancel each other every step.
        same_drop = ((left_hand != EMPTY) & (right_hand != EMPTY)
                     & (dest[:, 0] == empty_slot) & (dest[self.rows, active] == empty_slot))
        alternatives = (arranged == EMPTY) & (wall[order] != empty_slot[:, None])
        spare = np.where(alternatives.any(1), wall[order[self.rows, alternatives.argmax(1)]], 0)
        dest[self.rows, active] = np.where(same_drop, spare, dest[self.rows, active])
        blocked = np.column_stack((np.zeros(len(self.rows), int), pos[:, 2], pos[:, 1]))
        moves = self.paths[dest, blocked, pos].astype(np.int64)
        moves = np.where(moves >= 6, np.where(facing == moves - 6, INTERACT, moves - 6), moves)
        # A parking chef yields a contested destination to the working chef.
        offsets = np.array([*OFFSETS, 0, 0])
        next_pos = pos + offsets[moves]
        collision = next_pos[self.rows, active] == next_pos[self.rows, other]
        moves[self.rows, other] = np.where(collision, STAY, moves[self.rows, other])
        actions = np.empty_like(moves)
        actions[self.rows[:, None], self.chef_ids] = moves
        return torch.as_tensor(actions, device=self.env.device)



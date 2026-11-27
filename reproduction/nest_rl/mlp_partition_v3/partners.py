"""Observed-state, two-pot planner with explicit ingredient and station reservations."""
from copy import deepcopy
from collections import deque
from functools import lru_cache
from itertools import permutations
import numpy as np
import torch
from .legacy_partners import LegacyPartners, _routing
from overcooked_ai_py.cooperative.partition.model import (
    BITS, BOARDS, CUT_POTATO, CUT_TOMATO, DELIVERY_SIDE, EMPTY, FIRST_MEAL,
    FLOORS, INTERACT, OFFSETS, POTATO, POTS, RECIPES, SERVERS, SOURCES, STAY, TOMATO,
    WALL, WIDTH, cell,
)

@lru_cache(maxsize=1)
def _distances():
    """Static shortest work distances, including counter transfers."""
    size = WIDTH * 7
    d = np.full((size, size), 1000, dtype=np.int32)
    for floors in FLOORS:
        floor = set(floors)
        stations = set(SOURCES.values()) | set(BOARDS) | set(POTS) | set(SERVERS) | set(WALL)
        for start in floor | stations:
            queue = deque([(start, 0)]) if start in floor else deque((start + o, 1) for o in OFFSETS if start + o in floor)
            seen = set()
            while queue:
                point, distance = queue.popleft()
                if point in seen:
                    continue
                seen.add(point)
                d[start, point] = min(d[start, point], distance)
                for offset in OFFSETS:
                    nxt = point + offset
                    if nxt in floor and nxt not in seen:
                        queue.append((nxt, distance + 1))
                    elif nxt in stations:
                        d[start, nxt] = min(d[start, nxt], distance + 1)
    for wall in WALL:
        d = np.minimum(d, d[:, wall, None] + d[wall, None, :])
    np.fill_diagonal(d, 0)
    return d

class Partners(LegacyPartners):
    """Plan both pots independently; True learner_mask entries reserve the learner.

    Roles bind to physical chefs on reset. All progress is reconciled against
    observations, never inferred merely from a previously issued action.
    """
    version = 'parallel-v2'

    def __init__(self, env, roles, styles, learner_mask=None):
        mask = np.zeros(env.num_envs, dtype=bool) if learner_mask is None else np.asarray(
            learner_mask.detach().cpu() if torch.is_tensor(learner_mask) else learner_mask)
        if mask.shape != (env.num_envs,) or mask.dtype != np.bool_:
            raise ValueError('learner_mask must contain one boolean per kitchen')
        self.learner_mask = mask.copy()
        super().__init__(env, roles, styles)

    def reset(self):
        super().reset()
        n = self.env.num_envs
        self.job_ticket_ids = np.full((n, 2), -1, dtype=np.int64)
        self.job_recipes = np.full((n, 2), -1, dtype=np.int64)
        self.previous_mix = np.zeros((n, 2), dtype=np.int64)
        self.assignments = [[None] * 3 for _ in range(n)]
        self.reservations = [[] for _ in range(n)]
        self.destinations = np.zeros((n, 3), dtype=np.int64)
        self.eligible_work = np.zeros((n, 3), dtype=bool)
        self.idle_actionable = np.zeros((n, 3), dtype=bool)
        self._batch_keys = ('pos', 'facing', 'held', 'ids', 'due', 'recipe',
                            'board', 'counter', 'mix', 'locked', 'cuts', 'heat')
        boundaries = np.cumsum([0] + [self.env.state[k].shape[1] for k in self._batch_keys])
        self._batch_slices = tuple(slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]))

    def state_dict(self):
        result = {k: torch.from_numpy(getattr(self, k).copy()) for k in
                  ('roles', 'styles', 'chef_ids', 'ticket_ids', 'learner_mask',
                   'job_ticket_ids', 'job_recipes', 'previous_mix', 'destinations', 'eligible_work', 'idle_actionable')}
        result.update(version=self.version, assignments=deepcopy(self.assignments),
                      reservations=deepcopy(self.reservations))
        return result

    def load_state_dict(self, state):
        if state.get('version') != self.version:
            raise ValueError('Partner checkpoint version mismatch')
        for key in ('roles', 'styles', 'learner_mask'):
            if not np.array_equal(getattr(self, key), state[key].cpu().numpy()):
                raise ValueError('Partner checkpoint configuration mismatch: ' + key)
        expected = self.state_dict()
        for key, value in expected.items():
            if torch.is_tensor(value):
                incoming = state[key].detach().cpu().numpy()
                if incoming.shape != value.shape:
                    raise ValueError('Partner checkpoint shape mismatch: ' + key)
                setattr(self, key, incoming.copy())
        self.assignments = deepcopy(state['assignments'])
        self.reservations = deepcopy(state['reservations'])
        self._refresh_learner_ids()

    def diagnostics(self):
        return dict(version=self.version, job_ticket_ids=self.job_ticket_ids.copy(),
                    job_recipes=self.job_recipes.copy(), assignments=deepcopy(self.assignments),
                    reservations=deepcopy(self.reservations), destinations=self.destinations.copy(),
                    idle_actionable=self.idle_actionable.copy(), eligible_work=self.eligible_work.copy())

    @staticmethod
    def _distance(a, b):
        return int(_distances()[a, b])

    def _lane(self, lane, s):
        pos, hand = s['pos'], s['held']
        side = [int(p % WIDTH > 6) for p in pos]
        controlled = [i for i in range(3) if not (self.learner_mask[lane] and i == self._learner_cpu[lane])]
        rank = sorted(range(3), key=lambda j: ((s['due'][j] if self.styles[lane] == 1 else
                       int(RECIPES[s['recipe'][j]]).bit_count() if self.styles[lane] == 2 else j), j))
        self.ticket_ids[lane] = s['ids'][rank[0]]
        jobs, recipes = self.job_ticket_ids[lane], self.job_recipes[lane]
        used = set()
        # A meal already in transit owns a live ticket before new pots claim it.
        meals = [item for item in hand if item >= FIRST_MEAL]
        meals.extend(item for item in s['counter'] if item >= FIRST_MEAL)
        for meal in meals:
            slot = next((j for j in rank if s['recipe'][j] == meal - FIRST_MEAL and int(s['ids'][j]) not in used), None)
            if slot is not None:
                used.add(int(s['ids'][slot]))
        for pot in range(2):
            mix, lock = int(s['mix'][pot]), int(s['locked'][pot])
            if self.previous_mix[lane, pot] and not mix:
                jobs[pot] = recipes[pot] = -1
            valid = [j for j in rank if int(s['ids'][j]) not in used and
                     (lock < 0 or s['recipe'][j] == lock) and
                     (mix & RECIPES[s['recipe'][j]]) == mix]
            old = next((j for j in valid if s['ids'][j] == jobs[pot]), None)
            if old is not None:
                used.add(int(jobs[pot]))
            elif mix:
                slot = valid[0] if valid else None
                recipes[pot] = int(s['recipe'][slot]) if slot is not None else lock if lock >= 0 else next(r for r, bits in enumerate(RECIPES) if mix & bits == mix)
                jobs[pot] = int(s['ids'][slot]) if slot is not None else -1
                if slot is not None:
                    used.add(int(jobs[pot]))
            else:
                jobs[pot] = recipes[pot] = -1
        # Rank first, then enumerate both pot mappings jointly. Existing jobs
        # remain committed, so a new ticket cannot interrupt a partially filled pot.
        free = [p for p in range(2) if recipes[p] < 0]
        selected = [j for j in rank if int(s['ids'][j]) not in used][:len(free)]
        def cost(p, r):
            ingredient_cost = 0
            for item, bit in BITS.items():
                if not RECIPES[r] & bit:
                    continue
                raw = TOMATO if item == CUT_TOMATO else POTATO if item == CUT_POTATO else item
                options = [self._distance(SOURCES[raw], POTS[p])]
                if raw in (TOMATO, POTATO):
                    options = [self._distance(SOURCES[raw], b) + self._distance(b, POTS[p]) + 3 for b in BOARDS]
                options.extend(self._distance(int(pos[i]), POTS[p]) for i in range(3) if hand[i] == item)
                options.extend(self._distance(WALL[k], POTS[p]) for k in range(5) if s['counter'][k] == item)
                options.extend(self._distance(BOARDS[k], POTS[p]) + 3 - int(s['cuts'][k]) for k in range(4) if s['board'][k] == raw)
                ingredient_cost += min(options)
            workers = [self._distance(int(pos[i]), POTS[p]) for i in controlled if side[i] == p]
            return ingredient_cost + self._distance(POTS[p], SERVERS[DELIVERY_SIDE[r]]) + min(workers, default=20)
        mappings = list(permutations(free, len(selected)))
        if selected:
            mapping = min(mappings, key=lambda ps: sum(cost(p, int(s['recipe'][j])) for p, j in zip(ps, selected)))
            for pot, j in zip(mapping, selected):
                jobs[pot], recipes[pot] = int(s['ids'][j]), int(s['recipe'][j])
                used.add(int(jobs[pot]))
        self.previous_mix[lane] = s['mix']
        dest = [0] * 3
        assignment = [None] * 3
        claimed = set()
        busy = set()
        reservations = []
        def nearest(chef, stations):
            return min(stations, key=lambda x: self._distance(int(pos[chef]), x), default=0)
        def claim(chef, target, task, resource=None):
            dest[chef], assignment[chef] = int(target), task
            busy.add(chef)
            if resource is not None:
                claimed.add(resource)
            reservations.append((chef, int(target), task))
        def drop(chef):
            return nearest(chef, [w for k, w in enumerate(WALL) if s['counter'][k] == EMPTY and ('counter', k) not in claimed
                                   and w not in dest])
        for i in controlled:
            previous = self.assignments[lane][i]
            target = int(self.destinations[lane, i])
            if hand[i] != EMPTY or previous is None:
                continue
            if previous[0] == 'ingredient' and target in BOARDS:
                k = BOARDS.index(target)
                if s['board'][k] != EMPTY:
                    claim(i, target, previous, ('board', k))
            elif previous[0] == 'cook':
                pot = previous[1]
                if s['mix'][pot] and s['heat'][pot] > 0:
                    claim(i, POTS[pot], previous)
        # Deliver or transfer cooked dishes first; never leave a ready meal stranded.
        for i in controlled:
            if i not in busy and hand[i] >= FIRST_MEAL:
                r = int(hand[i]) - FIRST_MEAL
                target = SERVERS[side[i]] if side[i] == DELIVERY_SIDE[r] and r in s['recipe'] else drop(i)
                claim(i, target, ('meal', r), ('hand', i))
        for k, item in enumerate(s['counter']):
            if item >= FIRST_MEAL:
                r = int(item) - FIRST_MEAL
                if r not in s['recipe']:
                    continue
                candidates = [i for i in controlled if i not in busy and hand[i] == EMPTY and side[i] == DELIVERY_SIDE[r]]
                if candidates:
                    i = min(candidates, key=lambda i: self._distance(int(pos[i]), WALL[k]))
                    claim(i, WALL[k], ('meal', r), ('counter', k))
        # Tend/pick up each complete pot with exactly one available chef.
        for pot in range(2):
            if recipes[pot] >= 0 and s['mix'][pot] == RECIPES[recipes[pot]] and POTS[pot] not in dest:
                candidates = [i for i in controlled if i not in busy and hand[i] == EMPTY and side[i] == pot]
                if candidates:
                    i = min(candidates, key=lambda i: self._distance(int(pos[i]), POTS[pot]))
                    claim(i, POTS[pot], ('cook', pot))
        tasks = [(pot, item) for pot in range(2) if recipes[pot] >= 0 and s['locked'][pot] < 0
                 for item, bit in BITS.items() if RECIPES[recipes[pot]] & bit and not s['mix'][pot] & bit]
        # Existing resources are assigned once. Persistent task owners break ties.
        resources = []
        for i, item in enumerate(hand):
            if 0 < item < FIRST_MEAL:
                resources.append((('hand', i), int(item), int(pos[i]), side[i]))
        for k, item in enumerate(s['board']):
            if item:
                resources.append((('board', k), int(item), BOARDS[k], k // 2))
        for k, item in enumerate(s['counter']):
            if 0 < item < FIRST_MEAL:
                resources.append((('counter', k), int(item), WALL[k], -1))
        allocated = {}
        resources.sort(key=lambda r: 0 if r[0][0] == 'counter' else 1 if r[0][0] == 'board' else 2)
        for resource, value, where, rs in resources:
            equivalent = CUT_TOMATO if value == TOMATO else CUT_POTATO if value == POTATO else value
            choices = [task for task in tasks if task not in allocated and task[1] == equivalent]
            if not choices:
                continue
            old = self.assignments[lane][resource[1]] if resource[0] == 'hand' else None
            task = min(choices, key=lambda t: (0 if old == ('ingredient', *t) else 1,
                         0 if rs == t[0] or (rs == -1 and t[0] == (1 if value in (TOMATO, POTATO, CUT_TOMATO, CUT_POTATO) else 0)) else 1, self._distance(where, POTS[t[0]])))
            allocated[task] = (resource, value, where, rs)
        # Produce tasks already carried, then preserve owners, then nearest work.
        pending = list(tasks)
        pending.sort(key=lambda t: (0 if t in allocated and allocated[t][0][0] == 'hand' else 1,
                      0 if ('ingredient', *t) in self.assignments[lane] else 1,
                      next((k for k, j in enumerate(rank) if s['ids'][j] == jobs[t[0]]), 3)))
        for pot, item in pending:
            task = ('ingredient', pot, item)
            raw = TOMATO if item == CUT_TOMATO else POTATO if item == CUT_POTATO else item
            if jobs[pot] < 0 and (pot, item) not in allocated:
                continue
            resource, value, where, rs = allocated.get((pot, item), (('source', raw), raw, SOURCES[raw], 0 if raw in (TOMATO, POTATO) else 1))
            if resource in claimed:
                continue
            if resource[0] == 'hand':
                candidates = [resource[1]] if resource[1] in controlled and resource[1] not in busy else []
            else:
                work_side = pot if rs == -1 else rs
                candidates = [i for i in controlled if i not in busy and hand[i] == EMPTY and side[i] == work_side]
            if not candidates:
                continue
            i = min(candidates, key=lambda i: (self.assignments[lane][i] != task, self._distance(int(pos[i]), where)))
            target = where
            if resource[0] == 'hand':
                if side[i] != pot:
                    target = drop(i)
                elif value in (TOMATO, POTATO):
                    target = nearest(i, [b for k, b in enumerate(BOARDS) if k // 2 == side[i]
                                            and s['board'][k] == EMPTY and ('board', k) not in claimed])
                else:
                    target = POTS[pot]
            if target and target not in dest:
                claim(i, target, task, resource)
        # Surplus items remain reusable on counters/boards; empty workers clear
        # occupied ports so their colleague can reach its assigned station.
        for i in controlled:
            if i in busy:
                continue
            if hand[i] != EMPTY:
                target = drop(i)
                if hand[i] in (TOMATO, POTATO):
                    target = nearest(i, [b for k, b in enumerate(BOARDS) if k // 2 == side[i] and s['board'][k] == EMPTY and b not in dest]) or target
                claim(i, target, ('stash', int(hand[i])))
            elif side[i] == 1:
                dest[i] = cell(11, 1 if i == int(self.chef_ids[lane, 1]) else 5)
        # An idle chef counts only when an unclaimed ingredient operation is
        # executable now: inventory, side, reservation and teammate blockage
        # must all permit a non-STAY first action. Parking is not work.
        self.eligible_work[lane] = False
        for i in controlled:
            if i in busy or hand[i] != EMPTY:
                continue
            other = next((j for j in range(3) if j != i and side[j] == side[i]), None)
            blocked = int(pos[other]) if other is not None else 0
            for pot, item in tasks:
                raw = TOMATO if item == CUT_TOMATO else POTATO if item == CUT_POTATO else item
                if (pot, item) not in allocated and jobs[pot] < 0:
                    continue
                resource, value, where, rs = allocated.get((pot, item),
                    (('source', raw), raw, SOURCES[raw], 0 if raw in (TOMATO, POTATO) else 1))
                if resource in claimed or resource[0] == 'hand' or where in dest:
                    continue
                if side[i] != (pot if rs == -1 else rs):
                    continue
                if self.paths[where, blocked, pos[i]] != STAY:
                    self.eligible_work[lane, i] = True
                    break
        self.assignments[lane] = assignment
        self.reservations[lane] = reservations
        return dest

    def actions(self):
        keys = self._batch_keys
        state = self.env.state
        packed = torch.cat([state[k] for k in keys], dim=1).detach().cpu().numpy()
        # Lane planning performs scalar branching; native ints avoid repeatedly
        # constructing NumPy scalars. Keep the large routing lookup vectorized.
        rows = packed.tolist()
        self._learner_cpu = self.chef_ids[self.rows, self.roles]
        for lane in self.rows:
            self.destinations[lane] = self._lane(lane, {
                k: rows[lane][part] for k, part in zip(keys, self._batch_slices)})
        pos, facing = packed[:, self._batch_slices[0]], packed[:, self._batch_slices[1]]
        blocked = np.zeros_like(pos)
        a, b = self.chef_ids[:, 1], self.chef_ids[:, 2]
        blocked[self.rows, a], blocked[self.rows, b] = pos[self.rows, b], pos[self.rows, a]
        moves = self.paths[self.destinations, blocked, pos].astype(np.int64)
        moves = np.where(moves >= 6, np.where(facing == moves - 6, INTERACT, moves - 6), moves)
        external = self.rows[self.learner_mask]
        moves[external, self._learner_cpu[external]] = STAY
        off = np.asarray((*OFFSETS, 0, 0))
        collision = (pos[self.rows, a] + off[moves[self.rows, a]] ==
                     pos[self.rows, b] + off[moves[self.rows, b]])
        moves[self.rows[collision], b[collision]] = STAY
        self.idle_actionable = self.eligible_work & (moves == STAY)
        return torch.as_tensor(moves, device=self.env.device)

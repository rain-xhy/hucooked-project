"""V3 HA² subtask dispatch and goal-conditioned movement on batched kitchens."""

from collections import deque

import numpy as np
import torch

from overcooked_ai_py.cooperative.partition.model import (
    BEEF, BOARDS, EGG, FLOOR, INTERACT, OFFSETS, POTATO, POTS, SERVERS,
    SOURCES, STAY, TOMATO, WALL, WIDTH, HEIGHT, cell,
)


CELLS = WIDTH * HEIGHT
PARK_HELPER, PARK_COOK = cell(11, 1), cell(11, 5)
FACILITIES = tuple(dict.fromkeys((*SOURCES.values(), *BOARDS, *POTS, *SERVERS,
                                 WALL[1], WALL[3], PARK_HELPER, PARK_COOK, WALL[2])))
LEGACY_GOALS = len(FACILITIES) - 1
GOAL_ID = {station: i for i, station in enumerate(FACILITIES)}
LEFT, COOK, HELPER = range(3)


def pass_station(layout, recipe, variant):
    if variant and layout == "counter":
        return WALL[2]
    upper = (recipe == 0) != bool(variant)
    return WALL[1 if upper else 3]


def programs(layout, parallel, variant=0):
    def job(role, station, steps=1):
        return role, station, steps

    def transfer(wall, giver, receiver):
        if layout == "sync":
            return [(True, (job(giver, wall), job(receiver, wall)))]
        return [(False, (job(giver, wall),)), (False, (job(receiver, wall),))]

    def recipe(which):
        rows = []
        def stage(*jobs):
            rows.append((False, jobs))
        def vegetable(item, board, pot):
            stage(job(LEFT, SOURCES[item]))
            stage(job(LEFT, board, 4))
            stage(job(LEFT, pot))

        if which == 0:
            extra = [job(HELPER, SOURCES[EGG])] if parallel else []
            stage(job(LEFT, SOURCES[TOMATO]), *extra)
            extra = [job(HELPER, POTS[1])] if parallel else []
            stage(job(LEFT, BOARDS[0], 4), *extra)
            rows.extend(transfer(pass_station(layout, which, variant), LEFT, COOK))
            stage(job(COOK, POTS[1]))
            if not parallel:
                stage(job(COOK, SOURCES[EGG]))
                stage(job(COOK, POTS[1]))
            stage(job(COOK, POTS[1], 5))
            stage(job(COOK, SERVERS[1]))
        else:
            if which == 2 and parallel:
                stage(job(LEFT, SOURCES[POTATO]), job(HELPER, SOURCES[BEEF]))
                stage(job(LEFT, BOARDS[1], 4))
                stage(job(LEFT, POTS[0]))
            else:
                vegetable(POTATO, BOARDS[1], POTS[0])
            if which == 2:
                vegetable(TOMATO, BOARDS[0], POTS[0])
                if not parallel:
                    stage(job(HELPER, SOURCES[BEEF]))
                rows.extend(transfer(pass_station(layout, which, variant), HELPER, LEFT))
                stage(job(LEFT, POTS[0]))
            stage(job(LEFT, POTS[0], 5))
            stage(job(LEFT, SERVERS[0]))
        return rows
    return [recipe(r) for r in range(3)]


def path_table():
    floors = sorted(FLOOR)
    neighbor = np.arange(CELLS)[:, None] + np.asarray(OFFSETS)
    floor_pairs = {p: [int(n) for n in neighbor[p] if n in FLOOR] for p in floors}
    ports = [[p for p in floors if goal in neighbor[p]] if goal not in FLOOR else [goal]
             for goal in FACILITIES]
    table = np.full((CELLS, CELLS, len(FACILITIES), CELLS), 127, dtype=np.uint8)
    for i, one in enumerate(floors):
        for two in floors[i + 1:]:
            blocked = {one, two}
            for index, targets in enumerate(ports):
                state = {p: 0 for p in targets if p not in blocked}
                queue = deque(state)
                while queue:
                    p = queue.popleft()
                    for n in floor_pairs[p]:
                        if n not in blocked and n not in state:
                            state[n] = state[p] + 1
                            queue.append(n)
                view = table[one, two, index]
                for p, dist in state.items():
                    view[p] = dist
                table[two, one, index] = view
    return table


class Dispatcher:
    def __init__(self, env, distances, parallel=True, *, deadline_aware=False, prestage=False, gate_choice=False, protect_supply=False):
        self.env = env
        self.deadline_aware = deadline_aware
        self.prestage = prestage and parallel
        self.protect_supply = protect_supply
        self.variants = 2 if gate_choice else 1
        self.recipe_action_width = 2 * self.variants
        self.wait_action = 3 * self.recipe_action_width
        self.device = env.device
        self.distances = distances
        self.rows = torch.arange(env.num_envs, device=self.device)
        self.roles = torch.arange(3, device=self.device)
        self.shifts = torch.tensor(OFFSETS, device=self.device)
        self.locations = torch.tensor(FACILITIES, device=self.device)
        self.idle = torch.tensor([-1, GOAL_ID[PARK_COOK], GOAL_ID[PARK_HELPER]], device=self.device)
        goal_features = len(FACILITIES) if gate_choice else LEGACY_GOALS
        self.goal_eye = torch.eye(len(FACILITIES), device=self.device)[:, :goal_features]
        self.role_eye = torch.eye(3, device=self.device)
        self._programs([programs(env.config.layout, parallel, variant)
                        for variant in range(self.variants)])
        self.gates = torch.tensor([[GOAL_ID[pass_station(env.config.layout, recipe, v)]
                                    for v in range(self.variants)] for recipe in (0, 1, 2)], device=self.device)
        service = (37, 27, 59) if env.config.layout == "sync" else (46, 27, 64)
        if prestage:
            service = (37, 27, 55) if env.config.layout == "sync" else (45, 27, 61)
        self.service_time = torch.tensor(service, device=self.device)
        self.action_count = self.wait_action + int(deadline_aware)
        self.chosen = torch.full((env.num_envs,), -1, device=self.device)
        self.phase = torch.zeros_like(self.chosen)
        self.variant = torch.zeros_like(self.chosen)
        self.progress = torch.zeros((env.num_envs, 3), dtype=torch.long, device=self.device)
        self.assignment = self.canonical()

    def _programs(self, steps):
        width = max(map(len, steps[0]))
        station = np.full((3, self.variants, width, 3), -1)
        counts = np.zeros_like(station)
        sync = np.zeros((3, self.variants, width), dtype=bool)
        for variant, recipes in enumerate(steps):
            for recipe, phases in enumerate(recipes):
                for phase, (paired, jobs) in enumerate(phases):
                    for role, target, count in jobs:
                        station[recipe, variant, phase, role] = GOAL_ID[target]
                        counts[recipe, variant, phase, role] = count
                    sync[recipe, variant, phase] = paired
        self.stations, self.counts, self.sync, self.lengths = [
            torch.as_tensor(values, device=self.device)
            for values in (station, counts, sync, [len(recipe) for recipe in steps[0]])
        ]

    def canonical(self):
        pos = self.env.state["pos"]
        left = pos % WIDTH < 6
        left = left.long().argmax(dim=1)
        right = torch.where(pos % WIDTH > 6, pos, CELLS + pos).argsort(dim=1)[:, :2]
        return torch.cat((left[:, None], right), dim=1)

    def manager_state(self, obs):
        live = self.env.state
        team = self.canonical()
        features = obs[:, 0, :-1].clone()
        for offset in (2, 5, 8):
            features[:, offset:offset + 3] = features[:, offset:offset + 3].gather(1, team)
        valid = (live["recipe"][:, :, None] == self.roles).any(dim=1)
        deadline = (live["due"] - live["tick"])[:, :, None]
        slack = torch.where(live["recipe"][:, :, None] == self.roles, deadline, -1).amax(dim=1)
        if self.deadline_aware:
            valid &= slack >= self.service_time
        mask = valid.repeat_interleave(self.recipe_action_width, dim=1)
        if self.deadline_aware:
            mask = torch.cat((mask, ~valid.any(dim=1, keepdim=True)), dim=1)
        return torch.cat((features, slack / max(self.env.config.deadlines)), dim=1), mask, slack

    def start(self, decisions):
        pending = (self.chosen < 0) & (decisions < self.wait_action)
        chosen = decisions // self.recipe_action_width
        variant = (decisions // 2) % self.variants
        team = self.canonical()
        flip = decisions % 2 == 1
        assignment = torch.stack((team[:, 0], torch.where(flip, team[:, 2], team[:, 1]),
                                  torch.where(flip, team[:, 1], team[:, 2])), dim=1)
        self.assignment = torch.where(pending[:, None], assignment, self.assignment)
        self.chosen = torch.where(pending, chosen, self.chosen)
        self.variant = torch.where(pending, variant, self.variant)

    def _mask(self, role, chef, destination, enabled, reserved):
        s = self.env.state
        p = s["pos"][self.rows, chef]
        face = s["facing"][self.rows, chef]
        one = s["pos"][self.rows, self.assignment[:, (role + 1) % 3]]
        two = s["pos"][self.rows, self.assignment[:, (role + 2) % 3]]
        safe_destination = destination.clamp_min(0)
        reach = self.distances[one, two, safe_destination]
        current = reach[self.rows, p]
        next_position = (p[:, None] + self.shifts).clamp(0, CELLS - 1)
        after = reach.gather(1, next_position)
        moves = (after.long() + 1 == current[:, None]) & (current[:, None] != 127)
        awaiting = current == 0
        at_facility = self.locations[safe_destination][:, None] == next_position
        turn = awaiting[:, None] & at_facility
        turn &= torch.arange(4, device=self.device) != face[:, None]
        blocked = reserved.gather(1, next_position)
        moves = enabled[:, None] & ((moves & ~blocked) | turn)
        interact = enabled & awaiting & at_facility.gather(1, face[:, None]).squeeze(1)
        return moves, interact, next_position

    def propose(self, obs, worker=None, examples=None):
        target = self.stations[self.chosen, self.variant, self.phase]
        executing = self.chosen >= 0
        counts = self.counts[self.chosen, self.variant, self.phase] * executing[:, None]
        active = self.progress < counts
        jobs = torch.where(active, target, self.idle)
        jobs = torch.where((counts > 0) & ~active, -1, jobs)
        if self.prestage:
            stage_receiver = (self.chosen == 0) & (self.phase < 2) & ~active[:, COOK]
            if self.protect_supply:
                stage_receiver &= self.phase > 0
            stage_beef = (self.chosen == 2) & (self.phase > 0) & (self.phase < 6) & ~active[:, HELPER]
            if self.env.config.layout == "sync":
                jobs[:, COOK] = torch.where(stage_receiver, self.gates[0, self.variant], jobs[:, COOK])
            jobs[:, HELPER] = torch.where(stage_beef, self.gates[2, self.variant], jobs[:, HELPER])
        paired = self.sync[self.chosen, self.variant, self.phase]
        s = self.env.state
        chef_position = s["pos"].gather(1, self.assignment)
        chef_facing = s["facing"].gather(1, self.assignment)
        pretarget = chef_position + self.shifts[chef_facing]
        ready = (pretarget == self.locations[target.clamp_min(0)]) & active
        whole_pair = (ready | ~active).all(dim=1)
        deliver = active & (self.phase == (self.lengths[self.chosen] - 1))[:, None]
        order_exists = (s["recipe"] == self.chosen[:, None]).any(dim=1)
        actions = torch.full_like(self.progress, STAY)
        reservations = torch.zeros((self.env.num_envs, CELLS), dtype=torch.bool, device=self.device)
        # Plan work before the left chef, whose paths cannot visit the right half.
        for role in (COOK, HELPER, LEFT):
            index = self.assignment[:, role]
            job = jobs[:, role]
            moves, interact, next_pos = self._mask(role, index, job, job >= 0, reservations)
            interact &= active[:, role] & (~paired | whole_pair) & (~deliver[:, role] | order_exists)
            mask = torch.cat((moves, (~moves.any(dim=1) & ~interact)[:, None], interact[:, None]), dim=1)
            inputs = torch.cat((obs[self.rows, index], self.goal_eye[job.clamp_min(0)],
                                self.role_eye[role].expand(len(index), -1)), dim=1)
            labels = mask.long().argmax(dim=1)
            if examples is not None:
                examples.append((inputs.detach(), mask.detach().clone(), labels.detach()))
            decision = labels if worker is None else worker(inputs).masked_fill(~mask, -1e9).argmax(dim=1)
            actions[self.rows, index] = decision
            dest = next_pos[self.rows, decision.clamp_max(3)]
            reservations[self.rows, dest] |= (decision < STAY) & self.env.floor[dest]
        return actions

    def advance(self, actions, info):
        executing = self.chosen >= 0
        counts = self.counts[self.chosen, self.variant, self.phase] * executing[:, None]
        active = self.progress < counts
        presses = (actions.gather(1, self.assignment) == INTERACT) & active
        last_phase = self.phase == self.lengths[self.chosen] - 1
        accepted = info["delivered"].any(dim=1)
        presses &= (~last_phase | accepted)[:, None]
        self.progress += presses
        finished = executing & (self.progress >= counts).all(dim=1)
        self.phase += finished
        self.progress.masked_fill_(finished[:, None], 0)
        completed = finished & last_phase
        self.phase.masked_fill_(completed, 0)
        self.chosen.masked_fill_(completed, -1)

    def teacher(self, mask, slack):
        # Approximate service times of the three parallel recipes in a clean kitchen.
        duration = self.service_time if self.deadline_aware else torch.tensor([39, 28, 53], device=self.device)
        delay = (duration - slack).clamp_min(0)
        scores = -duration - 1.6 * delay
        team = self.canonical()
        pos = self.env.state["pos"].gather(1, team)
        x, y = pos % WIDTH, pos // WIDTH
        to_synch = (x - 7).abs() + (y - 2).abs()
        to_egg = (x - 11).abs() + (y - 2).abs()
        to_beef = (x - 11).abs() + (y - 4).abs()
        burden = torch.stack((
            torch.stack((to_synch[:, 1] + to_egg[:, 2], to_synch[:, 2] + to_egg[:, 1]), dim=1),
            torch.zeros_like(to_synch[:, 1:]),
            torch.stack((to_beef[:, 2], to_beef[:, 1]), dim=1),
        ), dim=1)
        logits = scores[:, :, None, None] - 0.35 * burden[:, :, None, :]
        if self.variants == 2:
            alternate = torch.tensor([-1.0, -1.0, -1.0] if self.env.config.layout == "sync" else
                                     [2.0, -1.0, 3.0], device=self.device)
            adjustment = torch.stack((torch.zeros_like(alternate), alternate), dim=1)
            logits = logits + adjustment[None, :, :, None]
        logits = logits.flatten(1)
        if self.deadline_aware:
            logits = torch.cat((logits, torch.zeros_like(logits[:, :1])), dim=1)
        return logits.masked_fill(~mask, -1e9).argmax(dim=1)

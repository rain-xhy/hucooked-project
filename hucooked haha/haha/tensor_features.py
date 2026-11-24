"""Batched BC observations, teammate inference, and HA² task availability."""

import numpy as np
import torch
from torch.nn.functional import one_hot

from overcooked_ai_py.mdp.overcooked_mdp import Direction

from .bc import BCPartner
from .encoding import make_planner
from .subtasks import UNKNOWN
from .tensor_kitchen import COUNTER, DISH, DISHES, EMPTY, FLOOR, ONION, ONIONS, POT, SERVE, SOUP, STAY

UNREACHABLE = 1 << 24


class TensorFeatures:
    def __init__(self, kitchen):
        self.kitchen = kitchen
        cells = kitchen.num_cells
        distances = np.full((len(kitchen.mdps), cells, 4, cells), UNREACHABLE, dtype=np.int64)
        accessible = np.zeros((len(kitchen.mdps), 2, cells), dtype=bool)
        for index, mdp in enumerate(kitchen.mdps.values()):
            planner = make_planner(mdp)
            motion = planner.motion_planner
            for position in mdp.get_valid_player_positions():
                for direction, orientation in enumerate(Direction.ALL_DIRECTIONS):
                    for target in motion.motion_goals_for_pos:
                        distance = motion.min_cost_to_feature((position, orientation), (target,))
                        if np.isfinite(distance):
                            distances[index, kitchen._cell(position), direction, kitchen._cell(target)] = distance
            start = mdp.get_standard_start_state()
            for player in (0, 1):
                counters = mdp.find_free_counters_valid_for_player(start, planner, player)
                accessible[index, player, [kitchen._cell(position) for position in counters]] = True
        self.distances = kitchen._tensor(distances)
        self.accessible = torch.as_tensor(accessible, device=kitchen.device)[kitchen.layout_ids]
        self.forced = torch.tensor([name == "forced_coordination" for name in kitchen.layouts],
                                   device=kitchen.device)
        self.coordinates = torch.stack((kitchen.cells % kitchen.width, kitchen.cells // kitchen.width), dim=-1)
        self.requested_objects = kitchen._tensor([ONION, -1, DISH, SOUP, -1, -1])
        self.held_codes = kitchen._tensor([ONION, SOUP, DISH, -1])

    def compile_kernels(self):
        self.kitchen.step = torch.compile(self.kitchen.step, fullgraph=True)
        self.kitchen.observation = torch.compile(self.kitchen.observation, fullgraph=True)
        self.bc_observation = torch.compile(self.bc_observation, fullgraph=True)
        self.skill_mask = torch.compile(self.skill_mask, fullgraph=True)

    def _closest(self, distances, candidates, order):
        """Resolve equal distances in native feature-list order."""
        distance = torch.where(candidates, distances, UNREACHABLE)
        minimum = distance.amin(-1, keepdim=True)
        ties = (distance == minimum) & candidates & (distance < UNREACHABLE)
        index = torch.where(ties, order, torch.iinfo(torch.long).max).argmin(-1)
        return index, minimum.squeeze(-1) < UNREACHABLE

    def bc_observation(self, state):
        """Return the two native player perspectives, each with 96 features."""
        kitchen = self.kitchen
        terrain, cells = kitchen.terrain, kitchen.cells
        kind, onions, ticks = state.objects.unbind(-1)
        held, held_onions, _ = state.held.unbind(-1)
        position = self.coordinates[state.position]
        distance = self.distances[kitchen.layout_ids[:, None], state.position, state.orientation]
        counter = terrain == COUNTER
        loose_onion = counter & (kind == ONION)
        loose_dish = counter & (kind == DISH)
        loose_soup = counter & (kind == SOUP)
        zeros = torch.zeros_like(counter)
        dispensers = torch.stack((terrain == ONIONS, zeros, terrain == DISHES), dim=1)
        loose = torch.stack((loose_onion, zeros, loose_dish), dim=1)
        candidates = torch.cat((dispensers | loose,
                                torch.stack((loose_soup, terrain == SERVE, counter & (kind == EMPTY)), 1)), dim=1)
        # Native dispensers precede counter objects. Counter objects preserve
        # insertion order when the planner breaks a tie.
        loose_order = state.object_order + 2 * kitchen.num_cells
        order = torch.cat((torch.where(dispensers, cells, loose_order[:, None]),
                           loose_order[:, None], cells.expand(kitchen.batch_size, 2, -1)), dim=1)
        nearest, reachable = self._closest(distance[:, :, None], candidates[:, None], order[:, None])
        carrying = held[:, :, None] == self.requested_objects
        delta = self.coordinates[nearest] - position[:, :, None]
        delta = torch.where((reachable & ~carrying)[..., None], delta, 0)
        soup_count = onions.gather(1, nearest[:, :, 3])
        soup_count = torch.where(reachable[:, :, 3], soup_count, 0)
        soup_count = torch.where(held == SOUP, held_onions, soup_count)
        no_tomatoes = torch.zeros_like(soup_count)

        pot_distance = torch.where(terrain[:, None] == POT, distance, UNREACHABLE)
        pot_priority = pot_distance * kitchen.num_cells + cells
        pots = pot_priority.topk(2, dim=-1, largest=False, sorted=True).indices
        exists = pot_distance.gather(2, pots) < UNREACHABLE
        row = kitchen.rows[:, None, None]
        pot_kind, pot_onions, pot_ticks = state.objects[row, pots].unbind(-1)
        empty = pot_kind == EMPTY
        full = (pot_kind == SOUP) & ((pot_onions == 3) | (pot_ticks >= 0))
        ready = (pot_kind == SOUP) & (pot_ticks >= kitchen.cook_time[:, None, None])
        cooking = (pot_kind == SOUP) & (pot_ticks >= 0) & ~ready
        remaining = torch.where(cooking, kitchen.cook_time[:, None, None] - pot_ticks, 0)
        pot_delta = self.coordinates[pots] - position[:, :, None]
        pot_features = torch.cat((
            torch.stack((exists, empty, full, cooking, ready, pot_onions,
                         torch.zeros_like(pot_onions), remaining), dim=-1),
            pot_delta,
        ), dim=-1)
        pot_features = torch.where(exists[..., None], pot_features, 0).flatten(2)
        adjacent = kitchen.front[state.position]
        walls = terrain[kitchen.rows[:, None, None], adjacent] != FLOOR
        personal = torch.cat((
            one_hot(state.orientation, 4), held[..., None] == self.held_codes, delta[:, :, :4].flatten(2),
            torch.stack((soup_count, no_tomatoes), dim=-1), delta[:, :, 4:].flatten(2), pot_features, walls,
        ), dim=-1)
        return torch.cat((personal, personal.flip(1), position.flip(1) - position, position), dim=-1).float()

    def skill_mask(self, state, players, previous):
        """Original task mask for each controlled player and previous subtask."""
        kitchen = self.kitchen
        held = state.held[kitchen.rows, players, 0]
        kind, onions, ticks = state.objects.unbind(-1)
        loose = (kitchen.terrain == COUNTER) & (kind != EMPTY)
        counter = loose & self.accessible[kitchen.rows, players]
        below_limit = loose.sum(-1) < 2
        source_side = ~(self.forced & (players == 0))
        pot_side = ~(self.forced & (players == 1))
        pot = kitchen.terrain == POT
        full = (pot & (kind == SOUP) & ((onions == 3) | (ticks >= 0))).sum(-1)
        put_onion = pot_side & (full < pot.sum(-1))
        get_soup = pot_side & (pot & (kind == SOUP) & (ticks >= 0)).any(-1)
        empty = held == EMPTY
        return torch.stack((
            empty & source_side,
            empty & (previous != 3) & (counter & (kind == ONION)).any(-1),
            (held == ONION) & put_onion,
            (held == ONION) & (previous != 1) & (below_limit | ~put_onion),
            empty & source_side,
            empty & (previous != 6) & (counter & (kind == DISH)).any(-1),
            (held == DISH) & (previous != 5) & (below_limit | ~get_soup),
            (held == DISH) & get_soup,
            empty & (previous != 9) & (counter & (kind == SOUP)).any(-1),
            (held == SOUP) & (below_limit | ~pot_side),
            (held == SOUP) & pot_side,
            torch.ones_like(empty),
        ), dim=-1)


class TensorBCPartner:
    """Run each map's frozen BC network once per batch on the kitchen device."""

    def __init__(self, features, weights):
        self.features = features
        kitchen = features.kitchen
        self.groups = [
            (kitchen._tensor([i for i, value in enumerate(kitchen.layouts) if value == layout]),
             BCPartner.load(weights[layout], kitchen.device).network)
            for layout in kitchen.mdps
        ]

    @torch.no_grad()
    def probabilities(self, state, players):
        kitchen = self.features.kitchen
        observation = self.features.bc_observation(state)[kitchen.rows, players]
        probabilities = torch.empty((kitchen.batch_size, 6), device=kitchen.device)
        for rows, network in self.groups:
            probabilities[rows] = network(observation[rows]).softmax(-1)
        return probabilities

    @torch.no_grad()
    def act(self, state, players, *, generator=None):
        return torch.multinomial(self.probabilities(state, players), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def manager_rollout_step(features, partner, state, players, previous, manager, worker,
                         *, deterministic=False):
    """One evaluation step with CUDA-resident Manager, Worker, and partner inference.

    The caller manages episode resets. This path returns native sparse rewards
    and does not yet implement the Manager training penalties or PPO collector.
    """
    kitchen = features.kitchen
    rows = kitchen.rows
    mask = features.skill_mask(state, players, previous[rows, players])
    tasks, _, _ = manager(
        {"visual_obs": kitchen.observation(state, players), "subtask_mask": mask},
        action_masks=mask, deterministic=deterministic,
    )
    result, previous, _ = manager_task_step(features, partner, state, players, previous,
                                            tasks, worker, deterministic=deterministic)
    return result, previous


@torch.no_grad()
def manager_task_step(features, partner, state, players, previous, tasks, worker,
                      *, deterministic=False, generator=None):
    kitchen = features.kitchen
    rows = kitchen.rows
    actions, _, _ = worker(
        {"visual_obs": kitchen.observation(state, players, tasks)}, deterministic=deterministic,
    )
    unknown = tasks == UNKNOWN
    actions = torch.where(unknown, STAY, actions)
    previous = previous.clone()
    previous[rows, players] = torch.where(unknown, UNKNOWN, previous[rows, players])
    return joint_step(features, partner, state, players, previous, actions, generator=generator)


def joint_step(features, partner, state, players, previous, actions, *, generator):
    teammates = partner.act(state, 1 - players, generator=generator)
    joint = torch.stack((torch.where(players == 0, actions, teammates),
                         torch.where(players == 1, actions, teammates)), dim=1)
    result = features.kitchen.step(state, joint)
    return result, torch.where(result.completed >= 0, result.completed, previous), joint

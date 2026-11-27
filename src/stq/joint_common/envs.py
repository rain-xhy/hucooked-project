"""Unchanged native tasks, with explicit training reward profiles."""
from collections import deque
from functools import lru_cache
import multiprocessing as mp
import traceback

import numpy as np

from baseline_common.envs import (
    ACTION_DIM, ENVIRONMENT_COMMIT, HEIGHT, NUM_AGENTS, OBS_DIM, SHAPING_SCALE,
    SPATIAL_CHANNELS, STATE_DIM, WIDTH, KitchenBatch as NativeBatch,
    KitchenSlot as NativeSlot, curriculum, encode, potential, task_config,
)
from overcooked_ai_py.cooperative.shared.layouts import (
    ASSEMBLY_PORTS, COOKER_PORTS, FLOOR, STATIONS,
)

REWARD_PROFILES = {
    'legacy': {'description': 'native reward + existing .2 discounted inventory potential'},
    'ppo_task': {
        'native_scale': .05, 'inventory_weight': .5, 'navigation_weight': .3,
        'load': 1., 'main_cooked': .5, 'unload': 1.5, 'joint_unload': 3.,
        'place_cutting': .25, 'chop_complete': .75, 'take_cutting': .25,
        'place_side': .75, 'place_plate': .5, 'assemble': 2., 'take_meal': .5,
        'joint_serve': 2., 'joint_wait_cost': .002, 'conflict_cost': .002,
        'description': 'native score plus bounded once-per-order/item task milestones and discounted public-state progress; all evaluation is native',
    },
}


def _ports(station):
    x, y = STATIONS[station]
    return tuple(point for point in ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
                 if point in FLOOR)


@lru_cache(maxsize=128)
def _distances(goals):
    distances = {point: 0 for point in goals}
    queue = deque(goals)
    while queue:
        x, y = queue.popleft()
        for point in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if point in FLOOR and point not in distances:
                distances[point] = distances[(x, y)] + 1
                queue.append(point)
    return distances


def navigation_potential(core):
    """Distance to currently useful facilities; no action or fixed-role oracle."""
    state = core.state
    table = state.assembly
    held_kinds = {chef.held.kind for chef in state.chefs if chef.held is not None}
    empty_goals = set()
    if state.cooker.item is not None and state.cooker.remaining == 0 and table.main is None:
        empty_goals.update(COOKER_PORTS)
    if state.cutting.item is not None:
        empty_goals.update(_ports('K'))
    if (table.plate is not None and table.plate.kind == 'meal') or (
            table.main is not None and table.side is not None and table.plate is not None):
        empty_goals.update(ASSEMBLY_PORTS)
    if state.cooker.item is None and 'raw_main' not in held_kinds:
        empty_goals.update(_ports('M'))
    if (table.side is None and state.cutting.item is None
            and not held_kinds.intersection({'raw_side', 'chopped_side'})):
        empty_goals.update(_ports('L'))
    if table.plate is None and not held_kinds.intersection({'plate', 'meal'}):
        empty_goals.update(_ports('D'))
    if state.buffer is not None and not empty_goals:
        empty_goals.update(_ports('B'))
    value = 0.
    for chef in state.chefs:
        if chef.position is None:
            continue
        item = chef.held
        if item is None:
            goals = empty_goals
        elif item.kind == 'meal':
            goals = _ports('S')
        elif item.kind == 'raw_main':
            goals = COOKER_PORTS if state.cooker.item is None else _ports('B' if state.buffer is None else 'M')
        elif item.kind == 'raw_side':
            goals = _ports('K') if state.cutting.item is None else _ports('B' if state.buffer is None else 'L')
        elif item.kind == 'chopped_side':
            goals = ASSEMBLY_PORTS if table.side is None else _ports('B' if state.buffer is None else 'L')
        else:
            goals = ASSEMBLY_PORTS if table.plate is None else _ports('B' if state.buffer is None else 'D')
        if goals:
            distance = _distances(tuple(sorted(goals))).get(chef.position)
            if distance is not None:
                value += 1. / (1. + distance)
    return value


class TaskMilestones:
    """Credit useful irreversible stages once; preparatory credits have quotas."""
    def __init__(self):
        self.credited = {}

    def award(self, stage, key, amount, *, quota=None):
        seen = self.credited.setdefault(stage, set())
        if key in seen or (quota is not None and len(seen) >= quota):
            return 0.
        seen.add(key)
        return amount

    def reward(self, old, new, events, config):
        weights = REWARD_PROFILES['ppo_task']
        completed = new.metrics['normal_completed'] + new.metrics['joint_completed']
        quota = completed + config.active_limit
        total = 0.
        for event in events:
            kind = event['type']
            order = event.get('order_id')
            units = tuple(event.get('units', ()))
            actors = event.get('actors', ())
            held = old.chefs[actors[0]].held if actors else None
            if kind in ('load', 'main_cooked', 'assemble', 'take_meal'):
                total += self.award(kind, order, weights[kind])
            elif kind == 'unload':
                joint = old.orders[order].kind == 'joint'
                total += self.award(kind, order, weights['joint_unload' if joint else 'unload'])
            elif kind == 'serve' and old.orders[order].kind == 'joint':
                total += self.award(kind, order, weights['joint_serve'])
            elif kind == 'place_cutting' and held is not None:
                total += self.award(kind, tuple(held.units), weights[kind], quota=quota)
            elif kind == 'chop' and event.get('progress') == config.chop_steps:
                total += self.award('chop_complete', units, weights['chop_complete'], quota=quota)
            elif kind == 'take_cutting' and old.cutting.item is not None:
                total += self.award(kind, tuple(old.cutting.item.units), weights[kind], quota=quota)
            elif kind == 'place_assembly' and held is not None:
                # Replacing an already assembled meal never earns another plate credit.
                stage = {'chopped_side': 'place_side', 'plate': 'place_plate'}.get(held.kind)
                if stage:
                    total += self.award(stage, tuple(held.units), weights[stage], quota=quota)
        return total


class JointKitchenSlot(NativeSlot):
    def __init__(self, seed, index, stage, train, gamma, reward_profile):
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f'Unknown reward profile: {reward_profile}')
        self.reward_profile = reward_profile
        super().__init__(seed, index, stage, train, gamma)

    def reset(self):
        observation = super().reset()
        self.milestones = TaskMilestones()
        self.native_training_total = 0.
        self.component_totals = {}
        return observation

    def reset(self):
        result = super().reset()
        self.milestones = TaskMilestones()
        self.native_training_total = 0.
        self.component_totals = {}
        return result

    def step(self, actions):
        if self.reward_profile == 'legacy' or not self.train:
            return super().step(actions)
        weights = REWARD_PROFILES['ppo_task']
        old = self.core.state
        old_inventory = potential(self.core)
        old_navigation = navigation_potential(self.core)
        _, raw, terminated, truncated, info = self.core.step(tuple(int(a) for a in actions))
        done = bool(terminated or truncated)
        new = self.core.state
        inventory = weights['inventory_weight'] * (
            self.gamma * (0. if done else potential(self.core)) - old_inventory)
        navigation = weights['navigation_weight'] * (
            self.gamma * (0. if done else navigation_potential(self.core)) - old_navigation)
        stage = self.milestones.reward(old, new, info['events'], self.core.config)
        waiting = bool(old.cooker.item is not None and old.cooker.remaining == 0
                       and old.assembly.main is None
                       and old.orders[old.cooker.item.order_id].kind == 'joint'
                       and not any(event['type'] == 'unload' for event in info['events']))
        conflicts = sum(new.metrics[key] - old.metrics[key]
                        for key in ('movement_conflicts', 'resource_conflicts'))
        scale = weights['native_scale']
        components = {
            'native': float(raw) * scale,
            'inventory': inventory * scale, 'navigation': navigation * scale,
            'stage': stage * scale,
            'joint_wait': -float(waiting) * weights['joint_wait_cost'] * scale,
            'conflicts': -conflicts * weights['conflict_cost'] * scale,
        }
        reward = sum(components.values())
        shaping = reward - components['native']
        self.reward_total += reward
        self.shaping_total += shaping
        self.native_training_total += components['native']
        for key, value in components.items():
            self.component_totals[key] = self.component_totals.get(key, 0.) + value
        info = dict(info)
        info['training_shaping'] = shaping
        info['task_reward_components'] = components
        if done:
            summary = dict(info['metrics'])
            summary.update(episode=self.episode, training_reward=self.reward_total,
                           training_shaping=self.shaping_total,
                           training_native_reward=self.native_training_total)
            summary.update({f'reward_{key}': value for key, value in self.component_totals.items()})
            info['episode_summary'] = summary
            observation = self.reset()
        else:
            observation = encode(self.core)
        return observation, float(reward), done, info


def _joint_worker(conn, indices, seed, stage, train, gamma, reward_profile):
    slots = [JointKitchenSlot(seed, index, stage, train, gamma, reward_profile)
             for index in indices]
    try:
        while True:
            command, payload = conn.recv()
            if command == 'close':
                break
            if command == 'reset':
                conn.send(('ok', [slot.reset() for slot in slots]))
            elif command == 'stage':
                for slot in slots:
                    slot.stage = payload
                conn.send(('ok', None))
            elif command == 'step':
                conn.send(('ok', [slot.step(action) for slot, action in zip(slots, payload)]))
            else:
                raise ValueError(command)
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


class KitchenBatch(NativeBatch):
    def __init__(self, num_envs=8, seed=1, stage='standard', workers=0,
                 train=True, gamma=.9985, reward_profile='legacy'):
        task_config(stage, gamma, train)
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f'Unknown reward profile: {reward_profile}')
        self.num_envs, self.stage = int(num_envs), stage
        self.reward_profile = reward_profile
        self.workers, self.slots = [], []
        self.closed = False
        if workers:
            context = mp.get_context('spawn')
            for indices in np.array_split(np.arange(self.num_envs), min(workers, self.num_envs)):
                indices = indices.tolist()
                parent, child = context.Pipe()
                process = context.Process(
                    target=_joint_worker,
                    args=(child, indices, seed, stage, train, gamma, reward_profile),
                    daemon=True,
                )
                process.start()
                child.close()
                self.workers.append((indices, parent, process))
        else:
            self.slots = [JointKitchenSlot(seed, index, stage, train, gamma, reward_profile)
                          for index in range(self.num_envs)]

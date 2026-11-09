"""Check batched CUDA rule-partner actions against the original CPU policy."""

import argparse
from collections import Counter
from dataclasses import dataclass
import os
import random
import sys
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import torch

from check_gpu_kitchen import INTERACT, LEFT, RIGHT, WAIT, approach, capture_layout, salad_actions
from overcooked_setup import CONFIG, RECIPES, ROOT, activate_upstream, rule_partner
from torch_kitchen import TorchKitchen


class Decision(NamedTuple):
    draw: float
    action: int
    event: int
    target: int
    joint: tuple[int, int]


@dataclass
class Episode:
    seed: int
    layout: tuple
    decisions: list[Decision]


@dataclass
class Trace:
    scenario: str
    recipe: str
    weights: np.ndarray
    episodes: list[Episode]


def cpu_action(partner, world, draw):
    def choose(count, *, p):
        cdf = p.cumsum()
        return int(cdf.searchsorted(draw * cdf[-1], side="right"))

    with patch("numpy.random.choice", side_effect=choose):
        return partner(world)


def event_key(event):
    return event.action, event.dynamic_obj, event.static_obj


def cpu_reference(player, random_episodes):
    from environment.overcooked.overcooked_maker import OvercookedMaker
    from gym_cooking import Arrdict

    random.seed(0)
    env = OvercookedMaker.make_env_from_config(CONFIG)
    core = env.env._env.unwrapped
    rng = np.random.default_rng(1234)
    coverage = Counter()
    traces = []

    def record(scenario, recipe, seeds, left_first=None):
        partner = rule_partner(recipe)
        partner.set_id(player)
        left_food, right_food = RECIPES[recipe]
        support = [event_key(event) for event in partner.support_set]
        weights = rng.uniform(1, 2, size=len(support))
        if scenario == "retained-target":
            food = RECIPES[recipe][player]
            weights[support.index(("Put", food, "Divider"))] = 100
        partner.event_probs = weights / weights.sum()
        trace = Trace(scenario, recipe, weights, [])
        traces.append(trace)
        for episode, episode_seed in enumerate(seeds):
            random.seed(episode_seed)
            env.reset()
            partner.reset()
            world = core.world
            layout = capture_layout(world)
            action_rng = random.Random(episode_seed)
            plan = None
            if left_first is not None:
                plan = salad_actions(world, left_food, right_food, left_first)
            elif scenario == "retained-target":
                food_row = world.world_objects[RECIPES[recipe][player]][0].location[1]
                shelf = (1, food_row, LEFT) if player == 0 else (5, food_row, RIGHT)
                movements = [*approach(world.agents[player].location, shelf), INTERACT]
                plan = [(act, WAIT) if player == 0 else (WAIT, act) for act in movements]
                plan += [(WAIT, WAIT)] * (env.horizon - len(plan))

            decisions = []
            for step in range(env.horizon):
                draw = 0.5 if scenario == "retained-target" else rng.random()
                previous = partner.cur_event
                action = cpu_action(partner, world, draw)
                event = partner.cur_event
                if event is not None and event.target_location is not None:
                    event_id = support.index(event_key(event))
                    x, y = event.target_location
                    target = y * world.width + x
                    coverage[event_key(event)] += 1
                else:
                    event_id = target = -1
                if episode and step == 0 and event is previous and target >= 0:
                    coverage["retained targets"] += 1
                    if scenario == "retained-target":
                        fresh = rule_partner(recipe)
                        fresh.set_id(player)
                        fresh.event_probs = partner.event_probs
                        if cpu_action(fresh, world, draw) != action:
                            coverage["retention changes action"] += 1
                joint = plan[step] if plan is not None else tuple(
                    action_rng.choices(range(6), weights=(1, 2, 2, 2, 2, 4), k=2)
                )
                decisions.append(Decision(draw, action, event_id, target, joint))
                data, info = env.step(Arrdict(
                    (name, Arrdict(action=act)) for name, act in zip(env.players, joint, strict=True)
                ))
                if left_first is not None:
                    for food in (left_food, right_food):
                        obj = world.world_objects[food][0]
                        if obj.done() and any(obj.location == board.location for board in world.world_objects["CutBoard"]):
                            coverage["chopped on board"] += 1
                    plate = world.world_objects["Plate"][0]
                    coverage[f"plate holds {len(plate.content)}"] += 1
                if data[env.players[0]].done:
                    break

            if left_first is not None:
                termination = info[env.players[0]]["termination_info"]
                assert termination == f"Terminating because {left_food}{right_food}Salad is completed", (
                    scenario, recipe, episode_seed, termination
                )
                coverage[f"delivered {recipe}"] += 1
            trace.episodes.append(Episode(episode_seed, layout, decisions))

    try:
        # On these layouts the held food leaves a valid divider target at
        # timeout. After reset, that target sends the chef in a different
        # direction than a new policy choosing from the reset kitchen.
        retention_reset_seeds = ((2, 1, 1), (1, 3, 5))[player]
        for index, recipe in enumerate(RECIPES):
            seed = 11 + 4 * index
            record("left-first", recipe, (seed, seed + 1), left_first=True)
            record("right-first", recipe, (seed + 2, seed + 3), left_first=False)
            record("retained-target", recipe, (30 + 2 * index, retention_reset_seeds[index]))
        for index in range(random_episodes):
            recipe = tuple(RECIPES)[index % len(RECIPES)]
            seed = 1000 + 2 * index
            record("random", recipe, (seed, seed + 1))
    finally:
        env.env._env.close()

    for required in ("chopped on board", "plate holds 1", "plate holds 2"):
        assert coverage[required], f"CPU reference for chef {player} did not cover {required}"
    assert coverage["retention changes action"] == len(RECIPES), f"CPU reset fixtures for chef {player}"
    return traces, coverage


@torch.inference_mode()
def replay(traces, player, device):
    from cuda_l0 import CudaRulePartner

    size = len(traces)
    kitchen = TorchKitchen(size, device)
    policy = CudaRulePartner(
        kitchen, [trace.recipe for trace in traces], player=player,
        event_weights=np.stack([trace.weights for trace in traces]),
    )
    episodes = [0] * size
    offsets = [0] * size

    def reset(mask=None):
        layouts = [trace.episodes[min(episode, 1)].layout for trace, episode in zip(traces, episodes, strict=True)]
        layout = tuple(torch.tensor(part, dtype=torch.long, device=device) for part in zip(*layouts, strict=True))
        kitchen.reset(mask=mask, layout=layout)

    reset()
    while active := [lane for lane, episode in enumerate(episodes) if episode < 2]:
        draws = np.zeros(size)
        joint = np.zeros((size, 2), dtype=np.int64)
        for lane in active:
            decision = traces[lane].episodes[episodes[lane]].decisions[offsets[lane]]
            draws[lane] = decision.draw
            joint[lane] = decision.joint

        actions = policy.act(torch.as_tensor(draws, device=device))
        assert (actions.shape, actions.dtype, actions.device.type) == ((size,), torch.long, device)
        observed = torch.stack((actions, policy.current_event, policy.current_target), dim=1).tolist()
        for lane in active:
            trace, step = traces[lane], offsets[lane]
            episode = trace.episodes[episodes[lane]]
            decision = episode.decisions[step]
            action, event, target = observed[lane]
            # The original policy briefly records targetless follow-up events.
            # They cannot survive the next validity check.
            actual = (action, event if target >= 0 else -1, target)
            assert actual == (decision.action, decision.event, decision.target), (
                f"{trace.scenario}, chef {player}, {trace.recipe}, seed {episode.seed}, "
                f"lane {lane}, step {step}, draw {decision.draw}, joint {decision.joint}: "
                f"(action, event, target) CPU={decision[1:4]} {device}={actual}"
            )

        kitchen.step(torch.as_tensor(joint, device=device))
        mask = [False] * size
        for lane in active:
            offsets[lane] += 1
            if offsets[lane] == len(traces[lane].episodes[episodes[lane]].decisions):
                episodes[lane] += 1
                offsets[lane] = 0
                mask[lane] = episodes[lane] < 2
        if any(mask):
            # Completed lanes will never be selected by this masked reset.
            reset(torch.tensor(mask, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-reference-only", action="store_true")
    parser.add_argument("--random-episodes", type=int, default=24, help="random two-episode traces per chef")
    args = parser.parse_args()
    if args.random_episodes < 1:
        parser.error("--random-episodes must be positive")
    if not args.cpu_reference_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    for path in (ROOT / "runs/cuda-deps", ROOT / "deps-legacy", ROOT / ".conda/Lib/site-packages"):
        if path.is_dir():
            sys.path.append(str(path))
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if (ROOT / "runs/cuda-tmp").is_dir():
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/cuda-tmp"))
    activate_upstream()
    torch.set_num_threads(1)

    for player in (0, 1):
        traces, coverage = cpu_reference(player, args.random_episodes)
        decisions = sum(len(ep.decisions) for trace in traces for ep in trace.episodes)
        retained = coverage["retained targets"]
        changed = coverage["retention changes action"]
        print(f"CPU chef {player} recorded {decisions} decisions across {2 * len(traces)} episodes, "
              f"{sum(coverage[f'delivered {recipe}'] for recipe in RECIPES)} deliveries, and "
              f"{retained} retained targets across reset ({changed} changed the next action)", flush=True)
        if not args.cpu_reference_only:
            replay(traces, player, args.device)
            print(f"{args.device} matched all {decisions} actions and active targets for chef {player}", flush=True)


if __name__ == "__main__":
    main()

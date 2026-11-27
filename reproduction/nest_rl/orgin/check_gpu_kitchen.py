"""Replay joint actions from the pinned fc_small_test CPU kitchen on TorchKitchen."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
import os
import random
from typing import NamedTuple

import numpy as np
import torch

from overcooked_setup import CONFIG, ROOT, activate_upstream


WAIT, LEFT, RIGHT, DOWN, UP, INTERACT = range(6)
OBJECT_NAMES = ("Tomato", "Lettuce", "Onion", "Carrot", "Potato", "Broccoli", "Plate")


class State(NamedTuple):
    obs: np.ndarray
    reward: np.float32
    done: bool
    truncated: bool
    recipe_index: int


@dataclass
class Trace:
    name: str
    seed: int
    layout: tuple
    actions: list
    states: list[State]


def capture_layout(world):
    def flatten(location):
        x, y = location
        return y * world.width + x

    return (
        [flatten(agent.location) for agent in world.agents],
        [flatten(world.world_objects[name][0].location) for name in OBJECT_NAMES],
    )


def approach(start, target):
    """Move within one chef's rectangular aisle, then face a neighboring counter."""
    x, y, facing = target
    dx, dy = x - start[0], y - start[1]
    horizontal = [LEFT if dx < 0 else RIGHT] * abs(dx)
    vertical = [UP if dy < 0 else DOWN] * abs(dy)
    return [*horizontal, *vertical, facing]


def prepare(start, shelf, board, divider):
    return [
        *approach(start, shelf), INTERACT,
        *approach(shelf, board), INTERACT, INTERACT, INTERACT,
        *approach(board, divider),
    ]


def salad_actions(world, left_food, right_food, left_first):
    objects = world.world_objects
    plate_row = objects["Plate"][0].location[1]
    left_row = plate_row if left_first else plate_row + 1
    left = prepare(
        world.agents[0].location, (1, objects[left_food][0].location[1], LEFT),
        (1, 4, LEFT), (2, left_row, RIGHT),
    )
    right_divider = (4, plate_row, LEFT)
    right = prepare(
        world.agents[1].location, (5, objects[right_food][0].location[1], RIGHT),
        (5, 6, DOWN), right_divider,
    )
    actions = list(zip_longest(left, right, fillvalue=WAIT))
    # Both chefs interact at once. The left chef adds to the plate or stages
    # its food on an empty divider tile before the right chef interacts.
    actions.extend([(INTERACT, INTERACT), (WAIT, INTERACT)])
    if not left_first:
        staged_food = (4, left_row, LEFT)
        actions.extend((WAIT, action) for action in approach(right_divider, staged_food))
        actions.append((WAIT, INTERACT))
        right_divider = staged_food
    actions.extend((WAIT, action) for action in approach(right_divider, (5, 3, RIGHT)))
    actions.append((WAIT, INTERACT))
    return actions


def reference_traces(random_episodes):
    from environment.overcooked.overcooked_maker import OvercookedMaker
    from gym_cooking import Arrdict

    random.seed(0)
    env = OvercookedMaker.make_env_from_config(CONFIG)
    core = env.env._env.unwrapped
    recipe_names = core.recipes
    recipe_messages = {
        f"Terminating because {recipe} is completed": idx
        for idx, recipe in enumerate(recipe_names)
    }
    players = env.players
    horizon = env.horizon

    def observations(data):
        return np.asarray([data[player].obs for player in players], dtype=np.float32)

    def record(name, seed, plan):
        random.seed(seed)
        data = env.reset()
        layout = capture_layout(core.world)
        states = [State(observations(data), np.float32(0), False, False, -1)]
        actions = []
        for joint_action in plan(core.world):
            decision = Arrdict((player, Arrdict(action=action)) for player, action in zip(players, joint_action))
            data, info = env.step(decision)
            first, second = (data[player] for player in players)
            first_info, second_info = (info[player] for player in players)
            assert (first.reward, first.done, first_info) == (second.reward, second.done, second_info)
            states.append(State(
                observations(data), first.reward, first.done,
                first_info.get("bad_transition", False), recipe_messages.get(first_info["termination_info"], -1),
            ))
            actions.append(joint_action)
            if first.done:
                break
        assert states[-1].done, f"CPU scenario {name} failed to terminate"
        return Trace(name, seed, layout, actions, states)

    def salad(left, right, seed, *, left_first=True, on_horizon=False, recipe=None):
        plated = (left, right) if left_first else (right, left)
        name = "-then-".join(plated) + ("-on-horizon" if on_horizon else "")

        def plan(world):
            actions = salad_actions(world, left, right, left_first)
            assert len(actions) < horizon
            waiting = [(WAIT, WAIT)] * (horizon - len(actions))
            return waiting + actions if on_horizon else actions + waiting

        trace = record(name, seed, plan)
        final = trace.states[-1]
        expected_recipe = -1 if recipe is None else recipe_names.index(recipe)
        assert final.recipe_index == expected_recipe and final.truncated == (recipe is None), name
        assert (len(trace.actions) == horizon) == (on_horizon or recipe is None), name
        plate = core.world.world_objects["Plate"][0]
        assert plate.location == core.world.world_objects["DeliverSquare"][0].location, name
        assert tuple(type(food).__name__ for food in plate.content) == plated, name
        return trace

    traces = []
    pairs = [("Potato", "Broccoli"), ("Lettuce", "Onion"), ("Tomato", "Carrot")]
    for idx, (left, right) in enumerate(pairs):
        recipe = f"{left}{right}Salad"
        traces.append(salad(left, right, 11 + 2 * idx, recipe=recipe))
        traces.append(salad(left, right, 12 + 2 * idx, left_first=False, recipe=recipe))
    traces.append(salad("Potato", "Broccoli", 31, on_horizon=True, recipe="PotatoBroccoliSalad"))
    traces.append(salad("Tomato", "Onion", 32))
    idle = record("wait-only", 33, lambda world: [(WAIT, WAIT)] * horizon)
    assert len(idle.actions) == horizon and idle.states[-1].truncated
    traces.append(idle)

    for idx in range(random_episodes):
        seed = 1000 + idx
        weights = (1, 2, 2, 2, 2, 4) if idx % 2 else None
        rng = random.Random(seed)
        actions = [tuple(rng.choices(range(6), weights=weights, k=2)) for _ in range(horizon)]
        traces.append(record(f"random-{idx}", seed, lambda world: actions))
    env.env._env.close()
    return traces, recipe_names


def check_field(field, tensor, expected, lanes, active, offsets, device):
    actual = tensor.cpu().numpy()
    shape = (len(lanes), *expected.shape[1:])
    assert (actual.shape, actual.dtype, tensor.device.type) == (shape, expected.dtype, device), (
        f"{field} returned {actual.shape}, {actual.dtype}, {tensor.device}; "
        f"expected {shape}, {expected.dtype}, {device}"
    )
    actual = actual[active]
    if np.array_equal(actual, expected):
        return
    row, *index = np.argwhere(actual != expected)[0].tolist()
    lane = active[row]
    trace, step = lanes[lane], offsets[lane]
    action = trace.actions[step - 1] if step else None
    element = (row, *index)
    raise AssertionError(
        f"{trace.name} (layout seed {trace.seed}, lane {lane}, step {step}, action {action}) "
        f"{field}{tuple(index)} differs, CPU={expected[element]!r}, {device}={actual[element]!r}"
    )


@torch.inference_mode()
def replay(traces, device):
    from torch_kitchen import TorchKitchen

    pending = iter(traces)
    size = min(32, len(traces))
    lanes = [next(pending) for _ in range(size)]
    offsets = [0] * size
    kitchen = TorchKitchen(size, device)

    def check_reset(mask=None):
        layouts = [trace.layout if trace is not None else traces[0].layout for trace in lanes]
        layout = tuple(torch.tensor(part, dtype=torch.long, device=device) for part in zip(*layouts))
        obs = kitchen.reset(mask=mask, layout=layout)
        active = [idx for idx, trace in enumerate(lanes) if trace is not None]
        expected = np.stack([lanes[idx].states[offsets[idx]].obs for idx in active])
        check_field("reset obs", obs, expected, lanes, active, offsets, device)

    check_reset()
    resets = 0
    while active := [idx for idx, trace in enumerate(lanes) if trace is not None]:
        actions = np.zeros((size, 2), dtype=np.int64)
        for idx in active:
            actions[idx] = lanes[idx].actions[offsets[idx]]
            offsets[idx] += 1
        actual = kitchen.step(torch.as_tensor(actions, device=device))
        expected_states = [lanes[idx].states[offsets[idx]] for idx in active]
        for field, tensor, expected, dtype in zip(
            State._fields, actual, zip(*expected_states), (np.float32, np.float32, np.bool_, np.bool_, np.int64),
            strict=True,
        ):
            check_field(field, tensor, np.asarray(expected, dtype=dtype), lanes, active, offsets, device)

        mask = [False] * size
        for idx, state in zip(active, expected_states):
            if state.done:
                lanes[idx] = next(pending, None)
                offsets[idx] = 0
                mask[idx] = lanes[idx] is not None
        if any(mask):
            check_reset(torch.tensor(mask, device=device))
            resets += sum(mask)
    return resets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-reference-only", action="store_true", help="validate the scripted CPU episodes")
    parser.add_argument("--random-episodes", type=int, default=96)
    args = parser.parse_args()
    if args.random_episodes < 1:
        parser.error("--random-episodes must be positive")
    if not args.cpu_reference_only and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if (ROOT / "runs/cuda-tmp").is_dir():
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/cuda-tmp"))
    activate_upstream()
    torch.set_num_threads(1)

    traces, recipes = reference_traces(args.random_episodes)
    deliveries = Counter(recipes[t.states[-1].recipe_index] for t in traces if t.states[-1].recipe_index >= 0)
    timeouts = sum(t.states[-1].truncated for t in traces)
    transitions = sum(len(t.actions) for t in traces)
    print(f"CPU reference recorded {transitions} transitions across {len(traces)} episodes, "
          f"{sum(deliveries.values())} deliveries {dict(deliveries)} and {timeouts} timeouts", flush=True)
    if not args.cpu_reference_only:
        resets = replay(traces, args.device)
        print(f"{args.device} matched all {transitions} transitions exactly for both chefs "
              f"and matched {resets} masked episode resets")


if __name__ == "__main__":
    main()

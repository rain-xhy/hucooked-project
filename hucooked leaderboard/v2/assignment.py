"""Learn initial role allocation from comparative SharedKitchen rollouts."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, smooth_l1_loss

from overcooked_ai_py.cooperative.shared import SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.controller import ScriptedController
from overcooked_ai_py.cooperative.shared.kitchen import DIRECTIONS, STAY
from overcooked_ai_py.cooperative.shared.layouts import FLOOR

from haha.artifacts import portable_path
from .commitment import ScatteredKitchen
from .policy import GOALS, GET_MAIN, GET_PLATE, GET_SIDE, mlp, reserve, scripted_action, targets


PERMUTATIONS = tuple(permutations(range(3)))
BENCHMARK_CONFIG = {"normal_orders": 5, "joint_orders": 5, "cook_steps": 6, "chop_steps": 3}


def distances_from(target):
    distances = {target: 0}
    frontier = deque([target])
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in DIRECTIONS:
            cell = x + dx, y + dy
            if cell in FLOOR and cell not in distances:
                distances[cell] = distances[(x, y)] + 1
                frontier.append(cell)
    return distances


ROLE_STARTS = [distances_from(GOALS[index][0]) for index in (GET_PLATE, GET_MAIN, GET_SIDE)]


def features(env, assignment):
    values = []
    for role, chef_id in enumerate(assignment):
        chef = env.state.chefs[chef_id]
        x, y = chef.position
        values.extend((x / 6, y / 6, ROLE_STARTS[role][chef.position] / 18))
        values.extend(float(chef.orientation == facing) for facing in DIRECTIONS)
    values.append(float(env.state.orders[0].kind == "joint"))
    values.append(float(env.state.orders[1].kind == "joint"))
    return values


def controller_for(env, assignment):
    controller = ScriptedController(env)
    controller.server, controller.main, controller.side = assignment
    return controller


def rollout(env, assignment, *, first_service=False):
    controller = controller_for(env, assignment)
    limit = min(env.config.horizon, 32) if first_service else env.config.horizon
    while not (env.terminated or env.truncated) and env.state.tick < limit:
        plan = targets(env, controller, 4, pipeline=True, lock_commitment=True)
        actions = [STAY] * 3
        reserved = set()
        for chef, subtask in zip(assignment, plan):
            action = scripted_action(env, chef, subtask, reserved, synchronize=True)
            actions[chef] = action
            reserve(env, chef, action, reserved)
        _, reward, _, _, _ = env.step(actions)
        if first_service and reward:
            break
    return env.state.score, env.state.tick, env.state.metrics["joint_completed"]


def distance_cost(env, team):
    return sum(ROLE_STARTS[role][env.state.chefs[i].position] for role, i in enumerate(team))


def prefix_choice(env):
    def cost(team):
        # No command is applied to the live kitchen. Stop before acting on
        # any ticket released by the first service.
        score, tick, _ = rollout(copy(env), team, first_service=True)
        return score, -tick, -distance_cost(env, team)
    return max(PERMUTATIONS, key=cost)


def distance_choice(env):
    return min(PERMUTATIONS, key=lambda team: distance_cost(env, team))


def learned_choice(env, net):
    obs = torch.tensor([[features(env, team) for team in PERMUTATIONS]], dtype=torch.float32)
    return PERMUTATIONS[int(prediction(net, obs)[0])]


def evaluate_seed(case):
    seed, horizon = case
    env = ScatteredKitchen(SharedKitchenConfig(horizon=horizon, **BENCHMARK_CONFIG))
    env.reset(seed)
    controller = ScriptedController(env)
    reference = PERMUTATIONS.index((controller.server, controller.main, controller.side))
    min_distance = PERMUTATIONS.index(distance_choice(env))
    candidate_features = [features(env, team) for team in PERMUTATIONS]
    rows = []
    for assignment in PERMUTATIONS:
        env.reset(seed)
        rows.append(rollout(env, assignment))
    return candidate_features, rows, (reference, min_distance)


def data(seeds, horizon, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluate_seed, ((int(seed), horizon) for seed in seeds), chunksize=3))
    a, b, c = zip(*rows)
    return np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.int64)


def report(trials, indices):
    rows = trials[np.arange(len(trials)), indices]
    full = rows[:, 0] == 200
    return {
        "episodes": len(rows), "score": float(rows[:, 0].mean()),
        "minimum": float(rows[:, 0].min()), "full_rate": float(full.mean()),
        "finish_steps_when_full": float(rows[full, 1].mean()) if full.any() else None,
        "joint_completed": float(rows[:, 2].mean()),
    }


def model(input_size):
    return mlp(input_size, 2)


def prediction(net, obs):
    with torch.no_grad():
        estimates = net(obs)
    full_probability = estimates[..., 0].sigmoid()
    ticks = estimates[..., 1].clamp(0, 1)
    return (full_probability - ticks * 0.15).argmax(-1).cpu().numpy()


def oracle_indices(trials):
    return np.asarray([max(range(len(PERMUTATIONS)), key=lambda c: (run[c, 0], -run[c, 1])) for run in trials])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=420)
    parser.add_argument("--test", type=int, default=160)
    parser.add_argument("--horizon", type=int, default=165)
    parser.add_argument("--objective", choices=("full", "score"), default="full")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=405)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    train_seeds = rng.choice(1_000_000, size=args.train, replace=False)
    heldout_seeds = rng.choice(1_000_000, size=args.test, replace=False) + 1_000_000
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    print("Collecting training role comparisons", flush=True)
    train_x, train_y, train_ref = data(train_seeds, args.horizon, args.workers)
    print("Collecting held-out role comparisons", flush=True)
    test_x, test_y, test_ref = data(heldout_seeds, args.horizon, args.workers)
    net = model(train_x.shape[-1]).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    x = torch.as_tensor(train_x, device=args.device)
    y = torch.as_tensor(train_y, device=args.device)
    for epoch in range(60):
        for ids in torch.randperm(len(x), device=args.device).split(128):
            predicted = net(x[ids])
            achieved = y[ids, :, 0] / 200.0
            targets = (achieved == 1).float() if args.objective == "full" else achieved
            classification = binary_cross_entropy_with_logits(predicted[..., 0], targets)
            ranking = smooth_l1_loss(predicted[..., 1], y[ids, :, 1] / args.horizon)
            loss = classification + 3 * ranking
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(), args.output / "role-assignment.pt")
    pred = prediction(net, torch.as_tensor(test_x, device=args.device))
    candidates = oracle_indices(test_y)
    metrics = {
        "config": vars(args) | {"output": portable_path(args.output)},
        "reference": report(test_y, test_ref[:, 0]), "geodesic": report(test_y, test_ref[:, 1]),
        "learned": report(test_y, pred), "all_permutations_oracle": report(test_y, candidates),
    }
    np.savez_compressed(args.output / "assignment-comparisons.npz", train_seeds=train_seeds,
                        train_features=train_x, train_outcomes=train_y, train_baselines=train_ref,
                        heldout_seeds=heldout_seeds, heldout_features=test_x, heldout_outcomes=test_y,
                        heldout_baselines=test_ref, learned_choices=pred, oracle_choices=candidates)
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

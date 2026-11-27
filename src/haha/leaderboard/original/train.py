"""Train a goal-conditioned Worker and a team-reward PPO Manager on Counter Circuit."""

import argparse
from collections import defaultdict, deque
from functools import lru_cache
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState

from haha.artifacts import portable_path
from haha.goal_training import imitation, mlp, ppo_update, tensor
from haha.subtasks import NAMES, UNKNOWN, held_name
from .simulation import Episode


LAYOUT = "counter_circuit_o_1order"
HORIZON = 400
STAY, INTERACT = Action.ACTION_TO_INDEX[Action.STAY], Action.ACTION_TO_INDEX[Action.INTERACT]
ITEMS = (None, "onion", "dish", "soup")
GOALS = np.eye(len(NAMES), dtype=np.float32)
SEATS = np.eye(2, dtype=np.float32)


class Navigation:
    def __init__(self, mdp):
        self.floor = frozenset(mdp.get_valid_player_positions())
        self.poses = tuple((p, d) for p in self.floor for d in Direction.ALL_DIRECTIONS)
        self.facing = {(p, d): Action.move_in_direction(p, d) for p, d in self.poses}

    @lru_cache(maxsize=None)
    def graph(self, occupied):
        transitions, reverse = {}, defaultdict(list)
        for pose in self.poses:
            position, _ = pose
            if position == occupied:
                continue
            moves = {}
            for action in Direction.ALL_DIRECTIONS:
                target = self.facing[position, action]
                if target == occupied:
                    continue
                end = (target if target in self.floor else position, action)
                moves[Action.ACTION_TO_INDEX[action]] = end
                reverse[end].append(pose)
            transitions[pose] = moves
        return transitions, reverse

    @lru_cache(maxsize=4096)
    def distances(self, targets, occupied):
        transitions, reverse = self.graph(occupied)
        goals = {pose: 0 for pose in transitions if self.facing[pose] in targets}
        frontier = deque(goals)
        while frontier:
            pose = frontier.popleft()
            for previous in reverse[pose]:
                if previous not in goals:
                    goals[previous] = goals[pose] + 1
                    frontier.append(previous)
        return goals

    def plan(self, position, orientation, targets, occupied):
        pose = (position, orientation)
        distances = self.distances(tuple(sorted(targets)), occupied)
        distance = distances.get(pose)
        if distance is None:
            return None
        if distance == 0:
            return 1, (INTERACT,)
        transitions, _ = self.graph(occupied)
        choices = tuple(action for action, end in transitions[pose].items() if distances.get(end) == distance - 1)
        return distance + 1, choices


class Trial:
    def __init__(self, mdp, navigation, seed, mode, seat=None, *, standard=False):
        self.mdp, self.navigation = mdp, navigation
        rng = np.random.default_rng(seed)
        sampled_seat = int(rng.integers(2))
        self.seat = sampled_seat if seat is None else seat
        start = mdp.get_standard_start_state()
        if not standard:
            positions = sorted(navigation.floor)
            selected = rng.choice(len(positions), size=2, replace=False)
            start.players = tuple(PlayerState(positions[index], Direction.INDEX_TO_DIRECTION[rng.integers(4)])
                                  for index in selected)
        self.game = Episode(mdp, start, HORIZON)
        self.learning = np.ones(2, dtype=bool) if mode == "multi" else np.arange(2) == self.seat
        self.score = 0.0
        self.stall = 0

    def step(self, actions):
        previous = tuple(p.position for p in self.game.state.players)
        reward, done = self.game.step(actions)
        current = tuple(p.position for p in self.game.state.players)
        self.stall = self.stall + 1 if previous == current else 0
        self.score += reward
        return reward, done

    def view(self):
        mdp, state = self.mdp, self.game.state
        pots = mdp.get_pot_locations()
        counters = mdp.get_counter_locations()
        supplies = defaultdict(list)
        for pos in counters:
            obj = state.objects.get(pos)
            supplies[obj.name if obj is not None else None].append(pos)
        empty_pots, ready_pots, remaining = [], [], []
        common = [state.timestep / HORIZON, min(self.stall, 10) / 10]
        for chef in state.players:
            common.extend((chef.position[0] / 8, chef.position[1] / 5))
            common.extend(float(direction == chef.orientation) for direction in Direction.ALL_DIRECTIONS)
            common.extend(float(kind == held_name(chef)) for kind in ITEMS)
        for pos in pots:
            soup = state.objects.get(pos)
            common.extend((len(soup.ingredients) / 3 if soup else 0, float(bool(soup and soup.is_ready)),
                           float(bool(soup and soup.is_cooking))))
            if soup is None or not soup.is_full:
                empty_pots.append(pos)
                remaining.append(3 - (len(soup.ingredients) if soup else 0))
            elif soup.is_ready:
                ready_pots.append(pos)
        for pos in counters:
            obj = state.objects.get(pos)
            common.extend(float(kind == (obj.name if obj else None)) for kind in ITEMS)
        common = np.asarray(common, dtype=np.float32)
        obs = np.concatenate((np.broadcast_to(common, (2, len(common))), SEATS), axis=1)
        moves, priorities = [], []
        cooking = len(pots) - len(empty_pots) - len(ready_pots)
        need = sum(remaining)
        for index, chef in enumerate(state.players):
            other = state.players[1 - index]
            hand, other_hand = held_name(chef), held_name(other)
            tasks = {}
            if hand is None:
                if supplies["soup"]:
                    tasks[8] = (0, supplies["soup"])
                if len(ready_pots) + cooking > int(other_hand in ("dish", "soup")):
                    priority = 1 if ready_pots or need == 0 else 3
                    tasks[4] = (priority, mdp.get_dish_dispenser_locations())
                    tasks[5] = (priority - 0.2, supplies["dish"])
                if need > int(other_hand == "onion"):
                    tasks[0] = (2, mdp.get_onion_dispenser_locations())
                    tasks[1] = (1.8, supplies["onion"])
            elif hand == "onion":
                tasks[2] = (0, empty_pots)
                if not empty_pots and (ready_pots or other_hand == "onion"):
                    tasks[3] = (4, supplies[None])
            elif hand == "dish":
                tasks[7] = (0, ready_pots)
                if not (ready_pots or cooking) and other_hand != "onion":
                    tasks[6] = (4, supplies[None])
            elif hand == "soup":
                tasks[10] = (0, mdp.get_serving_locations())
            legal = {UNKNOWN: (0, (STAY,))}
            rank = {}
            for task, (priority, targets) in tasks.items():
                if targets:
                    route = self.navigation.plan(chef.position, chef.orientation, targets, other.position)
                    if route is not None:
                        legal[task] = route
                        rank[task] = (priority, route[0], task)
            best = min(rank, key=rank.get) if rank else UNKNOWN
            moves.append(legal)
            priorities.append(best)
        masks = np.asarray([[i in goals for i in range(len(NAMES))] for goals in moves])
        return common, obs, masks, np.asarray(priorities), moves


class Policy(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.manager = mlp(width + 2, len(NAMES))
        self.worker = mlp(width + 2 + len(NAMES) + 1, 6)
        self.critic = mlp(width, 1)

    def logits(self, module, obs, masks):
        return module(obs).masked_fill(~masks, -1e9)


def worker_inputs(obs, goal, route):
    distance, choices = route
    mask = np.zeros(6, dtype=bool)
    mask[list(choices)] = True
    return np.r_[obs, GOALS[goal], np.float32(distance / 32)], mask


def teachers(views, primitive, rng):
    rows = []
    for _, obs, masks, labels, moves in views:
        rows.append((obs, masks, labels))
        for chef, (features, candidates) in enumerate(zip(obs, moves)):
            chosen = labels[chef]
            alternative = int(rng.choice(sorted(candidates)))
            for goal in sorted({chosen, alternative}):
                route = candidates[goal]
                state, mask = worker_inputs(features, goal, route)
                primitive.append((state, mask, route[1][0]))
    return rows


@torch.no_grad()
def act(policy, trials, views, stochastic=False):
    obs = np.stack([row[1] for row in views])
    masks = np.stack([row[2] for row in views])
    labels = np.stack([row[3] for row in views])
    learning = np.stack([trial.learning for trial in trials])
    selected, logp = labels, np.zeros(labels.shape, dtype=np.float32)
    if policy is not None:
        dist = Categorical(logits=policy.logits(policy.manager, tensor(obs), tensor(masks)))
        proposals = dist.sample() if stochastic else dist.logits.argmax(-1)
        selected = np.where(learning, proposals.numpy(), labels)
        logp = dist.log_prob(tensor(selected)).numpy()
    actions = np.empty(selected.shape, dtype=np.int64)
    inputs, masks, indices = [], [], []
    for index, (trial, view) in enumerate(zip(trials, views)):
        for chef, (goal, candidates) in enumerate(zip(selected[index], view[4])):
            route = candidates[goal]
            actions[index, chef] = route[1][0]
            if policy is not None and trial.learning[chef]:
                state, allowed = worker_inputs(obs[index, chef], goal, route)
                inputs.append(state)
                masks.append(allowed)
                indices.append((index, chef))
    if indices:
        logits = policy.logits(policy.worker, tensor(np.stack(inputs)), tensor(np.stack(masks)))
        for (i, j), action in zip(indices, logits.argmax(-1).tolist()):
            actions[i, j] = action
    return actions, selected, logp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=488)
    parser.add_argument("--test-seed-base", type=int, default=84031)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--lanes", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir()
    start = monotonic()
    mdp = OvercookedGridworld.from_layout_name(LAYOUT)
    nav = Navigation(mdp)

    def log(stage, **values):
        data = {"stage": stage, "seconds": round(monotonic() - start, 2), **values}
        with (args.output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(data) + "\n")
        print(json.dumps(data), flush=True)

    def new_trial():
        return Trial(mdp, nav, int(rng.integers(1_000_000, 2_000_000_000)), args.mode)

    def evaluate(checkpoint, seeds, *, trace=False, standard=False):
        rows = []
        seats = (0, 1) if args.mode == "single" else (None,)
        cases = [(s, seat) for s in seeds for seat in seats]
        trials = [Trial(mdp, nav, seed, args.mode, seat, standard=standard) for seed, seat in cases]
        starts = [trial.game.state.to_dict() for trial in trials] if trace else []
        histories = [[] for _ in trials]
        for _ in range(HORIZON):
            moves, goals, _ = act(checkpoint, trials, [trial.view() for trial in trials])
            for i, (trial, joint) in enumerate(zip(trials, moves)):
                reward, _ = trial.step(joint)
                if trace:
                    histories[i].append({"actions": joint.tolist(), "goals": goals[i].tolist(), "reward": reward})
        for i, (trial, (seed, seat)) in enumerate(zip(trials, cases)):
            rows.append({"layout": LAYOUT, "horizon": HORIZON, "seed": seed, "learning_role": seat,
                         "score": trial.score, "normal_completed": int(trial.score / 20),
                         **({"initial_state": starts[i], "trajectory": histories[i]} if trace else {})})
        return rows

    def report(rows):
        return {LAYOUT: {"episodes": len(rows), "mean": float(np.mean([x["score"] for x in rows])),
                         "max": max(x["score"] for x in rows)}}

    log("config", **(vars(args) | {"output": portable_path(args.output), "layout": LAYOUT, "horizon": HORIZON,
         "starts": "seeded distinct floor tiles and orientations, plus separate standard-start evaluation",
         "team_rewards": "native sparse delivery rewards"}))
    host = Policy(len(new_trial().view()[0]))
    model = Policy(len(new_trial().view()[0])).to(args.device)
    examples, actions = [], []
    demos = [new_trial() for _ in range(28)]
    for _ in range(HORIZON):
        views = [trial.view() for trial in demos]
        examples.extend(teachers(views, actions, rng))
        joint, _, _ = act(None, demos, views)
        for trial, move in zip(demos, joint):
            trial.step(move)
    log("imitation", **imitation(model, (examples, actions), 24, args.device, rng))
    del examples, actions, demos
    validation_seeds = range(20031, 20043)
    optimizer = torch.optim.Adam([
        {"params": model.manager.parameters(), "lr": 0.0003},
        {"params": model.critic.parameters(), "lr": 0.001},
    ])
    best, checkpoint = float("-inf"), args.output / "best.pt"
    trials = [new_trial() for _ in range(args.lanes)]
    for update in range(args.updates + 1):
        host.load_state_dict(model.state_dict())
        if update % 3 == 0 or update == args.updates:
            scores = report(evaluate(host, validation_seeds))
            log("after_imitation" if update == 0 else "validation", update=update, results=scores)
            value = scores[LAYOUT]["mean"]
            if value > best:
                torch.save(model.state_dict(), checkpoint)
                best = value
        if update == args.updates:
            break
        records, completed_scores = [], []
        for _ in range(160):
            views = [trial.view() for trial in trials]
            common, local, masks, labels, _ = zip(*views)
            learning = np.stack([trial.learning for trial in trials])
            moves, selected, logp = act(host, trials, views, stochastic=True)
            rewards, ended = [], []
            for i, (trial, move) in enumerate(zip(trials, moves)):
                reward, done = trial.step(move)
                rewards.append(reward / 20)
                ended.append(done)
                if done:
                    completed_scores.append(trial.score)
                    trials[i] = new_trial()
            records.append((common, local, masks, learning, labels, selected, logp, rewards, ended))
        final = np.stack([trial.view()[0] for trial in trials])
        ppo_update(model, optimizer, (records, final, completed_scores), args.device, 0.12)
        mean_score = float(np.mean(completed_scores)) if completed_scores else None
        log("update", update=update + 1, env_steps=(update + 1) * args.lanes * 160, sampled_mean=mean_score)
    torch.save(model.state_dict(), args.output / "last.pt")
    host.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    seeds = range(args.test_seed_base, args.test_seed_base + args.test_episodes)
    final = evaluate(host, seeds, trace=True)
    (args.output / "evaluation.json").write_text(json.dumps(final))
    log("evaluation", results=report(final))
    standard = evaluate(host, [args.test_seed_base], standard=True)
    rules = evaluate(None, seeds)
    (args.output / "standard-start.json").write_text(json.dumps(standard, indent=2))
    (args.output / "rules-comparison.json").write_text(json.dumps(rules, indent=2))
    log("standard-start", results=report(standard))
    log("rules-comparison", results=report(rules))


if __name__ == "__main__":
    main()

"""Train HA² on all three original MixedKitchen geometry variants."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from overcooked_ai_py.cooperative.controller import ReferenceController
from overcooked_ai_py.cooperative.kitchen import DIRECTIONS, MixedKitchen, STAY
from overcooked_ai_py.cooperative.layouts import LAYOUTS
from overcooked_ai_py.cooperative.model import KitchenConfig

from haha.artifacts import portable_path
from haha.goal_training import imitation, mlp, ppo_update, tensor
from .schedule import GOAL_COUNT, ScheduledTeam, choices, goals, navigation, reserve, worker_mask


ENVIRONMENT_COMMIT = "671c17bf4ba48ecb11e056dbf1e077b0bd434b2d"
GOAL_FEATURES = np.eye(GOAL_COUNT + 1, dtype=np.float32)
SLOTS = np.eye(3, dtype=np.float32)
PORT_FEATURES = np.eye(4, dtype=np.float32)


class Policy(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        self.manager = mlp(common_dim + 3, GOAL_COUNT + 1)
        self.worker = mlp(common_dim + 3 + GOAL_COUNT + 1 + 2 + 4 + 1 + 77, 6)
        self.critic = mlp(common_dim, 1)

    def logits(self, module, obs, mask):
        return module(obs).masked_fill(~mask, -1e9)


class Trial:
    def __init__(self, layout, seed, mode, horizon, planning, delay, learned_role=None, routing=False):
        self.env = MixedKitchen(KitchenConfig(layout=layout, horizon=horizon))
        self.env.reset(seed)
        self.scheduler = ScheduledTeam(planning)
        self.random = np.random.default_rng(np.random.SeedSequence((seed, 2094)))
        self.delay = delay
        self.routing = routing
        self.learned_role = seed % 3 if learned_role is None else learned_role
        self.learn = np.full(3, mode == "multi", dtype=bool)
        if mode == "single":
            self.learn[self.learned_role] = True

    def step(self, actions):
        applied = [int(a) if v >= self.delay else STAY for a, v in zip(actions, self.random.random(3))]
        _, reward, done, truncated, info = self.env.step(applied)
        return reward, done or truncated, info, applied


def encode(trials):
    common, labels, masks, maps = [], [], [], []
    for trial in trials:
        env = trial.env
        roles = trial.scheduler.assign(env)
        mapped_goals = goals(env, roles)
        options = [choices(env, roles, role, goal) for role, goal in enumerate(mapped_goals)]
        common.append(np.r_[env.centralized_state(), SLOTS[list(roles)].flatten()])
        labels.append([goal.subtask for goal in mapped_goals])
        maps.append((roles, options))
        mask = np.zeros((3, GOAL_COUNT + 1), dtype=bool)
        for role, opts in enumerate(options):
            mask[role, list(opts)] = True
        masks.append(mask)
    shared = np.asarray(common, dtype=np.float32)
    obs = np.concatenate((np.broadcast_to(shared[:, None, :], (len(trials), 3, shared.shape[-1])),
                          np.broadcast_to(SLOTS, (len(trials), 3, 3))), axis=-1)
    return shared, obs, np.stack(masks), np.stack([trial.learn for trial in trials]), np.asarray(labels), maps


def worker_observation(obs, goal, reserved):
    occupied = np.zeros(77, dtype=np.float32)
    for x, y in reserved:
        occupied[y * 11 + x] = 1.0
    x, y = goal.cell
    geometry = (x / 10, y / 6, float(goal.interact))
    return np.r_[obs, GOAL_FEATURES[goal.subtask], PORT_FEATURES[DIRECTIONS.index(goal.facing)],
                 np.asarray(geometry, dtype=np.float32), occupied]


@torch.no_grad()
def act(policy, trials, encoded, *, stochastic=False, teacher_rate=0.0, rng=None,
        demonstrations=None, worker_teacher=False):
    _, obs, masks, learning, labels, maps = encoded
    if policy is None:
        selection = labels
        logp = np.zeros(labels.shape, dtype=np.float32)
    else:
        distribution = Categorical(logits=policy.logits(policy.manager, tensor(obs), tensor(masks)))
        actions = distribution.sample() if stochastic else distribution.logits.argmax(-1)
        selection = np.where(learning, actions.numpy(), labels)
        if teacher_rate:
            selection = np.where(rng.random(selection.shape) < teacher_rate, labels, selection)
        logp = distribution.log_prob(tensor(selection)).numpy()

    joint = np.full((len(trials), 3), STAY, dtype=np.int64)
    reservations = [set() for _ in trials]
    for role in (2, 1, 0):
        inputs, permitted, items = [], [], []
        for i, (trial, (assignment, options)) in enumerate(zip(trials, maps)):
            chef = assignment[role]
            env = trial.env
            goal = options[role][selection[i, role]]
            blocked = reservations[i]
            if demonstrations is not None:
                sample = rng.choice(list(options[role]))
                for subtask in sorted({int(sample), int(goal.subtask)}):
                    option = options[role][subtask]
                    mask = worker_mask(env, chef, option, blocked, progress=trial.routing)
                    action = navigation(env, chef, option, blocked)
                    if not mask[action]:
                        raise ValueError("Teacher action is outside the Worker mask")
                    demonstrations.append((worker_observation(obs[i, role], option, blocked), mask, action))
            if policy is None or not learning[i, role] or worker_teacher or (
                teacher_rate and rng.random() < teacher_rate
            ):
                action = navigation(env, chef, goal, blocked)
                joint[i, chef] = action
                reserve(env, chef, action, blocked)
            else:
                inputs.append(worker_observation(obs[i, role], goal, blocked))
                permitted.append(worker_mask(env, chef, goal, blocked, progress=trial.routing))
                items.append((i, chef))
        if items:
            logits = policy.logits(policy.worker, tensor(np.stack(inputs)), tensor(np.stack(permitted)))
            for (i, chef), action in zip(items, logits.argmax(-1).tolist()):
                joint[i, chef] = action
                reserve(trials[i].env, chef, action, reservations[i])
    return joint, selection, logp


def evaluate(policy, cases, args, *, trace=False, worker_teacher=False):
    rows = []
    for layout, seed, role in cases:
        trial = Trial(layout, seed, args.mode, args.horizon, args.lookahead, args.delay, role, args.route_progress)
        reference = ReferenceController() if args.mode == "reference" else None
        history = []
        while True:
            if reference:
                actions = reference(trial.env)
                selected = None
            else:
                joint, decisions, _ = act(policy, [trial], encode([trial]), worker_teacher=worker_teacher)
                actions, selected = joint[0], decisions[0].tolist()
            reward, done, info, applied = trial.step(actions)
            if trace:
                history.append({"tick": info["tick"], "actions": applied, "goals": selected,
                                "roles": trial.scheduler.team, "reward": reward, "events": info["events"]})
            if done:
                state = trial.env.state
                rows.append({"layout": layout, "seed": seed, "learning_role": role,
                             "score": state.score, "ticks": state.tick, **info["metrics"],
                             "orders": state.history, **({"trajectory": history} if trace else {})})
                break
    return rows


def summarize(rows):
    rates = {}
    for layout in LAYOUTS:
        group = [r for r in rows if r["layout"] == layout]
        if not group:
            continue
        winners = [r for r in group if r["score"] == 480.0]
        rates[layout] = {
            "episodes": len(group), "score": float(np.mean([r["score"] for r in group])),
            "minimum": min(r["score"] for r in group), "full": len(winners) / len(group),
            "ticks": float(np.mean([r["ticks"] for r in group])),
            **{k: float(np.mean([r[k] for r in group])) for k in
               ("normal_completed", "joint_completed", "joint_aborts", "collisions", "invalid_joint_presses")},
        }
    return rates


def rank(rows):
    return (float(np.mean([r["score"] for r in rows])), min(r["score"] for r in rows),
            -float(np.mean([r["ticks"] for r in rows])))


def collect(args, rng, episodes, policy=None):
    manager, worker = [], []
    for index in range(episodes):
        layout = args.train_layout[index % len(args.train_layout)]
        trial = Trial(layout, int(rng.integers(2**31)), args.mode, args.horizon,
                      args.lookahead, args.train_delay, routing=args.route_progress)
        while True:
            encoded = encode([trial])
            _, obs, masks, _, labels, _ = encoded
            manager.append((obs[0], masks[0], labels[0]))
            joint, _, _ = act(policy, [trial], encoded, teacher_rate=0.3 if policy else 0.0,
                              rng=rng, demonstrations=worker)
            _, done, _, _ = trial.step(joint[0])
            if done:
                break
    return manager, worker


def rollout(args, rng, trials, policy):
    records, scores = [], []
    for _ in range(args.steps):
        encoded = encode(trials)
        common, obs, masks, learning, labels, _ = encoded
        joint, selected, logp = act(policy, trials, encoded, stochastic=True)
        rewards, ended = [], []
        for i, (trial, actions) in enumerate(zip(trials, joint)):
            reward, done, info, _ = trial.step(actions)
            rewards.append(reward / 20.0)
            ended.append(done)
            if done:
                scores.append(info["score"])
                trials[i] = Trial(str(rng.choice(args.train_layout)), int(rng.integers(2**31)),
                                  args.mode, args.horizon, args.lookahead, args.train_delay, routing=args.route_progress)
        records.append((common, obs, masks, learning, labels, selected, logp, rewards, ended))
    return records, encode(trials)[0], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "planner", "single", "multi"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--reset-manager", action="store_true")
    parser.add_argument("--anchor", type=float, default=0.12)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--worker-teacher", action="store_true")
    parser.add_argument("--lookahead", action="store_true")
    parser.add_argument("--route-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=72)
    parser.add_argument("--recovery", type=int, default=2)
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--train-delay", type=float, default=0.01)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--test-seed-base", type=int, default=9020)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--train-layout", choices=tuple(LAYOUTS), nargs="+", default=list(LAYOUTS))
    parser.add_argument("--test-layout", choices=tuple(LAYOUTS), nargs="+", default=list(LAYOUTS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.validate_every, args.test_episodes, args.lanes, args.steps) < 1:
        parser.error("Validation interval, test episodes, lanes, and rollout steps must be positive")
    if not 0 <= args.delay < 1 or not 0 <= args.train_delay < 1:
        parser.error("Action delay probabilities must be in [0, 1)")
    if args.reset_manager and (not args.initialize or args.evaluation_only):
        parser.error("--reset-manager requires --initialize in training mode")
    if args.evaluation_only and args.mode in ("single", "multi") and not args.initialize:
        parser.error("evaluation of a learned policy requires --initialize")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    args.output.mkdir(exist_ok=False, parents=True)
    start = monotonic()
    test_roles = range(3) if args.mode == "single" else (None,)
    test_cases = [(layout, args.test_seed_base + i, role)
                  for layout in args.test_layout for i in range(args.test_episodes)
                  for role in test_roles]
    validation = [(layout, 3001 + i, None) for layout in args.train_layout for i in range(6)]

    def record(stage, **fields):
        row = {"stage": stage, "seconds": round(monotonic() - start, 2), **fields}
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def final(policy):
        rows = evaluate(policy, test_cases, args, trace=True, worker_teacher=args.worker_teacher)
        (args.output / "evaluation.json").write_text(json.dumps(rows), encoding="utf-8")
        record("evaluation", results=summarize(rows))

    cfg = vars(args) | {"output": portable_path(args.output), "initialize": portable_path(args.initialize)}
    record("config", **cfg, environment=ENVIRONMENT_COMMIT, rules=asdict(KitchenConfig(horizon=args.horizon)))
    if args.mode in ("reference", "planner"):
        final(None)
        return
    width = len(MixedKitchen().centralized_state()) + 9
    policy = Policy(width).to(args.device)
    if args.initialize:
        weights = torch.load(args.initialize, map_location=args.device, weights_only=True)
        if args.reset_manager:
            policy.worker.load_state_dict({key.removeprefix("worker."): value for key, value in weights.items()
                                           if key.startswith("worker.")})
        else:
            policy.load_state_dict(weights)
    if args.evaluation_only:
        final(policy.cpu())
        return
    best = None
    best_stage = None

    def validate(stage, **extras):
        nonlocal best, best_stage
        host = deepcopy(policy).cpu()
        rows = evaluate(host, validation, args)
        record(stage, results=summarize(rows), **extras)
        score = rank(rows)
        if best is None or score > best:
            torch.save(policy.state_dict(), args.output / "best.pt")
            best, best_stage = score, stage
        return host

    if args.initialize:
        validate("initialized")
    else:
        data = collect(args, rng, args.episodes)
        record("imitation", **imitation(policy, data, 28, args.device, rng))
        host = validate("after_imitation")
        for i in range(args.recovery):
            extension = collect(args, rng, max(15, args.episodes // 4), host)
            data[0].extend(extension[0])
            data[1].extend(extension[1])
            record("recovery_imitation", iteration=i + 1, **imitation(policy, data, 12, args.device, rng))
            host = validate("after_recovery", iteration=i + 1)
    policy.load_state_dict(torch.load(args.output / "best.pt", map_location=args.device, weights_only=True))
    host = deepcopy(policy).cpu()
    optimizer = torch.optim.Adam([
        {"params": policy.manager.parameters(), "lr": 0.00025},
        {"params": policy.critic.parameters(), "lr": 0.001},
    ])
    trials = [Trial(str(rng.choice(args.train_layout)), int(rng.integers(2**31)),
                    args.mode, args.horizon, args.lookahead, args.train_delay, routing=args.route_progress) for _ in range(args.lanes)]
    for update in range(1, args.updates + 1):
        data = rollout(args, rng, trials, host)
        ppo_update(policy, optimizer, data, args.device, args.anchor)
        metrics = {"update": update, "environment_steps": update * args.steps * args.lanes,
                   "sampled_mean": float(np.mean(data[2])) if data[2] else None}
        if update % args.validate_every == 0 or update == args.updates:
            host = validate("ppo", **metrics)
        else:
            record("ppo_update", **metrics)
            host = deepcopy(policy).cpu()
    torch.save(policy.state_dict(), args.output / "last.pt")
    host.load_state_dict(torch.load(args.output / "best.pt", map_location="cpu", weights_only=True))
    record("selected", best_stage=best_stage)
    final(host)


if __name__ == "__main__":
    main()

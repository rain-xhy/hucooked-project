"""Train and evaluate HA² policies on hucooked's SharedKitchen."""

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.distributions import Categorical

from overcooked_ai_py.cooperative.shared import SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.controller import ScriptedController
from overcooked_ai_py.cooperative.shared.kitchen import STAY

from haha.artifacts import portable_path
from haha.goal_training import imitation, ppo_update, tensor
from .commitment import CommitmentKitchen, ScatteredKitchen
from .assignment import BENCHMARK_CONFIG, distance_choice, learned_choice, prefix_choice
from .assignment import model as assignment_model
from .policy import (
    SharedPolicy, actor_inputs, features, goal_mask, ordered_chefs, reserve,
    reserved_features, scripted_action, targets, valid_actions, worker_inputs,
)

ENVIRONMENT_COMMIT = "10aba7415fb0ce8340b2cc99025ca1f0627a92e0"
CONTINUOUS_ENVIRONMENT_COMMIT = "508bcbf066216acb7bb66bdd2ecf9148bf73c7c1"
COMPATIBLE_CONTINUOUS_REVISIONS = (
    CONTINUOUS_ENVIRONMENT_COMMIT,
    "de37e0818c2e76311e105ab8689e5162bd84133f",
)
FIXED_ORDERS = {"normal_orders": 4, "joint_orders": 4}
PROFILES = {
    "default": FIXED_ORDERS,
    "continuous": {},
    "regular": {"normal_orders": 4, "joint_orders": 0},
    "cooperative": {"normal_orders": 0, "joint_orders": 4},
    "high_volume": {"normal_orders": 8, "joint_orders": 8},
    "slow_stations": {**FIXED_ORDERS, "cook_steps": 8, "chop_steps": 4},
    "one_active_order": {**FIXED_ORDERS, "active_limit": 1},
    "commitment": BENCHMARK_CONFIG,
    "scattered": BENCHMARK_CONFIG,
}


def profile_ceiling(name):
    params = SharedKitchenConfig(**PROFILES[name])
    return params.normal_orders * params.normal_reward + params.joint_orders * params.joint_reward


@dataclass(frozen=True)
class Execution:
    early: int = 2
    pipeline: bool = False
    synchronize: bool = False
    horizon: int = 300
    delay: float = 0.0
    goal_lock: bool = False
    profile: str = "default"
    lock_commitment: bool = False
    queue_prefetch: bool = False
    route_progress: bool = False
    role_allocator: str = "reference"
    role_policy: torch.nn.Module | None = None


class Trial:
    def __init__(self, seed, mode, execution, learn_role=None):
        self.execution = execution
        environment = {"commitment": CommitmentKitchen, "scattered": ScatteredKitchen}.get(
            execution.profile, SharedKitchen
        )
        self.env = environment(SharedKitchenConfig(horizon=execution.horizon, **PROFILES[execution.profile]))
        self.env.reset(seed)
        self.controller = ScriptedController(self.env)
        if execution.profile == "scattered" and execution.role_allocator != "reference":
            if execution.role_allocator == "learned":
                assignment = learned_choice(self.env, execution.role_policy)
            else:
                allocate = distance_choice if execution.role_allocator == "geodesic" else prefix_choice
                assignment = allocate(self.env)
            self.controller.server, self.controller.main, self.controller.side = assignment
        self.mode = mode
        self.learn_role = seed % 3 if learn_role is None else learn_role
        self.learning = np.ones(3, dtype=bool)
        if mode == "single":
            self.learning = np.arange(3) == self.learn_role
        if mode in ("reference", "planner"):
            self.learning[:] = False
        self.reward = 0.0
        self.delay_rng = np.random.default_rng(np.random.SeedSequence([seed, 917]))

    def delayed(self, actions):
        if not self.execution.delay:
            return actions
        stalled = self.delay_rng.random(3) < self.execution.delay
        if self.mode == "single":
            stalled &= ~self.learning
            stalled = stalled[np.argsort(ordered_chefs(self.controller))]
        return np.where(stalled, STAY, actions)

    def step(self, actions):
        _, reward, ended, truncated, info = self.env.step(actions)
        self.reward += reward
        return reward, ended or truncated, info


def encode(trials):
    common = np.stack([features(t.env, t.controller) for t in trials])
    masks = np.stack([goal_mask(t.env, t.controller, lock_commitment=t.execution.lock_commitment,
                                queue_prefetch=t.execution.queue_prefetch) for t in trials])
    learning = np.stack([t.learning for t in trials])
    return common, actor_inputs(common), masks, learning


def teacher_goals(trials):
    return np.stack([targets(t.env, t.controller, t.execution.early, pipeline=t.execution.pipeline, lock_commitment=t.execution.lock_commitment,
                             queue_prefetch=t.execution.queue_prefetch) for t in trials])


@torch.no_grad()
def act(trials, policy, obs, masks, learning, labels, *, sample=False, scripted_worker=False):
    if policy is not None:
        dist = Categorical(logits=policy.logits(policy.manager, tensor(obs), tensor(masks)))
        proposals = dist.sample() if sample else dist.logits.argmax(-1)
        goals = proposals.numpy() if labels is None else np.where(learning, proposals.numpy(), labels)
        logp = dist.log_prob(tensor(goals)).numpy()
    else:
        goals, logp = labels, np.zeros(learning.shape, dtype=np.float32)
    actions = np.full((len(trials), 3), STAY, dtype=np.int64)
    reservations = [set() for _ in trials]
    for role in range(3):
        chefs = [ordered_chefs(t.controller)[role] for t in trials]
        active = learning[:, role]
        proposed = np.full(len(trials), STAY, dtype=np.int64)
        if policy is not None and active.any() and not scripted_worker:
            indices = np.flatnonzero(active)
            worker_obs = worker_inputs(obs[indices, role], goals[indices, role],
                                       np.stack([reserved_features(reservations[i]) for i in indices]))
            worker_masks = np.stack([valid_actions(trials[i].env, chefs[i], goals[i, role], synchronize=trials[i].execution.synchronize,
                                                   goal_lock=trials[i].execution.goal_lock,
                                                   route_progress=trials[i].execution.route_progress,
                                                   reserved=reservations[i]) for i in indices])
            proposed[indices] = policy.logits(policy.worker, tensor(worker_obs), tensor(worker_masks)).argmax(-1).numpy()
        for i, (trial, chef) in enumerate(zip(trials, chefs)):
            if not active[i] or scripted_worker:
                proposed[i] = scripted_action(trial.env, chef, goals[i, role], reservations[i], synchronize=trial.execution.synchronize)
            actions[i, chef] = proposed[i]
            reserve(trial.env, chef, proposed[i], reservations[i])
    return actions, goals, logp


def evaluate(policy, mode, seeds, execution, *, scripted_worker=False, trace=False):
    trials = [Trial(seed, mode, execution, role) for seed, role in seeds]
    pending = list(range(len(trials)))
    histories = [[] for _ in trials]
    results = [None] * len(trials)
    while pending:
        batch = [trials[i] for i in pending]
        if mode == "reference":
            actions = [trial.controller.actions(trial.env) for trial in batch]
            selected = [None] * len(batch)
        else:
            _, obs, masks, learning = encode(batch)
            labels = None if policy is not None and mode == "multi" else teacher_goals(batch)
            actions, selected, _ = act(batch, policy, obs, masks, learning, labels,
                                       scripted_worker=scripted_worker)
        running = []
        for index, trial, action, goals in zip(pending, batch, actions, selected):
            action = trial.delayed(action)
            reward, done, info = trial.step(action)
            if trace:
                histories[index].append({
                    "tick": trial.env.state.tick, "actions": list(map(int, action)),
                    "goals": None if goals is None else goals.tolist(),
                    "reward": reward, "events": info["events"],
                })
            if done:
                seed, _ = seeds[index]
                results[index] = {
                    "seed": seed, "learned_role": trial.learn_role if mode == "single" else None,
                    "horizon": trial.env.config.horizon,
                    **info["metrics"], "orders": info["orders"],
                }
                if trace:
                    results[index]["trajectory"] = histories[index]
            else:
                running.append(index)
        pending = running
    return results


def summary(rows):
    report = {
        "episodes": len(rows), "mean_score": float(np.mean([r["team_score"] for r in rows])),
        "min_score": min(r["team_score"] for r in rows),
        **{key: float(np.mean([r[key] for r in rows])) for key in (
            "normal_completed", "joint_completed", "joint_successes", "unmatched_attempt_steps",
            "three_contributor_meals", "movement_conflicts", "resource_conflicts",
        )},
        **{key: float(np.mean([r[key] for r in rows])) for key in ("joint_run_resets", "chopping_resets")
           if key in rows[0]},
    }
    if "meals_per_100_steps" in rows[0]:
        report.update({key: float(np.mean([r[key] for r in rows])) for key in (
            "meals_per_100_steps", "joint_attempt_success_rate", "joint_wait_steps",
        )})
    else:
        full = [r["ticks"] for r in rows if r["meals_completed"] == r["orders_total"]]
        report.update(full_rate=len(full) / len(rows), full_ticks=float(np.mean(full)) if full else None)
    return report


def rank(report):
    if "full_ticks" not in report:
        return report["mean_score"], report["min_score"]
    ticks = report["full_ticks"]
    return report["mean_score"], report["min_score"], report["full_rate"], -(ticks if ticks is not None else 300)


def sample_trial(mode, profiles, execution, rng):
    profile = str(rng.choice(profiles)) if profiles else execution.profile
    return Trial(int(rng.integers(2**31)), mode, replace(execution, profile=profile))


def collect(policy, mode, episodes, execution, rng, beta, profiles=None):
    manager, worker = [], []
    for _ in range(episodes):
        trial = sample_trial(mode, profiles, execution, rng)
        env = trial.env
        for _ in range(env.config.horizon):
            _, obs, masks, learning = encode([trial])
            obs, masks = obs[0], masks[0]
            labels = targets(env, trial.controller, execution.early, pipeline=execution.pipeline, lock_commitment=execution.lock_commitment, queue_prefetch=execution.queue_prefetch)
            manager.append((obs, masks, labels))
            chosen = labels.copy()
            if policy is not None:
                with torch.no_grad():
                    logits = policy.logits(policy.manager, tensor(obs), tensor(masks))
                    chosen = np.where(learning[0] & (rng.random(3) > beta), logits.argmax(-1).numpy(), labels)
            actions = np.full(3, STAY, dtype=np.int64)
            reserved = set()
            for role, chef in enumerate(ordered_chefs(trial.controller)):
                goal = chosen[role]
                flags = reserved_features(reserved)
                permitted = np.flatnonzero(masks[role])
                alternatives = set((int(goal), int(rng.choice(permitted))))
                for candidate in sorted(alternatives):
                    action_mask = valid_actions(env, chef, candidate, synchronize=execution.synchronize,
                                                goal_lock=execution.goal_lock, route_progress=execution.route_progress,
                                                reserved=reserved)
                    worker_obs = worker_inputs(obs[role], candidate, flags)
                    desired = scripted_action(env, chef, candidate, reserved, synchronize=execution.synchronize)
                    if not action_mask[desired]:
                        raise ValueError(f"Worker demonstration action {desired} is masked for goal {candidate}")
                    worker.append((worker_obs, action_mask, desired))
                if policy is not None and learning[0, role] and rng.random() > beta:
                    with torch.no_grad():
                        choice = policy.logits(policy.worker, tensor(worker_inputs(obs[role], goal, flags)),
                                               tensor(valid_actions(env, chef, goal, synchronize=execution.synchronize,
                                                                    goal_lock=execution.goal_lock, route_progress=execution.route_progress,
                                                                    reserved=reserved))).argmax().item()
                else:
                    choice = scripted_action(env, chef, goal, reserved, synchronize=execution.synchronize)
                if rng.random() < (0.07 if policy is None else 0.025):
                    choice = STAY
                actions[chef] = choice
                reserve(env, chef, choice, reserved)
            _, done, _ = trial.step(trial.delayed(actions))
            if done:
                break
    return manager, worker


def collect_ppo(policy, trials, mode, execution, horizon, rng, profiles=None):
    records = []
    scores = []
    for _ in range(horizon):
        common, obs, masks, learning = encode(trials)
        labels = teacher_goals(trials)
        actions, goals, logp = act(trials, policy, obs, masks, learning, labels, sample=True)
        rewards, dones = [], []
        for index, (trial, joint) in enumerate(zip(trials, actions)):
            reward, done, _ = trial.step(trial.delayed(joint))
            rewards.append(reward / 20.0)
            dones.append(done)
            if done:
                scores.append(trial.reward)
                trials[index] = sample_trial(mode, profiles, execution, rng)
        records.append((common, obs, masks, learning, labels, goals, logp, rewards, dones))
    return records, encode(trials)[0], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multi", "reference"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-seed-base", type=int, default=9012)
    parser.add_argument("--test-episodes", type=int, default=18)
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--imitate-episodes", type=int, default=60)
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--anchor", type=float, default=0.15)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--reset-manager", action="store_true",
                        help="Load the pretrained Worker and start the Manager and critic from scratch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early", type=int, default=2)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--synchronize", action="store_true")
    parser.add_argument("--goal-lock", action="store_true",
                        help="At the target tile, face the station and finish or wait for a legal interaction.")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--lock-commitment", action="store_true")
    parser.add_argument("--queue-prefetch", action="store_true")
    parser.add_argument("--route-progress", action="store_true")
    parser.add_argument("--role-allocator", choices=("reference", "geodesic", "learned", "lookahead"), default="reference")
    parser.add_argument("--role-weights", type=Path)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--save-training-data", action="store_true")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="default")
    parser.add_argument("--training-profiles", choices=tuple(PROFILES), nargs="+")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Independent chance of a stay replacing each eligible action."
                             " In single-agent mode only the scripted teammates are delayed.")
    args = parser.parse_args()
    if args.reset_manager and (not args.initialize or args.evaluation_only):
        parser.error("--reset-manager requires --initialize in training mode")
    if args.evaluation_only and not args.initialize:
        parser.error("--evaluation-only requires --initialize")
    if not 0 <= args.delay < 1:
        parser.error("--delay must be in [0, 1)")
    if args.validate_every < 1 or args.test_episodes < 1:
        parser.error("--validate-every and --test-episodes must be positive")
    if args.profile == "continuous" or "continuous" in (args.training_profiles or ()):
        if not getattr(SharedKitchenConfig(), "continuous_orders", False):
            parser.error("The continuous profile requires the latest Shared environment")
        if args.training_profiles:
            parser.error("Continuous orders cannot be normalized against fixed order ceilings")
    if args.role_allocator == "learned" and args.role_weights is None:
        parser.error("--role-allocator learned requires --role-weights")
    role_policy = None
    if args.role_allocator == "learned":
        state = torch.load(args.role_weights, map_location="cpu", weights_only=True)
        role_policy = assignment_model(state["0.weight"].shape[-1])
        role_policy.load_state_dict(state)
        role_policy.eval()
    execution = Execution(early=args.early, pipeline=args.pipeline, synchronize=args.synchronize,
                          horizon=args.horizon, delay=args.delay, goal_lock=args.goal_lock,
                          profile=args.profile, lock_commitment=args.lock_commitment,
                          queue_prefetch=args.queue_prefetch, route_progress=args.route_progress,
                          role_allocator=args.role_allocator, role_policy=role_policy)
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    val = [(2001 + i, i % 3) for i in range(12)]
    test = [(args.test_seed_base + i, role if args.mode == "single" else None)
            for i in range(args.test_episodes) for role in (range(3) if args.mode == "single" else (0,))]
    start = monotonic()

    def record(stage, **values):
        entry = {"stage": stage, "seconds": round(monotonic() - start, 2), **values}
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def write_results(name, rows):
        (args.output / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")

    revision = CONTINUOUS_ENVIRONMENT_COMMIT if args.profile == "continuous" else ENVIRONMENT_COMMIT
    record("config", environment=revision, **vars(args) | {
        "output": portable_path(args.output), "initialize": portable_path(args.initialize),
        "role_weights": portable_path(args.role_weights),
    })
    if args.mode == "reference":
        rows = evaluate(None, "reference", test, execution, trace=True)
        write_results("native-reference", rows)
        record("native-reference", **summary(rows))
        for early in range(5):
            variant = replace(execution, early=early)
            rows = evaluate(None, "planner", test, variant, trace=True)
            write_results(f"planner-{early}", rows)
            record("planner", early=early, **summary(rows))
        return

    policy = SharedPolicy().to(args.device)
    if args.initialize:
        state = torch.load(args.initialize, map_location=args.device, weights_only=True)
        if args.reset_manager:
            state = {key.removeprefix("worker."): value for key, value in state.items()
                     if key.startswith("worker.")}
            policy.worker.load_state_dict(state)
        else:
            policy.load_state_dict(state)
    if args.evaluation_only:
        rows = evaluate(policy.cpu(), args.mode, test, execution, trace=True)
        write_results("evaluation", rows)
        record("evaluation", **summary(rows))
        return
    best = None
    best_stage = None

    def validate(stage):
        nonlocal best, best_stage
        cpu = deepcopy(policy).cpu()
        current = summary(evaluate(cpu, args.mode, val, execution))
        score = rank(current)
        if args.training_profiles:
            profiles = tuple(dict.fromkeys((args.profile, *args.training_profiles)))
            reports = {name: current if name == args.profile else summary(
                evaluate(cpu, args.mode, val[:6], replace(execution, profile=name))
            ) for name in profiles}
            ratio = {name: r["mean_score"] / profile_ceiling(name) for name, r in reports.items()}
            score = (float(np.mean(list(ratio.values()))), min(ratio.values()),
                     float(np.mean([r["full_rate"] for r in reports.values()])),
                     -sum(r["full_ticks"] or args.horizon for r in reports.values()))
            record(stage, profiles=reports, normalized_score=score[0], **current)
        else:
            record(stage, **current)
        if best is None or score > best:
            best = score
            best_stage = stage
            torch.save(policy.state_dict(), args.output / "best.pt")
        return cpu

    if args.initialize:
        cpu = validate("initialized")
    else:
        examples = collect(None, args.mode, args.imitate_episodes, execution, rng, beta=1.0, profiles=args.training_profiles)
        record("imitation", **imitation(policy, examples, 25, args.device, rng))
        cpu = validate("after_imitation")
        for iteration in range(args.refine):
            additions = collect(cpu, args.mode, max(16, args.imitate_episodes // 2), execution, rng, beta=0.35, profiles=args.training_profiles)
            examples[0].extend(additions[0])
            examples[1].extend(additions[1])
            record("recovery_imitation", iteration=iteration + 1,
                   **imitation(policy, examples, 12, args.device, rng))
            cpu = validate("after_recovery")
        if args.save_training_data:
            for name, rows in zip(("manager", "worker"), examples):
                obs, masks, choices = (np.stack(column) for column in zip(*rows))
                np.savez_compressed(args.output / f"{name}-demonstrations.npz",
                                    observations=obs, masks=masks, labels=choices)

    policy.load_state_dict(torch.load(args.output / "best.pt", map_location=args.device, weights_only=True))
    cpu = deepcopy(policy).cpu()
    optimizer = torch.optim.Adam([
        {"params": policy.manager.parameters(), "lr": 0.00025},
        {"params": policy.critic.parameters(), "lr": 0.001},
    ])
    trials = [sample_trial(args.mode, args.training_profiles, execution, rng) for _ in range(args.lanes)]
    for update in range(1, args.updates + 1):
        rollout = collect_ppo(cpu, trials, args.mode, execution, 192, rng, profiles=args.training_profiles)
        ppo_update(policy, optimizer, rollout, args.device, args.anchor)
        if args.save_training_data:
            keys = ("state", "observation", "mask", "learning", "teacher", "goals",
                    "old_logp", "reward_in_meals", "done")
            arrays = {key: np.asarray(column) for key, column in zip(keys, zip(*rollout[0]))}
            np.savez_compressed(args.output / f"rollout-{update:03d}.npz",
                                **arrays, bootstrap_state=rollout[1])
        record("ppo_update", update=update, environment_steps=update * args.lanes * 192,
               episodes=len(rollout[2]), sampled_mean=float(np.mean(rollout[2])) if rollout[2] else None)
        if update % args.validate_every == 0 or update == args.updates:
            cpu = validate("after_ppo")
        else:
            cpu = deepcopy(policy).cpu()
    torch.save(policy.state_dict(), args.output / "last.pt")
    cpu.load_state_dict(torch.load(args.output / "best.pt", map_location="cpu", weights_only=True))
    rows = evaluate(cpu, args.mode, test, execution, trace=True)
    write_results("best-evaluation", rows)
    record("best-evaluation", selected_stage=best_stage, **summary(rows))
    if args.profile == "continuous":
        reference = evaluate(None, "reference", [(s, None) for s in sorted({seed for seed, _ in test})], execution)
        write_results("official-reference", reference)
        record("official-reference", **summary(reference))
    guided = evaluate(cpu, args.mode, test, execution, scripted_worker=True)
    write_results("scripted-worker-evaluation", guided)
    record("scripted-worker-evaluation", **summary(guided))


if __name__ == "__main__":
    main()

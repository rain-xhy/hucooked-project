"""Train a sequential CoMeDi population with a centralized three-chef actor."""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
from typing import Any

import numpy as np
import torch

DEFAULT_STEPS = 16_000_000
DEFAULT_POPULATION = 8
DEFAULT_ROUNDS = 128
DEFAULT_ENVS = 16
DEFAULT_ROLLOUT = 300
DEFAULT_WORKERS = 2
DEFAULT_EPOCHS = 4
DEFAULT_MINIBATCHES = 8
DEFAULT_LR = 3e-4
DEFAULT_SAVE_INTERVAL = 250_000
DEFAULT_EVAL_INTERVAL = 200_000
DEFAULT_EVAL_EPISODES = 8
DEFAULT_MAX_SECONDS = 0
GAMMA = 0.9985
GAE_LAMBDA = 0.95
CLIP = 0.2
ENTROPY_COEF = 0.001
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
XP_WEIGHT = 0.5
MP_WEIGHT = 1.0
MP_PREFIX_MAX = 64
MP_CURRENT_PROB = 0.5
CBR_FRACTION = 0.10
CBR_SP_FRACTION = 0.5
CBR_BC_COEF = 0.5
BC_EPOCHS = 64
BC_BATCH_SIZE = 512

NATIVE_METRICS = (
    "team_score", "meals_completed", "normal_completed", "joint_completed",
    "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
    "three_contributor_meals", "resource_conflicts", "movement_conflicts",
    "ticks", "meals_per_100_steps",
)
SUMMARY_TOTALS = ("training_reward", "training_shaping")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("rl", "bc_rl"), default="rl")
    result.add_argument("--output", required=True, help="unique run artifact directory")
    result.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="total joint transitions, including BC demonstrations once")
    result.add_argument("--physical-gpu", type=int, choices=(1,), default=1)
    result.add_argument("--memory-fraction", type=float, default=0.035)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                        help="wall-clock limit in seconds; 0 disables the limit")
    result.add_argument("--reward-profile", choices=("legacy", "ppo_task"), default="legacy")
    result.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    result.add_argument("--rollout", type=int, default=DEFAULT_ROLLOUT)
    result.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    result.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    result.add_argument("--rounds-per-convention", type=int, default=DEFAULT_ROUNDS)
    result.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    result.add_argument("--minibatches", type=int, default=DEFAULT_MINIBATCHES)
    result.add_argument("--lr", type=float, default=DEFAULT_LR)
    result.add_argument("--xp-probe-fraction", type=float, default=0.20)
    result.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    result.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    result.add_argument("--checkpoint-interval", "--save-interval", dest="save_interval",
                        type=int, default=DEFAULT_SAVE_INTERVAL)
    result.add_argument("--demonstrations", default=None)
    result.add_argument("--bc-epochs", type=int, default=BC_EPOCHS)
    result.add_argument("--smoke", action="store_true",
                        help="run small complete SP/XP/MP/CBR stages on CPU-friendly settings")
    return result


def _round_down(value: int, quantum: int) -> int:
    return max(0, int(value) // int(quantum) * int(quantum))


def _round_up(value: int, quantum: int) -> int:
    value, quantum = int(value), int(quantum)
    return max(0, ((value + quantum - 1) // quantum) * quantum)


def effective_args(args: argparse.Namespace, *, demonstration_steps=None) -> argparse.Namespace:
    """Apply smoke overrides once while preserving the requested BC epoch count."""
    args = deepcopy(args)
    if getattr(args, "_joint_comedi_effective_args", False):
        return args
    args.cbr_fraction = float(getattr(args, "cbr_fraction", CBR_FRACTION))
    args.cbr_sp_fraction = CBR_SP_FRACTION
    args.cbr_bc_coef = CBR_BC_COEF
    args.xp_weight = XP_WEIGHT
    args.mp_weight = MP_WEIGHT
    args.mp_prefix_max = MP_PREFIX_MAX
    args.mp_current_probability = MP_CURRENT_PROB
    args.gamma = GAMMA
    args.gae_lambda = GAE_LAMBDA
    args.clip = CLIP
    args.entropy_coef = ENTROPY_COEF
    args.value_coef = VALUE_COEF
    args.max_grad_norm = MAX_GRAD_NORM
    if args.mode == "bc_rl" and not args.demonstrations:
        raise ValueError("--demonstrations is required for --mode bc_rl")
    if args.smoke:
        if args.population < 2:
            raise ValueError("CoMeDi smoke requires at least two conventions for XP and MP")
        args.population = 2
        args.rounds_per_convention = 2
        args.num_envs = min(int(args.num_envs), 2)
        args.rollout = min(int(args.rollout), 8)
        args.workers = 0
        args.epochs = 1
        args.minibatches = 1
        args.eval_episodes = 1
        args.cbr_fraction = 0.25
        demo_steps = 0
        if args.mode == "bc_rl":
            if demonstration_steps is None:
                from joint_common.demonstrations import demonstration_steps as get_steps

                demo_steps = int(get_steps(args.demonstrations))
            else:
                demo_steps = int(demonstration_steps)
        smoke_rl_steps = max(args.num_envs * args.rollout * 6, args.num_envs * 48)
        args.steps = min(int(args.steps), int(demo_steps) + smoke_rl_steps)

    if args.population < 2:
        raise ValueError("population must be >= 2 so XP and MP are trained")
    if args.num_envs <= 0 or args.rollout < 2 or args.steps <= 0:
        raise ValueError("steps/num-envs must be positive and rollout must be >= 2")
    if args.workers < 0 or args.rounds_per_convention <= 0:
        raise ValueError("workers cannot be negative and rounds-per-convention must be positive")
    if args.epochs <= 0 or args.minibatches <= 0 or args.lr <= 0.0:
        raise ValueError("PPO epochs, minibatches, and learning rate must be positive")
    if args.bc_epochs <= 0:
        raise ValueError("bc-epochs must remain positive")
    if args.max_seconds < 0 or args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("max-seconds must be nonnegative and checkpoint/evaluation intervals must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("eval-episodes must be positive")
    if not 0.0 < args.cbr_fraction < 1.0:
        raise ValueError("CBR fraction must be in (0, 1)")
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in (0, 1]")
    if not 0.0 < args.xp_probe_fraction < 0.25:
        raise ValueError("xp-probe-fraction must be in (0, 0.25)")
    if torch.device(args.device).type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if args.mode not in ("rl", "bc_rl"):
        raise ValueError("mode must be rl or bc_rl")
    args._joint_comedi_effective_args = True
    return args


def allocate_budgets(args: argparse.Namespace, *, demonstration_env_steps: int = 0) -> dict[str, Any]:
    """Allocate one total vector-step budget to conventions and final CBR."""
    quantum = int(args.num_envs)
    demonstration_env_steps = int(demonstration_env_steps)
    if demonstration_env_steps < 0 or demonstration_env_steps >= args.steps:
        raise ValueError("demonstration transitions must be nonnegative and below total steps")
    requested_rl = int(args.steps) - demonstration_env_steps
    effective_rl = _round_up(requested_rl, quantum)
    effective_total = demonstration_env_steps + effective_rl
    cbr_total = _round_down(effective_rl * (args.cbr_fraction if hasattr(args, "cbr_fraction") else CBR_FRACTION), quantum)
    population_total = effective_rl - cbr_total
    per_policy = _round_down(population_total // args.population, quantum)
    policy_budgets = [per_policy] * args.population
    policy_budgets[-1] += population_total - per_policy * args.population

    conventions = []
    for index, policy_total in enumerate(policy_budgets):
        if index == 0:
            phases = {"sp": policy_total, "xp": 0, "mp": 0}
        else:
            sp_fraction, xp_fraction = 0.5, 0.25
            sp = _round_down(policy_total * sp_fraction, quantum)
            xp = _round_down(policy_total * xp_fraction, quantum)
            phases = {"sp": sp, "xp": xp, "mp": policy_total - sp - xp}
        conventions.append(phases)
    cbr_sp = _round_down(cbr_total * CBR_SP_FRACTION, quantum)
    cbr = {"cbr_sp": cbr_sp, "cbr_bc": cbr_total - cbr_sp}

    round_min = quantum * int(args.rounds_per_convention)
    for index, phases in enumerate(conventions):
        for phase in (("sp",) if index == 0 else ("sp", "xp", "mp")):
            minimum = round_min * (2 if phase in ("xp", "mp") else 1)
            if phases[phase] < minimum:
                raise ValueError(
                    f"budget too small: convention {index} {phase} needs {minimum} steps"
                )
    for phase in ("cbr_sp", "cbr_bc"):
        if cbr[phase] < round_min:
            raise ValueError(f"budget too small: {phase} needs {round_min} steps")
    return {
        "requested_total": int(args.steps),
        "effective_total": effective_total,
        "demonstration_env_steps": demonstration_env_steps,
        "effective_rl_total": effective_rl,
        "population_total": population_total,
        "cbr_total": cbr_total,
        "conventions": conventions,
        "cbr": cbr,
    }


def _round_budgets(total: int, rounds: int, quantum: int) -> list[int]:
    if total == 0:
        return [0] * rounds
    each = _round_down(total // rounds, quantum)
    if each == 0:
        raise ValueError("budget per round is below one vectorized environment step")
    result = [each] * rounds
    result[-1] += total - sum(result)
    return result


def _episode_means(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"episodes": 0}
    metrics: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        missing = [key for key in (*NATIVE_METRICS, *SUMMARY_TOTALS) if key not in summary]
        if missing:
            raise ValueError(f"episode summary is missing required native fields: {missing}")
        for key in (*NATIVE_METRICS, *SUMMARY_TOTALS):
            metrics[key].append(float(summary[key]))
    result = {"episodes": len(summaries)}
    result.update({f"mean_{key}": float(np.mean(values)) for key, values in metrics.items()})
    return result


def evaluate_policy(actor, args, device, *, eval_index: int, dependencies,
                    should_stop=lambda: False) -> dict[str, Any]:
    from .algorithm import sample_joint_actions

    env = dependencies.envs.KitchenBatch(
        num_envs=min(int(args.num_envs), int(args.eval_episodes)),
        seed=int(args.seed) + 100_000 + int(eval_index) * 1009,
        stage="standard",
        workers=0,
        train=False,
        gamma=GAMMA,
        reward_profile=args.reward_profile,
    )
    summaries: list[dict[str, Any]] = []
    env_steps = 0
    num_envs = int(env.num_envs)
    try:
        observation = env.reset()
        max_steps = max(1, int(args.eval_episodes) * 2_000 // num_envs)
        for _ in range(max_steps):
            if should_stop():
                break
            actions, _ = sample_joint_actions(actor, observation, device, deterministic=True)
            observation, _rewards, _dones, infos = env.step(actions)
            env_steps += num_envs
            summaries.extend(
                dict(info["episode_summary"])
                for info in infos if "episode_summary" in info
            )
            if len(summaries) >= args.eval_episodes:
                summaries = summaries[:args.eval_episodes]
                break
    finally:
        env.close()
    result = _episode_means(summaries)
    result.update({"env_steps": env_steps, "complete": len(summaries) >= args.eval_episodes})
    return result


def main() -> None:
    from .runner import run

    run(effective_args(parser().parse_args()))


if __name__ == "__main__":
    main()

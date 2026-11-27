"""Train a sequential population of three-role CoMeDi conventions.

Run from ``src/stq`` with ``python -m comedi.train``.
The step budget counts vectorized joint environment transitions, not individual
chef actions. ``--smoke`` runs the complete SP, XP, MP, and convention-aware
training path with a two-policy population and a small CPU-friendly budget.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch

from baseline_common.runtime import atomic_json, atomic_save, setup_device, source_hashes
from .algorithm import (
    collect_cross_play,
    collect_mixed_play,
    collect_self_play,
    collect_xp_probe,
)
from .model import ACTION_DIM, NUM_ROLES, ComediPolicy, select_most_compatible_prior


POPULATION_PHASES = ("sp", "xp", "mp")
CBR_PHASES = ("cbr_sp", "cbr_bc")
NATIVE_METRICS = (
    "team_score",
    "meals_completed",
    "normal_completed",
    "joint_completed",
    "joint_successes",
    "joint_attempt_success_rate",
    "joint_wait_steps",
    "three_contributor_meals",
    "resource_conflicts",
    "movement_conflicts",
    "ticks",
    "meals_per_100_steps",
)
SUMMARY_TOTALS = ("training_reward", "training_shaping")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True, help="run artifact directory")
    result.add_argument("--steps", type=int, default=16_000_000,
                        help="total joint transitions, including demos and CBR")
    result.add_argument("--num-envs", type=int, default=8)
    result.add_argument("--rollout", type=int, default=128,
                        help="maximum vector steps collected before each PPO update")
    result.add_argument("--population", type=int, default=8)
    result.add_argument("--rounds-per-convention", type=int, default=16,
                        help="joint SP/XP/MP collection and optimization rounds per policy")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--gamma", type=float, default=0.9985)
    result.add_argument("--gae-lambda", type=float, default=0.95)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--minibatches", type=int, default=8)
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--clip", type=float, default=0.2)
    result.add_argument("--entropy-coef", type=float, default=0.001)
    result.add_argument("--value-coef", type=float, default=0.5)
    result.add_argument("--max-grad-norm", type=float, default=0.5)
    result.add_argument("--sp-fraction", type=float, default=0.5,
                        help="share of each later convention's budget assigned to SP")
    result.add_argument("--xp-fraction", type=float, default=0.25,
                        help="share of each later convention's budget assigned to XP")
    result.add_argument("--xp-weight", type=float, default=0.5,
                        help="positive alpha; the clipped XP actor objective is weighted by -alpha")
    result.add_argument("--xp-probe-fraction", type=float, default=0.20,
                        help="share of each XP round used to refresh the max-compatible prior")
    result.add_argument("--mp-weight", type=float, default=1.0,
                        help="positive mixed-play actor objective weight")
    result.add_argument("--mp-prefix-max", type=int, default=64)
    result.add_argument("--mp-current-prob", type=float, default=0.5)
    result.add_argument("--cbr-fraction", type=float, default=0.10,
                        help="share of total transitions reserved for convention-aware training")
    result.add_argument("--cbr-sp-fraction", type=float, default=0.5,
                        help="share of CBR transitions for its SP PPO phase; remainder is SP PPO+BC")
    result.add_argument("--cbr-bc-coef", type=float, default=0.5)
    result.add_argument("--eval-interval", type=int, default=200_000)
    result.add_argument("--eval-episodes", type=int, default=8)
    result.add_argument("--save-interval", type=int, default=250_000)
    result.add_argument("--max-seconds", type=int, default=None,
                        help="stop safely after this many wall-clock seconds")
    result.add_argument("--resume", default=None, help="checkpoint path to resume")
    result.add_argument("--warm-start-conventions", action="store_true",
                        help="initialize each new convention from the previous actor")
    result.add_argument("--demonstrations", default=None,
                        help="native SharedKitchen demonstration dataset for actor pretraining")
    result.add_argument("--bc-epochs", type=int, default=64,
                        help="behavior-cloning epochs for each new convention actor")
    result.add_argument("--smoke", action="store_true",
                        help="shrink the budget while still running SP, XP, MP, and CBR")
    return result


def _round_down(value: int, quantum: int) -> int:
    return max(0, int(value) // int(quantum) * int(quantum))


def _round_up(value: int, quantum: int) -> int:
    value, quantum = int(value), int(quantum)
    return max(0, ((value + quantum - 1) // quantum) * quantum)


def effective_args(args: argparse.Namespace) -> argparse.Namespace:
    args = deepcopy(args)
    if getattr(args, "_comedi_effective_args", False):
        return args
    if args.smoke:
        if args.population < 2:
            raise ValueError("CoMeDi smoke mode requires population >= 2 for XP and MP")
        args.population = 2
        args.rounds_per_convention = 2
        args.num_envs = min(args.num_envs, 2)
        args.rollout = min(args.rollout, 8)
        args.epochs = 1
        args.minibatches = 1
        args.eval_episodes = 1
        args.cbr_fraction = 0.25
        demonstration_env_steps = 0
        if args.demonstrations:
            from baseline_common.demonstrations import demonstration_steps

            demonstration_env_steps = int(demonstration_steps(args.demonstrations))
        smoke_rl_steps = max(args.num_envs * args.rollout * 6, args.num_envs * 48)
        smoke_total_steps = demonstration_env_steps + smoke_rl_steps
        args.steps = min(args.steps, smoke_total_steps)
    if args.population < 2:
        raise ValueError("population must be >= 2 so the XP and MP phases are trained")
    if args.num_envs <= 0 or args.rollout < 2 or args.steps <= 0:
        raise ValueError("steps and num-envs must be positive, and rollout must be at least 2")
    if args.rounds_per_convention <= 0:
        raise ValueError("rounds-per-convention must be positive")
    if args.epochs <= 0 or args.minibatches <= 0 or args.lr <= 0 or args.bc_epochs <= 0:
        raise ValueError("PPO/BC epochs, minibatches, and learning rate must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max-seconds must be a positive integer")
    if args.mp_prefix_max <= 0:
        raise ValueError("mp-prefix-max must be positive")
    if not 0.0 <= args.mp_current_prob <= 1.0:
        raise ValueError("mp-current-prob must be in [0, 1]")
    if not 0.0 < args.cbr_fraction < 1.0 or not 0.0 < args.cbr_sp_fraction < 1.0:
        raise ValueError("CBR fractions must be strictly between zero and one")
    if not 0.0 < args.xp_probe_fraction < 0.25:
        raise ValueError("xp-probe-fraction must be in (0, 0.25)")
    if not 0.0 < args.sp_fraction < 1.0 or not 0.0 < args.xp_fraction < 1.0:
        raise ValueError("SP and XP fractions must be strictly between zero and one")
    if args.sp_fraction + args.xp_fraction >= 1.0:
        raise ValueError("sp-fraction + xp-fraction must be less than one (MP gets the remainder)")
    if args.xp_weight < 0.0 or args.mp_weight < 0.0 or args.cbr_bc_coef < 0.0:
        raise ValueError("XP, MP, and BC coefficients must be nonnegative")
    args._comedi_effective_args = True
    return args


def allocate_budgets(
    args: argparse.Namespace, *, demonstration_env_steps: int = 0
) -> dict[str, Any]:
    """Allocate total joint-step budget after reserving dataset transitions."""
    quantum = args.num_envs
    demonstration_env_steps = int(demonstration_env_steps)
    if demonstration_env_steps < 0 or demonstration_env_steps >= args.steps:
        raise ValueError("demonstration transitions must be nonnegative and below the total budget")
    requested_rl = args.steps - demonstration_env_steps
    total_rl = _round_up(requested_rl, quantum)
    total = demonstration_env_steps + total_rl
    cbr_total = _round_down(total_rl * args.cbr_fraction, quantum)
    population_total = total_rl - cbr_total
    per_policy = _round_down(population_total // args.population, quantum)
    policy_budgets = [per_policy] * args.population
    policy_budgets[-1] += population_total - per_policy * args.population

    conventions: list[dict[str, int]] = []
    for index, policy_total in enumerate(policy_budgets):
        if index == 0:
            phases = {"sp": policy_total, "xp": 0, "mp": 0}
        else:
            sp = _round_down(policy_total * args.sp_fraction, quantum)
            xp = _round_down(policy_total * args.xp_fraction, quantum)
            phases = {"sp": sp, "xp": xp, "mp": policy_total - sp - xp}
        conventions.append(phases)

    cbr_sp = _round_down(cbr_total * args.cbr_sp_fraction, quantum)
    cbr = {"cbr_sp": cbr_sp, "cbr_bc": cbr_total - cbr_sp}
    min_phase = quantum * args.rounds_per_convention
    for index, phases in enumerate(conventions):
        for phase in (("sp",) if index == 0 else POPULATION_PHASES):
            min_required = min_phase * (2 if phase in ("xp", "mp") else 1)
            if phases[phase] < min_required:
                raise ValueError(
                    f"budget too small: convention {index} phase {phase} needs at least "
                    f"{min_required} jointenvsteps for {args.rounds_per_convention} rounds"
                )
    for phase in CBR_PHASES:
        if cbr[phase] < min_phase:
            raise ValueError(f"budget too small: {phase} needs at least {min_phase} jointenvsteps")
    return {
        "requested_total": args.steps,
        "effective_total": total,
        "demonstration_env_steps": demonstration_env_steps,
        "effective_rl_total": total_rl,
        "population_total": population_total,
        "cbr_total": cbr_total,
        "conventions": conventions,
        "cbr": cbr,
    }


def _episode_means(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"episodes": 0}
    metrics: dict[str, list[float]] = defaultdict(list)
    for summary in summaries:
        missing = [key for key in (*NATIVE_METRICS, *SUMMARY_TOTALS) if key not in summary]
        if missing:
            raise ValueError(f"SharedKitchen episode summary is missing fields: {missing}")
        for key in (*NATIVE_METRICS, *SUMMARY_TOTALS):
            metrics[key].append(float(summary[key]))
    result: dict[str, Any] = {"episodes": len(summaries)}
    result.update({f"mean_{key}": float(np.mean(values)) for key, values in metrics.items()})
    return result


def evaluate_policy(actor, args, device, *, eval_index: int, should_stop) -> dict[str, Any]:
    """Validate native SharedKitchen episode metrics with shaping disabled."""
    from baseline_common.envs import KitchenBatch

    num_envs = min(args.num_envs, max(1, args.eval_episodes))
    env = KitchenBatch(
        num_envs=num_envs,
        seed=args.seed + 100_000 + eval_index * 1009,
        workers=0,
        train=False,
        gamma=args.gamma,
    )
    summaries: list[dict[str, Any]] = []
    env_steps = 0
    try:
        observation = env.reset()
        max_vector_steps = max(1, args.eval_episodes * 2_000 // num_envs)
        for _ in range(max_vector_steps):
            if should_stop():
                break
            actions, _ = _deterministic_actions(actor, observation, device)
            observation, _, dones, infos = env.step(actions)
            env_steps += num_envs
            for info in infos:
                if "episode_summary" in info:
                    summaries.append(info["episode_summary"])
                    if len(summaries) >= args.eval_episodes:
                        break
            if len(summaries) >= args.eval_episodes:
                break
    finally:
        env.close()
    result = _episode_means(summaries)
    result.update({"env_steps": env_steps, "complete": len(summaries) >= args.eval_episodes})
    return result


def _deterministic_actions(actor, observation, device):
    from .algorithm import sample_joint_actions

    return sample_joint_actions(actor, observation, device, deterministic=True)


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _log_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def _phase_name(cursor: dict[str, Any]) -> str:
    phase = cursor.get("phase", "initializing")
    if phase == "sp" or phase == "xp" or phase == "mp":
        return f"convention_{cursor.get('policy_index', 0)}_{phase}"
    return str(phase)


def main() -> None:
    from .runner import run

    run(effective_args(parser().parse_args()))


if __name__ == "__main__":
    main()

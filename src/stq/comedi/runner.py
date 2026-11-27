"""Round-based CoMeDi orchestration with joint SP/XP/MP actor updates."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import signal
import time
from typing import Any, Callable

import numpy as np
import torch

from .algorithm import (
    CROSSPLAY_ROLE_MASKS,
    collect_cross_play,
    collect_mixed_play,
    collect_self_play,
    collect_teacher_self_play,
    collect_xp_probe,
    update_bundle,
)
from .model import ComediPolicy, select_most_compatible_prior
from .train import (
    _episode_means,
    _round_down,
    allocate_budgets,
    effective_args,
    evaluate_policy,
    parser,
)
from baseline_common.runtime import atomic_json, atomic_save, setup_device, source_hashes


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_budgets(total: int, rounds: int, quantum: int) -> list[int]:
    if total == 0:
        return [0] * rounds
    each = _round_down(total // rounds, quantum)
    if each == 0:
        raise ValueError("budget per round is below one vectorized environment step")
    result = [each] * rounds
    result[-1] += total - sum(result)
    return result


def _mp_vector_chunk(remaining_units: int, rollout: int) -> int:
    """Keep MP chunks long enough to contain prefix and suffix data."""
    remaining_units, rollout = map(int, (remaining_units, rollout))
    if remaining_units < 2:
        raise ValueError("MP phase budget must retain at least two vector steps")
    chunk = min(remaining_units, rollout)
    if remaining_units - chunk == 1:
        chunk += 1
    return chunk


def _merge_batches(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("cannot merge an empty objective buffer")
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks):
        raise ValueError("rollout chunks for an objective have different fields")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _policy_summary(policy: ComediPolicy) -> dict[str, int]:
    return {
        "actor_parameters": sum(parameter.numel() for parameter in policy.actor.parameters()),
        "sp_critic_parameters": sum(parameter.numel() for parameter in policy.sp_critics.parameters()),
        "mp_critic_parameters": sum(parameter.numel() for parameter in policy.mp_critics.parameters()),
        "xp_critic_count": len(policy.xp_critics),
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
    }


def run(args) -> None:
    args = effective_args(args)
    from baseline_common.envs import ENVIRONMENT_COMMIT, KitchenBatch

    device = setup_device(args.device, args.seed)
    demonstration_env_steps = 0
    demonstration_hash = None
    if args.demonstrations:
        from baseline_common.demonstrations import demonstration_steps

        demonstration_path = Path(args.demonstrations).expanduser().resolve()
        if not demonstration_path.is_file():
            raise FileNotFoundError(f"demonstration dataset not found: {demonstration_path}")
        args.demonstrations = str(demonstration_path)
        demonstration_env_steps = int(demonstration_steps(demonstration_path))
        demonstration_hash = _sha256_file(demonstration_path)
    budgets = allocate_budgets(args, demonstration_env_steps=demonstration_env_steps)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    status_path = output / "status.json"
    latest_path = output / "latest.pt"
    metrics_path = output / "metrics.jsonl"
    if config_path.exists() and not args.resume:
        raise FileExistsError("output contains config.json; choose another directory or use --resume")

    source_manifest = source_hashes("comedi")
    if demonstration_hash is not None:
        source_manifest[f"demonstrations:{args.demonstrations}"] = demonstration_hash
    training_config = {
        key: value for key, value in vars(args).items() if not key.startswith("_")
    }
    manifest = {
        "algorithm": "CoMeDi",
        "arguments": training_config,
        "budgets_jointenvsteps": budgets,
        "source_hashes": source_manifest,
        "protocol": {
            "population": "sequential growth; each new convention starts from fresh random weights",
            "actor_update": "one joint SP + negative weighted XP + positive MP update per round",
            "xp_roles": "all six nonempty proper current/prior role masks for three agents",
            "xp_selection": "maximum current-round cross-play score; low-env-count probes are sampled approximations",
            "mp": "random old/new prefix, current-policy suffix only",
            "cbr": "each round collects student SP for PPO and frozen-teacher self-play for BC only; teacher IDs stay fixed within each episode",
            "demonstration_warm_start": "the native rules dataset initializes every new convention and the CBR actor; its environment steps are budgeted once",
        },
        "budget_rounding": "requested joint steps round up to a vector boundary; phase allocations end at complete rounds",
        "checkpoint_retention": "latest.pt plus immutable step_<actual_joint_steps>.pt snapshots at save intervals and final completion",
        "environment": "SharedKitchen",
        "environment_commit": ENVIRONMENT_COMMIT,
        "demonstrations": {
            "path": args.demonstrations,
            "env_steps": demonstration_env_steps,
            "bc_epochs": args.bc_epochs,
            "batch_size": 512,
        },
        "observation": {"vector": [3, 866], "state": [863], "spatial": [3, 26, 7, 7]},
        "native_episode_metrics": [
            "team_score", "meals_completed", "normal_completed", "joint_completed",
            "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
            "three_contributor_meals", "resource_conflicts", "movement_conflicts",
            "ticks", "meals_per_100_steps",
        ],
        "episode_summary_additions": ["episode", "training_reward", "training_shaping"],
        "action_dim": 6,
        "roles": 3,
        "validation": "SharedKitchen team_score with KitchenBatch(train=False)",
        "resume_semantics": "checkpoints are saved at complete round boundaries; environments reset on resume",
    }

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("source_hashes") != source_manifest:
            raise ValueError("existing run config has different source hashes")
    else:
        launcher_metadata = {"queue.json", "train.pid", "train.log"}
        unexpected = [path.name for path in output.iterdir() if path.name not in launcher_metadata]
        if unexpected:
            raise FileExistsError(f"output contains unexpected files: {unexpected}")
        atomic_json(config_path, manifest)

    quantum = args.num_envs
    conventions: list[ComediPolicy] = []
    current: ComediPolicy | None = None
    cbr: ComediPolicy | None = None
    cursor: dict[str, Any] = {
        "policy_index": 0,
        "round_index": 0,
        "policy_steps": 0,
        "xp_combo_cursor": 0,
        "xp_role_cursor": 0,
        "bc_warmstarted_policies": [],
        "cbr_round_index": 0,
        "cbr_steps": 0,
        "cbr_teacher_cursor": 0,
        "cbr_bc_warmstarted": False,
    }
    counters: dict[str, Any] = {
        "total_train_env_steps": 0,
        "total_demonstration_env_steps": 0,
        "total_rl_env_steps": 0,
        "phases": {
            "sp": 0,
            "xp_selection": 0,
            "xp_optimization": 0,
            "mp_prefix": 0,
            "mp_suffix": 0,
            "cbr_sp": 0,
            "cbr_bc": 0,
        },
        "cbr_teacher_env_steps": [0] * args.population,
        "phase_wallclock_seconds": {},
        "round_wallclock_seconds": {"population": 0.0, "cbr": 0.0},
        "validation_env_steps": 0,
    }
    best_validation_score: float | None = None
    best_validation: dict[str, Any] | None = None
    best_population_score: float | None = None
    best_population_validation: dict[str, Any] | None = None
    elapsed_base = 0.0
    eval_index = 0

    if args.resume:
        snapshot = torch.load(Path(args.resume).expanduser().resolve(), map_location=device, weights_only=False)
        if snapshot["source_hashes"] != source_manifest:
            raise ValueError("resume source hashes differ; use the original code revision")
        saved_config = snapshot["training_config"]
        for key in ("steps", "num_envs", "rollout", "population", "rounds_per_convention",
                    "gamma", "gae_lambda", "seed", "warm_start_conventions",
                    "demonstrations", "bc_epochs"):
            if saved_config.get(key) != training_config.get(key):
                raise ValueError(f"resume protocol mismatch: {key}")
        conventions = []
        for index, state in enumerate(snapshot["conventions"]):
            policy = ComediPolicy(index, lr=args.lr).to(device)
            policy.load_checkpoint(state)
            policy.freeze_actor()
            conventions.append(policy)
        cursor = snapshot["cursor"]
        counters = snapshot["counters"]
        best_validation_score = snapshot.get("best_validation_score")
        best_validation = snapshot.get("best_validation")
        best_population_score = snapshot.get("best_population_score")
        best_population_validation = snapshot.get("best_population_validation")
        elapsed_base = float(snapshot.get("elapsed_seconds", 0.0))
        eval_index = int(snapshot.get("eval_index", 0))
        if snapshot.get("current") is not None:
            index = int(cursor["policy_index"])
            current = ComediPolicy(index, lr=args.lr).to(device)
            current.load_checkpoint(snapshot["current"])
        if snapshot.get("cbr") is not None:
            cbr = ComediPolicy(0, lr=args.lr).to(device)
            cbr.load_checkpoint(snapshot["cbr"])
        if not config_path.exists():
            atomic_json(config_path, manifest)
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        torch.set_rng_state(snapshot["torch_rng"].cpu())
        if torch.cuda.is_available() and snapshot.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
    else:
        if any(output.iterdir()) and config_path.exists():
            allowed = {"config.json", "queue.json", "train.pid", "train.log"}
            unexpected = [path.name for path in output.iterdir() if path.name not in allowed]
            if unexpected:
                raise FileExistsError(f"output contains unexpected files: {unexpected}")
        current = ComediPolicy(0, lr=args.lr).to(device)

    next_checkpoint_at = (
        (int(counters["total_train_env_steps"]) // int(args.save_interval) + 1)
        * int(args.save_interval)
        if int(args.save_interval) > 0 else None
    )

    stop_requested = False
    previous_sigterm = None

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    try:
        previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    except (ValueError, AttributeError):
        previous_sigterm = None
    run_started = time.monotonic()

    def elapsed() -> float:
        return elapsed_base + time.monotonic() - run_started

    def step_counts() -> dict[str, int]:
        return {
            "demonstration_env_steps": int(counters["total_demonstration_env_steps"]),
            "rl_env_steps": int(counters["total_rl_env_steps"]),
            "total_env_steps": int(counters["total_train_env_steps"]),
        }

    def log_event(event: dict[str, Any]) -> None:
        event.setdefault("step_counts", step_counts())
        _log_event(metrics_path, event)

    def should_stop() -> bool:
        return stop_requested or (
            args.max_seconds is not None and time.monotonic() - run_started >= args.max_seconds
        )

    def payload() -> dict[str, Any]:
        return _cpu_tree({
            "schema_version": 2,
            "training_config": training_config,
            "source_hashes": source_manifest,
            "budgets_jointenvsteps": budgets,
            "conventions": [policy.checkpoint() for policy in conventions],
            "current": current.checkpoint() if current is not None else None,
            "cbr": cbr.checkpoint() if cbr is not None else None,
            "cursor": cursor,
            "counters": counters,
            "best_validation_score": best_validation_score,
            "best_validation": best_validation,
            "best_population_score": best_population_score,
            "best_population_validation": best_population_validation,
            "eval_index": eval_index,
            "elapsed_seconds": elapsed(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        })

    def save_step_snapshot(snapshot: dict[str, Any] | None = None) -> Path:
        step = int(counters["total_train_env_steps"])
        step_path = output / f"step_{step}.pt"
        if not step_path.exists():
            atomic_save(step_path, snapshot if snapshot is not None else payload())
        return step_path

    def save_latest(state: str = "running", *, retain_step: bool = False) -> None:
        snapshot = payload()
        atomic_save(latest_path, snapshot)
        if retain_step:
            save_step_snapshot(snapshot)
        atomic_json(status_path, {
            "state": state,
            "total_train_env_steps": counters["total_train_env_steps"],
            **step_counts(),
            "requested_budget_jointenvsteps": budgets["requested_total"],
            "budget_jointenvsteps": budgets["effective_total"],
            "cursor": cursor.copy(),
            "wallclock_seconds": elapsed(),
            "updated_at_unix": time.time(),
            "checkpoint_boundary": "complete SP/XP/MP or CBR round",
        })

    def save_convention_weights(policy_index: int, policy: ComediPolicy) -> None:
        weight_path = output / f"convention_{int(policy_index)}.pt"
        if weight_path.exists():
            return
        atomic_save(weight_path, {
            "kind": "completed_convention_actor",
            "policy_index": int(policy_index),
            "train_env_steps": int(counters["total_train_env_steps"]),
            "policy_env_steps": int(cursor.get("policy_steps", 0)),
            "seed": int(args.seed),
            "source_hashes": source_manifest,
            "actor_state_dict": _cpu_tree(policy.actor.state_dict()),
        })

    def save_cbr_initial_weights(policy: ComediPolicy) -> None:
        weight_path = output / "cbr_initial.pt"
        if weight_path.exists():
            return
        atomic_save(weight_path, {
            "kind": "initial_convention_aware_actor",
            "train_env_steps": int(counters["total_train_env_steps"]),
            "seed": int(args.seed),
            "warm_start_conventions": bool(args.warm_start_conventions),
            "source_hashes": source_manifest,
            "actor_state_dict": _cpu_tree(policy.actor.state_dict()),
        })

    if not args.resume:
        budget_event = {
            "event": "budget_plan",
            "requested_total_joint_env_steps": budgets["requested_total"],
            "effective_total_joint_env_steps": budgets["effective_total"],
            "reserved_demonstration_env_steps": budgets["demonstration_env_steps"],
            "effective_rl_env_steps": budgets["effective_rl_total"],
            "rounded_up_joint_env_steps": budgets["effective_total"] - budgets["requested_total"],
            "vector_step_quantum": quantum,
            "complete_round_boundary": True,
        }
        print(
            "CoMeDi budget: requested="
            f"{budget_event['requested_total_joint_env_steps']}, effective="
            f"{budget_event['effective_total_joint_env_steps']}, rounded_up="
            f"{budget_event['rounded_up_joint_env_steps']} joint steps; "
            f"vector quantum={quantum}, checkpoint at complete rounds"
        )
        log_event(budget_event)

    if not args.resume:
        atomic_save(output / "initial.pt", payload())

    env = KitchenBatch(
        num_envs=args.num_envs,
        seed=args.seed,
        workers=args.workers,
        train=True,
        gamma=args.gamma,
    )
    next_eval_at = (
        counters["total_train_env_steps"] // max(1, args.eval_interval) + 1
    ) * max(1, args.eval_interval)

    def record_wall(key: str, seconds: float) -> None:
        counters["phase_wallclock_seconds"][key] = (
            float(counters["phase_wallclock_seconds"].get(key, 0.0)) + seconds
        )

    def collect_chunks(
        target_steps: int,
        *,
        collector: Callable[[int], tuple[dict[str, np.ndarray] | None, dict[str, Any]]],
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]], int]:
        """Collect a full objective budget, resetting between rollout chunks."""
        batches: list[dict[str, np.ndarray]] = []
        summaries: list[dict[str, Any]] = []
        completed = 0
        while completed < target_steps:
            remaining_units = (target_steps - completed) // quantum
            vector_steps = min(args.rollout, remaining_units)
            if vector_steps <= 0:
                raise RuntimeError("objective budget is not aligned to the vector quantum")
            batch, info = collector(vector_steps)
            actual = int(info["env_steps"])
            if actual <= 0:
                raise RuntimeError("collector returned no joint environment transitions")
            if batch is not None:
                batches.append(batch)
            summaries.extend(info.get("episode_summaries", []))
            completed += actual
        return batches, summaries, completed

    def maybe_validate(active_actor, policy_name: str) -> None:
        nonlocal best_validation_score, best_validation
        nonlocal best_population_score, best_population_validation, eval_index, next_eval_at
        if args.eval_interval <= 0 or counters["total_train_env_steps"] < next_eval_at:
            return
        eval_index += 1
        result = evaluate_policy(
            active_actor, args, device, eval_index=eval_index, should_stop=should_stop
        )
        counters["validation_env_steps"] += int(result["env_steps"])
        result.update({
            "event": "validation",
            "policy": policy_name,
            "train_env_steps": counters["total_train_env_steps"],
            "wallclock_seconds": elapsed(),
        })
        log_event(result)
        if result.get("complete") and "mean_team_score" in result and policy_name == "convention_aware":
            score = float(result["mean_team_score"])
            if best_validation_score is None or score > best_validation_score:
                best_validation_score = score
                best_validation = result.copy()
                atomic_save(output / "best.pt", payload())
        elif result.get("complete") and "mean_team_score" in result:
            score = float(result["mean_team_score"])
            if best_population_score is None or score > best_population_score:
                best_population_score = score
                best_population_validation = result.copy()
                atomic_save(output / "best_population.pt", payload())
        while next_eval_at <= counters["total_train_env_steps"]:
            next_eval_at += max(1, args.eval_interval)

    def add_steps(amount: int) -> None:
        amount = int(amount)
        counters["total_train_env_steps"] += amount
        counters["total_rl_env_steps"] += amount

    def evaluate_demonstration_phase(actor, policy_index: int, phase: str) -> dict[str, Any]:
        nonlocal eval_index
        eval_index += 1
        result = evaluate_policy(
            actor, args, device, eval_index=eval_index, should_stop=should_stop
        )
        counters["validation_env_steps"] += int(result["env_steps"])
        result.update({
            "event": "demonstration_validation",
            "policy_index": int(policy_index),
            "phase": phase,
            "train_env_steps": int(counters["total_train_env_steps"]),
            "wallclock_seconds": elapsed(),
        })
        log_event(result)
        return result

    try:
        save_latest("running")
        while not should_stop():
            policy_index = int(cursor["policy_index"])
            if policy_index < args.population:
                if current is None:
                    actor_state = None
                    if args.warm_start_conventions and conventions:
                        actor_state = conventions[-1].actor.state_dict()
                    current = ComediPolicy(policy_index, lr=args.lr, actor_state=actor_state).to(device)
                if (args.demonstrations
                        and policy_index not in cursor["bc_warmstarted_policies"]):
                    evaluate_demonstration_phase(current.actor, policy_index, "before_bc")
                    from baseline_common.demonstrations import warm_start

                    actor = current.actor
                    bc_started = time.monotonic()
                    bc_metrics = warm_start(
                        actor,
                        lambda spatial, vector: actor(spatial, vector),
                        dataset_path=args.demonstrations,
                        device=device,
                        seed=args.seed + policy_index,
                        epochs=args.bc_epochs,
                        batch_size=512,
                    )
                    if int(bc_metrics["demo_env_steps"]) != demonstration_env_steps:
                        raise ValueError(
                            "warm_start dataset step count differs from demonstration metadata"
                        )
                    if counters["total_demonstration_env_steps"] == 0:
                        counters["total_demonstration_env_steps"] = demonstration_env_steps
                        counters["total_train_env_steps"] += demonstration_env_steps
                    cursor["bc_warmstarted_policies"].append(policy_index)
                    log_event({
                        "event": "demonstration_warm_start",
                        "policy_index": policy_index,
                        "dataset_env_steps": demonstration_env_steps,
                        "dataset_steps_counted_once": len(cursor["bc_warmstarted_policies"]) == 1,
                        "bc_metrics": bc_metrics,
                        "bc_epochs": args.bc_epochs,
                        "batch_size": 512,
                        "elapsed_seconds": time.monotonic() - bc_started,
                    })
                    evaluate_demonstration_phase(current.actor, policy_index, "after_bc")
                    save_latest("running")
                policy_plan = budgets["conventions"][policy_index]
                policy_total = sum(policy_plan.values())
                round_index = int(cursor["round_index"])
                if round_index >= args.rounds_per_convention:
                    current.freeze_actor()
                    save_convention_weights(policy_index, current)
                    conventions.append(current)
                    current = None
                    cursor.update({
                        "policy_index": policy_index + 1,
                        "round_index": 0,
                        "policy_steps": 0,
                        "xp_combo_cursor": 0,
                        "xp_role_cursor": 0,
                    })
                    save_latest("running")
                    continue

                round_start = time.monotonic()
                phase_wall: dict[str, float] = {}
                current.actor.eval()  # fixed actor parameters for all buffers in this round
                sp_steps = _round_budgets(
                    policy_plan["sp"], args.rounds_per_convention, quantum
                )[round_index]
                xp_steps = _round_budgets(
                    policy_plan["xp"], args.rounds_per_convention, quantum
                )[round_index]
                mp_steps = _round_budgets(
                    policy_plan["mp"], args.rounds_per_convention, quantum
                )[round_index]
                progress = int(cursor["policy_steps"])

                sp_started = time.monotonic()
                sp_batches, sp_summaries, got_sp = collect_chunks(
                    sp_steps,
                    collector=lambda n: collect_self_play(
                        env, current, n, device=device, gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )[::2],
                )
                phase_wall["sp"] = time.monotonic() - sp_started
                sp_batch = _merge_batches(sp_batches)
                progress += got_sp

                objectives: list[tuple[str, dict[str, np.ndarray], float]] = [("sp", sp_batch, 1.0)]
                xp_scores: dict[int, list[float]] = {}
                xp_role_scores: dict[str, list[float]] = {}
                selected_prior = None
                xp_probe_steps = 0
                xp_train_steps = 0
                xp_summaries: list[dict[str, Any]] = []
                xp_batch = None
                if policy_index > 0:
                    probe_target = min(
                        max(quantum, _round_down(xp_steps * args.xp_probe_fraction, quantum)),
                        max(quantum, xp_steps - quantum),
                    )
                    probe_started = time.monotonic()
                    def probe_collector(vector_count: int):
                        probe_result = collect_xp_probe(
                            env, current, conventions, vector_count, device=device,
                            combo_cursor=int(cursor["xp_combo_cursor"]),
                        )
                        cursor["xp_combo_cursor"] = int(probe_result["combo_cursor"])
                        for prior, values in probe_result["scores"].items():
                            xp_scores.setdefault(int(prior), []).extend(values)
                        for mask_key, values in probe_result["role_scores"].items():
                            xp_role_scores.setdefault(mask_key, []).extend(values)
                        return None, probe_result

                    probe_batches, _, xp_probe_steps = collect_chunks(
                        probe_target,
                        collector=probe_collector,
                    )
                    phase_wall["xp_probe"] = time.monotonic() - probe_started
                    progress += xp_probe_steps
                    if not xp_scores:
                        raise RuntimeError("XP probe produced no rewards")
                    selected_prior = select_most_compatible_prior(
                        xp_scores
                    )

                    xp_train_target = xp_steps - xp_probe_steps
                    xp_started = time.monotonic()
                    xp_role_cursor = int(cursor["xp_role_cursor"])
                    xp_batches, xp_summaries, xp_train_steps = collect_chunks(
                        xp_train_target,
                        collector=lambda n: collect_cross_play(
                            env, current, conventions[selected_prior], selected_prior, n,
                            device=device, gamma=args.gamma, gae_lambda=args.gae_lambda,
                            start_config=xp_role_cursor,
                        )[::2],
                    )
                    phase_wall["xp"] = time.monotonic() - xp_started
                    xp_batch = _merge_batches(xp_batches)
                    progress += xp_train_steps
                    objectives.append(("xp", xp_batch, -args.xp_weight))
                    cursor["xp_role_cursor"] = (xp_role_cursor + args.num_envs) % len(CROSSPLAY_ROLE_MASKS)

                mp_batches: list[dict[str, np.ndarray]] = []
                mp_summaries: list[dict[str, Any]] = []
                mp_prefix_steps = mp_suffix_steps = 0
                if policy_index > 0:
                    assert selected_prior is not None
                    mp_started = time.monotonic()
                    mp_elapsed = 0
                    while mp_elapsed < mp_steps:
                        remaining_units = (mp_steps - mp_elapsed) // quantum
                        vector_steps = _mp_vector_chunk(
                            remaining_units, args.rollout
                        )
                        batch, _, info = collect_mixed_play(
                            env, current, conventions[selected_prior], vector_steps,
                            device=device, gamma=args.gamma, gae_lambda=args.gae_lambda,
                            rng=np.random.default_rng(args.seed + counters["total_train_env_steps"] + mp_elapsed),
                            prefix_max=args.mp_prefix_max,
                            current_probability=args.mp_current_prob,
                        )
                        if batch is not None:
                            mp_batches.append(batch)
                        mp_summaries.extend(info.get("episode_summaries", []))
                        mp_elapsed += int(info["env_steps"])
                        mp_prefix_steps += int(info["prefix_env_steps"])
                        mp_suffix_steps += int(info["suffix_env_steps"])
                    phase_wall["mp"] = time.monotonic() - mp_started
                    mp_batch = _merge_batches(mp_batches)
                    objectives.append(("mp", mp_batch, args.mp_weight))
                    progress += mp_elapsed

                current.actor.train()
                update_started = time.monotonic()
                train_info = update_bundle(
                    current,
                    objectives,
                    device=device,
                    epochs=args.epochs,
                    minibatches=args.minibatches,
                    clip=args.clip,
                    entropy_coef=args.entropy_coef,
                    value_coef=args.value_coef,
                    max_grad_norm=args.max_grad_norm,
                    prior_count=policy_index,
                    rng=np.random.default_rng(args.seed + counters["total_train_env_steps"] + policy_index * 1_000_003),
                )
                phase_wall["joint_update"] = time.monotonic() - update_started

                round_steps = got_sp + xp_probe_steps + xp_train_steps + mp_steps
                if policy_index == 0:
                    round_steps = got_sp
                add_steps(round_steps)
                counters["phases"]["sp"] += got_sp
                counters["phases"]["xp_selection"] += xp_probe_steps
                counters["phases"]["xp_optimization"] += xp_train_steps
                counters["phases"]["mp_prefix"] += mp_prefix_steps
                counters["phases"]["mp_suffix"] += mp_suffix_steps
                cursor["policy_steps"] = progress
                cursor["round_index"] = round_index + 1
                cursor["selected_prior"] = selected_prior
                for key, seconds in phase_wall.items():
                    record_wall(f"convention_{policy_index}_{key}", seconds)
                round_seconds = time.monotonic() - round_start
                counters["round_wallclock_seconds"]["population"] += round_seconds
                log_event({
                    "event": "joint_round",
                    "policy_index": policy_index,
                    "round_index": round_index,
                    "policy_progress": float(cursor["policy_steps"] / max(1, policy_total)),
                    "selected_prior": selected_prior,
                    "xp_mean_reward_by_prior": {
                        str(key): float(np.mean(values)) for key, values in xp_scores.items()
                    },
                    "xp_role_or_mask_samples": {key: len(values) for key, values in xp_role_scores.items()},
                    "env_steps": {
                        "sp": got_sp,
                        "xp_probe": xp_probe_steps,
                        "xp_optimization": xp_train_steps,
                        "mp_prefix": mp_prefix_steps,
                        "mp_suffix": mp_suffix_steps,
                        "total": round_steps,
                    },
                    "phase_wallclock_seconds": phase_wall,
                    "round_wallclock_seconds": round_seconds,
                    "wallclock_seconds": elapsed(),
                    "train": train_info,
                    "episodes": {
                        "sp": _episode_means(sp_summaries),
                        "xp": _episode_means(xp_summaries),
                        "mp": _episode_means(mp_summaries),
                    },
                })
                maybe_validate(current.actor, f"convention_{policy_index}")
                checkpoint_due = (
                    next_checkpoint_at is not None
                    and counters["total_train_env_steps"] >= next_checkpoint_at
                )
                if (checkpoint_due
                        or cursor["round_index"] >= args.rounds_per_convention or should_stop()):
                    save_latest(
                        "stopped" if should_stop() else "running",
                        retain_step=checkpoint_due or should_stop(),
                    )
                    if checkpoint_due:
                        next_checkpoint_at = (
                            counters["total_train_env_steps"] // int(args.save_interval) + 1
                        ) * int(args.save_interval)
                if cursor["round_index"] >= args.rounds_per_convention:
                    current.freeze_actor()
                    save_convention_weights(policy_index, current)
                    conventions.append(current)
                    current = None
                    cursor.update({
                        "policy_index": policy_index + 1,
                        "round_index": 0,
                        "policy_steps": 0,
                        "xp_combo_cursor": 0,
                        "xp_role_cursor": 0,
                        "selected_prior": None,
                    })
                if should_stop():
                    break
                continue

            # The convention-aware actor uses the same joint-round optimizer:
            # an SP buffer plus teacher action labels in the BC half of each round.
            if cbr is None:
                # best.pt is reserved for convention-aware validation only.
                best_validation_score = None
                best_validation = None
                actor_state = conventions[-1].actor.state_dict() if args.warm_start_conventions else None
                cbr = ComediPolicy(0, lr=args.lr, actor_state=actor_state).to(device)
                save_cbr_initial_weights(cbr)
            if args.demonstrations and not cursor.get("cbr_bc_warmstarted", False):
                from baseline_common.demonstrations import warm_start

                actor = cbr.actor
                bc_started = time.monotonic()
                bc_metrics = warm_start(
                    actor,
                    lambda spatial, vector: actor(spatial, vector),
                    dataset_path=args.demonstrations,
                    device=device,
                    seed=args.seed + args.population,
                    epochs=args.bc_epochs,
                    batch_size=512,
                )
                if int(bc_metrics["demo_env_steps"]) != demonstration_env_steps:
                    raise ValueError(
                        "CBR warm_start dataset step count differs from demonstration metadata"
                    )
                cursor["cbr_bc_warmstarted"] = True
                atomic_save(output / "bc_cbr_initial.pt", {
                    "kind": "bc_warm_started_convention_aware_actor",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "demonstration_env_steps": int(counters["total_demonstration_env_steps"]),
                    "seed": int(args.seed + args.population),
                    "bc_metrics": _cpu_tree(bc_metrics),
                    "source_hashes": source_manifest,
                    "actor_state_dict": _cpu_tree(actor.state_dict()),
                })
                log_event({
                    "event": "cbr_demonstration_warm_start",
                    "dataset_env_steps": demonstration_env_steps,
                    "dataset_steps_counted_once": False,
                    "bc_metrics": bc_metrics,
                    "bc_epochs": args.bc_epochs,
                    "batch_size": 512,
                    "elapsed_seconds": time.monotonic() - bc_started,
                })
                eval_index += 1
                candidate_eval_args = deepcopy(args)
                candidate_eval_args.eval_episodes = 1
                candidate_validation = evaluate_policy(
                    cbr.actor, candidate_eval_args, device,
                    eval_index=eval_index, should_stop=should_stop,
                )
                counters["validation_env_steps"] += int(candidate_validation["env_steps"])
                candidate_validation.update({
                    "event": "cbr_demonstration_validation",
                    "policy": "convention_aware_bc_initial",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "wallclock_seconds": elapsed(),
                })
                log_event(candidate_validation)
                if (candidate_validation.get("complete")
                        and "mean_team_score" in candidate_validation):
                    best_validation_score = float(candidate_validation["mean_team_score"])
                    best_validation = candidate_validation.copy()
                    atomic_save(output / "best.pt", payload())
                save_latest("running")
            round_index = int(cursor["cbr_round_index"])
            if round_index >= args.rounds_per_convention:
                break
            round_start = time.monotonic()
            cbr.actor.eval()
            sp_round = _round_budgets(
                budgets["cbr"]["cbr_sp"], args.rounds_per_convention, quantum
            )[round_index]
            bc_round = _round_budgets(
                budgets["cbr"]["cbr_bc"], args.rounds_per_convention, quantum
            )[round_index]
            cbr_total = budgets["cbr_total"]
            cbr_progress = int(cursor["cbr_steps"])

            cbr_sp_started = time.monotonic()
            cbr_sp_batches, cbr_sp_summaries, got_cbr_sp = collect_chunks(
                sp_round,
                collector=lambda n: collect_self_play(
                    env, cbr, n, device=device, gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )[::2],
            )
            cbr_sp_batch = _merge_batches(cbr_sp_batches)
            cbr_sp_seconds = time.monotonic() - cbr_sp_started
            cbr_progress += got_cbr_sp
            cbr_bc_started = time.monotonic()
            teacher_start_cursor = int(cursor.get("cbr_teacher_cursor", 0))
            teacher_next_cursor = teacher_start_cursor
            teacher_batch_chunks: dict[int, list[dict[str, np.ndarray]]] = defaultdict(list)
            teacher_env_steps: dict[int, int] = defaultdict(int)
            teacher_episode_counts: dict[int, int] = defaultdict(int)
            teacher_summaries: dict[int, list[dict[str, Any]]] = defaultdict(list)

            def collect_teacher_batch(vector_count: int):
                nonlocal teacher_next_cursor
                batches_by_teacher, info = collect_teacher_self_play(
                    env, conventions, vector_count, device=device,
                    start_teacher=teacher_next_cursor,
                )
                teacher_next_cursor = int(info["next_teacher_cursor"])
                for teacher_id, batch in batches_by_teacher.items():
                    teacher_batch_chunks[int(teacher_id)].append(batch)
                for teacher_id in range(len(conventions)):
                    key = str(teacher_id)
                    teacher_env_steps[teacher_id] += int(info["teacher_env_steps"].get(key, 0))
                    teacher_episode_counts[teacher_id] += int(
                        info["teacher_episode_counts"].get(key, 0)
                    )
                    teacher_summaries[teacher_id].extend(
                        info["episode_summaries_by_teacher"].get(key, [])
                    )
                return None, {
                    "env_steps": int(info["env_steps"]),
                    "episode_summaries": [
                        summary
                        for teacher_id in range(len(conventions))
                        for summary in info["episode_summaries_by_teacher"].get(
                            str(teacher_id), []
                        )
                    ],
                }

            _, cbr_bc_summaries, got_cbr_bc = collect_chunks(
                bc_round,
                collector=collect_teacher_batch,
            )
            cbr_bc_batches = [
                (teacher_id, _merge_batches(chunks))
                for teacher_id, chunks in sorted(teacher_batch_chunks.items())
                if chunks
            ]
            cbr_bc_seconds = time.monotonic() - cbr_bc_started
            cbr_progress += got_cbr_bc
            cbr.actor.train()
            update_started = time.monotonic()
            train_info = update_bundle(
                cbr,
                (("sp", cbr_sp_batch, 1.0),),
                device=device,
                epochs=args.epochs,
                minibatches=args.minibatches,
                clip=args.clip,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                prior_count=0,
                bc_coef=args.cbr_bc_coef,
                bc_objectives=cbr_bc_batches,
                rng=np.random.default_rng(args.seed + counters["total_train_env_steps"] + 91),
            )
            update_seconds = time.monotonic() - update_started
            round_seconds = time.monotonic() - round_start
            add_steps(got_cbr_sp + got_cbr_bc)
            counters["phases"]["cbr_sp"] += got_cbr_sp
            counters["phases"]["cbr_bc"] += got_cbr_bc
            for teacher_id in range(len(conventions)):
                counters["cbr_teacher_env_steps"][teacher_id] += teacher_env_steps[teacher_id]
            cursor["cbr_teacher_cursor"] = (
                teacher_next_cursor + 1
            ) % len(conventions)
            cursor["cbr_steps"] = cbr_progress
            cursor["cbr_round_index"] = round_index + 1
            counters["round_wallclock_seconds"]["cbr"] += round_seconds
            record_wall("cbr_sp", cbr_sp_seconds)
            record_wall("cbr_bc", cbr_bc_seconds)
            record_wall("cbr_joint_update", update_seconds)
            log_event({
                "event": "cbr_joint_round",
                "round_index": round_index,
                "cbr_progress": float(cursor["cbr_steps"] / max(1, cbr_total)),
                "env_steps": {"cbr_sp": got_cbr_sp, "cbr_bc": got_cbr_bc,
                              "total": got_cbr_sp + got_cbr_bc},
                "phase_wallclock_seconds": {
                    "cbr_sp": cbr_sp_seconds,
                    "cbr_bc": cbr_bc_seconds,
                    "joint_update": update_seconds,
                    "round": round_seconds,
                },
                "wallclock_seconds": elapsed(),
                "bc_teachers": len(conventions),
                "bc_teacher_start_cursor": teacher_start_cursor,
                "bc_teacher_next_cursor": int(cursor["cbr_teacher_cursor"]),
                "bc_teacher_env_steps": {
                    str(teacher_id): int(teacher_env_steps[teacher_id])
                    for teacher_id in range(len(conventions))
                },
                "bc_teacher_episode_counts": {
                    str(teacher_id): int(teacher_episode_counts[teacher_id])
                    for teacher_id in range(len(conventions))
                },
                "bc_teacher_total_env_steps": {
                    str(teacher_id): int(counters["cbr_teacher_env_steps"][teacher_id])
                    for teacher_id in range(len(conventions))
                },
                "train": train_info,
                "episodes": {
                    "sp": _episode_means(cbr_sp_summaries),
                    "teacher_self_play": {
                        str(teacher_id): _episode_means(teacher_summaries[teacher_id])
                        for teacher_id in range(len(conventions))
                    },
                },
            })
            maybe_validate(cbr.actor, "convention_aware")
            checkpoint_due = (
                next_checkpoint_at is not None
                and counters["total_train_env_steps"] >= next_checkpoint_at
            )
            if (checkpoint_due
                    or cursor["cbr_round_index"] >= args.rounds_per_convention or should_stop()):
                save_latest(
                    "stopped" if should_stop() else "running",
                    retain_step=checkpoint_due or should_stop(),
                )
                if checkpoint_due:
                    next_checkpoint_at = (
                        counters["total_train_env_steps"] // int(args.save_interval) + 1
                    ) * int(args.save_interval)
            if should_stop():
                break

        complete = (
            int(cursor["policy_index"]) >= args.population
            and int(cursor["cbr_round_index"]) >= args.rounds_per_convention
        )
        if complete and not should_stop():
            eval_index += 1
            validation = evaluate_policy(
                cbr.actor, args, device, eval_index=eval_index, should_stop=should_stop
            )
            counters["validation_env_steps"] += int(validation["env_steps"])
            validation.update({
                "event": "validation", "policy": "convention_aware",
                "train_env_steps": counters["total_train_env_steps"],
                "wallclock_seconds": elapsed(), "final_validation": True,
            })
            log_event(validation)
            if validation.get("complete") and "mean_team_score" in validation:
                score = float(validation["mean_team_score"])
                if best_validation_score is None or score > best_validation_score:
                    best_validation_score = score
                    best_validation = validation.copy()
                    atomic_save(output / "best.pt", payload())
            if not should_stop():
                final_snapshot = payload()
                atomic_save(output / "final.pt", final_snapshot)
                atomic_save(latest_path, final_snapshot)
                save_step_snapshot(final_snapshot)
                atomic_json(status_path, {
                    "state": "completed",
                    "total_train_env_steps": counters["total_train_env_steps"],
                    **step_counts(),
                    "requested_budget_jointenvsteps": budgets["requested_total"],
                    "budget_jointenvsteps": budgets["effective_total"],
                    "cursor": cursor.copy(),
                    "best_validation": best_validation,
                    "wallclock_seconds": elapsed(),
                    "updated_at_unix": time.time(),
                })
            else:
                save_latest("stopped", retain_step=True)
        else:
            save_latest("stopped" if should_stop() else "running", retain_step=should_stop())
    except KeyboardInterrupt:
        stop_requested = True
        save_latest("stopped", retain_step=True)
    except BaseException:
        save_latest("failed")
        atomic_json(status_path, {
            "state": "failed",
            "total_train_env_steps": counters["total_train_env_steps"],
            **step_counts(),
            "budget_jointenvsteps": budgets["effective_total"],
            "cursor": cursor.copy(),
            "wallclock_seconds": elapsed(),
            "updated_at_unix": time.time(),
        })
        raise
    finally:
        env.close()
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except (ValueError, AttributeError):
                pass


def main() -> None:
    run(effective_args(parser().parse_args()))

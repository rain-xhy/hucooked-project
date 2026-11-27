"""Sequential population and CBR lifecycle for centralized CoMeDi."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

from .algorithm import (
    CROSSPLAY_ROLE_MASKS,
    _mp_vector_chunk,
    collect_cross_play,
    collect_mixed_play,
    collect_self_play,
    collect_teacher_self_play,
    collect_xp_probe,
    update_bundle,
)
from .model import (
    ACTOR_PARAMETER_COUNT,
    NUM_ROLES,
    ComediPolicy,
    build_actor,
    select_most_compatible_prior,
)
from .train import (
    BC_BATCH_SIZE,
    CBR_BC_COEF,
    CBR_FRACTION,
    CBR_SP_FRACTION,
    CLIP,
    ENTROPY_COEF,
    GAMMA,
    GAE_LAMBDA,
    MAX_GRAD_NORM,
    MP_CURRENT_PROB,
    MP_PREFIX_MAX,
    MP_WEIGHT,
    VALUE_COEF,
    XP_WEIGHT,
    _episode_means,
    _round_down,
    _round_budgets,
    allocate_budgets,
    effective_args,
    evaluate_policy,
)


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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_event(output: Path, event: dict[str, Any]) -> None:
    row = {"time_unix": time.time(), **_json_safe(event)}
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    print(encoded, flush=True)


def _merge_batches(chunks: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("cannot merge an empty objective buffer")
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks):
        raise ValueError("objective rollout chunks have inconsistent fields")
    merged: dict[str, Any] = {}
    for key in keys:
        if key == "prior_id":
            values = {int(chunk[key]) for chunk in chunks}
            if len(values) != 1:
                raise ValueError("XP chunks in a round must use the same selected prior")
            merged[key] = values.pop()
        else:
            merged[key] = np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0)
    return merged


def _policy_summary(policy: ComediPolicy) -> dict[str, int]:
    counts = policy.parameter_counts()
    return {
        "actor_parameters": counts["actor"],
        "sp_critic_parameters": sum(p.numel() for p in policy.sp_critics.parameters()),
        "mp_critic_parameters": sum(p.numel() for p in policy.mp_critics.parameters()),
        "xp_critic_count": len(policy.xp_critics),
        "xp_critic_parameters": sum(p.numel() for p in policy.xp_critics.parameters()),
        "total_parameters": counts["total"],
    }


def _load_dependencies():
    from joint_common import demonstrations, envs, runtime

    return SimpleNamespace(demonstrations=demonstrations, envs=envs, runtime=runtime)


def _validate_dataset(data: dict[str, Any]) -> tuple[int, str]:
    required = {"state", "spatial", "actions", "validation", "episode_lengths", "env_steps", "dataset_sha256"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"demonstration dataset missing fields: {sorted(missing)}")
    states = np.asarray(data["state"])
    spatial = np.asarray(data["spatial"])
    actions = np.asarray(data["actions"])
    validation = np.asarray(data["validation"])
    lengths = np.asarray(data["episode_lengths"])
    frames = len(states)
    if states.shape != (frames, 863) or spatial.shape != (frames, 26, 7, 7):
        raise ValueError("joint demonstrations must use state[frames,863] and spatial[frames,26,7,7]")
    if actions.shape != (frames, 3) or validation.shape != (frames,):
        raise ValueError("joint demo actions/split must use shapes [frames,3]/[frames]")
    if lengths.ndim != 1 or lengths.sum() != frames:
        raise ValueError("episode_lengths do not cover all demonstration frames")
    if not np.isfinite(states).all() or not np.isfinite(spatial).all():
        raise FloatingPointError("demonstration dataset has non-finite observations")
    env_steps = int(data["env_steps"])
    if env_steps != int((~validation).sum()):
        raise ValueError("demo env_steps must count training joint frames once")
    return env_steps, str(data["dataset_sha256"])


def _checkpoint_payload(*, args, budgets, manifest, source_manifest, counters,
                        conventions, current, cbr, stage, policy_index,
                        round_index, best_population, best_cbr,
                        threshold=None):
    return _cpu_tree({
        "schema_version": 1,
        "kind": "joint_comedi_run_checkpoint",
        "method": "joint_comedi",
        "mode": args.mode,
        "seed": int(args.seed),
        "stage": stage,
        "policy_index": int(policy_index),
        "round_index": int(round_index),
        "training_config": manifest["arguments"],
        "source_hashes": source_manifest,
        "budgets_jointenvsteps": budgets,
        "actor_schema": manifest["actor_schema"],
        "convention_actors": [actor.state_dict() for actor in conventions],
        "current_policy": current.checkpoint() if current is not None else None,
        "cbr_policy": cbr.checkpoint() if cbr is not None else None,
        "counters": counters,
        "best_population": best_population,
        "best_cbr": best_cbr,
        "save_threshold_env_steps": threshold,
    })


def run(args, *, dependencies=None) -> dict[str, Any]:
    dependencies = _load_dependencies() if dependencies is None else dependencies
    args = effective_args(args)
    runtime = dependencies.runtime
    demonstration_data = None
    demonstration_steps = 0
    demonstration_hash = None
    demonstration_path = None
    if args.mode == "bc_rl":
        demonstration_path = Path(args.demonstrations).expanduser().resolve()
        if not demonstration_path.is_file():
            raise FileNotFoundError(f"demonstration dataset not found: {demonstration_path}")
        args.demonstrations = str(demonstration_path)
        demonstration_data = dependencies.demonstrations.load_joint_dataset(demonstration_path)
        demonstration_steps, demonstration_hash = _validate_dataset(demonstration_data)
        if _sha256_file(demonstration_path) != demonstration_hash:
            raise ValueError("demonstration dataset hash changed while loading")

    budgets = allocate_budgets(args, demonstration_env_steps=demonstration_steps)
    device = runtime.setup_device(
        device=args.device,
        seed=int(args.seed),
        physical_gpu=int(args.physical_gpu),
        memory_fraction=float(args.memory_fraction),
    )
    output = Path(args.output).expanduser().resolve()
    runtime.prepare_output(output)
    source_manifest = runtime.source_hashes("joint_comedi")
    if demonstration_hash is not None:
        source_manifest[f"demonstrations:{demonstration_path}"] = demonstration_hash

    effective_config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    manifest = {
        "method": "joint_comedi",
        "mode": args.mode,
        "arguments": effective_config,
        "budgets_jointenvsteps": budgets,
        "source_hashes": source_manifest,
        "demonstrations": {
            "path": str(demonstration_path) if demonstration_path is not None else None,
            "dataset_sha256": demonstration_hash,
            "env_steps_counted_once": demonstration_steps,
            "bc_epochs_per_convention_and_cbr": args.bc_epochs if args.mode == "bc_rl" else 0,
            "batch_size_action_labels": BC_BATCH_SIZE if args.mode == "bc_rl" else 0,
        },
        "environment": "SharedKitchen",
        "environment_commit": dependencies.envs.ENVIRONMENT_COMMIT,
        "reward_profile": args.reward_profile,
        "observation": {"state": [863], "canonical_spatial": [26, 7, 7],
                        "native_spatial": [3, 26, 7, 7], "native_vector": [3, 866]},
        "actor_schema": {
            "body": "Conv2d(26,32,kernel=3,padding=0) -> flatten 800 + state863 -> FC64 -> FC64 -> Linear18 -> reshape [B,3,6]",
            "parameter_count": ACTOR_PARAMETER_COUNT,
            "input_encode_count": 1,
            "canonical_view": "spatial[:,0] with fixed agent channels 0/1/2",
        },
        "critic_schema": "independent old-style ConvCritic(spatial[:,role],vector[:,role]) per phase/role/prior",
        "native_episode_metrics": [
            "team_score", "meals_completed", "normal_completed", "joint_completed",
            "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
            "three_contributor_meals", "resource_conflicts", "movement_conflicts",
            "ticks", "meals_per_100_steps", "training_reward", "training_shaping",
        ],
        "protocol": {
            "population": "8 sequential independent conventions; old actors are actor-only CPU copies",
            "actor_update": "one accumulated SP + signed XP + MP actor step each round; PPO ratios/clipping remain per role",
            "xp": "current-round max-compatible prior probe plus 6 one/two-current role masks",
            "mp": "old/new random prefix; actor/critic PPO validity is limited to the all-current suffix",
            "cbr": "student SP PPO plus teacher-own-self-play BC from each frozen convention, all 3 role labels retained",
            "reward_profile": args.reward_profile,
            "validation_reward": "native reward with KitchenBatch(train=False)",
        },
        "checkpoint_policy": "single output directory; immutable convention and threshold artifacts; no resume",
    }
    runtime.atomic_json(output / "config.json", _json_safe(manifest))

    counters: dict[str, Any] = {
        "total_train_env_steps": 0,
        "total_rl_env_steps": 0,
        "total_demonstration_env_steps": 0,
        "validation_env_steps": 0,
        "phases": {
            "sp": 0, "xp_probe": 0, "xp_train": 0,
            "mp_prefix": 0, "mp_suffix": 0, "cbr_sp": 0, "cbr_teacher_bc": 0,
        },
        "teacher_env_steps": [0] * int(args.population),
    }
    conventions = []
    current = None
    cbr = None
    best_population: dict[str, Any] | None = None
    best_cbr: dict[str, Any] | None = None
    eval_index = 0
    next_eval_at = int(args.eval_interval)
    next_save_at = int(args.save_interval)
    started = time.monotonic()
    status_path = output / "status.json"

    def elapsed() -> float:
        return time.monotonic() - started

    def should_stop() -> bool:
        return int(args.max_seconds) > 0 and elapsed() >= int(args.max_seconds)

    def step_counts() -> dict[str, int]:
        return {
            "total_env_steps": int(counters["total_train_env_steps"]),
            "rl_env_steps": int(counters["total_rl_env_steps"]),
            "demonstration_env_steps": int(counters["total_demonstration_env_steps"]),
        }

    def log_event(event: dict[str, Any]) -> None:
        event.setdefault("step_counts", step_counts())
        event.setdefault("wallclock_seconds", elapsed())
        _log_event(output, event)

    def payload(stage, policy_index, round_index, threshold=None):
        return _checkpoint_payload(
            args=args,
            budgets=budgets,
            manifest=manifest,
            source_manifest=source_manifest,
            counters=counters,
            conventions=conventions,
            current=current,
            cbr=cbr,
            stage=stage,
            policy_index=policy_index,
            round_index=round_index,
            best_population=best_population,
            best_cbr=best_cbr,
            threshold=threshold,
        )

    def save_latest(state, stage, policy_index, round_index, *, threshold=None):
        snapshot = payload(stage, policy_index, round_index, threshold=threshold)
        runtime.atomic_save(output / "latest.pt", snapshot)
        runtime.atomic_json(status_path, _json_safe({
            "state": state,
            **step_counts(),
            "requested_budget_jointenvsteps": budgets["requested_total"],
            "effective_budget_jointenvsteps": budgets["effective_total"],
            "stage": stage,
            "policy_index": int(policy_index),
            "round_index": int(round_index),
            "wallclock_seconds": elapsed(),
            "updated_at_unix": time.time(),
        }))
        return snapshot

    def save_step_archive(snapshot, threshold):
        actual = int(counters["total_train_env_steps"])
        path = output / f"step_{actual}.pt"
        if not path.exists():
            archive = dict(snapshot)
            archive["save_threshold_env_steps"] = int(threshold)
            runtime.atomic_save(path, archive)
            log_event({"event": "checkpoint", "threshold_env_steps": threshold,
                       "actual_env_steps": actual, "path": str(path)})
        return path

    def add_train_steps(amount):
        amount = int(amount)
        counters["total_train_env_steps"] += amount
        counters["total_rl_env_steps"] += amount

    def _call_eval(actor, policy_name, *, candidate_kind=None):
        nonlocal eval_index, best_population, best_cbr
        eval_index += 1
        result = evaluate_policy(
            actor, args, device, eval_index=eval_index, dependencies=dependencies,
            should_stop=lambda: False,
        )
        counters["validation_env_steps"] += int(result["env_steps"])
        result.update({
            "event": "validation",
            "policy": policy_name,
            "candidate_kind": candidate_kind,
            "train_env_steps": counters["total_train_env_steps"],
        })
        log_event(result)
        if not result.get("complete") or "mean_team_score" not in result:
            return result
        score = float(result["mean_team_score"])
        if policy_name.startswith("convention_"):
            if best_population is None or score > float(best_population["mean_team_score"]):
                best_population = result.copy()
                runtime.atomic_save(output / "best_population.pt", {
                    "kind": "best_population_actor",
                    "policy_index": int(policy_name.rsplit("_", 1)[1]),
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "seed": int(args.seed),
                    "mode": args.mode,
                    "actor_state_dict": _cpu_tree(actor.state_dict()),
                    "evaluation": result.copy(),
                    "source_hashes": source_manifest,
                })
        elif policy_name == "convention_aware":
            if best_cbr is None or score > float(best_cbr["mean_team_score"]):
                best_cbr = result.copy()
                runtime.atomic_save(output / "best.pt", {
                    "kind": "best_cbr_policy_bundle",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "seed": int(args.seed),
                    "mode": args.mode,
                    "policy": _cpu_tree(cbr.checkpoint()),
                    "evaluation": result.copy(),
                    "source_hashes": source_manifest,
                })
        return result

    def maybe_validate(actor, policy_name):
        nonlocal next_eval_at
        if counters["total_train_env_steps"] < next_eval_at:
            return
        _call_eval(actor, policy_name)
        while next_eval_at <= counters["total_train_env_steps"]:
            next_eval_at += int(args.eval_interval)

    def collect_chunks(target_steps, collector):
        batches: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        completed = 0
        while completed < int(target_steps):
            remaining_units = (int(target_steps) - completed) // int(args.num_envs)
            vector_steps = min(int(args.rollout), remaining_units)
            if vector_steps <= 0:
                raise RuntimeError("phase budget is not aligned to num_envs")
            batch, info = collector(vector_steps)
            actual = int(info["env_steps"])
            if actual != vector_steps * int(args.num_envs):
                raise RuntimeError("collector did not return the full vectorized step count")
            if batch is not None:
                batches.append(batch)
            summaries.extend(info.get("episode_summaries", []))
            completed += actual
        return batches, summaries, completed

    def finish_round(stage, policy_index, round_index):
        nonlocal next_save_at
        snapshot = save_latest("running", stage, policy_index, round_index)
        while counters["total_train_env_steps"] >= next_save_at:
            save_step_archive(snapshot, next_save_at)
            next_save_at += int(args.save_interval)

    env = None
    try:
        current = ComediPolicy(0, lr=args.lr).to(device)
        runtime.atomic_save(output / "initial.pt", payload("convention", 0, -1))
        log_event({
            "event": "budget_plan",
            "requested_total_joint_env_steps": budgets["requested_total"],
            "effective_total_joint_env_steps": budgets["effective_total"],
            "reserved_demonstration_env_steps": budgets["demonstration_env_steps"],
            "effective_rl_env_steps": budgets["effective_rl_total"],
            "rounded_up_joint_env_steps": budgets["effective_total"] - budgets["requested_total"],
            "vector_step_quantum": int(args.num_envs),
            "smoke_mode": bool(args.smoke),
        })
        save_latest("running", "convention", 0, 0)
        env = dependencies.envs.KitchenBatch(
            num_envs=int(args.num_envs),
            seed=int(args.seed),
            stage="standard",
            workers=int(args.workers),
            train=True,
            gamma=GAMMA,
            reward_profile=args.reward_profile,
        )

        demo_steps_counted = False
        state = "completed"
        for policy_index in range(int(args.population)):
            if should_stop():
                state = "time_limit"
                break
            if policy_index > 0:
                current = ComediPolicy(policy_index, lr=args.lr).to(device)
            convention_bc_metrics = None
            if args.mode == "bc_rl":
                bc_metrics = dependencies.demonstrations.warm_start_joint(
                    current.actor,
                    lambda spatial, state: current.actor(spatial, state),
                    dataset_path=args.demonstrations,
                    device=device,
                    seed=int(args.seed) + policy_index,
                    epochs=int(args.bc_epochs),
                    batch_size=BC_BATCH_SIZE,
                )
                if int(bc_metrics["demo_env_steps"]) != demonstration_steps:
                    raise ValueError("convention BC demo steps differ from the loaded dataset")
                if bc_metrics.get("dataset_sha256") != demonstration_hash:
                    raise ValueError("convention BC dataset hash differs from the loaded dataset")
                convention_bc_metrics = _json_safe(dict(bc_metrics))
                if not demo_steps_counted:
                    counters["total_demonstration_env_steps"] = demonstration_steps
                    counters["total_train_env_steps"] += demonstration_steps
                    demo_steps_counted = True
                candidate = _call_eval(current.actor, f"convention_{policy_index}", candidate_kind="post_bc")
                runtime.atomic_save(output / f"bc_convention_{policy_index}.pt", {
                    "kind": "bc_initialized_convention_actor",
                    "policy_index": policy_index,
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "seed": int(args.seed) + policy_index,
                    "mode": args.mode,
                    "demo_env_steps_counted_once": demo_steps_counted,
                    "demo_env_steps": demonstration_steps,
                    "dataset_sha256": demonstration_hash,
                    "bc_metrics": convention_bc_metrics,
                    "native_evaluation": candidate,
                    "source_hashes": source_manifest,
                    "actor_state_dict": _cpu_tree(current.actor.state_dict()),
                })
                log_event({
                    "event": "convention_bc_warm_start",
                    "policy_index": policy_index,
                    "dataset_steps_counted_once": policy_index == 0,
                    "bc_metrics": convention_bc_metrics,
                    "native_evaluation": candidate,
                })
                save_latest("running", "convention", policy_index, 0)

            plan = budgets["conventions"][policy_index]
            phase_rounds = {
                phase: _round_budgets(amount, int(args.rounds_per_convention), int(args.num_envs))
                for phase, amount in plan.items()
            }
            xp_role_cursor = 0
            xp_combo_cursor = 0
            for round_index in range(int(args.rounds_per_convention)):
                if should_stop():
                    state = "time_limit"
                    break
                round_started = time.monotonic()
                current.actor.eval()
                sp_target = phase_rounds["sp"][round_index]
                sp_batches, sp_summaries, sp_steps = collect_chunks(
                    sp_target,
                    lambda n: collect_self_play(
                        env, current, n, device=device, gamma=GAMMA, gae_lambda=GAE_LAMBDA
                    )[::2],
                )
                sp_batch = _merge_batches(sp_batches)
                objectives = [("sp", sp_batch, 1.0)]
                xp_probe_steps = xp_train_steps = 0
                selected_prior = None
                xp_scores: dict[int, list[float]] = defaultdict(list)
                xp_mask_scores: dict[str, list[float]] = defaultdict(list)
                xp_summaries: list[dict[str, Any]] = []
                if policy_index > 0:
                    xp_target = phase_rounds["xp"][round_index]
                    probe_target = min(
                        max(int(args.num_envs), _round_down(xp_target * float(args.xp_probe_fraction), int(args.num_envs))),
                        max(int(args.num_envs), xp_target - int(args.num_envs)),
                    )

                    def probe_chunk(vector_count):
                        nonlocal xp_combo_cursor
                        result = collect_xp_probe(
                            env, current, conventions, vector_count,
                            device=device, combo_cursor=xp_combo_cursor,
                        )
                        xp_combo_cursor = int(result["combo_cursor"])
                        for prior, values in result["scores"].items():
                            xp_scores[int(prior)].extend(values)
                        for key, values in result["role_scores"].items():
                            xp_mask_scores[key].extend(values)
                        return None, result

                    _, _, xp_probe_steps = collect_chunks(probe_target, probe_chunk)
                    selected_prior = select_most_compatible_prior(xp_scores)
                    xp_train_target = xp_target - xp_probe_steps

                    def cross_chunk(vector_count):
                        nonlocal xp_role_cursor
                        batch, _observation, info = collect_cross_play(
                            env, current, conventions[selected_prior], selected_prior, vector_count,
                            device=device, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                            start_config=xp_role_cursor,
                        )
                        xp_role_cursor = int(info["next_role_cursor"])
                        return batch, info

                    xp_batches, xp_summaries, xp_train_steps = collect_chunks(
                        xp_train_target, cross_chunk,
                    )
                    if xp_batches:
                        objectives.append(("xp", _merge_batches(xp_batches), -XP_WEIGHT))

                mp_batches: list[dict[str, Any]] = []
                mp_summaries: list[dict[str, Any]] = []
                mp_prefix_steps = mp_suffix_steps = 0
                mp_steps = 0
                if policy_index > 0:
                    selected_actor = conventions[selected_prior]
                    mp_target = phase_rounds["mp"][round_index]
                    completed = 0
                    while completed < mp_target:
                        remaining_units = (mp_target - completed) // int(args.num_envs)
                        vector_count = _mp_vector_chunk(remaining_units, int(args.rollout))
                        batch, _obs, info = collect_mixed_play(
                            env, current, selected_actor, vector_count,
                            device=device, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
                            rng=np.random.default_rng(
                                int(args.seed) + int(counters["total_train_env_steps"]) + completed
                            ),
                            prefix_max=MP_PREFIX_MAX,
                            current_probability=MP_CURRENT_PROB,
                        )
                        mp_batches.append(batch)
                        mp_summaries.extend(info["episode_summaries"])
                        completed += int(info["env_steps"])
                        mp_prefix_steps += int(info["prefix_env_steps"])
                        mp_suffix_steps += int(info["suffix_env_steps"])
                    mp_steps = completed
                    objectives.append(("mp", _merge_batches(mp_batches), MP_WEIGHT))

                current.actor.train()
                update_info = update_bundle(
                    current,
                    objectives,
                    device=device,
                    epochs=int(args.epochs),
                    minibatches=int(args.minibatches),
                    clip=CLIP,
                    entropy_coef=ENTROPY_COEF,
                    value_coef=VALUE_COEF,
                    max_grad_norm=MAX_GRAD_NORM,
                    prior_count=policy_index,
                    rng=np.random.default_rng(
                        int(args.seed) + int(counters["total_train_env_steps"]) + policy_index * 1_000_003
                    ),
                )
                round_steps = sp_steps + xp_probe_steps + xp_train_steps + mp_steps
                add_train_steps(round_steps)
                counters["phases"]["sp"] += sp_steps
                counters["phases"]["xp_probe"] += xp_probe_steps
                counters["phases"]["xp_train"] += xp_train_steps
                counters["phases"]["mp_prefix"] += mp_prefix_steps
                counters["phases"]["mp_suffix"] += mp_suffix_steps
                _log_event(output, {
                    "event": "joint_round",
                    "policy_index": policy_index,
                    "round_index": round_index,
                    "selected_prior": selected_prior,
                    "env_steps": {
                        "sp": sp_steps, "xp_probe": xp_probe_steps,
                        "xp_optimization": xp_train_steps,
                        "mp_prefix": mp_prefix_steps, "mp_suffix": mp_suffix_steps,
                        "total": round_steps,
                    },
                    "xp_mean_reward_by_prior": {
                        str(key): float(np.mean(value)) for key, value in xp_scores.items()
                    },
                    "xp_role_mask_samples": {key: len(value) for key, value in xp_mask_scores.items()},
                    "train": update_info,
                    "episodes": {
                        "sp": _episode_means(sp_summaries),
                        "xp": _episode_means(xp_summaries),
                        "mp": _episode_means(mp_summaries),
                    },
                    "round_wallclock_seconds": time.monotonic() - round_started,
                })
                maybe_validate(current.actor, f"convention_{policy_index}")
                finish_round("convention", policy_index, round_index + 1)

            if state != "completed":
                break
            actor_state = _cpu_tree(current.actor.state_dict())
            runtime.atomic_save(output / f"convention_{policy_index}.pt", {
                "kind": "completed_convention_actor",
                "policy_index": int(policy_index),
                "train_env_steps": int(counters["total_train_env_steps"]),
                "policy_env_steps": int(sum(plan.values())),
                "seed": int(args.seed),
                "mode": args.mode,
                "actor_parameter_count": ACTOR_PARAMETER_COUNT,
                "source_hashes": source_manifest,
                "demonstration_dataset_sha256": demonstration_hash,
                "actor_state_dict": actor_state,
            })
            completed_actor = build_actor(actor_state).cpu().eval()
            for parameter in completed_actor.parameters():
                parameter.requires_grad_(False)
            conventions.append(completed_actor)
            _call_eval(completed_actor, f"convention_{policy_index}", candidate_kind="completed")
            save_latest("running", "population", policy_index + 1, 0)
            del current
            current = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if state == "completed" and len(conventions) == int(args.population):
            if should_stop():
                state = "time_limit"
        if state == "completed" and len(conventions) == int(args.population):
            cbr = ComediPolicy(0, lr=args.lr).to(device)
            actor_state = _cpu_tree(cbr.actor.state_dict())
            random_eval = _call_eval(cbr.actor, "convention_aware", candidate_kind="random_initial")
            runtime.atomic_save(output / "random_initial.pt", {
                "kind": "cbr_random_initial_actor",
                "train_env_steps": int(counters["total_train_env_steps"]),
                "seed": int(args.seed), "mode": args.mode,
                "actor_parameter_count": ACTOR_PARAMETER_COUNT,
                "actor_state_dict": actor_state,
                "native_evaluation": random_eval,
                "source_hashes": source_manifest,
            })
            if random_eval.get("complete"):
                best_cbr = random_eval.copy()
                runtime.atomic_save(output / "best.pt", {
                    "kind": "best_cbr_policy_bundle",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "seed": int(args.seed), "mode": args.mode,
                    "policy": _cpu_tree(cbr.checkpoint()),
                    "evaluation": random_eval,
                    "source_hashes": source_manifest,
                })
            if args.mode == "bc_rl":
                bc_metrics = dependencies.demonstrations.warm_start_joint(
                    cbr.actor,
                    lambda spatial, state: cbr.actor(spatial, state),
                    dataset_path=args.demonstrations,
                    device=device,
                    seed=int(args.seed) + int(args.population),
                    epochs=int(args.bc_epochs),
                    batch_size=BC_BATCH_SIZE,
                )
                if int(bc_metrics["demo_env_steps"]) != demonstration_steps:
                    raise ValueError("CBR BC demo steps differ from the loaded dataset")
                if bc_metrics.get("dataset_sha256") != demonstration_hash:
                    raise ValueError("CBR BC dataset hash differs from the loaded dataset")
                cbr_bc_metrics = _json_safe(dict(bc_metrics))
                bc_eval = _call_eval(cbr.actor, "convention_aware", candidate_kind="bc_cbr_initial")
                runtime.atomic_save(output / "bc_cbr_initial.pt", {
                    "kind": "bc_initialized_cbr_actor",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "demonstration_env_steps_counted_once": True,
                    "demo_env_steps": demonstration_steps,
                    "dataset_sha256": demonstration_hash,
                    "seed": int(args.seed) + int(args.population), "mode": args.mode,
                    "bc_metrics": cbr_bc_metrics,
                    "native_evaluation": bc_eval,
                    "source_hashes": source_manifest,
                    "actor_state_dict": _cpu_tree(cbr.actor.state_dict()),
                })
                if bc_eval.get("complete") and (
                    best_cbr is None or float(bc_eval["mean_team_score"]) >= float(best_cbr["mean_team_score"])
                ):
                    best_cbr = bc_eval.copy()
                    runtime.atomic_save(output / "best.pt", {
                        "kind": "best_cbr_policy_bundle",
                        "train_env_steps": int(counters["total_train_env_steps"]),
                        "seed": int(args.seed), "mode": args.mode,
                        "policy": _cpu_tree(cbr.checkpoint()),
                        "evaluation": bc_eval,
                        "source_hashes": source_manifest,
                    })
                log_event({"event": "cbr_bc_warm_start", "dataset_steps_counted_once": False,
                           "bc_metrics": cbr_bc_metrics, "native_evaluation": bc_eval})
            save_latest("running", "cbr", int(args.population), 0)

            cbr_sp_rounds = _round_budgets(
                budgets["cbr"]["cbr_sp"], int(args.rounds_per_convention), int(args.num_envs)
            )
            cbr_bc_rounds = _round_budgets(
                budgets["cbr"]["cbr_bc"], int(args.rounds_per_convention), int(args.num_envs)
            )
            teacher_cursor = 0
            for round_index in range(int(args.rounds_per_convention)):
                if should_stop():
                    state = "time_limit"
                    break
                cbr.actor.eval()
                sp_batches, sp_summaries, sp_steps = collect_chunks(
                    cbr_sp_rounds[round_index],
                    lambda n: collect_self_play(
                        env, cbr, n, device=device, gamma=GAMMA, gae_lambda=GAE_LAMBDA
                    )[::2],
                )
                sp_batch = _merge_batches(sp_batches)
                teacher_batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
                teacher_env_steps: dict[int, int] = defaultdict(int)
                teacher_summaries: dict[int, list[dict[str, Any]]] = defaultdict(list)
                teacher_episode_counts: dict[int, int] = defaultdict(int)
                next_teacher_cursor = teacher_cursor

                def collect_teachers(vector_steps):
                    nonlocal next_teacher_cursor
                    batches, info = collect_teacher_self_play(
                        env, conventions, vector_steps, device=device,
                        start_teacher=next_teacher_cursor,
                    )
                    next_teacher_cursor = int(info["next_teacher_cursor"])
                    for teacher_id, data in batches.items():
                        teacher_batches[int(teacher_id)].append(data)
                    for teacher_id in range(len(conventions)):
                        key = str(teacher_id)
                        teacher_env_steps[teacher_id] += int(info["teacher_env_steps"][key])
                        teacher_episode_counts[teacher_id] += int(info["teacher_episode_counts"][key])
                        teacher_summaries[teacher_id].extend(info["episode_summaries_by_teacher"][key])
                    combined_summaries = [
                        summary for teacher_id in range(len(conventions))
                        for summary in info["episode_summaries_by_teacher"][str(teacher_id)]
                    ]
                    return None, {"env_steps": info["env_steps"], "episode_summaries": combined_summaries}

                _, teacher_all_summaries, teacher_steps = collect_chunks(
                    cbr_bc_rounds[round_index], collect_teachers
                )
                bc_batches = [
                    (teacher_id, _merge_batches(chunks))
                    for teacher_id, chunks in sorted(teacher_batches.items()) if chunks
                ]
                cbr.actor.train()
                update_info = update_bundle(
                    cbr,
                    (("sp", sp_batch, 1.0),),
                    device=device,
                    epochs=int(args.epochs),
                    minibatches=int(args.minibatches),
                    clip=CLIP,
                    entropy_coef=ENTROPY_COEF,
                    value_coef=VALUE_COEF,
                    max_grad_norm=MAX_GRAD_NORM,
                    prior_count=0,
                    bc_coef=CBR_BC_COEF,
                    bc_objectives=bc_batches,
                    rng=np.random.default_rng(
                        int(args.seed) + int(counters["total_train_env_steps"]) + 91
                    ),
                )
                round_steps = int(sp_steps + teacher_steps)
                add_train_steps(round_steps)
                counters["phases"]["cbr_sp"] += int(sp_steps)
                counters["phases"]["cbr_teacher_bc"] += int(teacher_steps)
                for teacher_id in range(len(conventions)):
                    counters["teacher_env_steps"][teacher_id] += teacher_env_steps[teacher_id]
                teacher_cursor = (next_teacher_cursor + 1) % len(conventions)
                log_event({
                    "event": "cbr_joint_round",
                    "round_index": round_index,
                    "env_steps": {"cbr_sp": sp_steps, "teacher_bc": teacher_steps,
                                  "total": round_steps},
                    "bc_teacher_env_steps": {str(i): teacher_env_steps[i]
                                              for i in range(len(conventions))},
                    "bc_teacher_episode_counts": {str(i): teacher_episode_counts[i]
                                                   for i in range(len(conventions))},
                    "bc_teacher_role_label_counts": {
                        str(i): [teacher_env_steps[i]] * NUM_ROLES
                        for i in range(len(conventions))
                    },
                    "train": update_info,
                    "episodes": {
                        "sp": _episode_means(sp_summaries),
                        "teacher_self_play": {
                            str(i): _episode_means(teacher_summaries[i])
                            for i in range(len(conventions))
                        },
                        "teacher_completed": _episode_means(teacher_all_summaries),
                    },
                })
                maybe_validate(cbr.actor, "convention_aware")
                finish_round("cbr", int(args.population), round_index + 1)

            if state == "completed":
                final_eval = _call_eval(cbr.actor, "convention_aware", candidate_kind="final")
                if final_eval.get("complete") and (
                    best_cbr is None or float(final_eval["mean_team_score"]) > float(best_cbr["mean_team_score"])
                ):
                    best_cbr = final_eval.copy()
                    runtime.atomic_save(output / "best.pt", {
                        "kind": "best_cbr_policy_bundle",
                        "train_env_steps": int(counters["total_train_env_steps"]),
                        "seed": int(args.seed), "mode": args.mode,
                        "policy": _cpu_tree(cbr.checkpoint()),
                        "evaluation": final_eval,
                        "source_hashes": source_manifest,
                    })
                runtime.atomic_save(output / "final.pt", {
                    "kind": "final_cbr_policy_bundle",
                    "status": "completed",
                    "train_env_steps": int(counters["total_train_env_steps"]),
                    "seed": int(args.seed), "mode": args.mode,
                    "policy": _cpu_tree(cbr.checkpoint()),
                    "best_evaluation": best_cbr,
                    "final_evaluation": final_eval,
                    "source_hashes": source_manifest,
                })
                save_latest("completed", "complete", int(args.population), int(args.rounds_per_convention))
            else:
                save_latest("time_limit", "cbr", int(args.population), int(counters["phases"]["cbr_sp"]))
        else:
            if state == "completed":
                state = "time_limit"
            save_latest(state, "population", len(conventions), 0)

        final_state = "completed" if state == "completed" and cbr is not None else state
        runtime.atomic_json(output / "status.json", _json_safe({
            "state": final_state,
            **step_counts(),
            "requested_budget_jointenvsteps": budgets["requested_total"],
            "effective_budget_jointenvsteps": budgets["effective_total"],
            "best_population": best_population,
            "best_cbr": best_cbr,
            "wallclock_seconds": elapsed(),
            "updated_at_unix": time.time(),
        }))
        log_event({"event": "finished", "state": final_state,
                   "requested_budget_jointenvsteps": budgets["requested_total"],
                   "effective_budget_jointenvsteps": budgets["effective_total"]})
        return {
            "state": final_state,
            "output": str(output),
            "total_env_steps": int(counters["total_train_env_steps"]),
            "best_population": best_population,
            "best_cbr": best_cbr,
        }
    except BaseException as error:
        runtime.atomic_json(output / "status.json", _json_safe({
            "state": "failed",
            **step_counts(),
            "stage": "unknown",
            "error_type": type(error).__name__,
            "error": str(error),
            "wallclock_seconds": elapsed(),
            "updated_at_unix": time.time(),
        }))
        if current is not None or cbr is not None:
            runtime.atomic_save(
                output / "latest.pt",
                payload("failed", len(conventions), -1),
            )
        raise
    finally:
        if env is not None:
            env.close()
        current = None
        cbr = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

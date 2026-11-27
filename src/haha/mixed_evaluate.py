"""Native reference scoring and batched CUDA evaluation for MixedKitchen."""

import time
from argparse import Namespace

import numpy as np
import torch

from .early_handoff import EarlyHandoffGate, GATE_METRICS
from .env import NoExecutableSubtask
from .mixed_env import MixedVecEnv
from .mixed_kitchen import GAME_METRICS
from .mixed_tasks import NAMES, PLANNING_NAMES, PREPARE_HANDOFF, STAY, UNKNOWN
from .tensor_ppo import actor_logits, sample_actions

PREPARATION_METRICS = ("handoff_preparation_steps", "handoff_preparation_stay_steps")
HANDOFF_DIAGNOSTICS = ("own_handoff_receives", "teammate_handoff_receives",
                       "own_joint_deliveries", "teammate_joint_deliveries", "handoff_backoffs",
                       "own_handoff_donations")
LOOKAHEAD_METRICS = ("handoff_lookahead_calls", "handoff_lookahead_switches")
COMMIT_METRICS = ("committed_handoff_steps", "committed_action_overrides")
ABORT_FIELDS = ("tick", "batch", "progress", "sync_steps", "sync_due", "subtask",
                *(f"{field}_{seat}" for field in ("in_pair", "action", "position", "orientation", "held")
                  for seat in range(3)))
TRACE_FIELDS = ("tick", "batch", "subtask", "sync_progress", "sync_steps", "sync_due", "delivery_due",
                *(f"{field}_{seat}" for field in ("position", "orientation", "held", "action") for seat in range(3)),
                "joint_started", "joint_aborted", *HANDOFF_DIAGNOSTICS)


def _result(stage, layout, role, player, seed, episodes, names):
    result = {"layout": layout, "partner": role, "player": player, "seed": seed,
              "status": "running", "requested_episodes": episodes, "returns": [], "lengths": [],
              "switches": [], "skills": {name: dict(success=0, failure=0, invalidated=0, successful_steps=0)
                                         for i, name in enumerate(names) if i != UNKNOWN},
              "return_kind": "team_sparse_reward" if stage == "manager" else "skill_reward"}
    if stage == "manager":
        result["game_metrics"] = []
    return result


def _record(result, reward, length, switches, task, success, invalidated, names):
    if result["return_kind"] == "skill_reward":
        record = result["skills"][names[task]]
        if success:
            record["success"] += 1
            record["successful_steps"] += length
        elif invalidated:
            record["invalidated"] += 1
        else:
            record["failure"] += 1
    result["returns"].append(reward)
    result["lengths"].append(length)
    result["switches"].append(switches)


def _summarize(result, base_steps):
    returns = result["returns"]
    result.update(episodes=len(returns), mean_return=float(np.mean(returns)) if returns else None,
                  episode_return_se=float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
                  if len(returns) > 1 else None, base_env_steps=base_steps)
    for record in result["skills"].values():
        valid = record["success"] + record["failure"]
        record["success_rate"] = record["success"] / valid if valid else None
        record["mean_success_steps"] = (record["successful_steps"] / record["success"]
                                        if record["success"] else None)


def measure_mixed(args, model, stage, layout, role, player, seed, episodes, *, deadline, persist):
    options = Namespace(**{**vars(args), "layouts": [layout], "n_envs": 1})
    env = MixedVecEnv(options, stage=stage, evaluation=True, player=player, role=role)
    result = _result(stage, layout, role, player, seed, episodes, env.names)
    env.seed(seed)
    torch.manual_seed(seed)
    try:
        obs = env.reset()
        for _ in range(episodes):
            switches, previous = 0, None
            preparation_steps, preparation_stays = 0, 0
            while time.monotonic() < deadline:
                kwargs = {"action_masks": obs["subtask_mask"]} if stage == "manager" else {}
                actions, _ = model.predict(obs, deterministic=args.deterministic, **kwargs)
                obs, _, done, infos = env.step(actions)
                info = infos[0]
                task = info["subtask"]
                if stage == "manager":
                    switches += int(previous is not None and task != previous)
                    previous = task
                    preparation_steps += task == PREPARE_HANDOFF
                    preparation_stays += task == PREPARE_HANDOFF and info["low_action"] == STAY
                if not done[0]:
                    continue
                length = info["episode"]["l"]
                if stage == "manager":
                    metrics = {key: info["game_metrics"][key] for key in GAME_METRICS}
                    metrics.update(zip(PREPARATION_METRICS, (preparation_steps, preparation_stays)))
                    result["game_metrics"].append(metrics)
                _record(result, info["episode"]["r"], length, switches, task,
                        info.get("task_success", False), info.get("task_invalidated", False), env.names)
                break
            else:
                result["status"] = "time_limit"
                break
        else:
            result["status"] = "complete"
    except NoExecutableSubtask as exc:
        result.update(status="no_executable_skill", error=str(exc))
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _summarize(result, sum(env.env_method("get_base_steps")))
        if persist:
            persist(result)
        env.close()
    return result


@torch.no_grad()
def measure_mixed_batch(args, model, stage, role, episodes, seed, deadline):
    """Collect CUDA results for all requested map/seat conditions in one batch."""
    manager = stage == "manager"
    planning = manager and getattr(args, "mixed_handoff_planning", None) not in (None, "baseline")
    names = PLANNING_NAMES if planning else NAMES
    conditions = [(layout, player) for layout in args.layouts for player in range(3)]
    if time.monotonic() >= deadline:
        results = [_result(stage, layout, role, player, seed, episodes, names) for layout, player in conditions]
        for result in results:
            result.update(status="time_limit", backend="cuda")
            _summarize(result, 0)
        return results
    # Manager games are independent. Worker skills retain the per-seat kitchen
    # between episodes, allowing later skills to use states earlier skills reached.
    replicas = episodes if manager else 1
    quota = 1 if manager else episodes
    lanes = [condition for condition in conditions for _ in range(replicas)]
    options = Namespace(**{**vars(args), "layouts": [layout for layout, _ in lanes],
                           "n_envs": len(lanes), "compile_kitchen": False})
    env = MixedVecEnv(options, stage=stage, evaluation=True, player=[player for _, player in lanes],
                      role=role, tensor_evaluation=True)
    env.seed(seed)
    rng = torch.Generator(device=env.device).manual_seed(seed)
    rows = env.kitchen.rows
    counts = torch.zeros_like(rows)
    switches = torch.zeros_like(rows)
    previous = torch.full_like(rows, -1)
    records = torch.empty((quota + 1, len(lanes), 6), device=env.device)
    metrics = torch.zeros((len(lanes), len(GAME_METRICS)), dtype=torch.long, device=env.device)
    preparation_metrics = torch.zeros((len(lanes), len(PREPARATION_METRICS)),
                                      dtype=torch.long, device=env.device)
    diagnostics = torch.zeros((len(lanes), len(HANDOFF_DIAGNOSTICS)), dtype=torch.long, device=env.device)
    lookahead_metrics = torch.zeros((len(lanes), len(LOOKAHEAD_METRICS)), dtype=torch.long, device=env.device)
    commit_metrics = torch.zeros((len(lanes), len(COMMIT_METRICS)), dtype=torch.long, device=env.device)
    capture_aborts = manager and getattr(args, "mixed_abort_events", False)
    abort_frames = []
    trace_count = args.trajectories if manager else 0
    if trace_count:
        trace_rows = (rows[::replicas, None] + rows[:trace_count]).flatten()
        frames = []
    planner = None
    gate = None
    try:
        observations = env.reset_tensors()
        early_handoff = getattr(args, "mixed_early_handoff", "off")
        if manager and early_handoff != "off":
            gate = EarlyHandoffGate(early_handoff, env)
        if manager and getattr(args, "mixed_lookahead", "off") != "off":
            from .handoff_lookahead import HandoffLookahead

            planner = HandoffLookahead(args, env, model.policy, seed + 73_000)
        while time.monotonic() < deadline:
            for _ in range(16):
                if time.monotonic() >= deadline:
                    break
                active = counts < quota
                logits = actor_logits(model.policy, observations)
                if manager:
                    logits = logits.masked_fill(~observations["subtask_mask"], -1e8)
                actions = sample_actions(logits, args.deterministic, rng)
                if manager:
                    if planner is not None:
                        planned, selected = planner.choose(env, actions, logits, active)
                        lookahead_metrics[:, 0] += selected
                        lookahead_metrics[:, 1] += selected & (actions != planned)
                        actions = planned
                    if gate is not None:
                        actions = gate.choose(actions, active)
                    before = env.state
                    switches += active & (previous >= 0) & (previous != actions)
                    observations, _, metadata = env.step_tensors(actions, active=active)
                    metrics += torch.where(active[:, None], env.step_metrics, 0)
                    diagnostics += active[:, None] & env.handoff_diagnostics
                    if env.handoff_commit:
                        commit_metrics += env.handoff_commit_stats
                    if capture_aborts:
                        aborted = active & env.step_metrics[:, 3].bool()
                        order = torch.stack((aborted, before.tick, before.batch, before.progress,
                                             before.sync_steps, before.sync_due, actions), dim=1)
                        abort_frames.append(torch.cat((order, before.pair, env.last_joint_actions,
                                                       before.position, before.orientation, before.held), dim=1))
                    if trace_count:
                        order = torch.stack((before.tick, before.batch, actions, before.progress, before.sync_steps,
                                             before.sync_due, before.due[:, 1]), dim=1)
                        frames.append(torch.cat((order, before.position, before.orientation, before.held,
                                                 env.last_joint_actions, env.step_metrics[:, 2:4],
                                                 env.handoff_diagnostics), dim=1)[trace_rows])
                    if planning:
                        preparing = active & (actions == PREPARE_HANDOFF)
                        preparation_metrics[:, 0] += preparing
                        preparation_metrics[:, 1] += preparing & (metadata[:, 4] == STAY)
                else:
                    observations, _, metadata, _ = env.step_worker_tick(actions, active)
                done = metadata[:, 1].bool() & active
                if gate is not None:
                    gate.update(before, active, done)
                slots = torch.where(done, counts, quota)
                records[slots, rows] = torch.stack(
                    (metadata[:, 7], metadata[:, 8], switches, metadata[:, 9],
                     metadata[:, 10], metadata[:, 11]), 1,
                )
                counts += done
                switches = torch.where(done, 0, switches)
                previous = torch.where(done, -1, actions)
            if bool(((counts >= quota) | env.exhausted).all()):
                break
        sizes, exhausted = counts.tolist(), env.exhausted.tolist()
        data = records[:-1].cpu().numpy()
        base_steps = env.base_steps.tolist()
        per_game = torch.cat((metrics, preparation_metrics, diagnostics, lookahead_metrics,
                              commit_metrics), dim=1) if manager else None
        metric_names = (*GAME_METRICS, *PREPARATION_METRICS, *HANDOFF_DIAGNOSTICS, *LOOKAHEAD_METRICS, *COMMIT_METRICS)
        if gate is not None:
            per_game = torch.cat((per_game, gate.metrics), dim=1)
            metric_names = (*metric_names, *GATE_METRICS)
        if manager:
            per_game = per_game.tolist()
        traces = torch.stack(frames, dim=1).tolist() if trace_count and frames else []
        aborts = [[] for _ in lanes]
        if abort_frames:
            frames_by_tick = torch.stack(abort_frames)
            ticks, lane_ids = frames_by_tick[:, :, 0].nonzero(as_tuple=True)
            for lane, event in zip(lane_ids.tolist(), frames_by_tick[ticks, lane_ids, 1:].tolist()):
                aborts[lane].append(event)
        results = []
        for index, (layout, player) in enumerate(conditions):
            result = _result(stage, layout, role, player, seed, episodes, names)
            result.update(backend="cuda", seed_lane=index * replicas)
            if capture_aborts:
                result["handoff_abort_fields"] = ABORT_FIELDS
                result["handoff_abort_events"] = []
            if len(env.partner_roles) == 2:
                result["teammate_roles"] = [env.partner_roles[::1 if lane % 2 == 0 else -1]
                                            for lane in range(index * replicas, (index + 1) * replicas)]
            seats = range(index * replicas, (index + 1) * replicas)
            for lane in seats:
                for reward, length, changes, task, success, invalidated in data[:sizes[lane], lane]:
                    _record(result, float(reward), int(length), int(changes), int(task),
                            bool(success), bool(invalidated), names)
                if manager and sizes[lane]:
                    result["game_metrics"].append(dict(zip(metric_names, per_game[lane])))
                    if capture_aborts:
                        result["handoff_abort_events"].append(aborts[lane])
            if traces:
                result["trace_fields"] = TRACE_FIELDS
                result["traces"] = [traces[index * trace_count + i][:length]
                                    for i, length in enumerate(result["lengths"][:trace_count])]
            result["status"] = ("complete" if len(result["returns"]) == episodes else
                                "no_executable_skill" if any(exhausted[lane] for lane in seats) else "time_limit")
            _summarize(result, sum(base_steps[lane] for lane in seats))
            results.append(result)
        return results
    finally:
        if planner is not None:
            planner.close()
        env.close()

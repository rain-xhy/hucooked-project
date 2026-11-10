"""Policy measurements and training-only BC validation; no work on import."""

import json
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from .artifacts import check_loaded_policy, checkpoint_info, write_json
from .env import NoExecutableSubtask
from .subtasks import NAMES, UNKNOWN


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported trajectory value: {type(value).__name__}")


def measure(args, model, stage, layout, role, player, seed, episodes, *,
            deadline, trace=None, persist=None):
    from .train import make_env

    result = {"layout": layout, "partner": role, "player": player, "seed": seed,
              "status": "running", "requested_episodes": episodes, "returns": [],
              "lengths": [], "switches": [], "skills": {
                  name: {"success": 0, "failure": 0, "invalidated": 0,
                         "successful_steps": 0} for name in NAMES[:-1]
              }, "return_kind": "team_sparse_reward" if stage == "manager" else "skill_reward"}
    torch.manual_seed(seed)
    env = make_env(args, stage, layout, 0, evaluation=True, partner_role=role, player=player)

    def update():
        returns = result["returns"]
        result.update(episodes=len(returns), mean_return=float(np.mean(returns)) if returns else None,
                      episode_return_se=float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
                      if len(returns) > 1 else None,
                      base_env_steps=env.base_steps, reset_wait_steps=env.reset_wait_steps)
        for record in result["skills"].values():
            valid = record["success"] + record["failure"]
            record["success_rate"] = record["success"] / valid if valid else None
            record["mean_success_steps"] = (record["successful_steps"] / record["success"]
                                            if record["success"] else None)
        if persist:
            persist(result)

    try:
        for episode in range(episodes):
            if time.monotonic() >= deadline:
                result["status"] = "time_limit"
                break
            obs, _ = env.reset(seed=seed if episode == 0 else None)
            total, length, changes, previous_task = 0.0, 0, 0, None
            while True:
                if time.monotonic() >= deadline:
                    result["status"] = "time_limit"
                    break
                kwargs = {"action_masks": env.action_masks()} if stage == "manager" else {}
                action, _ = model.predict(obs, deterministic=args.deterministic, **kwargs)
                action = int(np.asarray(action).item())
                before = env.state.to_dict() if trace and episode < args.trajectories else None
                if stage == "manager":
                    changes += int(previous_task is not None and action != previous_task)
                    previous_task = action
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(info["team_reward"] if stage == "manager" else reward)
                length += 1
                if before is not None:
                    trace.write(json.dumps({
                        "schema": "haha-native-v1", "layout": layout, "partner": role,
                        "player": player, "episode": episode, "step": length,
                        "state": before, "next_state": env.state.to_dict(),
                        "manager_task": action, "low_action": info["low_action"],
                        "joint_action": info["joint_action"], "team_reward": info["team_reward"],
                        "terminated": terminated, "truncated": truncated,
                    }, default=_json_value) + "\n")
                if terminated or truncated:
                    if stage == "worker" and info["subtask"] != UNKNOWN:
                        record = result["skills"][NAMES[info["subtask"]]]
                        if info["task_invalidated"] or truncated and not terminated:
                            record["invalidated"] += 1
                        elif info["task_success"]:
                            record["success"] += 1
                            record["successful_steps"] += length
                        else:
                            record["failure"] += 1
                    break
            if result["status"] == "time_limit":
                break  # Never mix incomplete episodes into the reported mean.
            result["returns"].append(total)
            result["lengths"].append(length)
            result["switches"].append(changes)
            update()
        else:
            result["status"] = "complete"
    except NoExecutableSubtask as exc:
        result.update(status="no_executable_skill", error=str(exc))
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        update()
        env.close()
    return result


def validate(args, model, stage, deadline):
    results = []
    for index, layout in enumerate(args.layouts):
        for player in (0, 1):
            if time.monotonic() >= deadline:
                return {"status": "time_limit", "score": None, "results": results}
            results.append(measure(
                args, model, stage, layout, "bc", player,
                args.seed + 500_000 + index * 1000 + player * 100,
                args.validation_episodes if stage == "worker" else args.validation_games,
                deadline=deadline,
            ))
    complete = all(r["status"] == "complete" for r in results)
    if stage == "manager":
        score = float(np.mean([r["mean_return"] for r in results])) if complete else None
        coverage = None
    else:
        # A fixed set prevents picking a checkpoint by succeeding only at easy pickups.
        core = [0, 2, 4, 7, 10]
        coverage, rates = {}, []
        for task in core:
            successes = sum(r["skills"][NAMES[task]]["success"] for r in results)
            valid = sum(r["skills"][NAMES[task]]["success"] +
                        r["skills"][NAMES[task]]["failure"] for r in results)
            coverage[NAMES[task]] = valid
            if valid >= args.validation_min_samples:
                rates.append(successes / valid)
        score = float(np.mean(rates)) if complete and len(rates) == len(core) else None
    return {"status": "complete" if complete else "incomplete", "score": score,
            "core_skill_samples": coverage, "partner": "bc", "results": results}


def evaluate(args, device):
    from .train import preflight

    stage = args.evaluate_stage
    path = args.worker if stage == "worker" else args.manager
    config = preflight(args, device)
    metadata = checkpoint_info(path)
    algorithm = PPO if stage == "worker" else MaskablePPO
    model = algorithm.load(path, device=device)
    check_loaded_policy(model, metadata)
    model.policy.set_training_mode(False)
    out = args.run_dir / "evaluation" / f"{stage}_{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    payload = {"config": config, "checkpoint": str(path), "checkpoint_metadata": metadata,
               "policy_parameters": sum(p.numel() for p in model.policy.parameters()),
               "status": "running", "results": [],
               "note": "Episode SE is not uncertainty across training seeds. Native JSONL is not "
                       "the group's replay format. Training BC, proxy and random are reported separately."}
    deadline = time.monotonic() + args.max_hours * 3600

    def persist(result):
        key = (result["layout"], result["partner"], result["player"])
        payload["results"] = [r for r in payload["results"]
                              if (r["layout"], r["partner"], r["player"]) != key] + [result]
        write_json(out / "results.json", payload)

    write_json(out / "results.json", payload)
    try:
        for index, layout in enumerate(args.layouts):
            for role_index, role in enumerate(args.partners):
                for player in (0, 1):
                    trace = None
                    try:
                        if args.trajectories and stage == "manager":
                            trace = (out / f"{layout}_{role}_seat{player}.jsonl").open("w", encoding="utf-8")
                        result = measure(args, model, stage, layout, role, player,
                                         args.seed + 1_000_000 + index * 10000 + role_index * 1000 + player * 100,
                                         args.episodes, deadline=deadline, trace=trace, persist=persist)
                        print(f"{layout} / {role} / seat {player}: {result['mean_return']} ({result['status']})", flush=True)
                    finally:
                        if trace:
                            trace.close()
        payload["status"] = ("complete" if all(r["status"] == "complete" for r in payload["results"])
                             else "partial")
    except BaseException as exc:
        payload.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(out / "results.json", payload)
        print(f"Evaluation saved to {out}", flush=True)

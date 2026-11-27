"""Train central HiPT on SharedKitchen. Run with ``python -m joint_hipt.train``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from joint_common import runtime
from joint_common.demonstrations import load_joint_dataset

from .behavior_cloning import warm_start_hierarchical
from .evaluate import evaluate, evaluate_population
from .model import CentralFlatActorCritic, JointHiPTPolicy
from .population import PopulationTrainer
from .trainer import HierarchicalTrainer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("rl", "bc_rl"), required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--steps", type=int, default=8_000_000,
                   help="Total vector-environment transitions, including BC dataset steps")
    p.add_argument("--pool-steps", type=int, default=None,
                   help="Population stage total steps including demonstration steps")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--rollout", type=int, default=300)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--z-dim", type=int, choices=(4, 5, 6), default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-gpu", type=int, default=1)
    p.add_argument("--memory-fraction", type=float, default=.035)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=.9985)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--value-coef", type=float, default=.5)
    p.add_argument("--low-entropy-coef", type=float, default=.01)
    p.add_argument("--high-entropy-coef", type=float, default=.01)
    p.add_argument("--population-entropy-coef", type=float, default=.01)
    p.add_argument("--jsd-coef", type=float, default=.05)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--p-min", type=int, default=20)
    p.add_argument("--p-max", type=int, default=40)
    p.add_argument("--influence-coef", type=float, default=1000.)
    p.add_argument("--influence-final", type=float, default=1.)
    p.add_argument("--selfplay-fraction", type=float, default=.5)
    p.add_argument("--eval-interval", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--checkpoint-interval", type=int, default=250_000)
    p.add_argument("--pool-eval-episodes", type=int, default=4)
    p.add_argument("--demonstrations", default=None)
    p.add_argument("--bc-epochs", type=int, default=64)
    p.add_argument("--max-seconds", type=float, default=158400.)
    p.add_argument("--smoke", action="store_true")
    return p


def _configure(args):
    if args.mode == "bc_rl":
        if not args.demonstrations:
            raise ValueError("--demonstrations is required in bc_rl mode")
        data = load_joint_dataset(args.demonstrations)
        demo_steps = int(data["env_steps"])
        dataset_hash = data["dataset_sha256"]
    else:
        # Pure RL never reads or validates a demonstration path.
        demo_steps, dataset_hash = 0, None
    if args.smoke:
        args.population = 2
        args.num_envs = min(args.num_envs, 2)
        args.rollout = max(41, min(args.rollout, 48))
        args.workers = 0
        args.eval_episodes = min(args.eval_episodes, args.num_envs)
        args.eval_interval = 0
        args.checkpoint_interval = args.num_envs * args.rollout
        args.pool_eval_episodes = min(2, args.num_envs)
        # Preserve the requested BC epoch count and include its data steps.
        args.steps = demo_steps + 2 * args.num_envs * args.rollout
        args.pool_steps = demo_steps + args.num_envs * args.rollout
    elif args.pool_steps is None:
        args.pool_steps = args.steps // 4
    if args.steps <= demo_steps or args.pool_steps is None or args.pool_steps <= demo_steps:
        raise ValueError("steps and pool-steps must exceed demonstration steps")
    if args.pool_steps >= args.steps:
        raise ValueError("Population training must leave at least one hierarchical rollout")
    if args.num_envs <= 0 or args.rollout <= 0 or args.workers < 0 or args.population < 2:
        raise ValueError("num-envs, rollout, population and workers are invalid")
    if not 20 <= args.p_min <= args.p_max <= 40:
        raise ValueError("HiPT option durations must satisfy 20 <= p-min <= p-max <= 40")
    if args.epochs <= 0 or args.minibatches <= 0 or args.bc_epochs <= 0:
        raise ValueError("epochs, minibatches, and bc-epochs must be positive")
    if args.lr <= 0 or not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("lr must be positive and gamma/gae-lambda in (0,1]")
    if not 0 <= args.selfplay_fraction <= 1:
        raise ValueError("selfplay-fraction must be in [0,1]")
    if not 0 < args.max_seconds <= 158400:
        raise ValueError("max-seconds must be positive and at most 44 hours")
    if args.eval_interval < 0 or args.checkpoint_interval < 0 or args.eval_episodes <= 0:
        raise ValueError("Invalid evaluation/checkpoint arguments")
    batch = args.num_envs * args.rollout
    total_updates = (args.steps - demo_steps) // batch
    pool_updates = (args.pool_steps - demo_steps) // batch
    if total_updates < 2 or pool_updates < 1:
        raise ValueError("Budget must include one full update in each training stage")
    pool_updates = min(total_updates - 1, pool_updates)
    return demo_steps, dataset_hash, total_updates, pool_updates, batch


def _append(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def _select_mid(team_scores_by_step, metrics_by_step, final_step):
    final_scores = np.asarray(team_scores_by_step[final_step], dtype=np.float64)
    selected, members = [], []
    for member, final_score in enumerate(final_scores):
        target = float(final_score / 2.)
        candidates = [(step, float(scores[member])) for step, scores in team_scores_by_step.items()
                      if step < final_step]
        if not candidates:
            candidates = [(final_step, float(final_score))]
        errors = [(abs(score - target), step, score) for step, score in candidates]
        minimum = min(row[0] for row in errors)
        error, step, score = max((row for row in errors if abs(row[0] - minimum) <= 1e-12),
                                 key=lambda row: row[1])
        failed = bool(final_score <= 0.)
        members.append({
            "member": member, "final_team_score": float(final_score),
            "target_half_team_score": target, "selected_step": int(step),
            "selected_team_score": score, "absolute_target_error": float(error),
            "quality": "qualityfailed" if failed else "measured_positive_final",
            "quality_warning": "final team_score <= 0" if failed else None,
            "selected_is_final": bool(step == final_step),
            "candidate_team_scores": {str(s): float(v[member])
                                       for s, v in sorted(team_scores_by_step.items())},
        })
        selected.append(int(step))
    return selected, {
        "metric": "standard unshaped team_score mean",
        "target": "final team_score / 2, including negative final scores",
        "tie_rule": "latest eligible step wins",
        "candidate_metric_means": {
            str(step): {key: value.tolist() for key, value in metrics.items()}
            for step, metrics in sorted(metrics_by_step.items())
        },
        "members": members,
        "qualityfailed_members": [row["member"] for row in members
                                  if row["quality"] == "qualityfailed"],
    }


def _pool_states(models):
    return [{key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for model in models]


def _make_pool_models(states):
    models = [CentralFlatActorCritic().cpu() for _ in states]
    for model, payload in zip(models, states):
        model.load_state_dict(payload)
    return models


def main(argv=None):
    args = parser().parse_args(argv)
    demo_steps, dataset_hash, total_updates, pool_updates, batch_size = _configure(args)
    device = runtime.setup_device(args.device, args.seed, physical_gpu=args.physical_gpu,
                                  memory_fraction=args.memory_fraction)
    from joint_common.envs import KitchenBatch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output or f"joint_hipt/{args.mode}/seed_{args.seed}/results")
    runtime.prepare_output(out)
    config_path = out / "config.json"
    sources = runtime.source_hashes("joint_hipt")
    config = {
        "method": "HiPT central-team",
        "mode": args.mode,
        "paper": "Loo et al., IJCAI 2023, arXiv:2305.16708",
        "arguments": vars(args),
        "demonstrations": None if args.mode == "rl" else {
            "path": args.demonstrations, "sha256": dataset_hash,
            "environment_steps": demo_steps, "epochs": args.bc_epochs,
            "population_batch_size_actions": 512,
            "population_joint_frames_per_batch": 170,
            "hierarchical_sequence_mode": "complete joint episodes; one central LSTM state per team",
            "transition_counted_once": True,
        },
        "architecture": {
            "central_inputs": "canonical obs.spatial[:,0] and shared 863-value public state",
            "one_encoder_and_lstm_stream_per_environment": True,
            "roles": 3, "z_dim": args.z_dim, "action_dim": 6,
            "population_jsd": "single JSD averaged over batch and 3 per-role 6-action distributions",
            "population_update": "CPU models/Adam with exact joint-logit-gradient staging through one GPU workspace",
            "partner_models": "frozen central snapshots stay on CPU",
        },
        "protocol": {
            "stage_order": ["population", "hierarchical"],
            "population": "16 central policies; shared-team PPO sum and full-member JSD gradients",
            "skill_levels": "pre-BC random, per-member checkpoint nearest half final native team_score, final",
            "hierarchical": "central state; three independent z policies, clocks, option returns and GAE streams",
            "option_duration": [args.p_min, args.p_max],
            "macro_reward": "observed-duration mean of legacy environment reward plus influence_coef × each chef's original directed KL",
            "recurrent_training": "one central hidden per environment; truncated context at full rollout length for low/high BPTT",
            "step_accounting": "one vector environment transition counts once; BC train transitions counted once in bc_rl",
            "evaluation": "native unshaped standard task; main central policy controls all three chefs",
            "best_metric": "mean unshaped standard team_score",
        },
        "planned_updates": {"total": total_updates, "population": pool_updates,
                            "hierarchical": total_updates - pool_updates},
        "step_budget": {
            "requested_total_env_steps": args.steps,
            "requested_population_total_env_steps": args.pool_steps,
            "demonstration_env_steps": demo_steps,
            "planned_population_rl_steps": pool_updates * batch_size,
            "planned_hierarchical_rl_steps": (total_updates - pool_updates) * batch_size,
            "planned_total_env_steps": demo_steps + total_updates * batch_size,
            "vector_env_batch_size": batch_size,
        },
        "source_hashes": sources,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "parameter_counts": {
            "population_policy": CentralFlatActorCritic().parameter_counts(),
            "hierarchical_policy": JointHiPTPolicy(args.z_dim).parameter_counts(),
        },
    }
    atomic_json = runtime.atomic_json
    atomic_save = runtime.atomic_save
    atomic_json(config_path, config)
    start_time = time.monotonic()
    stop = {"signal": None}

    def request_stop(signum, _frame):
        stop["signal"] = signum

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}

    def timed_out():
        return time.monotonic() - start_time >= args.max_seconds

    def status(state, stage, pool_rl_steps, main_rl_steps, **extra):
        value = {
            "state": state, "stage": stage,
            "population_env_steps": int(pool_rl_steps),
            "hierarchical_env_steps": int(main_rl_steps),
            "demonstration_env_steps": demo_steps,
            "total_env_steps": int(demo_steps + pool_rl_steps + main_rl_steps),
            "wall_seconds": float(time.monotonic() - start_time),
            "pid": os.getpid(), "physical_gpu": args.physical_gpu,
            "device": str(device),
        }
        value.update(extra)
        atomic_json(out / "status.json", value)

    def archive_pool(step, models):
        path = out / f"pool_step_{int(step)}.pt"
        if not path.exists():
            atomic_save(path, {"models": _pool_states(models), "step": int(step),
                               "source_hashes": sources})
        return path

    def save_pool_latest(trainer, pool_rl_steps):
        atomic_save(out / "pool_latest.pt", {
            "stage": "population", "population": runtime.cpu_tree(trainer.state_dict()),
            "population_env_steps": int(pool_rl_steps),
            "demonstration_env_steps": demo_steps,
            "total_env_steps": int(demo_steps + pool_rl_steps),
            "elapsed_seconds": time.monotonic() - start_time,
            "source_hashes": sources,
        })

    snapshots = None
    pool_trainer = None
    main_trainer = None
    pool_env = None
    main_env = None
    pool_rl_steps = main_rl_steps = 0
    try:
        status("running", "population", 0, 0)
        pool_env = KitchenBatch(num_envs=args.num_envs, seed=args.seed,
                                stage="standard", workers=args.workers,
                                train=True, gamma=args.gamma, reward_profile="legacy")
        pool_trainer = PopulationTrainer(
            pool_env, num_envs=args.num_envs, population_size=args.population,
            device=device, seed=args.seed, epochs=args.epochs, minibatches=args.minibatches,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip=args.clip, value_coef=args.value_coef,
            entropy_coef=args.population_entropy_coef, jsd_coef=args.jsd_coef,
            max_grad_norm=args.max_grad_norm,
        )
        snapshots = {"initial": _pool_states(pool_trainer.models)}
        atomic_save(out / "pool_initial_policy.pt", {
            "models": snapshots["initial"], "step": demo_steps,
            "phase": "pre_behavior_cloning_random" if args.mode == "bc_rl" else "random_initial",
        })
        behavior_metrics = []
        if args.mode == "bc_rl":
            behavior_metrics = pool_trainer.behavior_clone(
                args.demonstrations, seed=args.seed + 1000, epochs=args.bc_epochs, batch_size=512
            )
            config["behavior_cloning"] = {"population": behavior_metrics}
            pool_bc_metrics = evaluate_population(
                pool_trainer.models, episodes_per_member=min(args.eval_episodes, 4),
                workers=args.workers, seed=args.seed + 700_000, gamma=args.gamma,
            )
            atomic_json(out / "pool_bc_initial_eval.json", {
                "episodes_per_member": min(args.eval_episodes, 4),
                "seed": args.seed + 700_000, "action_mode": "greedy",
                "metric_means_by_member": {key: value.tolist() for key, value in pool_bc_metrics.items()},
                "metric_means_across_members": {key: float(value.mean()) for key, value in pool_bc_metrics.items()},
            })
        atomic_save(out / "pool_initial.pt", {
            "stage": "population_initial", "models": _pool_states(pool_trainer.models),
            "optimizers": runtime.cpu_tree([o.state_dict() for o in pool_trainer.optimizers]),
            "demonstration_env_steps": demo_steps, "source_hashes": sources,
        })
        config["behavior_cloning"] = config.get("behavior_cloning", {"population": [], "hierarchical": None})
        atomic_json(config_path, config)
        pool_updates = pool_updates
        for update_index in range(pool_updates):
            if stop["signal"] or timed_out():
                break
            metrics = pool_trainer.collect_and_update(args.rollout)
            pool_rl_steps = pool_trainer.total_env_steps
            total_so_far = demo_steps + pool_rl_steps
            metrics["total_env_steps"] = total_so_far
            _append(out / "metrics.jsonl", metrics)
            print(json.dumps(metrics, allow_nan=False), flush=True)
            checkpoint = args.checkpoint_interval
            crossed = checkpoint and (total_so_far // checkpoint >
                                      (demo_steps + pool_rl_steps - batch_size) // checkpoint)
            if crossed or update_index == pool_updates - 1:
                archive_pool(total_so_far, pool_trainer.models)
                save_pool_latest(pool_trainer, pool_rl_steps)
            status("running", "population", pool_rl_steps, 0,
                   update=pool_trainer.update, policy_action_steps=pool_trainer.agent_steps.tolist())
        pool_trainer.close()
        if pool_rl_steps < pool_updates * batch_size:
            save_pool_latest(pool_trainer, pool_rl_steps)
            archive_pool(demo_steps + pool_rl_steps, pool_trainer.models)
            state = "stopped_signal" if stop["signal"] else "stopped_time"
            status(state, "population", pool_rl_steps, 0,
                   policy_action_steps=pool_trainer.agent_steps.tolist())
            return 0

        final_pool_step = demo_steps + pool_rl_steps
        archive_pool(final_pool_step, pool_trainer.models)
        snapshots["final"] = _pool_states(pool_trainer.models)
        atomic_save(out / "pool_final.pt", {"models": snapshots["final"], "step": final_pool_step})
        candidate_paths = sorted(out.glob("pool_step_*.pt"),
                                 key=lambda path: int(path.stem.removeprefix("pool_step_")))
        candidate_steps, scores_by_step, metrics_by_step = [], {}, {}
        seed = args.seed + 700_000
        for candidate_index, path in enumerate(candidate_paths, 1):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("source_hashes") != sources:
                raise ValueError(f"Stale population archive: {path.name}")
            candidate_step = int(payload["step"])
            if candidate_step != int(path.stem.removeprefix("pool_step_")):
                raise ValueError(f"Pool archive step mismatch: {path.name}")
            models = _make_pool_models(payload["models"])
            member_metrics = evaluate_population(
                models, episodes_per_member=args.pool_eval_episodes, workers=args.workers,
                seed=seed, gamma=args.gamma,
            )
            candidate_steps.append(candidate_step)
            scores_by_step[candidate_step] = member_metrics["team_score"]
            metrics_by_step[candidate_step] = member_metrics
            print(json.dumps({"pool_snapshot_eval": candidate_index,
                              "candidate_count": len(candidate_paths), "step": candidate_step,
                              "team_score": member_metrics["team_score"].tolist(),
                              "seed": seed, "episodes_per_member": args.pool_eval_episodes},
                             allow_nan=False), flush=True)
        if final_pool_step not in scores_by_step:
            raise FileNotFoundError("Final population snapshot is missing")
        selected_steps, selection = _select_mid(scores_by_step, metrics_by_step, final_pool_step)
        selection["seed"] = seed
        selection["episodes_per_member"] = args.pool_eval_episodes
        selection["action_mode"] = "greedy"
        step_to_payload = {}
        for path in candidate_paths:
            step = int(path.stem.removeprefix("pool_step_"))
            if step in set(selected_steps):
                step_to_payload[step] = torch.load(path, map_location="cpu", weights_only=False)["models"]
        snapshots["mid"] = [step_to_payload[step][member]
                             for member, step in enumerate(selected_steps)]
        atomic_save(out / "pool_mid.pt", {
            "models": snapshots["mid"], "selected_steps": selected_steps,
            "selection": selection,
        })
        pool_artifact = out / "partner_pool.pt"
        atomic_save(pool_artifact, {
            "snapshots": snapshots, "population_size": args.population,
            "levels": ["initial", "mid", "final"],
            "population_env_steps": final_pool_step,
            "population_rl_env_steps": pool_rl_steps,
            "policy_action_steps": pool_trainer.agent_steps.tolist(),
            "selection": selection,
        })
        pool_hash = hashlib.sha256(pool_artifact.read_bytes()).hexdigest()
        config["pool_snapshot"] = {"file": "partner_pool.pt", "sha256": pool_hash,
                                   "levels": ["initial", "mid", "final"],
                                   "population_size": args.population}
        config["pool_selection"] = selection
        atomic_json(config_path, config)

        if stop["signal"] or timed_out():
            status("stopped_signal" if stop["signal"] else "stopped_time",
                   "population_selection", pool_rl_steps, 0)
            return 0

        pool_trainer.release_workspace()
        model = JointHiPTPolicy(z_dim=args.z_dim).to(device)
        best_score = -float("inf")
        best_eval = None
        initial_eval = None
        if args.mode == "bc_rl":
            initial_eval = evaluate(
                model, episodes=min(args.eval_episodes, 4),
                num_envs=min(args.num_envs, min(args.eval_episodes, 4)),
                workers=args.workers, seed=args.seed + 400_000, device=device,
                gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
            )
            initial_eval.update({"phase": "before_behavior_cloning", "seed": args.seed + 400_000,
                                 "demonstration_env_steps": demo_steps})
            atomic_json(out / "pre_bc_eval.json", initial_eval)
            atomic_save(out / "initial_uncloned.pt", {
                "method": "HiPT central-team", "model": runtime.cpu_tree(model.state_dict()),
                "z_dim": args.z_dim, "source_hashes": sources,
                "native_evaluation": initial_eval, "dataset_sha256": dataset_hash,
            })
            dataset = load_joint_dataset(args.demonstrations)
            hierarchical_bc = warm_start_hierarchical(
                model, dataset, device=device, seed=args.seed + 50_000,
                epochs=args.bc_epochs, episodes_per_batch=2,
            )
            after_bc = evaluate(
                model, episodes=min(args.eval_episodes, 4),
                num_envs=min(args.num_envs, min(args.eval_episodes, 4)),
                workers=args.workers, seed=args.seed + 400_000, device=device,
                gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
            )
            after_bc.update({"phase": "after_behavior_cloning", "seed": args.seed + 400_000,
                             "demonstration_env_steps": demo_steps})
            atomic_json(out / "bc_initial_eval.json", after_bc)
            atomic_save(out / "bc_initial.pt", {
                "method": "HiPT central-team", "model": runtime.cpu_tree(model.state_dict()),
                "z_dim": args.z_dim, "source_hashes": sources,
                "behavior_cloning_metrics": hierarchical_bc,
                "native_evaluations": {"before_behavior_cloning": initial_eval,
                                       "after_behavior_cloning": after_bc},
                "dataset_sha256": dataset_hash,
            })
            config["behavior_cloning"]["hierarchical"] = hierarchical_bc
            config["behavior_cloning"]["native_evaluations"] = {
                "population": "pool_bc_initial_eval.json",
                "hierarchical_before": "pre_bc_eval.json",
                "hierarchical_after": "bc_initial_eval.json",
            }
            atomic_json(config_path, config)
            best_eval, best_score = after_bc, float(after_bc["metric_means"]["team_score"])
        else:
            initial_eval = evaluate(
                model, episodes=min(args.eval_episodes, 4),
                num_envs=min(args.num_envs, min(args.eval_episodes, 4)),
                workers=args.workers, seed=args.seed + 400_000, device=device,
                gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
            )
            initial_eval.update({"phase": "initial_rl", "seed": args.seed + 400_000})
            atomic_json(out / "initial_eval.json", initial_eval)
            best_eval, best_score = initial_eval, float(initial_eval["metric_means"]["team_score"])

        if best_eval is not None:
            atomic_save(out / "best.pt", {
                "method": "HiPT central-team", "stage": "hierarchical",
                "model": runtime.cpu_tree(model.state_dict()),
                "best_team_score": best_score, "native_evaluation": best_eval,
                "source_hashes": sources, "config": config,
            })
        main_env = KitchenBatch(num_envs=args.num_envs, seed=args.seed + 100_000,
                                stage="standard", workers=args.workers, train=True,
                                gamma=args.gamma, reward_profile="legacy")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        main_trainer = HierarchicalTrainer(
            main_env, snapshots, num_envs=args.num_envs, population_size=args.population,
            z_dim=args.z_dim, device=device, seed=args.seed + 1, epochs=args.epochs,
            minibatches=args.minibatches, learning_rate=args.lr, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip=args.clip, value_coef=args.value_coef,
            low_entropy_coef=args.low_entropy_coef, high_entropy_coef=args.high_entropy_coef,
            max_grad_norm=args.max_grad_norm, p_min=args.p_min, p_max=args.p_max,
            selfplay_fraction=args.selfplay_fraction,
            target_env_steps=(total_updates - pool_updates) * batch_size,
            model=model, optimizer=optimizer, bptt_context=args.rollout,
        )
        planned_main_steps = (total_updates - pool_updates) * batch_size
        last_eval_steps = 0

        def save_main(latest=False):
            payload = {
                "method": "HiPT central-team", "stage": "hierarchical",
                "hierarchical": runtime.cpu_tree(main_trainer.state_dict()),
                "population_env_steps": pool_rl_steps,
                "hierarchical_env_steps": main_trainer.total_env_steps,
                "demonstration_env_steps": demo_steps,
                "total_env_steps": demo_steps + pool_rl_steps + main_trainer.total_env_steps,
                "best_team_score": best_score, "partner_pool": config["pool_snapshot"],
                "parameter_counts": config["parameter_counts"],
                "source_hashes": sources, "elapsed_seconds": time.monotonic() - start_time,
                "config": config,
            }
            if latest:
                atomic_save(out / "latest.pt", payload)
            return payload

        status("running", "hierarchical", pool_rl_steps, 0)
        try:
            for update_index in range(total_updates - pool_updates):
                if stop["signal"] or timed_out():
                    break
                progress = min(1., main_trainer.total_env_steps / max(1, planned_main_steps))
                influence = args.influence_coef + (args.influence_final - args.influence_coef) * progress
                metrics = main_trainer.collect_and_update(args.rollout, influence)
                main_rl_steps = main_trainer.total_env_steps
                metrics["total_env_steps"] = demo_steps + pool_rl_steps + main_rl_steps
                _append(out / "metrics.jsonl", metrics)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                checkpoint_due = (args.checkpoint_interval and
                                  main_rl_steps // args.checkpoint_interval >
                                  (main_rl_steps - batch_size) // args.checkpoint_interval)
                eval_due = (args.eval_interval and
                            main_rl_steps - last_eval_steps >= args.eval_interval)
                if checkpoint_due:
                    payload = save_main(latest=True)
                    step_name = demo_steps + pool_rl_steps + main_rl_steps
                    archive_path = out / f"checkpoint_step_{step_name}.pt"
                    if not archive_path.exists():
                        atomic_save(archive_path, payload)
                if eval_due or update_index == total_updates - pool_updates - 1:
                    result = evaluate(
                        model, episodes=args.eval_episodes,
                        num_envs=min(args.num_envs, args.eval_episodes),
                        workers=args.workers,
                        seed=args.seed + 200_000 + main_rl_steps,
                        device=device, gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
                    )
                    result.update({"phase": "periodic", "step": main_rl_steps,
                                   "seed": args.seed + 200_000 + main_rl_steps,
                                   "demonstration_env_steps": demo_steps})
                    atomic_json(out / f"eval_{main_rl_steps}.json", result)
                    score = float(result["metric_means"]["team_score"])
                    if score >= best_score:
                        best_score, best_eval = score, result
                        atomic_save(out / "best.pt", {
                            "method": "HiPT central-team", "stage": "hierarchical",
                            "model": runtime.cpu_tree(model.state_dict()),
                            "best_team_score": best_score, "native_evaluation": result,
                            "source_hashes": sources, "config": config,
                        })
                    last_eval_steps = main_rl_steps
                    status("running", "hierarchical", pool_rl_steps, main_rl_steps,
                           update=main_trainer.update, best_team_score=best_score)
                if stop["signal"] or timed_out():
                    break
        finally:
            main_trainer.close()
        save_main(latest=True)
        completed = main_rl_steps >= planned_main_steps
        status("complete" if completed else ("stopped_signal" if stop["signal"] else "stopped_time"),
               "complete" if completed else "hierarchical", pool_rl_steps, main_rl_steps,
               update=main_trainer.update, best_team_score=best_score)
        return 0
    except BaseException as error:
        status("failed", "unknown", pool_rl_steps, main_rl_steps, error=repr(error))
        raise
    finally:
        if main_trainer is not None:
            main_trainer.close()
        elif main_env is not None:
            main_env.close()
        if pool_trainer is not None:
            pool_trainer.close()
        elif pool_env is not None:
            pool_env.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())

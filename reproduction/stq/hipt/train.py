"""Train the HiPT partner population and SharedKitchen hierarchical policy.

Run from the SharedKitchen repository root with ``python -m hipt.train``. The
step budget counts one vector-environment transition once, regardless of the
number of chef actors. Training and evaluation share a single visible GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .behavior_cloning import HIERARCHICAL_BC_EPISODES_PER_BATCH, warm_start_hierarchical
from .model import (
    FlatActorCritic,
    HiPTPolicy,
    SPATIAL_CHANNELS,
    SPATIAL_HEIGHT,
    SPATIAL_WIDTH,
    VECTOR_DIM,
)
from .population import PopulationTrainer
from .trainer import HierarchicalTrainer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="hipt/results", help="Run directory (default: hipt/results)")
    p.add_argument("--steps", type=int, default=8_000_000,
                   help="Total vector-environment steps across both training stages")
    p.add_argument("--pool-steps", type=int, default=None,
                   help="Population-stage budget; default is one quarter of --steps")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=300)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--z-dim", type=int, choices=(4, 5, 6), default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="PPO fine-tuning learning rate after behavior cloning")
    p.add_argument("--gamma", type=float, default=0.9985)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--low-entropy-coef", type=float, default=0.01)
    p.add_argument("--high-entropy-coef", type=float, default=0.01)
    p.add_argument("--population-entropy-coef", type=float, default=0.01)
    p.add_argument("--jsd-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--p-min", type=int, default=20)
    p.add_argument("--p-max", type=int, default=40)
    p.add_argument("--influence-coef", type=float, default=1000.0)
    p.add_argument("--influence-final", type=float, default=1.0)
    p.add_argument("--selfplay-fraction", type=float, default=0.5)
    p.add_argument("--eval-interval", type=int, default=2_000_000)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--checkpoint-interval", type=int, default=250_000)
    p.add_argument("--pool-eval-episodes", type=int, default=4,
                   help="Standard-task episodes per member for matched-seed pool snapshot selection")
    p.add_argument("--demonstrations", required=True,
                   help="SharedKitchen demonstration dataset used to warm-start every actor")
    p.add_argument("--bc-epochs", type=int, default=64,
                   help="Behavior-cloning epochs for each population and hierarchical actor")
    p.add_argument("--max-seconds", type=float, default=158400.0,
                   help="Stop at the next update boundary after this 44-hour wall-clock budget")
    p.add_argument("--resume", default=None, help="Resume a checkpoint within the same run")
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny two-policy/two-update end-to-end protocol check")
    return p


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


def _snapshot_models(models):
    return [_cpu_tree(model.state_dict()) for model in models]


def _save_step_once(path: Path, payload, atomic_save):
    """Keep an update-boundary archive immutable while latest remains mutable."""
    if path.exists():
        return False
    atomic_save(path, payload)
    return True


def _select_mid_snapshots(team_scores_by_step, final_step, metrics_by_step):
    """Choose each member's pre-final snapshot nearest half its final team score."""
    final_scores = np.asarray(team_scores_by_step[final_step], dtype=np.float64)
    selected_steps, members = [], []
    for member, final_score in enumerate(final_scores):
        target = float(final_score / 2.0)
        candidates = [
            (step, float(scores[member]))
            for step, scores in team_scores_by_step.items() if step < final_step
        ]
        if not candidates:
            candidates = [(final_step, float(final_score))]
        errors = [(abs(score - target), step, score) for step, score in candidates]
        minimum = min(row[0] for row in errors)
        tied = [row for row in errors if abs(row[0] - minimum) <= 1e-12]
        error, selected_step, selected_score = max(tied, key=lambda row: row[1])
        quality_failed = bool(final_score <= 0.0)
        members.append({
            "member": member,
            "final_team_score": float(final_score),
            "target_half_team_score": target,
            "selected_step": int(selected_step),
            "selected_team_score": selected_score,
            "absolute_target_error": float(error),
            "quality": "qualityfailed" if quality_failed else "measured_positive_final",
            "quality_warning": (
                "final team_score <= 0; retain selected weights without claiming high skill"
                if quality_failed else None
            ),
            "selected_is_final": bool(selected_step == final_step),
            "candidate_team_scores": {
                str(step): float(scores[member])
                for step, scores in sorted(team_scores_by_step.items())
            },
        })
        selected_steps.append(int(selected_step))
    return selected_steps, {
        "metric": "standard unshaped team_score mean",
        "target": "final team_score / 2, including negative final scores",
        "tie_rule": "latest eligible step wins",
        "episodes_per_member": None,
        "candidate_metric_means": {
            str(step): {metric: values.tolist() for metric, values in metrics.items()}
            for step, metrics in sorted(metrics_by_step.items())
        },
        "members": members,
        "qualityfailed_members": [row["member"] for row in members
                                  if row["quality"] == "qualityfailed"],
    }


def _evaluate_pool_snapshot(models, args, device, seed, *, episodes_per_member=None):
    """Measure each flat actor's standard-task native metrics on matched episodes."""
    from baseline_common.envs import KitchenBatch
    from .evaluate import EPISODE_METRICS

    population = len(models)
    if population != args.population:
        raise ValueError("Pool snapshot policy count differs from the configured population")
    episodes = (args.pool_eval_episodes if episodes_per_member is None
                else int(episodes_per_member))
    if episodes <= 0:
        raise ValueError("episodes_per_member must be positive")
    num_envs = population * episodes
    training_modes = [model.training for model in models]
    for model in models:
        model.eval()
    env = KitchenBatch(num_envs=num_envs, seed=seed, stage="standard",
                       workers=args.workers, train=False, gamma=args.gamma)
    episode_metrics = {
        key: np.full(num_envs, np.nan, dtype=np.float64)
        for key in EPISODE_METRICS
    }
    captured = np.zeros(num_envs, dtype=bool)
    try:
        obs = env.reset()
        while not captured.all():
            actions = np.empty((num_envs, 3), dtype=np.int64)
            with torch.no_grad():
                for member, model in enumerate(models):
                    start = member * episodes
                    stop = start + episodes
                    spatial = torch.as_tensor(obs["spatial"][start:stop], device=device).reshape(
                        episodes * 3, SPATIAL_CHANNELS, SPATIAL_HEIGHT,
                        SPATIAL_WIDTH
                    )
                    vector = torch.as_tensor(obs["vector"][start:stop], device=device).reshape(
                        episodes * 3, VECTOR_DIM
                    )
                    logits, _ = model(spatial, vector)
                    actions[start:stop] = logits.argmax(-1).cpu().numpy().reshape(episodes, 3)
            next_obs, _, done, infos = env.step(actions)
            for index, (ended, info) in enumerate(zip(done, infos)):
                if ended and not captured[index]:
                    summary = info.get("episode_summary")
                    if summary is None:
                        raise RuntimeError("Pool evaluation ended without an episode summary")
                    for key in EPISODE_METRICS:
                        episode_metrics[key][index] = float(summary[key])
                    captured[index] = True
            obs = next_obs
    finally:
        env.close()
        for model, training in zip(models, training_modes):
            model.train(training)
    member_metrics = {
        key: values.reshape(population, episodes).mean(axis=1)
        for key, values in episode_metrics.items()
    }
    if not np.isfinite(member_metrics["team_score"]).all():
        raise FloatingPointError("Nonfinite standard team_score in population evaluation")
    return member_metrics


def _evaluate_pool_candidates(paths, args, device, sources, pool_steps_done):
    """Score saved pool steps with matched-seed greedy standard-task episodes."""
    from .evaluate import NATIVE_METRICS

    population = args.population
    episodes = args.pool_eval_episodes
    seed = args.seed + 700_000
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    models = [FlatActorCritic().to(device) for _ in range(population)]
    team_scores_by_step, metrics_by_step, path_by_step = {}, {}, {}
    ordered = sorted(paths, key=lambda path: int(path.stem.removeprefix("pool_step_")))

    try:
        for candidate_index, path in enumerate(ordered, 1):
            step = int(path.stem.removeprefix("pool_step_"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload.get("step", -1)) != step or payload.get("source_hashes") != sources:
                raise ValueError(f"Invalid or stale population archive: {path.name}")
            states = payload["models"]
            if len(states) != population:
                raise ValueError(f"Pool snapshot population mismatch: {path.name}")
            for model, state in zip(models, states):
                model.load_state_dict(state)
            del payload
            member_metrics = _evaluate_pool_snapshot(models, args, device, seed)
            scores = member_metrics["team_score"]
            team_scores_by_step[step] = scores
            metrics_by_step[step] = member_metrics
            path_by_step[step] = path
            print(json.dumps({"pool_snapshot_eval": candidate_index,
                              "candidate_count": len(ordered), "step": step,
                              "standard_team_score_mean": scores.tolist(),
                              "standard_metrics_mean": {
                                  key: member_metrics[key].tolist() for key in NATIVE_METRICS
                              }, "seed": seed,
                              "episodes_per_member": episodes}, allow_nan=False), flush=True)

        final_step = int(pool_steps_done)
        if final_step not in team_scores_by_step:
            raise FileNotFoundError("Final population step was not archived for score selection")
        selected_steps, selection = _select_mid_snapshots(
            team_scores_by_step, final_step, metrics_by_step
        )
        selection["seed"] = seed
        selection["episodes_per_member"] = episodes
        selection["action_mode"] = "greedy"
        mid_states = [None] * population
        for step in sorted(set(selected_steps)):
            payload = torch.load(path_by_step[step], map_location="cpu", weights_only=False)
            for member, selected_step in enumerate(selected_steps):
                if selected_step == step:
                    mid_states[member] = payload["models"][member]
            del payload
        if any(state is None for state in mid_states):
            raise RuntimeError("Failed to assemble all selected middle snapshots")
        selection["overall_quality"] = (
            "qualityfailed" if selection["qualityfailed_members"] else "measured_positive_final"
        )
        return mid_states, selection
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _append_metrics(path: Path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def _configure_args(args):
    demonstration_env_steps = int(getattr(args, "demonstration_env_steps", 9600))
    if demonstration_env_steps < 0:
        raise ValueError("demonstration environment steps cannot be negative")
    args.demonstration_env_steps = demonstration_env_steps
    if args.smoke:
        args.population = 2
        args.num_envs = min(args.num_envs, 2)
        args.rollout = max(41, min(args.rollout, 48))
        args.workers = 0
        args.eval_episodes = min(args.eval_episodes, args.num_envs)
        args.eval_interval = 0
        args.checkpoint_interval = 1
        args.pool_eval_episodes = min(2, args.num_envs)
        args.epochs = min(args.epochs, 1)
        args.steps = demonstration_env_steps + args.num_envs * args.rollout * 2
        args.pool_steps = demonstration_env_steps + args.num_envs * args.rollout
    elif args.pool_steps is None:
        args.pool_steps = args.steps // 4
    if (args.steps <= demonstration_env_steps or args.pool_steps is None
            or args.pool_steps <= demonstration_env_steps):
        raise ValueError("steps and pool-steps must be positive")
    if args.pool_steps >= args.steps:
        raise ValueError("Population training must leave at least one rollout for HiPT")
    if args.num_envs <= 0 or args.rollout <= 0 or args.workers < 0:
        raise ValueError("num-envs/rollout must be positive and workers nonnegative")
    if args.population < 2:
        raise ValueError("Population training needs at least two policies")
    if not 1 <= args.p_min <= args.p_max <= 40 or args.p_min < 20:
        raise ValueError("HiPT option durations must satisfy 20 <= p-min <= p-max <= 40")
    if args.epochs <= 0 or args.minibatches <= 0:
        raise ValueError("epochs and minibatches must be positive")
    if args.bc_epochs <= 0:
        raise ValueError("bc-epochs must be positive")
    if args.lr <= 0 or not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("lr must be positive and gamma/gae-lambda must be in (0, 1]")
    if not 0 <= args.selfplay_fraction <= 1:
        raise ValueError("selfplay-fraction must be in [0, 1]")
    if not 0 < args.max_seconds <= 158400:
        raise ValueError("max-seconds must be positive and at most the authorized 44-hour budget")
    if (args.eval_interval < 0 or args.checkpoint_interval < 0 or args.eval_episodes <= 0
            or args.pool_eval_episodes <= 0):
        raise ValueError("evaluation/checkpoint intervals must be nonnegative and episodes positive")
    batch_size = args.num_envs * args.rollout
    if args.smoke:
        # Exercise immutable step archives with exactly one update per stage.
        args.checkpoint_interval = batch_size
    total_updates = (args.steps - demonstration_env_steps) // batch_size
    if total_updates < 2:
        raise ValueError("Budget must include at least one update for each training stage")
    pool_updates = (args.pool_steps - demonstration_env_steps) // batch_size
    if pool_updates < 1:
        raise ValueError("Population budget must contain at least one complete rollout after BC")
    pool_updates = min(total_updates - 1, pool_updates)
    return total_updates, pool_updates


def _serializable_args(args):
    return {key: value for key, value in vars(args).items()}


def main(argv=None):
    args = parser().parse_args(argv)
    from baseline_common.demonstrations import demonstration_steps
    args.demonstration_env_steps = int(demonstration_steps(args.demonstrations))
    total_updates, pool_updates = _configure_args(args)
    from baseline_common.envs import KitchenBatch
    from baseline_common.runtime import atomic_json, atomic_save, setup_device, source_hashes

    device = setup_device(args.device, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and Path(args.resume).resolve().parent != out.resolve():
        raise ValueError("--resume must point to a checkpoint inside the same --output run directory")
    config_path = out / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError("Run directory already has config.json; pass --resume to continue it")
    sources = source_hashes("hipt")
    demonstrations_sha256 = hashlib.sha256(
        Path(args.demonstrations).read_bytes()
    ).hexdigest()
    protocol = {
        "task": "three-chef SharedKitchen at the standard stage; final evaluation uses the shared HiPT actor for all three chefs",
        "native_episode_metrics": [
            "team_score", "meals_completed", "normal_completed", "joint_completed",
            "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
            "three_contributor_meals", "resource_conflicts", "movement_conflicts",
            "ticks", "meals_per_100_steps",
        ],
        "episode_summary_fields": ["episode", "training_reward", "training_shaping"],
        "best_checkpoint_metric": "unshaped standard evaluation mean team_score",
        "behavior_cloning": "population actors use observation batches; the hierarchical actor uses complete episode sequences with LSTM state carried through time before its PPO optimizer is constructed; one dataset transition budget is counted once",
        "hierarchical_behavior_cloning": "full-length episode unrolls, two episodes/six chef streams per batch; low-level teacher actions supervise every z head, while the unsupervised high-level z head is left for RL",
        "population": "Three paper skill levels are retained: randomly initialized policies as low-skill partners, saved snapshots nearest half each final team score, and fully trained policies. Behavior cloning initializes the active learner population but does not replace the frozen random initial snapshot.",
        "skill_levels": "The partner population uses random initialization, approximately half-final-score historical policies, and fully trained policies, following arXiv:2305.16708 section 4.1.",
        "hierarchical": "shared current HiPT actor; 50% self-play and otherwise two current actors plus one fixed historical population actor; fixed partner role rotates over episodes; all three roles are training targets",
        "macro_reward": "option reward is the observed-duration average of environment reward plus influence_coef * KL(current low policy || high-weighted all-z action mixture); high GAE gamma/lambda are per completed option as in the paper",
        "recurrent_training": "low PPO replays the rollout sequence with detached initial LSTM state; high PPO delays each transition until natural option expiry or true terminal and replays up to one rollout of episode context with gradients through the LSTM",
        "optimization": "low and high PPO objectives are summed before each shared-parameter optimizer step; both entropy coefficients start at 0.01 and decay linearly over the hierarchical budget",
        "pool_selection": "the low-skill partner is the randomly initialized pre-BC snapshot; each mid partner is the saved 250k-step snapshot nearest half its final standard team_score; the high-skill partner is the fully trained final policy. Nonpositive final team scores are qualityfailed and are not claimed high-skill",
        "best_checkpoint_ties": "latest evaluated checkpoint wins equal team_score ties",
        "rollout_boundary": "an unfinished option keeps its z and timer across rollout updates; it is not shortened or resampled; its original log-probability/value and reward accumulator are retained, so the high PPO ratio may span low-policy updates and is clipped",
        "evaluation": "all three chefs use the learned hierarchical policy in standard, unshaped SharedKitchen; no population partners are used",
        "step_accounting": "one vector environment transition counts once, including the demonstration dataset's declared environment steps; actor count and chef slots do not multiply this budget; evaluation wall time counts toward max-seconds",
    }
    config = {
        "method": "HiPT",
        "paper": "Loo et al., IJCAI 2023, arXiv:2305.16708",
        "arguments": _serializable_args(args),
        "demonstrations": {
            "path": args.demonstrations,
            "sha256": demonstrations_sha256,
            "environment_steps": args.demonstration_env_steps,
            "epochs": args.bc_epochs,
            "population_batch_size": 512,
            "hierarchical_episode_batch_size": HIERARCHICAL_BC_EPISODES_PER_BATCH,
            "hierarchical_chef_stream_batch_size": 3 * HIERARCHICAL_BC_EPISODES_PER_BATCH,
            "hierarchical_sequence_mode": "complete episodes; recurrent state carried over the full sequence",
            "transition_counted_once_across_actors": True,
        },
        "protocol": protocol,
        "source_hashes": sources,
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "planned_updates": {"total": total_updates, "population": pool_updates,
                            "hierarchical": total_updates - pool_updates},
        "step_budget": {
            "requested_total_env_steps": args.steps,
            "requested_population_total_env_steps": args.pool_steps,
            "demonstration_env_steps": args.demonstration_env_steps,
            "planned_population_rl_steps": pool_updates * args.num_envs * args.rollout,
            "planned_hierarchical_rl_steps": (
                total_updates - pool_updates
            ) * args.num_envs * args.rollout,
            "planned_total_env_steps": (
                args.demonstration_env_steps + total_updates * args.num_envs * args.rollout
            ),
        },
        "behavior_cloning": {
            "population": [], "hierarchical": None, "native_evaluations": {},
        },
        "step_batch": args.num_envs * args.rollout,
        "parameter_counts": {
            "population_policy": FlatActorCritic().parameter_counts(),
            "hipt_policy": HiPTPolicy(args.z_dim).parameter_counts(),
        },
        "pool_snapshot": None,
    }
    if config_path.exists() and args.resume:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config["source_hashes"] != sources:
            raise ValueError("Checkpoint sources differ from the run manifest")
        if saved_config.get("demonstrations") != config["demonstrations"]:
            raise ValueError("Resume demonstration dataset or behavior-cloning settings differ")
        for key in ("population", "num_envs", "rollout", "z_dim", "p_min", "p_max", "gamma"):
            if saved_config["arguments"][key] != config["arguments"][key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        config = saved_config
    else:
        atomic_json(config_path, config)
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("source_hashes") != sources:
            raise ValueError("Checkpoint source hashes differ from the current source")

    stop = {"signal": None}
    def request_stop(signum, _frame):
        stop["signal"] = int(signum)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    elapsed_before_resume = float(resume.get("elapsed_seconds", 0.0)) if resume else 0.0
    start_time = time.monotonic() - elapsed_before_resume

    def timed_out():
        return bool(args.max_seconds > 0 and time.monotonic() - start_time >= args.max_seconds)

    def status(state, stage, pool_steps_done, main_steps_done, **extra):
        payload = {
            "state": state,
            "stage": stage,
            "population_env_steps": int(pool_steps_done),
            "hierarchical_env_steps": int(main_steps_done),
            "demonstration_env_steps": int(args.demonstration_env_steps),
            "total_env_steps": int(
                args.demonstration_env_steps + pool_steps_done + main_steps_done
            ),
            "wall_seconds": float(time.monotonic() - start_time),
            "pid": os.getpid(),
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
        }
        payload.update(extra)
        atomic_json(out / "status.json", payload)

    def archive_pool_step(step, models):
        path = out / f"pool_step_{int(step)}.pt"
        _save_step_once(path, {"models": _snapshot_models(models), "step": int(step),
                               "source_hashes": sources}, atomic_save)

    pool_trainer = None
    pool_snapshots = None
    pool_steps_done = 0
    main_steps_done = 0
    pool_start_update = 0
    main_start_update = 0

    if resume and resume.get("stage") == "hierarchical":
        pool_artifact = torch.load(out / "partner_pool.pt", map_location="cpu", weights_only=False)
        pool_snapshots = pool_artifact["snapshots"]
        main_state = resume["hierarchical"]
        main_steps_done = int(main_state["total_env_steps"])
        main_start_update = int(main_state["update"])
        pool_steps_done = int(resume["population_env_steps"])
    else:
        status("running", "population", pool_steps_done, 0)
        try:
            pool_env = KitchenBatch(num_envs=args.num_envs, seed=args.seed, stage="standard",
                                    workers=args.workers, train=True, gamma=args.gamma)
        except BaseException as error:
            status("failed", "population", pool_steps_done, 0, error=repr(error))
            raise
        try:
            pool_trainer = PopulationTrainer(
                pool_env, num_envs=args.num_envs, population_size=args.population,
                device=device, seed=args.seed, epochs=args.epochs, minibatches=args.minibatches,
                learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip=args.clip, value_coef=args.value_coef,
                entropy_coef=args.population_entropy_coef, jsd_coef=args.jsd_coef,
                max_grad_norm=args.max_grad_norm,
                demonstrations_path=args.demonstrations,
                bc_epochs=args.bc_epochs,
                initialize_from_demonstrations=not bool(
                    resume and resume.get("stage") == "population"
                ),
            )
        except BaseException as error:
            pool_env.close()
            status("failed", "population", pool_steps_done, 0, error=repr(error))
            raise
        snapshots = {}
        if pool_trainer.behavior_cloning_metrics:
            config["behavior_cloning"]["population"] = pool_trainer.behavior_cloning_metrics
        if resume and resume.get("stage") == "population":
            pool_trainer.load_state_dict(resume["population"])
            pool_steps_done = pool_trainer.total_env_steps
            pool_start_update = pool_trainer.update
            snapshot_paths = {
                "initial": out / "pool_initial_policy.pt",
                "mid": out / "pool_mid.pt",
            }
            for level, path in snapshot_paths.items():
                if path.exists():
                    snapshots[level] = torch.load(path, map_location="cpu", weights_only=False)["models"]
            if "initial" not in snapshots:
                raise FileNotFoundError("Resume needs pool_initial_policy.pt from the same run directory")
        else:
            snapshots["initial"] = _cpu_tree(pool_trainer.uncloned_initial_models)
            atomic_save(out / "pool_initial_policy.pt", {
                "models": snapshots["initial"],
                "step": args.demonstration_env_steps,
            })
            atomic_save(out / "pool_initial.pt", {
                "stage": "population", "population": pool_trainer.state_dict(),
                "snapshot_phase": "after_behavior_cloning",
                "demonstration_env_steps": args.demonstration_env_steps,
                "total_env_steps": args.demonstration_env_steps,
                "source_hashes": sources, "config": config,
            })
        if pool_trainer.behavior_cloning_metrics:
            pool_bc_episodes = min(args.eval_episodes, 4)
            pool_bc_seed = args.seed + 700_000
            pool_bc_metrics = _evaluate_pool_snapshot(
                pool_trainer.models, args, device, pool_bc_seed,
                episodes_per_member=pool_bc_episodes,
            )
            pool_bc_evaluation = {
                "episodes_per_member": pool_bc_episodes,
                "seed": pool_bc_seed,
                "action_mode": "greedy",
                "metric_means_by_member": {
                    key: values.tolist() for key, values in pool_bc_metrics.items()
                },
                "metric_means_across_members": {
                    key: float(values.mean()) for key, values in pool_bc_metrics.items()
                },
            }
            config["behavior_cloning"]["native_evaluations"]["population"] = {
                "file": "pool_bc_initial_eval.json",
                "episodes_per_member": pool_bc_episodes,
                "seed": pool_bc_seed,
                "selection_metric": "native standard team_score and meals_completed",
            }
            atomic_json(out / "pool_bc_initial_eval.json", pool_bc_evaluation)
            atomic_json(config_path, config)
        status("running", "population", pool_steps_done, 0)
        try:
            for update_index in range(pool_start_update, pool_updates):
                if stop["signal"] or timed_out():
                    break
                metrics = pool_trainer.collect_and_update(args.rollout)
                pool_steps_done = pool_trainer.total_env_steps
                metrics["total_env_steps"] = args.demonstration_env_steps + pool_steps_done
                _append_metrics(out / "metrics.jsonl", metrics)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                if args.checkpoint_interval and pool_steps_done % args.checkpoint_interval < \
                        args.num_envs * args.rollout:
                    archive_pool_step(
                        args.demonstration_env_steps + pool_steps_done, pool_trainer.models
                    )
                    atomic_save(out / "pool_latest.pt", {
                        "stage": "population", "population": pool_trainer.state_dict(),
                        "pool_snapshots": sorted(snapshots), "source_hashes": sources,
                        "population_env_steps": pool_steps_done,
                        "demonstration_env_steps": args.demonstration_env_steps,
                        "total_env_steps": args.demonstration_env_steps + pool_steps_done,
                        "elapsed_seconds": time.monotonic() - start_time, "config": config,
                    })
                status("running", "population", pool_steps_done, 0,
                       policy_action_steps=pool_trainer.agent_steps.tolist(), update=pool_trainer.update)
        except BaseException as error:
            status("failed", "population", pool_steps_done, 0, error=repr(error))
            raise
        finally:
            pool_trainer.close()
        atomic_save(out / "pool_latest.pt", {
            "stage": "population", "population": pool_trainer.state_dict(),
            "population_env_steps": pool_steps_done, "source_hashes": sources,
            "demonstration_env_steps": args.demonstration_env_steps,
            "total_env_steps": args.demonstration_env_steps + pool_steps_done,
            "elapsed_seconds": time.monotonic() - start_time, "config": config,
        })
        if pool_steps_done < pool_updates * args.num_envs * args.rollout:
            archive_pool_step(
                args.demonstration_env_steps + pool_steps_done, pool_trainer.models
            )
            state = "stopped_signal" if stop["signal"] else "stopped_time"
            status(state, "population", pool_steps_done, 0,
                   policy_action_steps=pool_trainer.agent_steps.tolist())
            return
        pool_total_steps = args.demonstration_env_steps + pool_steps_done
        archive_pool_step(pool_total_steps, pool_trainer.models)
        snapshots["final"] = _snapshot_models(pool_trainer.models)
        atomic_save(out / "pool_final.pt", {
            "models": snapshots["final"], "step": pool_total_steps,
        })
        candidate_paths = []
        for path in out.glob("pool_step_*.pt"):
            try:
                step = int(path.stem.removeprefix("pool_step_"))
            except ValueError:
                continue
            if step <= pool_total_steps:
                candidate_paths.append(path)
        if not candidate_paths:
            raise FileNotFoundError("No immutable pool step snapshots are available for selection")
        status("selecting", "population", pool_steps_done, 0,
               candidate_snapshots=len(candidate_paths), eval_episodes=args.pool_eval_episodes)
        snapshots["mid"], pool_selection = _evaluate_pool_candidates(
            candidate_paths, args, device, sources, pool_total_steps
        )
        atomic_save(out / "pool_mid.pt", {
            "models": snapshots["mid"],
            "selected_steps": [row["selected_step"] for row in pool_selection["members"]],
            "selection": pool_selection,
        })
        pool_artifact_path = out / "partner_pool.pt"
        atomic_save(pool_artifact_path, {
            "snapshots": snapshots,
            "population_size": args.population,
            "levels": ["initial", "mid", "final"],
            "population_env_steps": pool_total_steps,
            "population_rl_env_steps": pool_steps_done,
            "policy_action_steps": pool_trainer.agent_steps.tolist(),
            "selection": pool_selection,
        })
        pool_hash = hashlib.sha256(pool_artifact_path.read_bytes()).hexdigest()
        config["pool_snapshot"] = {"file": "partner_pool.pt", "sha256": pool_hash,
                                   "levels": ["initial", "mid", "final"],
                                   "population_size": args.population}
        config["pool_selection"] = pool_selection
        atomic_json(config_path, config)

    if pool_snapshots is None:
        pool_snapshots = snapshots
    else:
        pool_artifact_path = out / "partner_pool.pt"
        pool_hash = hashlib.sha256(pool_artifact_path.read_bytes()).hexdigest()
        if config.get("pool_snapshot", {}).get("sha256") != pool_hash:
            raise ValueError("Partner population artifact hash does not match config.json")

    main_env = None
    try:
        model = HiPTPolicy(z_dim=args.z_dim).to(device)
        hierarchical_bc_metrics = None
        if not (resume and resume.get("stage") == "hierarchical"):
            from baseline_common.demonstrations import warm_start
            from .evaluate import evaluate

            bc_eval_episodes = min(args.eval_episodes, 4)
            bc_eval_envs = min(args.num_envs, bc_eval_episodes)
            bc_eval_seed = args.seed + 400_000
            before_bc_evaluation = evaluate(
                model, episodes=bc_eval_episodes, num_envs=bc_eval_envs,
                workers=args.workers, seed=bc_eval_seed, device=device,
                gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
            )
            before_bc_evaluation.update({
                "phase": "before_behavior_cloning",
                "seed": bc_eval_seed,
                "demonstration_env_steps": args.demonstration_env_steps,
            })
            atomic_json(out / "pre_bc_eval.json", before_bc_evaluation)
            atomic_save(out / "initial_uncloned.pt", {
                "method": "HiPT",
                "stage": "hierarchical_initial_uncloned",
                "model": _cpu_tree(model.state_dict()),
                "z_dim": args.z_dim,
                "source_hashes": sources,
                "demonstrations": config["demonstrations"],
                "native_evaluation": before_bc_evaluation,
            })

            hierarchical_bc_metrics = warm_start_hierarchical(
                model,
                dataset_path=args.demonstrations,
                device=device,
                seed=args.seed + 50_000,
                epochs=args.bc_epochs,
            )
            after_bc_evaluation = evaluate(
                model, episodes=bc_eval_episodes, num_envs=bc_eval_envs,
                workers=args.workers, seed=bc_eval_seed, device=device,
                gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
            )
            after_bc_evaluation.update({
                "phase": "after_behavior_cloning",
                "seed": bc_eval_seed,
                "demonstration_env_steps": args.demonstration_env_steps,
            })
            atomic_json(out / "bc_initial_eval.json", after_bc_evaluation)
            config["behavior_cloning"]["hierarchical"] = hierarchical_bc_metrics
            config["behavior_cloning"]["native_evaluations"]["hierarchical"] = {
                "episodes": bc_eval_episodes,
                "seed": bc_eval_seed,
                "selection_metric": "mean team_score",
                "before_file": "pre_bc_eval.json",
                "after_file": "bc_initial_eval.json",
            }
            atomic_json(config_path, config)
            atomic_save(out / "bc_initial.pt", {
                "method": "HiPT",
                "stage": "hierarchical_initial_behavior_cloned",
                "model": _cpu_tree(model.state_dict()),
                "z_dim": args.z_dim,
                "source_hashes": sources,
                "demonstrations": config["demonstrations"],
                "behavior_cloning_metrics": hierarchical_bc_metrics,
                "native_evaluations": {
                    "before_behavior_cloning": before_bc_evaluation,
                    "after_behavior_cloning": after_bc_evaluation,
                },
            })
        main_env = KitchenBatch(num_envs=args.num_envs, seed=args.seed + 100_000,
                                stage="standard", workers=args.workers,
                                train=True, gamma=args.gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        main_trainer = HierarchicalTrainer(
            main_env, pool_snapshots, num_envs=args.num_envs,
            population_size=args.population, z_dim=args.z_dim, device=device,
            seed=args.seed + 1, epochs=args.epochs, minibatches=args.minibatches,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip=args.clip, value_coef=args.value_coef,
            low_entropy_coef=args.low_entropy_coef, high_entropy_coef=args.high_entropy_coef,
            max_grad_norm=args.max_grad_norm, p_min=args.p_min, p_max=args.p_max,
            selfplay_fraction=args.selfplay_fraction,
            target_env_steps=(total_updates - pool_updates) * args.num_envs * args.rollout,
            model=model, optimizer=optimizer,
            initial_update=main_start_update, bptt_context=args.rollout,
        )
    except BaseException as error:
        if main_env is not None:
            main_env.close()
        status("failed", "hierarchical", pool_steps_done, main_steps_done, error=repr(error))
        raise
    if resume and resume.get("stage") == "hierarchical":
        main_trainer.load_state_dict(resume["hierarchical"])
        main_steps_done = main_trainer.total_env_steps
    else:
        main_steps_done = 0
        main_start_update = 0

    if resume and resume.get("stage") == "hierarchical":
        best_team_score = float(resume.get("best_team_score", -float("inf")))
    else:
        best_team_score = float(after_bc_evaluation["metric_means"]["team_score"])
    last_eval_steps = main_steps_done
    def checkpoint_payload(stage="hierarchical"):
        return {
            "method": "HiPT",
            "stage": stage,
            "hierarchical": main_trainer.state_dict(),
            "population_env_steps": pool_steps_done,
            "hierarchical_env_steps": main_trainer.total_env_steps,
            "demonstration_env_steps": args.demonstration_env_steps,
            "total_env_steps": (
                args.demonstration_env_steps + pool_steps_done + main_trainer.total_env_steps
            ),
            "best_team_score": best_team_score,
            "partner_population": config["pool_snapshot"],
            "pool_snapshot": pool_snapshots,
            "parameter_counts": config["parameter_counts"],
            "source_hashes": sources,
            "elapsed_seconds": time.monotonic() - start_time,
            "config": config,
        }

    def archive_main_step():
        step = int(args.demonstration_env_steps + pool_steps_done + main_trainer.total_env_steps)
        if step > 0:
            _save_step_once(out / f"step_{step}.pt", checkpoint_payload(), atomic_save)

    if not resume or resume.get("stage") != "hierarchical":
        atomic_save(out / "initial.pt", checkpoint_payload())
        atomic_save(out / "best.pt", checkpoint_payload())
    status("running", "hierarchical", pool_steps_done, main_steps_done,
           selfplay_fraction=args.selfplay_fraction, z_dim=args.z_dim)
    main_updates = total_updates - pool_updates
    try:
        for update_index in range(main_start_update, main_updates):
            if stop["signal"] or timed_out():
                break
            progress = main_trainer.total_env_steps / max(
                1, main_updates * args.num_envs * args.rollout
            )
            influence_coef = args.influence_coef + progress * (
                args.influence_final - args.influence_coef
            )
            metrics = main_trainer.collect_and_update(args.rollout, influence_coef)
            main_steps_done = main_trainer.total_env_steps
            metrics["total_env_steps"] = (
                args.demonstration_env_steps + pool_steps_done + main_steps_done
            )
            _append_metrics(out / "metrics.jsonl", metrics)
            print(json.dumps(metrics, allow_nan=False), flush=True)
            status("running", "hierarchical", pool_steps_done, main_steps_done,
                   update=main_trainer.update, influence_coef=influence_coef)

            if args.eval_interval and main_steps_done - last_eval_steps >= args.eval_interval:
                from .evaluate import evaluate
                evaluation = evaluate(
                    main_trainer.model, episodes=args.eval_episodes,
                    num_envs=min(args.num_envs, max(1, args.eval_episodes)),
                    workers=args.workers, seed=args.seed + 200_000 + main_steps_done,
                    device=device, gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
                )
                evaluation["steps"] = main_steps_done
                evaluation["demonstration_env_steps"] = args.demonstration_env_steps
                evaluation["population_env_steps"] = pool_steps_done
                evaluation["total_env_steps"] = (
                    args.demonstration_env_steps + pool_steps_done + main_steps_done
                )
                atomic_json(out / f"eval_{main_steps_done}.json", evaluation)
                last_eval_steps = main_steps_done
                team_score = evaluation["metric_means"]["team_score"]
                if team_score >= best_team_score:
                    best_team_score = team_score
                    atomic_save(out / "best.pt", checkpoint_payload())
                status("running", "hierarchical", pool_steps_done, main_steps_done,
                       update=main_trainer.update, best_team_score=best_team_score)
            if args.checkpoint_interval and main_steps_done - \
                    (main_steps_done // args.checkpoint_interval) * args.checkpoint_interval < \
                    args.num_envs * args.rollout:
                archive_main_step()
                atomic_save(out / "latest.pt", checkpoint_payload())

        if stop["signal"] or timed_out() or main_steps_done < main_updates * args.num_envs * args.rollout:
            archive_main_step()
            atomic_save(out / "latest.pt", checkpoint_payload())
            state = "stopped_signal" if stop["signal"] else "stopped_time"
            status(state, "hierarchical", pool_steps_done, main_steps_done,
                   update=main_trainer.update, pending_macro_options=sum(
                       item is not None for row in main_trainer.pending for item in row
                   ))
            return

        from .evaluate import evaluate
        evaluation = evaluate(
            main_trainer.model, episodes=args.eval_episodes,
            num_envs=min(args.num_envs, max(1, args.eval_episodes)),
            workers=args.workers, seed=args.seed + 300_000 + main_steps_done,
            device=device, gamma=args.gamma, p_min=args.p_min, p_max=args.p_max,
        )
        evaluation["steps"] = main_steps_done
        evaluation["demonstration_env_steps"] = args.demonstration_env_steps
        evaluation["population_env_steps"] = pool_steps_done
        evaluation["total_env_steps"] = (
            args.demonstration_env_steps + pool_steps_done + main_steps_done
        )
        atomic_json(out / f"eval_{main_steps_done}_final.json", evaluation)
        team_score = evaluation["metric_means"]["team_score"]
        if team_score >= best_team_score:
            best_team_score = team_score
            atomic_save(out / "best.pt", checkpoint_payload())
        archive_main_step()
        atomic_save(out / "latest.pt", checkpoint_payload())
        atomic_save(out / "final.pt", checkpoint_payload())
        status("completed", "complete", pool_steps_done, main_steps_done,
               update=main_trainer.update, final_team_score=team_score,
               best_team_score=best_team_score)
    except BaseException as error:
        status("failed", "hierarchical", pool_steps_done, main_steps_done,
               error=repr(error))
        raise
    finally:
        main_trainer.close()


if __name__ == "__main__":
    main()

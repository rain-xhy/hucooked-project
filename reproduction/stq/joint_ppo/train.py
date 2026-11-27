"""Train a standard centralized three-agent PPO policy on SharedKitchen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from torch.distributions import Categorical

from .algorithm import compute_gae, joint_log_prob, ppo_update
from .models import HEAD_SHAPE, JointPolicy, build_policy

DEFAULT_STEPS = 20_000_000
DEFAULT_NUM_ENVS = 16
DEFAULT_ROLLOUT = 300
DEFAULT_WORKERS = 2
DEFAULT_MAX_SECONDS = 0
CHECKPOINT_INTERVAL = 1_000_000
EVALUATION_INTERVAL = 100_000
EVALUATION_EPISODES = 12
EVALUATION_SEED = 120_001
BC_BATCH_SIZE = 512
BC_EPOCHS = 64
PPO_EPOCHS = 2
PPO_MINIBATCHES = 32
PPO_GAMMA = 0.9985
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_LEARNING_RATE = 1e-4
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
REWARD_PROFILE = "ppo_task"
CHECKPOINT_FORMAT = "joint_ppo.policy_bundle.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rl", "bc_rl"), default="rl")
    parser.add_argument("--backbone", choices=("mlp", "cnn", "transformer"), default="mlp")
    parser.add_argument("--output", required=True, help="new per-task results directory")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-size", choices=("small", "large", "million"), default="small")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--demonstrations", default=None)
    parser.add_argument("--reward-profile", choices=("legacy", "ppo_task"), default=REWARD_PROFILE)
    parser.add_argument("--bc-epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--minibatches", type=int, default=PPO_MINIBATCHES)
    parser.add_argument("--lr", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--eval-interval", type=int, default=EVALUATION_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--rollout", type=int, default=DEFAULT_ROLLOUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="run at most two small PPO updates")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "bc_rl" and not args.demonstrations:
        raise ValueError("--demonstrations is required for --mode bc_rl")
    if args.steps <= 0 or args.max_seconds < 0:
        raise ValueError("steps must be positive and max-seconds cannot be negative")
    if args.model_size not in ("small", "large", "million"):
        raise ValueError("model-size must be small, large, or million")
    if args.num_envs <= 0 or args.rollout <= 0 or args.workers < 0:
        raise ValueError("num-envs and rollout must be positive; workers cannot be negative")
    if args.bc_epochs <= 0 or args.epochs <= 0 or args.minibatches <= 0:
        raise ValueError("BC/PPO epochs and minibatches must be positive")
    if args.lr <= 0.0 or not np.isfinite(args.lr):
        raise ValueError("lr must be finite and positive")
    if args.checkpoint_interval <= 0 or args.eval_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("checkpoint/eval intervals and eval-episodes must be positive")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"invalid device: {args.device}") from error
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index not in (None, 0)):
        raise ValueError("device must be cpu or the single exposed cuda:0")
    if args.memory_fraction is None:
        args.memory_fraction = 0.035 if args.model_size in ("large", "million") else 0.008
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in (0, 1]")


def derive_run_config(args: argparse.Namespace, demo_env_steps: int = 0) -> dict[str, Any]:
    """Resolve full-rollout budgets, counting BC demonstration steps once."""
    validate_args(args)
    envs = int(args.num_envs)
    rollout = int(args.rollout)
    workers = int(args.workers)
    requested_total = int(args.steps)
    demo_steps = int(demo_env_steps) if args.mode == "bc_rl" else 0
    if demo_steps < 0:
        raise ValueError("demonstration environment steps cannot be negative")

    if args.smoke:
        envs = min(envs, 2)
        rollout = min(rollout, 8)
        workers = 0
        smoke_budget = demo_steps + 2 * envs * rollout
        requested_total = min(requested_total, smoke_budget)

    rollout_batch_steps = envs * rollout
    rl_budget = requested_total - demo_steps
    if rl_budget < envs or rl_budget % envs:
        raise ValueError(
            "RL environment-step budget must include at least one vector step and be divisible by num-envs"
        )
    num_updates = (rl_budget + rollout_batch_steps - 1) // rollout_batch_steps
    last_rollout_steps = (rl_budget - (num_updates - 1) * rollout_batch_steps) // envs
    effective_rl_steps = rl_budget
    effective_total = demo_steps + effective_rl_steps

    return {
        "mode": args.mode,
        "backbone": args.backbone,
        "model_size": args.model_size,
        "output": str(Path(args.output).resolve()),
        "requested_total_env_steps": requested_total,
        "effective_total_env_steps": effective_total,
        "demo_env_steps": demo_steps,
        "requested_rl_env_steps": rl_budget,
        "effective_rl_env_steps": effective_rl_steps,
        "rollout_batch_env_steps": rollout_batch_steps,
        "num_updates": num_updates,
        "last_rollout_vector_steps": int(last_rollout_steps),
        "num_envs": envs,
        "rollout": rollout,
        "workers": workers,
        "device": str(torch.device(args.device)),
        "seed": int(args.seed),
        "eval_seed": EVALUATION_SEED,
        "evaluation_episodes": int(args.eval_episodes),
        "physical_gpu": int(args.physical_gpu),
        "memory_fraction": float(args.memory_fraction),
        "max_seconds": int(args.max_seconds),
        "smoke": bool(args.smoke),
        "reward_profile": args.reward_profile,
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_LAMBDA,
        "clip": PPO_CLIP,
        "learning_rate": float(args.lr),
        "epochs": int(args.epochs),
        "minibatches": int(args.minibatches),
        "checkpoint_interval": int(args.checkpoint_interval),
        "evaluation_interval": int(args.eval_interval),
        "entropy_coef": PPO_ENTROPY_COEF,
        "value_coef": PPO_VALUE_COEF,
        "max_grad_norm": PPO_MAX_GRAD_NORM,
        "bc_epochs": int(args.bc_epochs) if args.mode == "bc_rl" else 0,
        "bc_batch_size_labels": BC_BATCH_SIZE if args.mode == "bc_rl" else 0,
        "head_shape": list(HEAD_SHAPE),
    }


class StepThresholdSchedule:
    """Yield each crossed step threshold once, at a completed update boundary."""

    def __init__(self, interval: int, *, first_threshold: int | None = None) -> None:
        if interval <= 0:
            raise ValueError("threshold interval must be positive")
        self.interval = int(interval)
        self.next_threshold = int(first_threshold or interval)

    def due(self, actual_env_steps: int) -> list[int]:
        thresholds: list[int] = []
        while self.next_threshold <= int(actual_env_steps):
            thresholds.append(self.next_threshold)
            self.next_threshold += self.interval
        return thresholds


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_dependencies() -> SimpleNamespace:
    from joint_common import demonstrations, envs, runtime

    return SimpleNamespace(demonstrations=demonstrations, envs=envs, runtime=runtime)


def _dataset_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_demo_dataset(data: dict[str, Any]) -> int:
    required = {"state", "spatial", "actions", "validation", "episode_lengths", "env_steps", "dataset_sha256"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"joint demonstration dataset lacks fields: {', '.join(sorted(missing))}")
    states = np.asarray(data["state"])
    spatial = np.asarray(data["spatial"])
    actions = np.asarray(data["actions"])
    validation = np.asarray(data["validation"])
    if states.ndim != 2 or states.shape[1] != 863:
        raise ValueError("joint demonstrations must have state shape [frames, 863]")
    frames = len(states)
    if spatial.shape != (frames, 26, 7, 7):
        raise ValueError("joint demonstrations must have canonical spatial shape [frames, 26, 7, 7]")
    if actions.shape != (frames, 3) or validation.shape != (frames,):
        raise ValueError("joint demonstration actions/splits must have shapes [frames, 3]/[frames]")
    if not np.isfinite(states).all() or not np.isfinite(spatial).all():
        raise FloatingPointError("joint demonstrations contain non-finite observations")
    if np.any(actions < 0) or np.any(actions >= 6):
        raise ValueError("joint demonstration actions must be in [0, 5]")
    steps = int(data["env_steps"])
    if steps <= 0:
        raise ValueError("joint demonstrations must contain positive training environment steps")
    return steps


def _json_safe(value: Any) -> Any:
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


def _append_event(output: Path, event: dict[str, Any]) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(), **_json_safe(event)}
    encoded = json.dumps(row, sort_keys=True, allow_nan=False)
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    print(encoded, flush=True)


def _checkpoint_payload(
    policy: JointPolicy,
    *,
    config: dict[str, Any],
    counts: dict[str, int],
    env_steps: int,
    rl_env_steps: int,
    reason: str,
    selection: dict[str, Any] | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "backbone": policy.backbone,
        "model_size": policy.model_size,
        "policy_state_dict": {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
        "parameter_counts": counts,
        "head_shape": list(HEAD_SHAPE),
        "mode": config["mode"],
        "env_steps": int(env_steps),
        "rl_env_steps": int(rl_env_steps),
        "demo_env_steps": int(config["demo_env_steps"]),
        "checkpoint_reason": reason,
        "config": config,
        "selection": selection,
    }
    if actor_optimizer is not None and critic_optimizer is not None:
        payload["optimizers"] = {
            "actor": actor_optimizer.state_dict(),
            "critic": critic_optimizer.state_dict(),
        }
    return payload


def _save_checkpoint(
    runtime: Any,
    path: Path,
    policy: JointPolicy,
    *,
    config: dict[str, Any],
    counts: dict[str, int],
    env_steps: int,
    rl_env_steps: int,
    reason: str,
    selection: dict[str, Any] | None = None,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    preserve_existing: bool = False,
) -> bool:
    if preserve_existing and path.exists():
        return False
    payload = _checkpoint_payload(
        policy,
        config=config,
        counts=counts,
        env_steps=env_steps,
        rl_env_steps=rl_env_steps,
        reason=reason,
        selection=selection,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    runtime.atomic_save(path, payload)
    return True


def _extract_native_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "episodes", "mean_team_score", "std_team_score", "mean_meals_completed",
        "mean_joint_completed", "mean_joint_attempt_success_rate", "mean_joint_wait_steps",
        "episode_metrics", "episode_summaries", "seed",
    )
    result = {key: evaluation[key] for key in allowed if key in evaluation}
    required = (
        "mean_team_score", "mean_meals_completed", "mean_joint_completed",
        "mean_joint_attempt_success_rate", "mean_joint_wait_steps",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"native evaluator lacks required metrics: {', '.join(missing)}")
    return result


def _is_better_evaluation(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    candidate_score = float(candidate["mean_team_score"])
    incumbent_score = float(incumbent["mean_team_score"])
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    return float(candidate.get("mean_meals_completed", 0.0)) > float(
        incumbent.get("mean_meals_completed", 0.0)
    )


def _episode_summaries(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(info["episode_summary"]) for info in infos if "episode_summary" in info]


def collect_rollout(
    env: Any,
    observation: dict[str, np.ndarray],
    policy: JointPolicy,
    *,
    rollout_steps: int,
    device: torch.device | str,
    gamma: float,
    gae_lambda: float,
    team_inputs: Callable[[dict[str, np.ndarray], torch.device | str], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Collect one on-policy rollout; actor and critic each see E team rows."""
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    spatial_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    log_prob_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    training_reward_sum = 0.0

    policy.eval()
    for _ in range(rollout_steps):
        spatial, state = team_inputs(observation, device)
        spatial = spatial.to(dtype=torch.float32)
        state = state.to(dtype=torch.float32)
        with torch.no_grad():
            logits = policy.actor(spatial, state)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = joint_log_prob(logits, actions)
            values = policy.critic(state)
        action_array = actions.cpu().numpy().astype(np.int64, copy=False)
        next_observation, rewards, dones, infos = env.step(action_array)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones_array = np.asarray(dones, dtype=bool)
        if rewards_array.shape != (spatial.shape[0],) or dones_array.shape != rewards_array.shape:
            raise ValueError("environment must return team rewards/dones with shape [num_envs]")
        spatial_rows.append(spatial.detach().cpu().numpy().copy())
        state_rows.append(state.detach().cpu().numpy().copy())
        action_rows.append(action_array.copy())
        log_prob_rows.append(log_probs.cpu().numpy().astype(np.float32, copy=False))
        value_rows.append(values.cpu().numpy().astype(np.float32, copy=False))
        reward_rows.append(rewards_array)
        done_rows.append(dones_array)
        training_reward_sum += float(rewards_array.sum())
        summaries.extend(_episode_summaries(infos))
        observation = next_observation

    final_spatial, final_state = team_inputs(observation, device)
    with torch.no_grad():
        bootstrap_values = policy.critic(final_state.to(dtype=torch.float32)).cpu().numpy()
    done_array = np.asarray(done_rows, dtype=bool)
    if done_array[-1].any():
        bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32).copy()
        bootstrap_values[done_array[-1]] = 0.0

    rewards_array = np.asarray(reward_rows, dtype=np.float32)
    values_array = np.asarray(value_rows, dtype=np.float32)
    advantages, returns = compute_gae(
        rewards_array,
        values_array,
        done_array,
        bootstrap_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    batch = {
        "spatial": np.asarray(spatial_rows, dtype=np.float32).reshape(-1, 26, 7, 7),
        "state": np.asarray(state_rows, dtype=np.float32).reshape(-1, 863),
        "actions": np.asarray(action_rows, dtype=np.int64).reshape(-1, 3),
        "old_log_probs": np.asarray(log_prob_rows, dtype=np.float32).reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
    }
    info = {
        "env_steps": int(rollout_steps * len(rewards_array[0])),
        "team_reward_sum": training_reward_sum,
        "team_reward_mean": training_reward_sum / max(1, rollout_steps * len(rewards_array[0])),
        "episode_summaries": summaries,
    }
    return batch, observation, info


def _default_evaluator(
    actor: torch.nn.Module,
    *,
    env_factory: Callable[..., Any],
    num_episodes: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    from .evaluate import evaluate_policy

    return evaluate_policy(
        actor,
        env_factory=env_factory,
        num_episodes=num_episodes,
        seed=seed,
        device=device,
    )


def train_experiment(
    args: argparse.Namespace,
    *,
    dependencies: Any | None = None,
    device_override: torch.device | str | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one isolated experiment. Dependencies are injectable for CPU smoke tests."""
    validate_args(args)
    dependencies = _load_dependencies() if dependencies is None else dependencies
    runtime = dependencies.runtime
    demo_data = None
    demo_steps = 0
    dataset_sha = None
    if args.mode == "bc_rl":
        demo_data = dependencies.demonstrations.load_joint_dataset(args.demonstrations)
        demo_steps = _validate_demo_dataset(demo_data)
        dataset_sha = str(demo_data["dataset_sha256"])
        if dataset_sha != _dataset_digest(args.demonstrations):
            raise ValueError("demonstration dataset hash changed while loading")

    config = derive_run_config(args, demo_steps)
    output = Path(config["output"])
    if device_override is None:
        device = runtime.setup_device(
            device=args.device,
            seed=int(args.seed),
            physical_gpu=int(args.physical_gpu),
            memory_fraction=float(args.memory_fraction),
        )
    else:
        _seed_everything(int(args.seed))
        device = torch.device(device_override)
        torch.set_num_threads(1)
    runtime.prepare_output(output)
    rng = np.random.default_rng(int(args.seed))

    policy = build_policy(args.backbone, args.model_size).to(device)
    counts = policy.parameter_counts()
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=float(args.lr), eps=1e-5)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=float(args.lr), eps=1e-5)
    source_hashes = runtime.source_hashes("joint_ppo")
    reward_specs = getattr(dependencies.envs, "REWARD_PROFILES", {})
    config.update({
        "actual_device": str(device),
        "model_size": args.model_size,
        "actor_parameters": counts["actor"],
        "critic_parameters": counts["critic"],
        "policy_bundle_parameters": counts["total"],
        "source_hashes": source_hashes,
        "reward_profile_spec": reward_specs.get(config["reward_profile"]),
        "demonstrations": (
            {"path": str(Path(args.demonstrations).resolve()), "dataset_sha256": dataset_sha}
            if args.mode == "bc_rl" else None
        ),
    })
    runtime.atomic_json(output / "config.json", _json_safe(config))
    start_time = time.monotonic()

    def write_status(status: str, *, env_steps: int, rl_env_steps: int, **details: Any) -> None:
        runtime.atomic_json(output / "status.json", _json_safe({
            "state": status,
            "pid": os.getpid(),
            "physical_gpu": int(args.physical_gpu),
            "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'),
            "mode": args.mode,
            "backbone": args.backbone,
            "env_steps": int(env_steps),
            "rl_env_steps": int(rl_env_steps),
            "effective_total_env_steps": config["effective_total_env_steps"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        }))

    def make_env(*, train: bool, num_envs: int, seed: int, workers: int) -> Any:
        return dependencies.envs.KitchenBatch(
            num_envs=num_envs,
            seed=seed,
            stage="standard",
            workers=workers,
            train=train,
            gamma=PPO_GAMMA,
            reward_profile=config["reward_profile"],
        )

    def run_evaluation(
        step: int, reason: str, *, threshold_env_steps: int | None = None
    ) -> dict[str, Any]:
        evaluate = evaluator or _default_evaluator
        raw = evaluate(
            policy.actor,
            env_factory=lambda **overrides: make_env(
                train=False,
                num_envs=int(args.eval_episodes),
                seed=EVALUATION_SEED,
                workers=0,
            ),
            num_episodes=int(args.eval_episodes),
            seed=EVALUATION_SEED,
            device=device,
        )
        result = _extract_native_metrics(dict(raw))
        event = {"event": "evaluation", "env_steps": int(step), "reason": reason, **result}
        if threshold_env_steps is not None:
            event["threshold_env_steps"] = int(threshold_env_steps)
        _append_event(output, event)
        return result

    def save(path: Path, step: int, rl_steps: int, reason: str,
             selection: dict[str, Any] | None = None, *, optimizers: bool = False,
             preserve: bool = False) -> bool:
        return _save_checkpoint(
            runtime, path, policy,
            config=config,
            counts=counts,
            env_steps=step,
            rl_env_steps=rl_steps,
            reason=reason,
            selection=selection,
            actor_optimizer=actor_optimizer if optimizers else None,
            critic_optimizer=critic_optimizer if optimizers else None,
            preserve_existing=preserve,
        )

    env = None
    try:
        write_status("running", env_steps=0, rl_env_steps=0)
        save(output / "initial.pt", 0, 0, "random initialization")
        best_evaluation = run_evaluation(0, "initial random")
        best_selection = {"metric": "mean_team_score", **best_evaluation}
        save(output / "best.pt", 0, 0, "initial random evaluation", best_selection)
        _append_event(output, {
            "event": "initial_evaluation", "env_steps": 0, "candidate": "initial",
            **best_evaluation,
        })

        actual_total_steps = int(config["demo_env_steps"])
        rl_env_steps = 0
        if args.mode == "bc_rl":
            # Only this branch loads the expert dataset or performs BC.
            bc_result = dependencies.demonstrations.warm_start_joint(
                policy.actor,
                lambda spatial, state: policy.actor(spatial, state),
                dataset_path=args.demonstrations,
                device=device,
                seed=int(args.seed),
                epochs=int(args.bc_epochs),
                batch_size=BC_BATCH_SIZE,
            )
            if int(bc_result["demo_env_steps"]) != demo_steps:
                raise ValueError("BC dataset step count changed between load and warm start")
            if bc_result.get("dataset_sha256") != dataset_sha:
                raise ValueError("BC dataset hash changed between load and warm start")
            bc_result = _json_safe(dict(bc_result))
            bc_evaluation = run_evaluation(actual_total_steps, "post-BC candidate")
            save(
                output / "bc_initial.pt",
                actual_total_steps,
                0,
                "post-BC initialization",
                {
                    "metric": "mean_team_score",
                    "native_evaluation": bc_evaluation,
                    "behavior_cloning": bc_result,
                },
            )
            _append_event(output, {
                "event": "behavior_cloning", "env_steps": actual_total_steps,
                "candidate": "bc_initial", **bc_result,
                "native_evaluation": bc_evaluation,
            })
            if _is_better_evaluation(bc_evaluation, best_evaluation):
                best_evaluation = bc_evaluation
                best_selection = {"metric": "mean_team_score", **bc_evaluation}
                save(output / "best.pt", actual_total_steps, 0, "post-BC evaluation", best_selection)

        env = make_env(
            train=True,
            num_envs=int(config["num_envs"]),
            seed=int(args.seed),
            workers=int(config["workers"]),
        )
        observation = env.reset()
        checkpoint_schedule = StepThresholdSchedule(int(args.checkpoint_interval))
        evaluation_schedule = StepThresholdSchedule(int(args.eval_interval))
        status = "completed"
        for update_index in range(1, int(config["num_updates"]) + 1):
            if args.max_seconds > 0 and time.monotonic() - start_time >= float(args.max_seconds):
                status = "time_limit"
                break
            remaining_rl_steps = int(config["effective_rl_env_steps"]) - rl_env_steps
            rollout_steps = min(
                int(config["rollout"]),
                remaining_rl_steps // int(config["num_envs"]),
            )
            if rollout_steps <= 0:
                raise RuntimeError("effective step budget ended before its scheduled PPO update")
            batch, observation, rollout_info = collect_rollout(
                env,
                observation,
                policy,
                rollout_steps=rollout_steps,
                device=device,
                gamma=PPO_GAMMA,
                gae_lambda=PPO_LAMBDA,
                team_inputs=runtime.team_inputs,
            )
            update_metrics = ppo_update(
                policy,
                actor_optimizer,
                critic_optimizer,
                batch,
                device=device,
                rng=rng,
                epochs=int(args.epochs),
                minibatches=int(args.minibatches),
                clip=PPO_CLIP,
                entropy_coef=PPO_ENTROPY_COEF,
                value_coef=PPO_VALUE_COEF,
                max_grad_norm=PPO_MAX_GRAD_NORM,
            )
            rl_env_steps += int(rollout_info["env_steps"])
            actual_total_steps += int(rollout_info["env_steps"])
            completed_training_rewards = [
                float(summary["training_reward"])
                for summary in rollout_info["episode_summaries"]
                if "training_reward" in summary
            ]
            completed_native_scores = [
                float(summary["team_score"])
                for summary in rollout_info["episode_summaries"]
                if "team_score" in summary
            ]
            event = {
                "event": "update",
                "update": update_index,
                "env_steps": actual_total_steps,
                "rl_env_steps": rl_env_steps,
                "training_reward_sum": rollout_info["team_reward_sum"],
                "training_reward_mean": rollout_info["team_reward_mean"],
                "completed_episode_count": len(rollout_info["episode_summaries"]),
                "completed_episode_training_reward_mean": (
                    float(np.mean(completed_training_rewards)) if completed_training_rewards else None
                ),
                "completed_episode_native_team_score_mean": (
                    float(np.mean(completed_native_scores)) if completed_native_scores else None
                ),
                "episode_summaries": rollout_info["episode_summaries"],
                **update_metrics,
            }
            _append_event(output, event)
            save(output / "latest.pt", actual_total_steps, rl_env_steps,
                 "latest completed PPO update", optimizers=True)

            for threshold in checkpoint_schedule.due(actual_total_steps):
                archive = output / f"step_{threshold}.pt"
                saved = save(archive, actual_total_steps, rl_env_steps,
                             f"checkpoint threshold {threshold}", preserve=True)
                _append_event(output, {
                    "event": "checkpoint", "threshold_env_steps": threshold,
                    "actual_env_steps": actual_total_steps, "path": str(archive), "saved": saved,
                })

            for threshold in evaluation_schedule.due(actual_total_steps):
                evaluated = run_evaluation(
                    actual_total_steps,
                    "evaluation threshold",
                    threshold_env_steps=threshold,
                )
                if _is_better_evaluation(evaluated, best_evaluation):
                    best_evaluation = evaluated
                    best_selection = {"metric": "mean_team_score", **evaluated}
                    save(output / "best.pt", actual_total_steps, rl_env_steps,
                         "best native evaluation", best_selection)

            write_status(
                "running",
                env_steps=actual_total_steps,
                rl_env_steps=rl_env_steps,
                update=update_index,
                elapsed_seconds=time.monotonic() - start_time,
            )

        final_evaluation = run_evaluation(actual_total_steps, 'final policy')
        if _is_better_evaluation(final_evaluation, best_evaluation):
            best_evaluation = final_evaluation
            best_selection = {"metric": "mean_team_score", **best_evaluation}
            save(output / 'best.pt', actual_total_steps, rl_env_steps,
                 'best native evaluation', best_selection)
        if status == "completed" and actual_total_steps != int(config["effective_total_env_steps"]):
            raise RuntimeError(
                f"completed step mismatch: {actual_total_steps} != {config['effective_total_env_steps']}"
            )
        save(output / "final.pt", actual_total_steps, rl_env_steps, status,
             {"metric": "mean_team_score", **final_evaluation}, optimizers=True)
        write_status(
            status,
            env_steps=actual_total_steps,
            rl_env_steps=rl_env_steps,
            best_evaluation=best_evaluation,
            final_evaluation=final_evaluation,
            elapsed_seconds=time.monotonic() - start_time,
        )
        _append_event(output, {
            "event": "finished", "status": status,
            "env_steps": actual_total_steps,
            "rl_env_steps": rl_env_steps,
            "elapsed_seconds": time.monotonic() - start_time,
            "best_evaluation": best_evaluation,
            "final_evaluation": final_evaluation,
        })
        return {
            "status": status,
            "output": str(output),
            "env_steps": actual_total_steps,
            "rl_env_steps": rl_env_steps,
            "best_evaluation": best_evaluation,
            "final_evaluation": final_evaluation,
        }
    except BaseException as error:
        write_status(
            "failed",
            env_steps=int(config["demo_env_steps"]) + rl_env_steps if "rl_env_steps" in locals() else 0,
            rl_env_steps=rl_env_steps if "rl_env_steps" in locals() else 0,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        if env is not None:
            env.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_experiment(args)
    print(json.dumps(_json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

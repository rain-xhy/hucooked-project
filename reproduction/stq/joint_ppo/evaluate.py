"""Native-score evaluation for a joint PPO policy bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .models import HEAD_SHAPE, JointPolicy, build_policy
from .train import CHECKPOINT_FORMAT, _load_dependencies, _json_safe


def load_policy_checkpoint(path: str | Path, device: torch.device | str) -> tuple[JointPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint; expected a joint_ppo policy bundle")
    backbone = checkpoint.get("backbone")
    if backbone not in ("mlp", "cnn", "transformer"):
        raise ValueError("checkpoint contains an unsupported backbone")
    model_size = checkpoint.get(
        "model_size", checkpoint.get("config", {}).get("model_size", "small")
    )
    if model_size not in ("small", "large", "million"):
        raise ValueError("checkpoint contains an unsupported model size")
    if checkpoint.get("head_shape") != list(HEAD_SHAPE):
        raise ValueError("checkpoint action head must have shape [3, 6]")
    policy = build_policy(backbone, model_size)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.to(device)
    policy.eval()
    return policy, checkpoint


def evaluate_policy(
    actor: torch.nn.Module,
    *,
    env_factory: Callable[..., Any],
    num_episodes: int = 12,
    seed: int = 120_001,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run deterministic centralized action selection from one actor pass per team state."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    from joint_common.runtime import team_inputs

    env = env_factory(
        num_envs=num_episodes,
        seed=seed,
        workers=0,
        train=False,
    )
    was_training = actor.training
    actor.eval()
    summaries: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    try:
        observation = env.reset()
        while len(summaries) < num_episodes:
            spatial, state = team_inputs(observation, device)
            with torch.no_grad():
                logits = actor(spatial.to(torch.float32), state.to(torch.float32))
                actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int64, copy=False)
            observation, _, _, infos = env.step(actions)
            for info in infos:
                if "episode_summary" in info:
                    metrics = info.get("metrics")
                    if not isinstance(metrics, dict):
                        raise ValueError("native evaluation info is missing its metrics mapping")
                    required = (
                        "team_score", "meals_completed", "joint_completed",
                        "joint_attempt_success_rate", "joint_wait_steps",
                    )
                    missing = [key for key in required if key not in metrics]
                    if missing:
                        raise ValueError(
                            f"native evaluation metrics lack required fields: {', '.join(missing)}"
                        )
                    metric_records.append(dict(metrics))
                    summaries.append(dict(info["episode_summary"]))
                    if len(summaries) == num_episodes:
                        break
    finally:
        env.close()
        actor.train(was_training)

    if len(metric_records) != num_episodes:
        raise RuntimeError(f"evaluation completed {len(metric_records)} of {num_episodes} requested episodes")
    scores = np.asarray([float(row["team_score"]) for row in metric_records], dtype=np.float64)
    meals = np.asarray([float(row["meals_completed"]) for row in metric_records], dtype=np.float64)
    joint_meals = np.asarray([float(row["joint_completed"]) for row in metric_records], dtype=np.float64)
    joint_success = np.asarray(
        [float(row["joint_attempt_success_rate"]) for row in metric_records], dtype=np.float64
    )
    joint_wait = np.asarray([float(row["joint_wait_steps"]) for row in metric_records], dtype=np.float64)
    return {
        "episodes": len(summaries),
        "seed": int(seed),
        "mean_team_score": float(scores.mean()),
        "std_team_score": float(scores.std(ddof=0)),
        "mean_meals_completed": float(meals.mean()),
        "mean_joint_completed": float(joint_meals.mean()),
        "mean_joint_attempt_success_rate": float(joint_success.mean()),
        "mean_joint_wait_steps": float(joint_wait.mean()),
        "episode_metrics": metric_records,
        "episode_summaries": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.008)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=120_001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dependencies = _load_dependencies()
    device = dependencies.runtime.setup_device(
        device=args.device,
        seed=args.seed,
        physical_gpu=args.physical_gpu,
        memory_fraction=args.memory_fraction,
    )
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    result = evaluate_policy(
        policy.actor,
        env_factory=lambda **kwargs: dependencies.envs.KitchenBatch(
            num_envs=kwargs["num_envs"],
            seed=kwargs["seed"],
            stage="standard",
            workers=kwargs.get("workers", 0),
            train=False,
            gamma=float(checkpoint["config"]["gamma"]),
            reward_profile=str(checkpoint["config"]["reward_profile"]),
        ),
        num_episodes=args.episodes,
        seed=args.seed,
        device=device,
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backbone": checkpoint["backbone"],
        "model_size": checkpoint.get("model_size", checkpoint.get("config", {}).get("model_size", "small")),
        "device": str(device),
        "physical_gpu": int(args.physical_gpu),
        "checkpoint_env_steps": checkpoint["env_steps"],
        **result,
    }
    encoded = json.dumps(_json_safe(result), sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

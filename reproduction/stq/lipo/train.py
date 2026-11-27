"""Population PPO trainer for LIPO on the three-chef KitchenBatch interface."""

from __future__ import annotations

import argparse
import hashlib
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from baseline_common.runtime import atomic_json, atomic_save, setup_device, source_hashes

from .algorithm import (
    assert_finite_tensor,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    role_for_xp,
    select_max_compatible,
)
from .models import (
    LipoMember,
    ModelConfig,
    SHARED_KITCHEN_COMMIT,
    SHARED_KITCHEN_ENVIRONMENT,
)


PAPER_MAPPING = {
    "environment": f"{SHARED_KITCHEN_ENVIRONMENT} commit {SHARED_KITCHEN_COMMIT}; vector observation 866, centralized state 863; spatial 3x26x7x7 validated and not consumed by this method.",
    "policy": "Shared per-member MAPPO actor; 2x64 ELU MLP, local obs + role one-hot + independent episode z8 -> action logits.",
    "sp_objective": "PPO/GAE maximizes self-play shared team reward; one centralized state critic per member.",
    "xp_objective": "PPO/GAE reverses the policy-gradient sign on the sampled highest-return XP partner.",
    "xp_critics": "One independent centralized state critic per possible partner, updated on its XP rollout.",
    "latent_mi": "Independent categorical z8 per chef and episode; q(z|obs, role one-hot, action_probs) CE trains q and actor.",
    "population": "Each XP rollout has one focal A policy in a rotating role and frozen B policy in the other two roles.",
    "xp_selection": "Approximate max-compatible selection over a shuffled, no-replacement candidate cycle; recorded per update.",
    "warm_start": "Every actor is initialized with behavior cloning; random independent z8 and the role bits in the native vector are the only policy inputs added.",
    "warmup": "XP policy-gradient and MI coefficients ramp linearly from zero to their configured values over the first 10% of RL environment steps.",
    "reference": "https://github.com/51616/marl-lipo (IncompatMAPPOZ, IncompatTrainer, default.yaml, lipo.sh)",
}

BEST_CHECKPOINT_POLICY = {
    "criterion": "highest mean_team_score on standard SharedKitchen evaluation",
    "tie_rule": "on an exact score tie, keep the later iteration",
}

SHARED_KITCHEN_METRICS = (
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
EPISODE_SUMMARY_FIELDS = SHARED_KITCHEN_METRICS + (
    "episode",
    "training_reward",
    "training_shaping",
)
CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_INTERVAL_STEPS = 250_000

CHECKPOINT_ALIGNMENT = (
    "step thresholds are checked after complete population updates; the archive is named "
    "with the actual total_env_steps at the first update boundary meeting or exceeding a threshold"
)


class CheckpointSchedule:
    """Report step thresholds crossed at completed population update boundaries."""

    def __init__(self, interval_steps: int):
        if isinstance(interval_steps, bool) or not isinstance(interval_steps, int) or interval_steps <= 0:
            raise ValueError("checkpoint interval must be a positive integer")
        self.interval_steps = interval_steps
        self.next_threshold = interval_steps

    def crossed(self, total_env_steps: int) -> tuple[int, ...]:
        if isinstance(total_env_steps, bool) or not isinstance(total_env_steps, int) or total_env_steps < 0:
            raise ValueError("total_env_steps must be a nonnegative integer")
        thresholds = []
        while self.next_threshold <= total_env_steps:
            thresholds.append(self.next_threshold)
            self.next_threshold += self.interval_steps
        return tuple(thresholds)


def _save_archive_if_absent(path: Path, save: Callable[[Path], None]) -> bool:
    """Use the trainer's atomic saver for a new archive, never replacing an old one."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path)
    return True


def _warmup_scale(rl_steps: int, rl_budget: int, fraction: float = 0.10) -> float:
    """Linearly ramp XP and MI coefficients over the first share of RL steps."""
    if rl_steps < 0 or rl_budget <= 0 or not 0 < fraction <= 1:
        raise ValueError("warmup needs nonnegative RL steps, positive budget, and fraction in (0,1]")
    return min(1.0, rl_steps / max(1.0, rl_budget * fraction))


@dataclass
class Rollout:
    obs: np.ndarray
    latent_ids: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_action_probs: np.ndarray
    states: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    role: int | None = None
    partner: int | None = None
    mean_reward: float = 0.0


class PartnerCycle:
    """Sample a configurable candidate subset while covering every partner."""

    def __init__(self, population: int, candidates: int, seed: int):
        self.population = population
        self.candidates = candidates
        self.rng = np.random.default_rng(seed)
        self.queues: list[deque[int]] = [deque() for _ in range(population)]

    def sample(self, member_id: int) -> list[int]:
        result: list[int] = []
        while len(result) < self.candidates:
            if not self.queues[member_id]:
                partners = [j for j in range(self.population) if j != member_id]
                self.rng.shuffle(partners)
                self.queues[member_id].extend(partners)
            result.append(self.queues[member_id].popleft())
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "queues": [list(q) for q in self.queues],
            "rng_state": self.rng.bit_generator.state,
        }


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    """Write a new JSON artifact and fail if the destination already exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(_json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _behavior_cloning_forward_logits(member: LipoMember, seed: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the BC callback with only native role bits and random episode z."""
    latent_rng = torch.Generator(device="cpu").manual_seed(seed)

    def forward_logits(spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if spatial.ndim != 4 or spatial.shape[1:] != (26, 7, 7):
            raise ValueError(f"Expected demonstration spatial input [N,26,7,7], got {tuple(spatial.shape)}")
        if vector.ndim != 2 or vector.shape[1] != 866 or vector.shape[0] != spatial.shape[0]:
            raise ValueError(f"Expected demonstration vector input [N,866], got {tuple(vector.shape)}")
        role_ids = vector[:, -3:].argmax(dim=-1)
        latent_ids = torch.randint(
            member.config.latent_dim,
            (vector.shape[0],),
            generator=latent_rng,
            device="cpu",
        ).to(vector.device)
        return member.actor_logits(vector, role_ids, latent_ids)

    return forward_logits


class LipoTrainer:
    def __init__(self, args: argparse.Namespace, device: torch.device, output: Path):
        self.args = args
        self.device = device
        self.output = output
        self.args.output = output
        self.metrics_path = output / "metrics.jsonl"
        self.status_path = output / "status.json"
        self.config = ModelConfig(population=args.population)
        self.members = [LipoMember(self.config).to(device) for _ in range(args.population)]
        exemplar = self.members[0]
        self.parameter_counts = {
            "per_member": sum(parameter.numel() for parameter in exemplar.parameters()),
            "population": sum(parameter.numel() for member in self.members for parameter in member.parameters()),
            "actor_per_member": sum(parameter.numel() for parameter in exemplar.actor.parameters()),
            "sp_critic_per_member": sum(parameter.numel() for parameter in exemplar.sp_critic.parameters()),
            "xp_critics_per_member": sum(parameter.numel() for parameter in exemplar.xp_critics.parameters()),
            "discriminator_per_member": sum(parameter.numel() for parameter in exemplar.discriminator.parameters()),
        }
        self.optimizers: list[dict[str, Any]] = []
        for member in self.members:
            self.optimizers.append({
                "actor": torch.optim.Adam(member.actor.parameters(), lr=args.lr, eps=1e-5),
                "sp_critic": torch.optim.Adam(member.sp_critic.parameters(), lr=args.lr, eps=1e-5),
                "xp_critics": [
                    torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
                    for critic in member.xp_critics
                ],
                "discriminator": torch.optim.Adam(member.discriminator.parameters(), lr=args.lr, eps=1e-5),
            })
        self.partner_cycle = PartnerCycle(args.population, args.xp_candidates, args.seed + 811)
        self.latent_rng = np.random.default_rng(args.seed + 1777)
        self.total_env_steps = 0
        self.sp_env_steps = 0
        self.xp_env_steps = 0
        self.sp_steps_by_member = [0] * args.population
        self.xp_steps_by_member = [0] * args.population
        self.iteration = 0
        self.demonstration_env_steps = 0
        self.xp_coef = args.xp_coef
        self.mi_coef = args.mi_coef
        self.best_score = -float("inf")
        self.best_iteration: int | None = None
        self.last_eval: dict[str, Any] | None = None
        self.current_stage: str | None = None
        self.checkpoint_schedule = CheckpointSchedule(args.checkpoint_interval)
        self.saved_step_checkpoints: set[int] = set()
        self.started_at = time.monotonic()
        self.source_hashes = source_hashes(Path(__file__).resolve().parent)
        if self.args.demonstrations.is_file():
            self.source_hashes[f"demonstrations:{self.args.demonstrations}"] = hashlib.sha256(
                self.args.demonstrations.read_bytes()).hexdigest()
        self.static_config = self._static_config()
        self.output.mkdir(parents=True, exist_ok=True)

    def _static_config(self) -> dict[str, Any]:
        return _json_value(vars(self.args))

    def _run_manifest(self) -> dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": "LIPO",
            "environment": SHARED_KITCHEN_ENVIRONMENT,
            "environment_commit": SHARED_KITCHEN_COMMIT,
            "curriculum": "fixed standard task",
            "args": self.static_config,
            "model_config": asdict(self.config),
            "parameter_counts": self.parameter_counts,
            "paper_mapping": PAPER_MAPPING,
            "source_hashes": self.source_hashes,
            "best_checkpoint_policy": BEST_CHECKPOINT_POLICY,
            "checkpoint_policy": {
                "interval_steps": self.args.checkpoint_interval,
                "steps_include_demonstrations": True,
                "demonstration_steps_counted_once": "BC initialization across population members does not multiply environment transitions",
                "alignment": CHECKPOINT_ALIGNMENT,
                "step_snapshots": f"save step_<actual_total_env_steps>.pt at each {self.args.checkpoint_interval}-step threshold after a complete population update",
                "final_snapshot": "save final.pt and step_<actual_total_env_steps>.pt at stop",
                "archive_overwrite": "existing run directories are rejected; existing archive names are skipped",
            },
        }

    def _write_run_manifest(self) -> None:
        launcher_files = {"queue.json", "train.pid", "train.log"}
        existing = [path for path in self.output.iterdir() if path.name not in launcher_files]
        if existing:
            names = ", ".join(sorted(path.name for path in existing[:8]))
            raise FileExistsError(
                f"Run output directory is not empty; choose a new --output path: {self.output} ({names})"
            )
        _write_json_exclusive(self.output / "config.json", self._run_manifest())

    def _status(self, state: str, reason: str | None = None) -> None:
        elapsed = time.monotonic() - self.started_at
        data = {
            "state": state,
            "reason": reason,
            "iteration": self.iteration,
            "total_env_steps": self.total_env_steps,
            "demonstration_env_steps": self.demonstration_env_steps,
            "sp_env_steps": self.sp_env_steps,
            "xp_env_steps": self.xp_env_steps,
            "sp_steps_by_member": self.sp_steps_by_member,
            "xp_steps_by_member": self.xp_steps_by_member,
            "elapsed_seconds": elapsed,
            "best_validation_team_score": None if not math.isfinite(self.best_score) else self.best_score,
            "best_validation_iteration": self.best_iteration,
            "best_checkpoint_policy": BEST_CHECKPOINT_POLICY,
            "effective_coefficients": {"xp": self.xp_coef, "mi": self.mi_coef},
            "last_evaluation": self.last_eval,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(self.status_path, _json_value(data))

    def _checkpoint(self, *, path: Path, reason: str) -> None:
        for member in self.members:
            for parameter in member.parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("Refusing to save non-finite model parameters")
        states = []
        for member, optim in zip(self.members, self.optimizers):
            states.append({
                "model": _to_cpu(member.state_dict()),
                "optimizers": _to_cpu({
                    "actor": optim["actor"].state_dict(),
                    "sp_critic": optim["sp_critic"].state_dict(),
                    "xp_critics": [o.state_dict() for o in optim["xp_critics"]],
                    "discriminator": optim["discriminator"].state_dict(),
                }),
            })
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "algorithm": "LIPO",
            "environment": SHARED_KITCHEN_ENVIRONMENT,
            "environment_commit": SHARED_KITCHEN_COMMIT,
            "checkpoint_reason": reason,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_config": asdict(self.config),
            "parameter_counts": self.parameter_counts,
            "config": self.static_config,
            "paper_mapping": PAPER_MAPPING,
            "source_hashes": self.source_hashes,
            "iteration": self.iteration,
            "total_env_steps": self.total_env_steps,
            "demonstration_env_steps": self.demonstration_env_steps,
            "sp_env_steps": self.sp_env_steps,
            "xp_env_steps": self.xp_env_steps,
            "sp_steps_by_member": list(self.sp_steps_by_member),
            "xp_steps_by_member": list(self.xp_steps_by_member),
            "best_validation_team_score": None if not math.isfinite(self.best_score) else self.best_score,
            "best_validation_iteration": self.best_iteration,
            "best_checkpoint_policy": BEST_CHECKPOINT_POLICY,
            "effective_coefficients": {"xp": self.xp_coef, "mi": self.mi_coef},
            "last_evaluation": self.last_eval,
            "training_stage": self.current_stage,
            "partner_cycle": self.partner_cycle.state_dict(),
            "member_states": states,
        }
        atomic_save(path, payload)

    def _archive_step_checkpoint(
        self,
        *,
        reason: str,
        crossed_thresholds: tuple[int, ...] = (),
    ) -> Path | None:
        step = self.total_env_steps
        if step in self.saved_step_checkpoints:
            return None
        path = self.output / f"step_{step}.pt"
        saved = _save_archive_if_absent(
            path, lambda target: self._checkpoint(path=target, reason=reason)
        )
        self.saved_step_checkpoints.add(step)
        self._append_metrics({
            "kind": "checkpoint",
            "checkpoint": path.name,
            "checkpoint_reason": reason,
            "saved": saved,
            "iteration": self.iteration,
            "total_env_steps": step,
            "requested_thresholds": list(crossed_thresholds),
            "threshold_alignment": CHECKPOINT_ALIGNMENT,
        })
        return path

    def _append_metrics(self, row: dict[str, Any]) -> None:
        row = _json_value(row)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _check_observation(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(observation["vector"], dtype=np.float32)
        state = np.asarray(observation["state"], dtype=np.float32)
        spatial = np.asarray(observation["spatial"], dtype=np.float32)
        if obs.shape != (self.args.num_envs, 3, self.config.obs_dim):
            raise ValueError(f"Expected vector observations [E,3,866], got {obs.shape}")
        if state.shape != (self.args.num_envs, self.config.state_dim):
            raise ValueError(f"Expected states [E,863], got {state.shape}")
        if spatial.shape != (self.args.num_envs, 3, 26, 7, 7):
            raise ValueError(f"Expected spatial observations [E,3,26,7,7], got {spatial.shape}")
        if not np.isfinite(obs).all() or not np.isfinite(state).all() or not np.isfinite(spatial).all():
            raise FloatingPointError("Environment returned non-finite observations")
        return obs, state

    def _latent_start(self) -> np.ndarray:
        return self.latent_rng.integers(
            self.config.latent_dim, size=(self.args.num_envs, self.config.num_roles), dtype=np.int64
        )

    def _refresh_latents(self, latent_ids: np.ndarray, done: np.ndarray) -> None:
        count = int(np.count_nonzero(done))
        if count:
            latent_ids[done] = self.latent_rng.integers(
                self.config.latent_dim, size=(count, self.config.num_roles), dtype=np.int64
            )

    @torch.no_grad()
    def _value_numpy(self, member: LipoMember, states: np.ndarray, opponent: int | None = None) -> np.ndarray:
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        values = member.value(state_t, opponent)
        assert_finite_tensor(values, "critic values")
        return values.cpu().numpy()

    @torch.no_grad()
    def collect_sp(self, env: KitchenBatch, member_id: int, horizon: int, stage: str) -> Rollout:
        member = self.members[member_id]
        env.set_stage(stage)
        observation = env.reset()
        latent_ids = self._latent_start()
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in ("obs", "latent_ids", "actions", "old_log_probs", "old_action_probs", "states", "old_values", "rewards", "dones")
        }
        for _ in range(horizon):
            obs, states = self._check_observation(observation)
            obs_flat = torch.as_tensor(obs.reshape(-1, self.config.obs_dim), device=self.device)
            roles = torch.arange(3, device=self.device).repeat(self.args.num_envs)
            z_ids = torch.as_tensor(latent_ids.reshape(-1), dtype=torch.long, device=self.device)
            logits = member.actor_logits(obs_flat, roles, z_ids)
            assert_finite_tensor(logits, "SP actor logits")
            dist = Categorical(logits=logits)
            actions = dist.sample().reshape(self.args.num_envs, 3)
            log_probs = dist.log_prob(actions.reshape(-1)).reshape(self.args.num_envs, 3)
            probs = dist.probs.reshape(self.args.num_envs, 3, self.config.action_dim)
            state_t = torch.as_tensor(states, device=self.device)
            values = member.value(state_t)
            assert_finite_tensor(values, "SP value predictions")
            next_observation, rewards, dones, _ = env.step(actions.cpu().numpy().astype(np.int64))
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.bool_)
            if not np.isfinite(rewards).all():
                raise FloatingPointError("Environment returned non-finite SP reward")
            rows["obs"].append(obs.copy())
            rows["latent_ids"].append(latent_ids.copy())
            rows["actions"].append(actions.cpu().numpy().astype(np.int64))
            rows["old_log_probs"].append(log_probs.cpu().numpy().astype(np.float32))
            rows["old_action_probs"].append(probs.cpu().numpy().astype(np.float32))
            rows["states"].append(states.copy())
            rows["old_values"].append(values.cpu().numpy().astype(np.float32))
            rows["rewards"].append(rewards.copy())
            rows["dones"].append(dones.copy())
            self._refresh_latents(latent_ids, dones)
            observation = next_observation
        _, last_states = self._check_observation(observation)
        last_values = self._value_numpy(member, last_states)
        data = {key: np.stack(values, axis=0) for key, values in rows.items()}
        advantages, returns = compute_gae(
            data["rewards"], data["old_values"], data["dones"], last_values,
            self.args.gamma, self.args.gae_lambda,
        )
        return Rollout(**data, advantages=advantages, returns=returns, mean_reward=float(data["rewards"].mean()))

    @torch.no_grad()
    def collect_xp(
        self,
        env: KitchenBatch,
        member_id: int,
        partner_id: int,
        role: int,
        horizon: int,
        stage: str,
    ) -> Rollout:
        """Sample A in one role; B's two actions are generated under no-grad."""
        member_a, member_b = self.members[member_id], self.members[partner_id]
        env.set_stage(stage)
        observation = env.reset()
        latent_ids = self._latent_start()
        rows: dict[str, list[np.ndarray]] = {
            name: [] for name in ("obs", "latent_ids", "actions", "old_log_probs", "old_action_probs", "states", "old_values", "rewards", "dones")
        }
        role_ids = torch.full((self.args.num_envs,), role, dtype=torch.long, device=self.device)
        for _ in range(horizon):
            obs, states = self._check_observation(observation)
            all_actions = np.empty((self.args.num_envs, 3), dtype=np.int64)
            selected_log_probs: torch.Tensor | None = None
            selected_probs: torch.Tensor | None = None
            selected_values: torch.Tensor | None = None
            with torch.no_grad():
                for slot in range(3):
                    policy = member_a if slot == role else member_b
                    obs_t = torch.as_tensor(obs[:, slot, :], device=self.device)
                    z_t = torch.as_tensor(latent_ids[:, slot], dtype=torch.long, device=self.device)
                    logits = policy.actor_logits(obs_t, torch.full_like(role_ids, slot), z_t)
                    assert_finite_tensor(logits, "XP actor logits")
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    all_actions[:, slot] = action.cpu().numpy().astype(np.int64)
                    if slot == role:
                        selected_log_probs = dist.log_prob(action)
                        selected_probs = dist.probs
                state_t = torch.as_tensor(states, device=self.device)
                selected_values = member_a.value(state_t, partner_id)
                assert_finite_tensor(selected_values, "XP value predictions")
            next_observation, rewards, dones, _ = env.step(all_actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=np.bool_)
            if not np.isfinite(rewards).all():
                raise FloatingPointError("Environment returned non-finite XP reward")
            assert selected_log_probs is not None and selected_probs is not None and selected_values is not None
            rows["obs"].append(obs[:, role, :].copy())
            rows["latent_ids"].append(latent_ids[:, role].copy())
            rows["actions"].append(all_actions[:, role].copy())
            rows["old_log_probs"].append(selected_log_probs.cpu().numpy().astype(np.float32))
            rows["old_action_probs"].append(selected_probs.cpu().numpy().astype(np.float32))
            rows["states"].append(states.copy())
            rows["old_values"].append(selected_values.cpu().numpy().astype(np.float32))
            rows["rewards"].append(rewards.copy())
            rows["dones"].append(dones.copy())
            self._refresh_latents(latent_ids, dones)
            observation = next_observation
        _, last_states = self._check_observation(observation)
        last_values = self._value_numpy(member_a, last_states, partner_id)
        data = {key: np.stack(values, axis=0) for key, values in rows.items()}
        advantages, returns = compute_gae(
            data["rewards"], data["old_values"], data["dones"], last_values,
            self.args.gamma, self.args.gae_lambda,
        )
        return Rollout(
            **data,
            advantages=advantages,
            returns=returns,
            role=role,
            partner=partner_id,
            mean_reward=float(data["rewards"].mean()),
        )

    def _tensor_batch(self, rollout: Rollout, *, self_play: bool) -> dict[str, torch.Tensor]:
        n = rollout.states.shape[0] * rollout.states.shape[1]
        if self_play:
            shaped = {
                "obs": rollout.obs.reshape(n, 3, self.config.obs_dim),
                "z": rollout.latent_ids.reshape(n, 3),
                "actions": rollout.actions.reshape(n, 3),
                "old_log_probs": rollout.old_log_probs.reshape(n, 3),
                "old_action_probs": rollout.old_action_probs.reshape(n, 3, self.config.action_dim),
            }
        else:
            shaped = {
                "obs": rollout.obs.reshape(n, self.config.obs_dim),
                "z": rollout.latent_ids.reshape(n),
                "actions": rollout.actions.reshape(n),
                "old_log_probs": rollout.old_log_probs.reshape(n),
                "old_action_probs": rollout.old_action_probs.reshape(n, self.config.action_dim),
            }
        shaped.update({
            "states": rollout.states.reshape(n, self.config.state_dim),
            "old_values": rollout.old_values.reshape(n),
            "advantages": rollout.advantages.reshape(n),
            "returns": rollout.returns.reshape(n),
        })
        return {
            key: torch.as_tensor(value, device=self.device)
            for key, value in shaped.items()
        }

    def _value_loss(
        self,
        critic: torch.nn.Module,
        data: dict[str, torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        states = data["states"][indices]
        current = critic(states).squeeze(-1)
        old = data["old_values"][indices]
        returns = data["returns"][indices]
        clipped = old + torch.clamp(current - old, -self.args.clip_param, self.args.clip_param)
        loss = torch.maximum((current - returns).square(), (clipped - returns).square()).mean()
        assert_finite_tensor(loss, "critic loss")
        return loss

    def update_member(
        self,
        member_id: int,
        sp_rollout: Rollout,
        xp_rollouts: list[Rollout],
        selected_xp_index: int,
    ) -> dict[str, float]:
        member = self.members[member_id]
        optim = self.optimizers[member_id]
        sp = self._tensor_batch(sp_rollout, self_play=True)
        xp = [self._tensor_batch(r, self_play=False) for r in xp_rollouts]
        selected_xp = xp[selected_xp_index]
        selected_role = int(xp_rollouts[selected_xp_index].role)
        count = sp["states"].shape[0]
        if any(batch["states"].shape[0] != count for batch in xp):
            raise ValueError("SP and XP rollout sizes must match for joint population updates")
        totals: dict[str, list[float]] = {key: [] for key in ("loss", "sp_pg", "xp_pg", "sp_value", "xp_value", "mi", "entropy")}
        for _ in range(self.args.epochs):
            sp_perm = torch.randperm(count, device=self.device)
            xp_perms = [torch.randperm(count, device=self.device) for _ in xp]
            for start in range(0, count, self.args.minibatch_size):
                end = min(start + self.args.minibatch_size, count)
                sp_idx = sp_perm[start:end]
                batch_size = len(sp_idx)
                obs_sp = sp["obs"][sp_idx].reshape(-1, self.config.obs_dim)
                z_sp = sp["z"][sp_idx].reshape(-1)
                role_sp = torch.arange(3, device=self.device).expand(batch_size, 3).reshape(-1)
                action_sp = sp["actions"][sp_idx].reshape(-1)
                old_lp_sp = sp["old_log_probs"][sp_idx].reshape(-1)
                adv_sp = sp["advantages"][sp_idx].unsqueeze(1).expand(-1, 3).reshape(-1)
                adv_sp = normalize_advantages(adv_sp)
                dist_sp = Categorical(logits=member.actor_logits(obs_sp, role_sp, z_sp))
                new_lp_sp = dist_sp.log_prob(action_sp)
                sp_pg = ppo_policy_loss(new_lp_sp, old_lp_sp, adv_sp, self.args.clip_param)
                entropy_sp = dist_sp.entropy().mean()
                mi_logits_sp = member.infer_latent(obs_sp, role_sp, dist_sp.probs)
                mi_sp = F.cross_entropy(mi_logits_sp, z_sp)
                sp_value = self._value_loss(member.sp_critic, sp, sp_idx)

                critic_losses: list[torch.Tensor] = []
                for j, xp_data in enumerate(xp):
                    xp_idx = xp_perms[j][start:end]
                    partner_id = int(xp_rollouts[j].partner)
                    critic_losses.append(self._value_loss(member.xp_critics[partner_id], xp_data, xp_idx))

                xp_idx = xp_perms[selected_xp_index][start:end]
                obs_xp = selected_xp["obs"][xp_idx]
                z_xp = selected_xp["z"][xp_idx]
                actions_xp = selected_xp["actions"][xp_idx]
                adv_xp = normalize_advantages(selected_xp["advantages"][xp_idx])
                role_xp = torch.full_like(z_xp, selected_role)
                dist_xp = Categorical(logits=member.actor_logits(obs_xp, role_xp, z_xp))
                new_lp_xp = dist_xp.log_prob(actions_xp)
                xp_pg = ppo_policy_loss(
                    new_lp_xp,
                    selected_xp["old_log_probs"][xp_idx],
                    adv_xp,
                    self.args.clip_param,
                    minimize_reward=True,
                )
                entropy_xp = dist_xp.entropy().mean()
                mi_logits_xp = member.infer_latent(obs_xp, role_xp, dist_xp.probs)
                mi_xp = F.cross_entropy(mi_logits_xp, z_xp)
                xp_value = torch.stack(critic_losses).mean()
                mi = (mi_sp + mi_xp) / 2.0
                entropy = (entropy_sp + entropy_xp) / 2.0
                total_loss = (
                    sp_pg + self.xp_coef * xp_pg
                    - self.args.ent_coef * entropy
                    + self.mi_coef * mi
                    + self.args.vf_coef * (sp_value + xp_value)
                )
                assert_finite_tensor(total_loss, "combined PPO loss")
                optim["actor"].zero_grad(set_to_none=True)
                optim["discriminator"].zero_grad(set_to_none=True)
                optim["sp_critic"].zero_grad(set_to_none=True)
                for optimizer in optim["xp_critics"]:
                    optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                active_modules = [member.actor, member.discriminator, member.sp_critic]
                active_modules.extend(member.xp_critics[int(r.partner)] for r in xp_rollouts)
                for module in active_modules:
                    for parameter in module.parameters():
                        if parameter.grad is not None:
                            assert_finite_tensor(parameter.grad, "gradient")
                    torch.nn.utils.clip_grad_norm_(module.parameters(), self.args.grad_clip)
                optim["actor"].step()
                optim["discriminator"].step()
                optim["sp_critic"].step()
                for rollout in xp_rollouts:
                    optim["xp_critics"][int(rollout.partner)].step()
                for module in active_modules:
                    for parameter in module.parameters():
                        assert_finite_tensor(parameter, "updated parameter")
                totals["loss"].append(float(total_loss.detach().cpu()))
                totals["sp_pg"].append(float(sp_pg.detach().cpu()))
                totals["xp_pg"].append(float(xp_pg.detach().cpu()))
                totals["sp_value"].append(float(sp_value.detach().cpu()))
                totals["xp_value"].append(float(xp_value.detach().cpu()))
                totals["mi"].append(float(mi.detach().cpu()))
                totals["entropy"].append(float(entropy.detach().cpu()))
        return {key: float(np.mean(values)) for key, values in totals.items()}

    def evaluate(self, env: KitchenBatch, eval_episodes: int) -> dict[str, Any]:
        """Evaluate each member and aggregate SharedKitchen's native episode metrics."""
        summaries: list[dict[str, Any]] = []
        for member_id, member in enumerate(self.members):
            observation = env.reset()
            latent_ids = self._latent_start()
            found: list[dict[str, Any]] = []
            steps = 0
            max_steps = 10000
            while len(found) < eval_episodes and steps < max_steps:
                obs, _ = self._check_observation(observation)
                obs_t = torch.as_tensor(obs.reshape(-1, self.config.obs_dim), device=self.device)
                role_ids = torch.arange(3, device=self.device).repeat(self.args.num_envs)
                z_ids = torch.as_tensor(latent_ids.reshape(-1), dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = member.actor_logits(obs_t, role_ids, z_ids)
                    assert_finite_tensor(logits, "evaluation logits")
                    actions = logits.argmax(dim=-1).reshape(self.args.num_envs, 3).cpu().numpy().astype(np.int64)
                observation, _, dones, infos = env.step(actions)
                steps += 1
                dones = np.asarray(dones, dtype=np.bool_)
                for done, info in zip(dones, infos):
                    if done and len(found) < eval_episodes:
                        summary = info.get("episode_summary")
                        if not isinstance(summary, dict):
                            raise RuntimeError("SharedKitchen completed an episode without episode_summary")
                        missing = [key for key in EPISODE_SUMMARY_FIELDS if key not in summary]
                        if missing:
                            raise RuntimeError(
                                "SharedKitchen episode_summary is missing fields: " + ", ".join(missing)
                            )
                        invalid = [
                            key for key in EPISODE_SUMMARY_FIELDS
                            if not isinstance(summary[key], (int, float, np.integer, np.floating))
                            or not math.isfinite(float(summary[key]))
                        ]
                        if invalid:
                            raise RuntimeError(
                                "SharedKitchen episode_summary has non-finite or non-numeric fields: "
                                + ", ".join(invalid)
                            )
                        found.append(dict(summary))
                self._refresh_latents(latent_ids, dones)
            if len(found) < eval_episodes:
                raise RuntimeError(
                    f"Member {member_id} produced only {len(found)}/{eval_episodes} evaluation episodes in {max_steps} steps"
                )
            summaries.append({"member": member_id, **_aggregate_summaries(found[:eval_episodes])})
        mean_native_metrics = {
            key: _mean_metric(summaries, f"{key}_mean") for key in SHARED_KITCHEN_METRICS
        }
        if any(value is None for value in mean_native_metrics.values()):
            raise RuntimeError("Evaluation did not aggregate all SharedKitchen native metrics")
        episode_fields = sorted({
            key[:-5] for summary in summaries for key in summary if key.endswith("_mean")
        })
        return {
            "members": summaries,
            "mean_team_score": mean_native_metrics["team_score"],
            "mean_native_metrics": mean_native_metrics,
            "mean_episode_summary": {
                key: _mean_metric(summaries, f"{key}_mean") for key in episode_fields
            },
            "episodes_per_member": eval_episodes,
            "stage": "standard",
            "policy_actions": "greedy argmax; independent z sampled per episode and chef",
        }

    def run(self) -> str:
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_run_manifest()
        self.current_stage = "standard"
        self._checkpoint(path=self.output / "initial.pt", reason="initial")
        self._status("running")
        latest_state = "stopped"
        failed = False
        stop_reason = "step_budget"
        max_steps = int(self.args.steps)
        contexts_per_iteration = self.args.population * (1 + self.args.xp_candidates)
        steps_per_env_context = self.args.num_envs
        try:
            # Keep --help and the algorithm checks usable without importing
            # the optional Overcooked/Gymnasium runtime. Actual training still
            # fails clearly if configured environment dependencies are absent.
            from baseline_common.demonstrations import demonstration_steps, warm_start
            from baseline_common.envs import KitchenBatch, curriculum

            demo_env_steps = int(demonstration_steps(self.args.demonstrations))
            if demo_env_steps <= 0:
                raise ValueError("Demonstration dataset must contain at least one environment transition")
            if max_steps <= demo_env_steps:
                raise ValueError(
                    f"--steps ({max_steps}) must exceed demonstration transitions ({demo_env_steps})"
                )
            bc_member_metrics = []
            for member_id, member in enumerate(self.members):
                member_metrics = warm_start(
                    member.actor,
                    _behavior_cloning_forward_logits(member, self.args.seed + member_id),
                    dataset_path=self.args.demonstrations,
                    device=self.device,
                    seed=self.args.seed + member_id,
                    epochs=self.args.bc_epochs,
                )
                reported_demo_steps = member_metrics.get("demo_env_steps")
                if reported_demo_steps is None:
                    raise RuntimeError("BC helper omitted its demo_env_steps metric")
                if int(reported_demo_steps) != demo_env_steps:
                    raise RuntimeError(
                        f"BC helper reported {reported_demo_steps} demonstration steps; metadata reports {demo_env_steps}"
                    )
                bc_member_metrics.append({
                    "member": member_id,
                    "metrics": {key: value for key, value in member_metrics.items() if key != "demo_env_steps"},
                })
            self.demonstration_env_steps = demo_env_steps
            self.total_env_steps = demo_env_steps
            self.xp_coef = 0.0
            self.mi_coef = 0.0
            self._checkpoint(path=self.output / "bc_initial.pt", reason="behavior_cloning_complete")
            self._append_metrics({
                "kind": "behavior_cloning",
                "dataset": str(self.args.demonstrations),
                "demo_env_steps": demo_env_steps,
                "environment_steps_counted_once": True,
                "bc_epochs_per_member": self.args.bc_epochs,
                "members": bc_member_metrics,
                "total_env_steps": self.total_env_steps,
            })
            self._status("running")

            with KitchenBatch(
                num_envs=self.args.num_envs,
                seed=self.args.seed,
                stage="standard",
                workers=self.args.workers,
                train=True,
                gamma=self.args.gamma,
            ) as train_env, KitchenBatch(
                num_envs=self.args.num_envs,
                seed=self.args.seed + 100000,
                stage="standard",
                workers=self.args.workers,
                train=False,
                gamma=self.args.gamma,
            ) as eval_env:
                initial_eval = self.evaluate(eval_env, self.args.eval_episodes)
                self.last_eval = initial_eval
                initial_score = float(initial_eval["mean_team_score"])
                self.best_score = initial_score
                self.best_iteration = self.iteration
                self._checkpoint(path=self.output / "best.pt", reason="initial_validation_best")
                self._append_metrics({
                    "kind": "evaluation", "iteration": self.iteration,
                    "total_env_steps": self.total_env_steps,
                    "demonstration_env_steps": self.demonstration_env_steps,
                    "validation": initial_eval,
                    "best_checkpoint_updated": True,
                    "best_checkpoint_reason": "initial_validation_best",
                    "best_validation_team_score": self.best_score,
                    "best_validation_iteration": self.best_iteration,
                })
                self._status("running")
                while self.total_env_steps < max_steps:
                    if time.monotonic() - self.started_at >= self.args.max_seconds:
                        stop_reason = "time_limit"
                        break
                    remaining = max_steps - self.total_env_steps
                    denom = contexts_per_iteration * steps_per_env_context
                    horizon = min(self.args.rollout, remaining // denom)
                    if horizon < 1:
                        stop_reason = "step_budget_below_one_population_update"
                        break
                    stage = curriculum(self.total_env_steps / max(1, max_steps))
                    self.current_stage = stage
                    sp_rollouts: list[Rollout] = []
                    xp_rollouts: list[list[Rollout]] = []
                    pair_choices: list[list[int]] = []
                    selected_indices: list[int] = []
                    selected_partners: list[int] = []
                    for member_id in range(self.args.population):
                        sp_rollout = self.collect_sp(train_env, member_id, horizon, stage)
                        partners = self.partner_cycle.sample(member_id)
                        member_xp = []
                        for slot, partner_id in enumerate(partners):
                            role = role_for_xp(self.iteration, slot, member_id)
                            member_xp.append(self.collect_xp(
                                train_env, member_id, partner_id, role, horizon, stage,
                            ))
                        max_index = select_max_compatible([rollout.mean_reward for rollout in member_xp])
                        sp_rollouts.append(sp_rollout)
                        xp_rollouts.append(member_xp)
                        pair_choices.append(partners)
                        selected_indices.append(max_index)
                        selected_partners.append(partners[max_index])
                    sampled_env_steps = horizon * self.args.num_envs
                    self.sp_env_steps += sampled_env_steps * self.args.population
                    self.xp_env_steps += sampled_env_steps * self.args.population * self.args.xp_candidates
                    self.sp_steps_by_member = [n + sampled_env_steps for n in self.sp_steps_by_member]
                    self.xp_steps_by_member = [n + sampled_env_steps * self.args.xp_candidates for n in self.xp_steps_by_member]
                    self.total_env_steps = (
                        self.demonstration_env_steps + self.sp_env_steps + self.xp_env_steps
                    )
                    rl_budget = max_steps - self.demonstration_env_steps
                    collected_env_steps = sampled_env_steps * self.args.population * (
                        1 + self.args.xp_candidates
                    )
                    rl_steps_at_update_start = self.sp_env_steps + self.xp_env_steps - collected_env_steps
                    warmup = _warmup_scale(rl_steps_at_update_start, rl_budget)
                    self.xp_coef = self.args.xp_coef * warmup
                    self.mi_coef = self.args.mi_coef * warmup
                    update_metrics = []
                    for member_id in range(self.args.population):
                        update_metrics.append(self.update_member(
                            member_id, sp_rollouts[member_id], xp_rollouts[member_id], selected_indices[member_id],
                        ))
                    self.iteration += 1
                    self._checkpoint(path=self.output / "latest.pt", reason="iteration_boundary")
                    mean_metrics = {
                        key: float(np.mean([row[key] for row in update_metrics]))
                        for key in update_metrics[0]
                    }
                    self._append_metrics({
                        "kind": "train",
                        "iteration": self.iteration,
                        "total_env_steps": self.total_env_steps,
                        "sp_env_steps": self.sp_env_steps,
                        "xp_env_steps": self.xp_env_steps,
                        "demonstration_env_steps": self.demonstration_env_steps,
                        "sp_steps_by_member": self.sp_steps_by_member,
                        "xp_steps_by_member": self.xp_steps_by_member,
                        "sampled_env_steps_per_context": sampled_env_steps,
                        "stage": stage,
                        "xp_coef": self.xp_coef,
                        "mi_coef": self.mi_coef,
                        "xp_mi_warmup_fraction": 0.10,
                        "xp_partner_candidates": pair_choices,
                        "xp_candidate_scores": [[r.mean_reward for r in batch] for batch in xp_rollouts],
                        "selected_max_xp_partner": selected_partners,
                        "selected_max_xp_role": [xp_rollouts[i][j].role for i, j in enumerate(selected_indices)],
                        "selected_max_xp_score": [xp_rollouts[i][j].mean_reward for i, j in enumerate(selected_indices)],
                        "losses_mean_over_population": mean_metrics,
                        "elapsed_seconds": time.monotonic() - self.started_at,
                    })
                    self._status("running")
                    crossed_thresholds = self.checkpoint_schedule.crossed(self.total_env_steps)
                    archive_reasons = []
                    if crossed_thresholds:
                        archive_reasons.append("interval")
                    should_evaluate = (
                        self.iteration % self.args.eval_interval == 0
                        or self.total_env_steps >= max_steps
                    )
                    if should_evaluate:
                        validation = self.evaluate(eval_env, self.args.eval_episodes)
                        self.last_eval = validation
                        score = float(validation["mean_team_score"])
                        tie = score == self.best_score
                        best_updated = score >= self.best_score
                        best_reason = None
                        if best_updated:
                            self.best_score = score
                            self.best_iteration = self.iteration
                            best_reason = "validation_tie_latest" if tie else "validation_improved"
                            self._checkpoint(path=self.output / "best.pt", reason=best_reason)
                        self._append_metrics({
                            "kind": "evaluation", "iteration": self.iteration,
                            "total_env_steps": self.total_env_steps,
                            "demonstration_env_steps": self.demonstration_env_steps,
                            "validation": validation,
                            "best_checkpoint_updated": best_updated,
                            "best_checkpoint_reason": best_reason,
                            "best_validation_team_score": self.best_score,
                            "best_validation_iteration": self.best_iteration,
                            "best_checkpoint_policy": BEST_CHECKPOINT_POLICY,
                        })
                        self._status("running")
                    if archive_reasons:
                        self._archive_step_checkpoint(
                            reason="+".join(archive_reasons),
                            crossed_thresholds=crossed_thresholds,
                        )
                    if time.monotonic() - self.started_at >= self.args.max_seconds:
                        stop_reason = "time_limit"
                        break
                else:
                    stop_reason = "step_budget"
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        except BaseException as exc:
            failed = True
            self._checkpoint(path=self.output / "latest.pt", reason=f"failed: {type(exc).__name__}")
            self._status("failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not failed:
                self._archive_step_checkpoint(reason=f"final:{stop_reason}")
                self._checkpoint(path=self.output / "latest.pt", reason=stop_reason)
                self._checkpoint(path=self.output / "final.pt", reason=stop_reason)
        if stop_reason == "step_budget":
            latest_state = "completed"
        self._status(latest_state, stop_reason)
        return latest_state


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for summary in summaries for key in summary})
    result: dict[str, Any] = {"episode_count": len(summaries)}
    for key in keys:
        values = [summary[key] for summary in summaries if key in summary]
        if not values:
            continue
        if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            finite = [float(value) for value in values if math.isfinite(float(value))]
            if finite:
                result[f"{key}_mean"] = float(np.mean(finite))
                result[f"{key}_std"] = float(np.std(finite))
        else:
            mean_value = _nested_stat(values, "mean")
            std_value = _nested_stat(values, "std")
            if mean_value is not None:
                result[f"{key}_mean"] = mean_value
                result[f"{key}_std"] = std_value
    return result


def _nested_stat(values: list[Any], statistic: str) -> Any:
    if not values:
        return None
    first = values[0]
    if isinstance(first, dict) and all(isinstance(value, dict) for value in values):
        keys = sorted(set.intersection(*(set(value) for value in values)), key=str)
        result = {}
        for key in keys:
            value = _nested_stat([row[key] for row in values], statistic)
            if value is not None:
                result[str(key)] = value
        return result or None
    if all(isinstance(value, (list, tuple, np.ndarray)) for value in values):
        try:
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                return None
            result = array.mean(axis=0) if statistic == "mean" else array.std(axis=0)
            return result.tolist()
        except (TypeError, ValueError):
            return None
    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        result = finite.mean() if statistic == "mean" else finite.std()
        return float(result)
    return None


def _mean_metric(members: list[dict[str, Any]], key: str) -> float | None:
    values = [member[key] for member in members if key in member]
    return None if not values else float(np.mean(values))


def _mean_vector_metric(members: list[dict[str, Any]], key: str) -> list[float] | None:
    values = [member[key] for member in members if key in member]
    return None if not values else _nested_stat(values, "mean")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an 8-member LIPO joint-policy population.")
    parser.add_argument("--output", type=Path, default=None, help="Run directory (default: lipo/runs/<UTC timestamp>)")
    parser.add_argument("--steps", type=int, default=16_000_000, help="Total environment steps including demonstrations and RL SP/XP.")
    parser.add_argument("--demonstrations", type=Path, required=True, help="SharedKitchen demonstration dataset used to initialize every member actor.")
    parser.add_argument("--bc-epochs", type=int, default=64, help="Behavior-cloning epochs per population member (default: 64).")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL_STEPS,
        help="Step archive interval (default: 250000).",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=128, help="Horizon per population/context sample.")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0", help="Use cpu for local checks; CUDA uses physical GPU 1 via CUDA_VISIBLE_DEVICES=1.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=172800.0, help="Stop at an iteration boundary after this wall time.")
    parser.add_argument("--xp-candidates", type=int, default=None, help="Sampled B partners per A; cycles cover all partners (default min(2,K-1)).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.9985)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.3)
    parser.add_argument("--xp-coef", type=float, default=0.3)
    parser.add_argument("--mi-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true", help="Exercise the full population with a short rollout/update/evaluation.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = ("steps", "bc_epochs", "checkpoint_interval", "num_envs", "rollout", "population", "eval_interval", "eval_episodes", "max_seconds", "epochs", "minibatch_size")
    for key in positive:
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.population < 2:
        raise ValueError("LIPO requires at least two population members for XP")
    if args.xp_candidates is None:
        args.xp_candidates = min(2, args.population - 1)
    if not 1 <= args.xp_candidates <= args.population - 1:
        raise ValueError("--xp-candidates must be in [1, population-1]")
    if args.workers < 0:
        raise ValueError("--workers must be nonnegative")
    if not args.demonstrations.is_file():
        raise FileNotFoundError(f"Demonstration dataset does not exist: {args.demonstrations}")
    from baseline_common.demonstrations import demonstration_steps
    demo_env_steps = int(demonstration_steps(args.demonstrations))
    if demo_env_steps <= 0 or args.steps <= demo_env_steps:
        raise ValueError("--steps must exceed the positive demonstration environment-step count")
    if args.gamma <= 0 or args.gamma > 1 or args.gae_lambda < 0 or args.gae_lambda > 1:
        raise ValueError("gamma must be in (0,1] and gae-lambda in [0,1]")
    if args.clip_param <= 0 or args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("clip parameter, learning rate, and gradient clip must be positive")
    if min(args.xp_coef, args.mi_coef, args.ent_coef, args.vf_coef) < 0:
        raise ValueError("loss coefficients must be nonnegative")
    if args.smoke:
        args.rollout = min(args.rollout, 4)
        args.epochs = min(args.epochs, 1)
        args.eval_interval = 1
        args.eval_episodes = 1
        minimum_budget = demo_env_steps + args.num_envs * args.population * (1 + args.xp_candidates)
        if args.steps < minimum_budget:
            raise ValueError(f"--smoke needs at least {minimum_budget} steps including demonstrations for one population update")
        args.steps = min(
            args.steps,
            demo_env_steps + args.num_envs * args.rollout * args.population * (1 + args.xp_candidates),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        if args.device != "cuda:0":
            raise RuntimeError("CUDA training must use logical cuda:0 with CUDA_VISIBLE_DEVICES=1")
    device = setup_device(device=args.device, seed=args.seed, memory_fraction=0.24)
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = Path("lipo") / "runs" / f"lipo_{stamp}"
    if not output.is_absolute():
        output = Path.cwd() / output
    trainer = LipoTrainer(args, device, output)
    state = trainer.run()
    print(json.dumps({
        "state": state,
        "output": str(output),
        "iteration": trainer.iteration,
        "total_env_steps": trainer.total_env_steps,
        "sp_env_steps": trainer.sp_env_steps,
        "xp_env_steps": trainer.xp_env_steps,
        "demonstration_env_steps": trainer.demonstration_env_steps,
        "reason": json.loads(trainer.status_path.read_text(encoding="utf-8")).get("reason"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

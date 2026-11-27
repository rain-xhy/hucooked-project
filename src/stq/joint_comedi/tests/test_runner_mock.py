from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from joint_comedi.runner import run
from joint_comedi.train import allocate_budgets, effective_args, parser


class FakeKitchenBatch:
    def __init__(self, num_envs=2, seed=1, stage="standard", workers=0,
                 train=True, gamma=0.9985, reward_profile="legacy"):
        self.num_envs = int(num_envs)
        self.ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.closed = False
        self.train = bool(train)
        self.reward_profile = reward_profile

    def reset(self):
        self.ticks[:] = 0
        return self._observation()

    def _observation(self):
        state = np.zeros((self.num_envs, 863), dtype=np.float32)
        state[:, 0] = self.ticks
        vector = np.zeros((self.num_envs, 3, 866), dtype=np.float32)
        vector[:, :, 0] = self.ticks[:, None]
        spatial = np.zeros((self.num_envs, 3, 26, 7, 7), dtype=np.float32)
        for role in range(3):
            spatial[:, role, 12 + role * 4, role + 1, 1] = 1.0
        return {"state": state, "vector": vector, "spatial": spatial}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, 3)
        assert np.issubdtype(actions.dtype, np.integer)
        self.ticks += 1
        dones = self.ticks >= 4
        infos = []
        for env_id, done in enumerate(dones):
            if done:
                metrics = {
                    "team_score": 4.0,
                    "meals_completed": 1,
                    "normal_completed": 1,
                    "joint_completed": 0,
                    "joint_successes": 2,
                    "joint_attempt_success_rate": 0.5,
                    "joint_wait_steps": 1,
                    "three_contributor_meals": 1,
                    "resource_conflicts": 3,
                    "movement_conflicts": 2,
                    "ticks": 4,
                    "meals_per_100_steps": 25.0,
                }
                infos.append({"episode_summary": {
                    **metrics, "episode": 0, "training_reward": 4.0,
                    "training_shaping": 0.0,
                }})
                self.ticks[env_id] = 0
            else:
                infos.append({})
        return self._observation(), np.ones(self.num_envs, dtype=np.float32), dones, infos

    def close(self):
        self.closed = True


def _dependencies(dataset_file: Path | None = None):
    prepared = []
    env_calls = []
    warm_start_calls = []
    data = {
        "state": np.zeros((4, 863), dtype=np.float32),
        "spatial": np.zeros((4, 26, 7, 7), dtype=np.float32),
        "actions": np.zeros((4, 3), dtype=np.int64),
        "validation": np.array([False, False, True, True]),
        "episode_lengths": np.array([2, 2], dtype=np.int64),
        "env_steps": 2,
    }
    data_sha = hashlib.sha256(dataset_file.read_bytes()).hexdigest() if dataset_file else None

    def atomic_json(path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def atomic_save(path, value):
        torch.save(value, path)

    def prepare_output(path):
        prepared.append(str(Path(path).resolve()))
        Path(path).mkdir(parents=True, exist_ok=True)

    def setup_device(**kwargs):
        assert kwargs["physical_gpu"] == 1
        return torch.device(kwargs["device"])

    runtime = SimpleNamespace(
        atomic_json=atomic_json,
        atomic_save=atomic_save,
        prepare_output=prepare_output,
        setup_device=setup_device,
        source_hashes=lambda _method: {"joint_comedi/test.py": "test-hash"},
        cpu_tree=lambda value: value,
    )

    envs = SimpleNamespace(
        ENVIRONMENT_COMMIT="local-test-commit",
        KitchenBatch=lambda **kwargs: (env_calls.append(kwargs) or FakeKitchenBatch(**kwargs)),
    )

    demonstrations = SimpleNamespace(
        load_joint_dataset=lambda _path: {**data, "dataset_sha256": data_sha},
    )

    def warm_start_joint(model, forward_logits, **kwargs):
        warm_start_calls.append(kwargs.copy())
        logits = forward_logits(torch.zeros(2, 26, 7, 7), torch.zeros(2, 863))
        assert logits.shape == (2, 3, 6)
        with torch.no_grad():
            model.head.bias.add_(0.001)
        return {
            "demo_env_steps": 2,
            "epochs": kwargs["epochs"],
            "train_ce": 1.0,
            "validation_accuracy": 0.5,
            "dataset_sha256": data_sha,
        }

    demonstrations.warm_start_joint = warm_start_joint
    return SimpleNamespace(
        runtime=runtime,
        envs=envs,
        demonstrations=demonstrations,
        prepared=prepared,
        env_calls=env_calls,
        warm_start_calls=warm_start_calls,
    )


def _run_smoke(mode: str, output: Path, dataset_file: Path | None,
               *, reward_profile: str = "legacy", max_seconds: int = 0):
    args = parser().parse_args([
        "--mode", mode,
        "--output", str(output),
        "--device", "cpu",
        "--physical-gpu", "1",
        "--steps", "16000000",
        "--smoke",
        "--eval-interval", "10000000",
        "--eval-episodes", "2",
        "--save-interval", "20",
        "--bc-epochs", "64",
        "--max-seconds", str(max_seconds),
        "--reward-profile", reward_profile,
        *(["--demonstrations", str(dataset_file)] if dataset_file is not None else []),
    ])
    dependencies = _dependencies(dataset_file)
    effective = effective_args(args, demonstration_steps=2 if mode == "bc_rl" else 0)
    assert effective_args(effective).steps == effective.steps
    result = run(effective, dependencies=dependencies)
    return args, effective, dependencies, result


def test_rl_smoke_runs_sp_xp_mp_and_cbr_without_reading_demos():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "rl-run"
        missing_demo = root / "must-not-be-read.npz"
        args = parser().parse_args([
            "--mode", "rl", "--output", str(output), "--device", "cpu",
            "--physical-gpu", "1", "--steps", "16000000", "--smoke",
            "--eval-interval", "10000000", "--eval-episodes", "1",
            "--save-interval", "20", "--demonstrations", str(missing_demo),
        ])
        assert args.reward_profile == "legacy"
        dependencies = _dependencies()
        effective = effective_args(args)
        assert effective.steps == 96
        assert effective.max_seconds == 0
        result = run(effective, dependencies=dependencies)
        assert result["state"] == "completed"
        assert result["total_env_steps"] == 96
        assert dependencies.warm_start_calls == []
        assert dependencies.prepared == [str(output.resolve())]
        assert all(call["reward_profile"] == "legacy" for call in dependencies.env_calls)
        assert (output / "convention_0.pt").exists()
        assert (output / "convention_1.pt").exists()
        assert (output / "best_population.pt").exists()
        assert (output / "random_initial.pt").exists()
        assert (output / "best.pt").exists()
        assert (output / "final.pt").exists()
        assert not (output / "bc_convention_0.pt").exists()
        assert not (output / "bc_cbr_initial.pt").exists()
        manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert manifest["mode"] == "rl"
        assert manifest["reward_profile"] == "legacy"
        assert manifest["protocol"]["validation_reward"] == "native reward with KitchenBatch(train=False)"
        assert manifest["actor_schema"]["parameter_count"] == 119346
        assert manifest["demonstrations"]["dataset_sha256"] is None
        events = [json.loads(row) for row in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        convention_rounds = [event for event in events if event.get("event") == "joint_round"]
        assert len(convention_rounds) == 4
        learned_round = next(event for event in convention_rounds if event["policy_index"] == 1)
        assert set(learned_round["train"]["samples_by_phase"]) == {"sp", "xp", "mp"}
        assert learned_round["env_steps"]["mp_prefix"] > 0
        assert learned_round["env_steps"]["mp_suffix"] > 0
        cbr_rounds = [event for event in events if event.get("event") == "cbr_joint_round"]
        assert len(cbr_rounds) == 2
        assert all(event["train"]["bc_loss"] > 0.0 for event in cbr_rounds)
        assert all(set(event["train"]["bc_samples_by_teacher"]) == {"0", "1"}
                   for event in cbr_rounds)


def test_ppo_task_reward_profile_is_forwarded_and_manifested():
    with TemporaryDirectory() as directory:
        output = Path(directory) / "ppo-task-run"
        args, effective, dependencies, result = _run_smoke(
            "rl", output, None, reward_profile="ppo_task", max_seconds=0
        )
        assert args.reward_profile == effective.reward_profile == "ppo_task"
        assert effective.max_seconds == 0
        assert result["state"] == "completed"
        assert all(call["reward_profile"] == "ppo_task" for call in dependencies.env_calls)
        manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert manifest["reward_profile"] == "ppo_task"
        assert manifest["arguments"]["reward_profile"] == "ppo_task"
        assert manifest["protocol"]["reward_profile"] == "ppo_task"
        assert manifest["protocol"]["validation_reward"] == "native reward with KitchenBatch(train=False)"


def test_bc_rl_smoke_counts_demos_once_and_saves_all_bc_candidates():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "demo.npz"
        dataset.write_bytes(b"small local demonstration fixture")
        output = root / "bc-run"
        _args, effective, dependencies, result = _run_smoke("bc_rl", output, dataset)
        assert effective.steps == 98
        assert effective.bc_epochs == 64
        assert effective.rounds_per_convention == 2
        assert result["state"] == "completed"
        assert result["total_env_steps"] == 98
        assert len(dependencies.warm_start_calls) == 3
        assert all(call["epochs"] == 64 and call["batch_size"] == 512
                   for call in dependencies.warm_start_calls)
        assert (output / "bc_convention_0.pt").exists()
        assert (output / "bc_convention_1.pt").exists()
        assert (output / "bc_cbr_initial.pt").exists()
        manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert manifest["demonstrations"]["env_steps_counted_once"] == 2
        assert manifest["demonstrations"]["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
        events = [json.loads(row) for row in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        bc_events = [event for event in events if event.get("event") == "convention_bc_warm_start"]
        assert len(bc_events) == 2
        assert sum(bool(event["dataset_steps_counted_once"]) for event in bc_events) == 1
        cbr_event = next(event for event in events if event.get("event") == "cbr_bc_warm_start")
        assert cbr_event["dataset_steps_counted_once"] is False
        assert (output / "step_20.pt").exists() or any(output.glob("step_*.pt"))


def test_smoke_budget_includes_9600_demo_steps_once_and_keeps_bc_epochs():
    requested = parser().parse_args([
        "--mode", "bc_rl", "--output", "unused", "--smoke",
        "--steps", "16000000", "--demonstrations", "fixture.npz", "--bc-epochs", "64",
    ])
    args = effective_args(requested, demonstration_steps=9600)
    repeated = effective_args(args)
    budgets = allocate_budgets(args, demonstration_env_steps=9600)
    assert args.steps == repeated.steps == 9696
    assert args.bc_epochs == repeated.bc_epochs == 64
    assert budgets["demonstration_env_steps"] == 9600
    assert budgets["effective_rl_total"] == 96
    assert budgets["effective_total"] == 9696

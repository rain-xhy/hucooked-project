"""CPU end-to-end protocol check with a deterministic KitchenBatch stand-in."""

import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

from comedi.runner import _mp_vector_chunk, run
from comedi.train import allocate_budgets, effective_args, parser


class _FakeKitchenBatch:
    def __init__(self, num_envs=8, seed=1, workers=0, train=True, gamma=0.9985):
        self.num_envs = int(num_envs)
        self.train = train
        self.ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.closed = False

    def reset(self):
        self.ticks[:] = 0
        return self._observation()

    def _observation(self):
        vector = np.zeros((self.num_envs, 3, 866), dtype=np.float32)
        vector[:, :, 0] = self.ticks[:, None]
        spatial = np.zeros((self.num_envs, 3, 26, 7, 7), dtype=np.float32)
        state = np.zeros((self.num_envs, 863), dtype=np.float32)
        return {"vector": vector, "spatial": spatial, "state": state}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, 3)
        assert np.issubdtype(actions.dtype, np.integer)
        self.ticks += 1
        dones = self.ticks >= 4
        rewards = np.ones(self.num_envs, dtype=np.float32)
        infos = []
        for env_index, done in enumerate(dones):
            if done:
                summary = {
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
                    "episode": 0,
                    "training_reward": 4.0,
                    "training_shaping": 0.0,
                }
                infos.append({"episode_summary": summary})
                self.ticks[env_index] = 0
            else:
                infos.append({})
        return self._observation(), rewards, dones, infos

    def close(self):
        self.closed = True


def test_runner_finishes_joint_comedi_and_cbr_without_gpu(monkeypatch):
    fake_env_module = types.ModuleType("baseline_common.envs")
    fake_env_module.ENVIRONMENT_COMMIT = "de37e0818c2e76311e105ab8689e5162bd84133f"
    fake_env_module.KitchenBatch = _FakeKitchenBatch
    monkeypatch.setitem(sys.modules, "baseline_common.envs", fake_env_module)
    fake_demonstrations = types.ModuleType("baseline_common.demonstrations")
    fake_demonstrations.demonstration_steps = lambda _path: 6
    bc_calls = []

    def fake_warm_start(model, forward_logits, **kwargs):
        logits = forward_logits(
            torch.zeros((2, 26, 7, 7)), torch.zeros((2, 866))
        )
        assert logits.shape == (2, 6)
        bc_calls.append(kwargs)
        with torch.no_grad():
            next(model.parameters()).add_(0.001)
        return {
            "demo_env_steps": 6,
            "epochs": kwargs["epochs"],
            "train_ce": 1.0,
            "validation_accuracy": 0.5,
        }

    fake_demonstrations.warm_start = fake_warm_start
    monkeypatch.setitem(sys.modules, "baseline_common.demonstrations", fake_demonstrations)

    comedi_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mock-run-", dir=comedi_root) as scratch:
        demonstrations = Path(scratch) / "demonstrations.npz"
        demonstrations.write_bytes(b"test dataset")
        output = Path(scratch) / "run"
        output.mkdir()
        (output / "queue.json").write_text("{}", encoding="utf-8")
        (output / "train.pid").write_text("123", encoding="utf-8")
        (output / "train.log").write_text("launcher metadata", encoding="utf-8")
        args = effective_args(parser().parse_args([
            "--output", str(output),
            "--device", "cpu",
            "--steps", "97",
            "--num-envs", "2",
            "--rollout", "8",
            "--population", "2",
            "--rounds-per-convention", "2",
            "--epochs", "1",
            "--minibatches", "1",
            "--eval-interval", "1",
            "--eval-episodes", "1",
            "--save-interval", "20",
            "--cbr-fraction", "0.25",
            "--demonstrations", str(demonstrations),
            "--bc-epochs", "1",
        ]))
        run(args)

        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "completed"
        assert status["total_train_env_steps"] == 98
        assert status["demonstration_env_steps"] == 6
        assert status["rl_env_steps"] == 92
        assert status["requested_budget_jointenvsteps"] == 97
        assert status["budget_jointenvsteps"] == 98
        assert (output / "initial.pt").exists()
        assert (output / "latest.pt").exists()
        assert (output / "best.pt").exists()
        assert (output / "best_population.pt").exists()
        assert (output / "final.pt").exists()
        assert (output / "step_98.pt").exists()
        assert (output / "convention_0.pt").exists()
        assert (output / "convention_1.pt").exists()
        assert (output / "cbr_initial.pt").exists()
        assert (output / "bc_cbr_initial.pt").exists()

        manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert manifest["environment"] == "SharedKitchen"
        assert manifest["environment_commit"].startswith("de37e08")
        assert manifest["observation"] == {
            "vector": [3, 866], "state": [863], "spatial": [3, 26, 7, 7],
        }
        assert manifest["demonstrations"]["env_steps"] == 6
        assert manifest["budgets_jointenvsteps"]["effective_rl_total"] == 92
        assert "joint_successes" in manifest["native_episode_metrics"]
        assert "joint_completed" in manifest["native_episode_metrics"]

        events = [json.loads(line) for line in (output / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()]
        validation = next(event for event in events if event["event"] == "validation")
        assert validation["mean_team_score"] == 4.0
        assert validation["mean_joint_completed"] == 0.0
        assert validation["mean_joint_successes"] == 2.0
        assert validation["mean_meals_per_100_steps"] == 25.0
        assert validation["mean_training_reward"] == 4.0
        warm_starts = [event for event in events
                       if event["event"] == "demonstration_warm_start"]
        assert len(warm_starts) == 2
        assert sum(event["dataset_steps_counted_once"] for event in warm_starts) == 1
        assert all(event["bc_metrics"]["train_ce"] == 1.0 for event in warm_starts)
        bc_evaluations = [event for event in events
                          if event["event"] == "demonstration_validation"]
        assert len(bc_evaluations) == 4
        assert {event["phase"] for event in bc_evaluations} == {"before_bc", "after_bc"}
        assert len(bc_calls) == 3
        cbr_warm_starts = [event for event in events
                           if event["event"] == "cbr_demonstration_warm_start"]
        assert len(cbr_warm_starts) == 1
        assert cbr_warm_starts[0]["dataset_steps_counted_once"] is False
        assert cbr_warm_starts[0]["step_counts"]["demonstration_env_steps"] == 6
        cbr_candidate_validation = next(
            event for event in events
            if event["event"] == "cbr_demonstration_validation"
        )
        assert cbr_candidate_validation["mean_team_score"] == 4.0
        bc_cbr = torch.load(
            output / "bc_cbr_initial.pt", map_location="cpu", weights_only=False
        )
        assert bc_cbr["bc_metrics"]["validation_accuracy"] == 0.5
        convention_rounds = [event for event in events if event["event"] == "joint_round"]
        assert len(convention_rounds) == 4
        learned_round = next(event for event in convention_rounds if event["policy_index"] == 1)
        assert set(learned_round["train"]["samples_by_phase"]) == {"sp", "xp", "mp"}
        assert all(learned_round["train"]["samples_by_phase"][phase] > 0
                   for phase in ("sp", "xp", "mp"))
        assert learned_round["env_steps"]["mp_prefix"] > 0
        assert learned_round["env_steps"]["mp_suffix"] > 0
        cbr_rounds = [event for event in events if event["event"] == "cbr_joint_round"]
        assert len(cbr_rounds) == 2
        assert all(event["train"]["bc_loss"] > 0 for event in cbr_rounds)
        assert all(set(event["train"]["samples_by_phase"]) == {"sp"}
                   for event in cbr_rounds)
        assert all(set(event["train"]["bc_samples_by_teacher"]) == {"0", "1"}
                   for event in cbr_rounds)
        assert all(set(event["bc_teacher_env_steps"]) == {"0", "1"}
                   for event in cbr_rounds)
        assert all(all(steps > 0 for steps in event["bc_teacher_env_steps"].values())
                   for event in cbr_rounds)
        budget_event = next(event for event in events if event["event"] == "budget_plan")
        assert budget_event["rounded_up_joint_env_steps"] == 1
        assert budget_event["reserved_demonstration_env_steps"] == 6
        assert budget_event["effective_rl_env_steps"] == 92

        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        assert best["cbr"] is not None
        assert len(best["conventions"]) == 2
        assert all(
            torch.equal(value, best["cbr"]["model"][f"actor.{key}"])
            for key, value in bc_cbr["actor_state_dict"].items()
        )
        population_best = torch.load(
            output / "best_population.pt", map_location="cpu", weights_only=False
        )
        assert population_best["cbr"] is None
        assert all((output / name).exists() for name in ("queue.json", "train.pid", "train.log"))


def test_smoke_reserves_demo_steps_once_and_preserves_bc_epochs(monkeypatch, tmp_path):
    fake_demonstrations = types.ModuleType("baseline_common.demonstrations")
    fake_demonstrations.demonstration_steps = lambda _path: 6
    monkeypatch.setitem(sys.modules, "baseline_common.demonstrations", fake_demonstrations)
    dataset = tmp_path / "demo.npz"
    dataset.write_bytes(b"smoke data")
    args = parser().parse_args([
        "--output", "unused", "--smoke", "--steps", "16000000",
        "--demonstrations", str(dataset), "--bc-epochs", "64",
    ])

    effective = effective_args(args)
    repeated = effective_args(effective)
    budgets = allocate_budgets(effective, demonstration_env_steps=6)

    assert effective.steps == 102
    assert repeated.steps == 102
    assert effective.bc_epochs == repeated.bc_epochs == 64
    assert budgets["demonstration_env_steps"] == 6
    assert budgets["effective_rl_total"] == 96
    assert budgets["effective_total"] == 102

    no_demo = effective_args(parser().parse_args(["--output", "unused", "--smoke"]))
    assert no_demo.steps == 96
    assert effective_args(no_demo).steps == 96


@pytest.mark.parametrize(
    "remaining,rollout",
    ((2, 8), (3, 2), (5, 4), (9, 4), (11, 4)),
)
def test_mp_chunk_never_leaves_a_one_step_prefix_only_tail(remaining, rollout):
    chunk = _mp_vector_chunk(remaining, rollout)
    assert 2 <= chunk <= remaining
    assert remaining - chunk != 1

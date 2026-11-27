from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from joint_ppo.algorithm import compute_gae, joint_bc_loss, joint_entropy, joint_log_prob, ppo_update
from joint_ppo.evaluate import evaluate_policy, load_policy_checkpoint
from joint_ppo.models import build_policy
from joint_ppo.train import (
    CHECKPOINT_FORMAT,
    _save_checkpoint,
    _checkpoint_payload,
    collect_rollout,
    derive_run_config,
    parse_args,
    train_experiment,
    StepThresholdSchedule,
)


@pytest.mark.parametrize(
    ("backbone", "expected_actor_params"),
    (("mlp", 129426), ("cnn", 132482), ("transformer", 126914)),
)
def test_actor_uses_one_team_row_and_emits_three_heads(backbone, expected_actor_params):
    policy = build_policy(backbone)
    spatial = torch.zeros(2, 26, 7, 7)
    state = torch.zeros(2, 863)
    observed_shapes = []
    if backbone == "mlp":
        trunk = policy.actor.trunk
    elif backbone == "cnn":
        trunk = policy.actor.fusion
    else:
        trunk = policy.actor.encoder
    hook = trunk.register_forward_pre_hook(lambda _module, inputs: observed_shapes.append(tuple(inputs[0].shape)))
    logits = policy.actor(spatial, state)
    hook.remove()

    assert logits.shape == (2, 3, 6)
    assert sum(parameter.numel() for parameter in policy.actor.parameters()) == expected_actor_params
    assert policy.parameter_counts()["critic"] == 127233
    assert len(observed_shapes) == 1
    assert observed_shapes[0][0] == 2
    if backbone == "transformer":
        assert not torch.equal(
            policy.actor.encoder.layers[0].linear1.weight,
            policy.actor.encoder.layers[1].linear1.weight,
        )
    assert len({id(parameter) for parameter in policy.actor.parameters()}.intersection(
        id(parameter) for parameter in policy.critic.parameters()
    )) == 0


@pytest.mark.parametrize(
    ("backbone", "model_size", "expected_actor_params", "expected_critic_params"),
    (("mlp", "large", 291602, 287233),
     ("cnn", "large", 354034, 287233),
     ("transformer", "large", 310450, 287233),
     ("mlp", "million", 974738, 705537),
     ("cnn", "million", 1064402, 705537),
     ("transformer", "million", 993490, 705537)),
)
def test_expanded_backbones_keep_one_team_trunk_and_expected_counts(
    backbone, model_size, expected_actor_params, expected_critic_params
):
    policy = build_policy(backbone, model_size=model_size)
    spatial = torch.zeros(2, 26, 7, 7)
    state = torch.zeros(2, 863)
    seen = []
    if backbone == "mlp":
        trunk = policy.actor.trunk
    elif backbone == "cnn":
        trunk = policy.actor.fusion
    else:
        trunk = policy.actor.encoder
    hook = trunk.register_forward_pre_hook(lambda _module, inputs: seen.append(tuple(inputs[0].shape)))
    logits = policy.actor(spatial, state)
    hook.remove()
    assert logits.shape == (2, 3, 6)
    assert sum(parameter.numel() for parameter in policy.actor.parameters()) == expected_actor_params
    assert policy.parameter_counts()["critic"] == expected_critic_params
    assert len(seen) == 1 and seen[0][0] == 2


@pytest.mark.parametrize("model_size", ("small", "large", "million"))
def test_checkpoint_roundtrip_restores_saved_model_size_and_old_missing_size_defaults_small(model_size):
    policy = build_policy("transformer", model_size=model_size)
    config = {"mode": "rl", "model_size": model_size, "demo_env_steps": 0}
    payload = _checkpoint_payload(
        policy,
        config=config,
        counts=policy.parameter_counts(),
        env_steps=0,
        rl_env_steps=0,
        reason="unit test",
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(payload, path)
        restored, metadata = load_policy_checkpoint(path, "cpu")
        assert restored.model_size == model_size
        assert metadata["model_size"] == model_size
        for key, value in policy.state_dict().items():
            assert torch.equal(value.cpu(), restored.state_dict()[key].cpu())

        if model_size == "small":
            old_payload = dict(payload)
            old_payload.pop("model_size")
            old_payload["config"] = {"mode": "rl"}
            torch.save(old_payload, path)
            old_restored, old_metadata = load_policy_checkpoint(path, "cpu")
            assert old_restored.model_size == "small"
            assert old_metadata.get("model_size") is None


def test_joint_log_probability_entropy_and_bc_keep_three_labels_together():
    logits = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6) / 10
    actions = torch.tensor([[0, 3, 5], [2, 1, 4]])
    distribution = Categorical(logits=logits)
    assert torch.allclose(joint_log_prob(logits, actions), distribution.log_prob(actions).sum(-1))
    assert torch.allclose(joint_entropy(logits), distribution.entropy().sum(-1))
    assert torch.allclose(joint_bc_loss(logits, actions), torch.nn.functional.cross_entropy(
        logits.reshape(-1, 6), actions.reshape(-1)
    ))


def test_gae_uses_one_shared_team_return_and_stops_at_done():
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [0.25]], dtype=np.float32)
    dones = np.array([[False], [True]])
    advantages, returns = compute_gae(
        rewards, values, dones, np.array([9.0], dtype=np.float32), gamma=0.9, gae_lambda=1.0
    )
    np.testing.assert_allclose(advantages[:, 0], [2.3, 1.75], rtol=1e-6)
    np.testing.assert_allclose(returns[:, 0], [2.8, 2.0], rtol=1e-6)


def test_cpu_ppo_update_changes_actor_and_independent_critic():
    policy = build_policy("mlp")
    before_actor = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    before_critic = [parameter.detach().clone() for parameter in policy.critic.parameters()]
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-4)
    rng = np.random.default_rng(5)
    spatial = np.zeros((8, 26, 7, 7), dtype=np.float32)
    state = rng.normal(size=(8, 863)).astype(np.float32)
    actions = rng.integers(0, 6, size=(8, 3), dtype=np.int64)
    with torch.no_grad():
        logits = policy.actor(torch.from_numpy(spatial), torch.from_numpy(state))
        old_log_probs = joint_log_prob(logits, torch.from_numpy(actions)).numpy()
    batch = {
        "spatial": spatial,
        "state": state,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "advantages": rng.normal(size=8).astype(np.float32),
        "returns": rng.normal(size=8).astype(np.float32),
    }
    metrics = ppo_update(
        policy,
        actor_optimizer,
        critic_optimizer,
        batch,
        device="cpu",
        rng=rng,
        epochs=1,
        minibatches=2,
    )

    assert metrics["optimizer_updates"] == 2
    assert metrics["joint_minibatch_size_mean"] == 4
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before, after) for before, after in zip(before_actor, policy.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_critic, policy.critic.parameters()))


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.observation = {
            "spatial": np.zeros((num_envs, 3, 26, 7, 7), dtype=np.float32),
            "state": np.zeros((num_envs, 863), dtype=np.float32),
            "vector": np.zeros((num_envs, 3, 866), dtype=np.float32),
        }
        self.steps = 0

    def reset(self):
        return self.observation

    def step(self, actions):
        assert actions.shape == (self.num_envs, 3)
        self.steps += 1
        done = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return self.observation, np.ones(self.num_envs, dtype=np.float32), done, infos

    def close(self):
        pass


def test_rollout_encodes_each_team_once_not_three_agent_rows():
    policy = build_policy("mlp")
    env = _FakeEnv(2)
    trunk_shapes = []
    input_calls = []
    hook = policy.actor.trunk.register_forward_pre_hook(
        lambda _module, inputs: trunk_shapes.append(tuple(inputs[0].shape))
    )

    def team_inputs(observation, device):
        input_calls.append(len(observation["state"]))
        return (
            torch.as_tensor(observation["spatial"][:, 0], device=device),
            torch.as_tensor(observation["state"], device=device),
        )

    batch, _, info = collect_rollout(
        env,
        env.reset(),
        policy,
        rollout_steps=3,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        team_inputs=team_inputs,
    )
    hook.remove()
    assert info["env_steps"] == 6
    assert batch["actions"].shape == (6, 3)
    assert input_calls == [2, 2, 2, 2]
    assert trunk_shapes == [(2, 863), (2, 863), (2, 863)]


def test_native_evaluation_preserves_required_metrics_and_rejects_missing_fields():
    required_metrics = {
        "team_score": 20,
        "meals_completed": 1,
        "joint_completed": 1,
        "joint_attempt_success_rate": 0.75,
        "joint_wait_steps": 4,
        "completion_rate": 0.5,
    }

    class EvalEnv(_FakeEnv):
        def step(self, actions):
            assert actions.shape == (self.num_envs, 3)
            info = {
                "metrics": dict(required_metrics),
                "episode_summary": {**required_metrics, "training_reward": 2.0},
            }
            return self.observation, np.zeros(self.num_envs, dtype=np.float32), np.ones(self.num_envs, bool), [info]

    actor = build_policy("mlp").actor
    result = evaluate_policy(
        actor,
        env_factory=lambda **kwargs: EvalEnv(kwargs["num_envs"]),
        num_episodes=1,
        seed=7,
        device="cpu",
    )
    assert result["mean_team_score"] == 20.0
    assert result["mean_meals_completed"] == 1.0
    assert result["mean_joint_completed"] == 1.0
    assert result["mean_joint_attempt_success_rate"] == 0.75
    assert result["mean_joint_wait_steps"] == 4.0
    assert result["episode_metrics"] == [required_metrics]
    assert "mean_completion_rate" not in result
    assert actor.training

    class MissingMetricEnv(EvalEnv):
        def step(self, actions):
            observation, rewards, dones, infos = super().step(actions)
            del infos[0]["metrics"]["joint_wait_steps"]
            return observation, rewards, dones, infos

    with pytest.raises(ValueError, match="joint_wait_steps"):
        evaluate_policy(
            actor,
            env_factory=lambda **kwargs: MissingMetricEnv(kwargs["num_envs"]),
            num_episodes=1,
            seed=7,
            device="cpu",
        )


def test_step_budget_counts_bc_steps_once_and_rounds_to_full_updates():
    rl_args = parse_args([
        "--mode", "rl", "--backbone", "mlp", "--output", "unused",
        "--steps", "2000000",
    ])
    bc_args = parse_args([
        "--mode", "bc_rl", "--backbone", "mlp", "--output", "unused",
        "--steps", "2000000", "--demonstrations", "demos.npz",
    ])
    rl = derive_run_config(rl_args)
    bc = derive_run_config(bc_args, 9600)
    assert rl["effective_total_env_steps"] == 2_000_000
    assert rl["num_updates"] == 417
    assert rl["last_rollout_vector_steps"] == 200
    assert bc["demo_env_steps"] == 9600
    assert bc["effective_rl_env_steps"] == 1_990_400
    assert bc["effective_total_env_steps"] == 2_000_000
    assert bc["num_updates"] == 415
    assert bc["last_rollout_vector_steps"] == 200

    default_rl = parse_args(["--mode", "rl", "--output", "unused"])
    default_config = derive_run_config(default_rl)
    assert default_rl.steps == 20_000_000
    assert default_rl.max_seconds == 0
    assert default_config["effective_total_env_steps"] == 20_000_000
    assert default_config["num_updates"] == 4167
    assert default_config["last_rollout_vector_steps"] == 200
    assert default_config["checkpoint_interval"] == 1_000_000
    assert default_rl.memory_fraction == 0.008

    large_args = parse_args([
        "--mode", "rl", "--model-size", "large", "--output", "unused",
    ])
    assert large_args.memory_fraction == 0.035
    assert derive_run_config(large_args)["model_size"] == "large"

    million_args = parse_args([
        "--mode", "rl", "--model-size", "million", "--output", "unused",
        "--steps", "100000000",
    ])
    million_config = derive_run_config(million_args)
    assert million_args.memory_fraction == 0.035
    assert million_config["effective_total_env_steps"] == 100_000_000
    assert million_config["num_updates"] == 20_834
    assert million_config["last_rollout_vector_steps"] == 100
    assert million_config["max_seconds"] == 0

    custom = parse_args([
        "--mode", "rl", "--backbone", "mlp", "--output", "unused", "--steps", "8",
        "--device", "cpu", "--epochs", "3", "--minibatches", "4", "--lr", "0.0002",
        "--checkpoint-interval", "16", "--eval-interval", "24", "--eval-episodes", "5",
        "--num-envs", "2", "--rollout", "4", "--reward-profile", "legacy",
    ])
    custom_config = derive_run_config(custom)
    assert custom_config["device"] == "cpu"
    assert custom_config["epochs"] == 3
    assert custom_config["minibatches"] == 4
    assert custom_config["learning_rate"] == 0.0002
    assert custom_config["checkpoint_interval"] == 16
    assert custom_config["evaluation_interval"] == 24
    assert custom_config["evaluation_episodes"] == 5
    assert custom_config["reward_profile"] == "legacy"


def test_thresholds_fire_once_at_update_boundaries_and_archives_are_preserved():
    schedule = StepThresholdSchedule(1_000_000)
    assert schedule.due(999_999) == []
    assert schedule.due(1_000_000) == [1_000_000]
    assert schedule.due(1_000_000) == []
    assert schedule.due(20_000_000) == list(range(2_000_000, 20_000_001, 1_000_000))
    hundred_m_schedule = StepThresholdSchedule(1_000_000)
    hundred_m_archives = hundred_m_schedule.due(100_000_000)
    assert len(hundred_m_archives) == 100
    assert hundred_m_archives[0] == 1_000_000
    assert hundred_m_archives[-1] == 100_000_000

    class Runtime:
        @staticmethod
        def atomic_save(path, payload):
            torch.save(payload, path)

    policy = build_policy("mlp")
    counts = policy.parameter_counts()
    config = {"mode": "rl", "demo_env_steps": 0}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "step_50000.pt"
        assert _save_checkpoint(
            Runtime, path, policy, config=config, counts=counts, env_steps=52_800,
            rl_env_steps=52_800, reason="threshold", preserve_existing=True,
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["format"] == CHECKPOINT_FORMAT
        original = saved["policy_state_dict"]["actor.head.weight"].clone()
        with torch.no_grad():
            policy.actor.head.weight.add_(1.0)
        assert not _save_checkpoint(
            Runtime, path, policy, config=config, counts=counts, env_steps=52_800,
            rl_env_steps=52_800, reason="duplicate threshold", preserve_existing=True,
        )
        retained = torch.load(path, map_location="cpu", weights_only=False)
        assert torch.equal(retained["policy_state_dict"]["actor.head.weight"], original)


def test_rl_path_does_not_load_or_apply_demonstrations():
    calls = []
    device_calls = []
    env_calls = []

    def setup_device(**kwargs):
        device_calls.append(kwargs)
        return torch.device(kwargs["device"])

    def kitchen_batch(**kwargs):
        env_calls.append(kwargs)
        return _FakeEnv(kwargs["num_envs"])

    runtime = SimpleNamespace(
        atomic_json=lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8"),
        atomic_save=lambda path, payload: torch.save(payload, path),
        source_hashes=lambda method: {},
        prepare_output=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        setup_device=setup_device,
        team_inputs=lambda observation, device: (
            torch.as_tensor(observation["spatial"][:, 0], device=device),
            torch.as_tensor(observation["state"], device=device),
        ),
    )
    demonstrations = SimpleNamespace(
        load_joint_dataset=lambda path: calls.append(("load", path)),
        warm_start_joint=lambda *args, **kwargs: calls.append(("bc", kwargs)),
    )
    envs = SimpleNamespace(
        REWARD_PROFILES={
            "legacy": {"description": "native reward profile"},
            "ppo_task": {"native_scale": 0.05},
        },
        KitchenBatch=kitchen_batch,
    )
    dependencies = SimpleNamespace(runtime=runtime, demonstrations=demonstrations, envs=envs)
    evaluator = lambda actor, **kwargs: {
        "episodes": 12,
        "mean_team_score": 0.0,
        "mean_meals_completed": 0.0,
        "mean_joint_completed": 0.0,
        "mean_joint_attempt_success_rate": 0.0,
        "mean_joint_wait_steps": 0.0,
    }

    with TemporaryDirectory() as directory:
        args = parse_args([
            "--mode", "rl", "--backbone", "mlp", "--output", str(Path(directory) / "run"),
            "--steps", "8", "--num-envs", "2", "--rollout", "4", "--workers", "0",
            "--device", "cpu", "--physical-gpu", "2", "--reward-profile", "legacy",
        ])
        result = train_experiment(
            args,
            dependencies=dependencies,
            evaluator=evaluator,
        )
        assert result["env_steps"] == 8
        assert device_calls[0]["device"] == "cpu"
        assert device_calls[0]["physical_gpu"] == 2
        assert env_calls[0]["reward_profile"] == "legacy"
        assert calls == []
        assert (Path(args.output) / "initial.pt").exists()
        assert (Path(args.output) / "latest.pt").exists()
        assert (Path(args.output) / "final.pt").exists()
        assert not (Path(args.output) / "train.log").exists()


def test_bc_rl_calls_joint_warm_start_and_counts_demo_steps_once():
    calls = []
    runtime = SimpleNamespace(
        atomic_json=lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8"),
        atomic_save=lambda path, payload: torch.save(payload, path),
        source_hashes=lambda method: {},
        prepare_output=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        team_inputs=lambda observation, device: (
            torch.as_tensor(observation["spatial"][:, 0], device=device),
            torch.as_tensor(observation["state"], device=device),
        ),
    )
    dataset = {
        "state": np.zeros((4, 863), dtype=np.float32),
        "spatial": np.zeros((4, 26, 7, 7), dtype=np.float32),
        "actions": np.zeros((4, 3), dtype=np.int64),
        "validation": np.array([False, False, True, True]),
        "episode_lengths": np.array([2, 2], dtype=np.int64),
        "env_steps": 2,
    }
    envs = SimpleNamespace(
        REWARD_PROFILES={"ppo_task": {"native_scale": 0.05}},
        KitchenBatch=lambda **kwargs: _FakeEnv(kwargs["num_envs"]),
    )

    def load_joint_dataset(path):
        calls.append(("load", path))
        return {**dataset, "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    def warm_start_joint(model, forward_logits, **kwargs):
        calls.append(("bc", kwargs))
        logits = forward_logits(torch.zeros(2, 26, 7, 7), torch.zeros(2, 863))
        assert logits.shape == (2, 3, 6)
        return {"demo_env_steps": 2, "epochs": kwargs["epochs"], "train_ce": 1.0,
                "validation_accuracy": 0.5, "dataset_sha256": dataset["dataset_sha256"]}

    demonstrations = SimpleNamespace(
        load_joint_dataset=load_joint_dataset,
        warm_start_joint=warm_start_joint,
    )
    dependencies = SimpleNamespace(runtime=runtime, demonstrations=demonstrations, envs=envs)
    eval_count = []

    def evaluator(actor, **kwargs):
        eval_count.append(1)
        return {
            "episodes": 12,
            "mean_team_score": [0., 380., 0.][len(eval_count) - 1],
            "mean_meals_completed": 0.0,
            "mean_joint_completed": 0.0,
            "mean_joint_attempt_success_rate": 0.0,
            "mean_joint_wait_steps": 0.0,
        }

    with TemporaryDirectory() as directory:
        demo_path = Path(directory) / "demo.npz"
        demo_path.write_bytes(b"small immutable test dataset")
        dataset["dataset_sha256"] = hashlib.sha256(demo_path.read_bytes()).hexdigest()
        args = parse_args([
            "--mode", "bc_rl", "--backbone", "mlp", "--output", str(Path(directory) / "run"),
            "--steps", "4", "--num-envs", "1", "--rollout", "2", "--workers", "0",
            "--demonstrations", str(demo_path),
        ])
        result = train_experiment(
            args,
            dependencies=dependencies,
            device_override="cpu",
            evaluator=evaluator,
        )
        assert result["env_steps"] == 4
        assert result["rl_env_steps"] == 2
        assert len(eval_count) == 3
        assert [call[0] for call in calls] == ["load", "bc"]
        assert calls[1][1]["epochs"] == 64
        assert calls[1][1]["batch_size"] == 512
        assert (Path(args.output) / "bc_initial.pt").exists()
        best = torch.load(Path(args.output) / "best.pt", map_location="cpu", weights_only=False)
        assert best["env_steps"] == 2
        bc_initial = torch.load(Path(args.output) / "bc_initial.pt", map_location="cpu", weights_only=False)
        assert bc_initial["selection"]["native_evaluation"]["mean_team_score"] == 380.0
        final = torch.load(Path(args.output) / "final.pt", map_location="cpu", weights_only=False)
        assert final["env_steps"] == 4
        assert final["selection"]["mean_team_score"] == 0.0
        assert best["selection"]["mean_team_score"] == 380.0
        assert result["final_evaluation"]["mean_team_score"] == 0.0

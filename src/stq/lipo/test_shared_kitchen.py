"""Focused SharedKitchen contract, update, metric, and checkpoint checks."""

from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from lipo.models import (
    LipoMember,
    LipoPopulation,
    ModelConfig,
    SHARED_KITCHEN_COMMIT,
    load_population,
)
from lipo.train import (
    EPISODE_SUMMARY_FIELDS,
    LipoTrainer,
    Rollout,
    SHARED_KITCHEN_METRICS,
    _aggregate_summaries,
    _behavior_cloning_forward_logits,
    _warmup_scale,
    build_parser,
)


def _summary(offset=0):
    values = {
        "team_score": 40,
        "meals_completed": 2,
        "normal_completed": 1,
        "joint_completed": 1,
        "joint_attempt_steps": 3,
        "joint_successes": 1,
        "joint_attempt_success_rate": 0.5,
        "joint_wait_steps": 3,
        "three_contributor_meals": 1,
        "resource_conflicts": 4,
        "movement_conflicts": 5,
        "ticks": 100,
        "meals_per_100_steps": 2.0,
        "episode": 7,
        "training_reward": 43.5,
        "training_shaping": 1.5,
    }
    assert set(EPISODE_SUMMARY_FIELDS).issubset(values)
    return {key: value + offset for key, value in values.items()}


def _trainer_args(output):
    return SimpleNamespace(
        population=2,
        xp_candidates=1,
        seed=11,
        lr=1e-3,
        checkpoint_interval=250_000,
        epochs=1,
        minibatch_size=2,
        clip_param=0.3,
        xp_coef=0.3,
        ent_coef=0.01,
        mi_coef=0.1,
        vf_coef=0.5,
        grad_clip=0.5,
        steps=1,
        num_envs=1,
        rollout=1,
        eval_interval=1,
        eval_episodes=1,
        max_seconds=1,
        workers=0,
        gamma=0.99,
        gae_lambda=0.95,
        smoke=False,
        output=Path(output),
        demonstrations=Path("demo.npz"),
        bc_epochs=8,
    )


def _rollouts():
    rng = np.random.default_rng(5)
    time_steps, envs = 2, 2
    sp = Rollout(
        obs=rng.normal(size=(time_steps, envs, 3, 866)).astype(np.float32),
        latent_ids=rng.integers(0, 8, size=(time_steps, envs, 3)),
        actions=rng.integers(0, 6, size=(time_steps, envs, 3)),
        old_log_probs=np.full((time_steps, envs, 3), -np.log(6), dtype=np.float32),
        old_action_probs=np.full((time_steps, envs, 3, 6), 1 / 6, dtype=np.float32),
        states=rng.normal(size=(time_steps, envs, 863)).astype(np.float32),
        old_values=np.zeros((time_steps, envs), dtype=np.float32),
        rewards=np.ones((time_steps, envs), dtype=np.float32),
        dones=np.zeros((time_steps, envs), dtype=np.bool_),
        advantages=np.ones((time_steps, envs), dtype=np.float32),
        returns=np.ones((time_steps, envs), dtype=np.float32),
        mean_reward=1.0,
    )
    xp = Rollout(
        obs=rng.normal(size=(time_steps, envs, 866)).astype(np.float32),
        latent_ids=rng.integers(0, 8, size=(time_steps, envs)),
        actions=rng.integers(0, 6, size=(time_steps, envs)),
        old_log_probs=np.full((time_steps, envs), -np.log(6), dtype=np.float32),
        old_action_probs=np.full((time_steps, envs, 6), 1 / 6, dtype=np.float32),
        states=rng.normal(size=(time_steps, envs, 863)).astype(np.float32),
        old_values=np.zeros((time_steps, envs), dtype=np.float32),
        rewards=np.ones((time_steps, envs), dtype=np.float32),
        dones=np.zeros((time_steps, envs), dtype=np.bool_),
        advantages=np.ones((time_steps, envs), dtype=np.float32),
        returns=np.ones((time_steps, envs), dtype=np.float32),
        role=0,
        partner=1,
        mean_reward=1.0,
    )
    return sp, xp


def test_shared_kitchen_dimensions_and_inference_shape():
    config = ModelConfig(population=2)
    assert config.obs_dim == 866
    assert config.state_dim == 863
    assert config.hidden_dim == 64 and config.hidden_layers == 2

    population = LipoPopulation([LipoMember(config), LipoMember(config)])
    observations = np.zeros((2, 3, 866), dtype=np.float32)
    actions = population.act(0, observations, np.zeros((2, 3), dtype=np.int64))
    assert actions.shape == (2, 3)
    with pytest.raises(ValueError, match="Expected observations"):
        population.act(0, np.zeros((2, 3, 865), dtype=np.float32))


def test_bc_callback_uses_vector_role_bits_and_random_latent_ids():
    member = LipoMember(ModelConfig(population=2))
    callback = _behavior_cloning_forward_logits(member, seed=23)
    roles = torch.arange(32) % 3
    vector = torch.zeros(32, 866)
    vector[:, -3:] = torch.nn.functional.one_hot(roles, num_classes=3)
    spatial = torch.zeros(32, 26, 7, 7)
    captured = {}
    original_actor_logits = member.actor_logits

    def capture_actor_logits(observations, role_ids, latent_ids):
        captured["roles"] = role_ids.detach().clone()
        captured["latents"] = latent_ids.detach().clone()
        return original_actor_logits(observations, role_ids, latent_ids)

    member.actor_logits = capture_actor_logits
    logits = callback(spatial, vector)
    assert logits.shape == (32, 6)
    assert torch.equal(captured["roles"], roles)
    assert torch.all((captured["latents"] >= 0) & (captured["latents"] < 8))
    assert torch.unique(captured["latents"]).numel() > 1


def test_observation_contract_checks_vector_state_and_spatial_shapes(tmp_path):
    trainer = LipoTrainer(_trainer_args(tmp_path), torch.device("cpu"), tmp_path)
    valid = {
        "vector": np.zeros((1, 3, 866), dtype=np.float32),
        "state": np.zeros((1, 863), dtype=np.float32),
        "spatial": np.zeros((1, 3, 26, 7, 7), dtype=np.float32),
    }
    obs, state = trainer._check_observation(valid)
    assert obs.shape == (1, 3, 866)
    assert state.shape == (1, 863)
    invalid = dict(valid, spatial=np.zeros((1, 3, 26, 7, 11), dtype=np.float32))
    with pytest.raises(ValueError, match="spatial observations"):
        trainer._check_observation(invalid)


def test_checkpoint_interval_can_be_overridden_from_250k_default(tmp_path):
    default_args = build_parser().parse_args(["--demonstrations", "dataset.npz"])
    assert default_args.checkpoint_interval == 250_000
    assert default_args.bc_epochs == 64
    overridden_args = build_parser().parse_args([
        "--demonstrations", "dataset.npz", "--checkpoint-interval", "1",
    ])
    assert overridden_args.checkpoint_interval == 1

    args = _trainer_args(tmp_path)
    args.checkpoint_interval = 1
    trainer = LipoTrainer(args, torch.device("cpu"), tmp_path)
    assert trainer.checkpoint_schedule.interval_steps == 1
    assert trainer._run_manifest()["checkpoint_policy"]["interval_steps"] == 1


def test_native_metric_aggregation_keeps_rate_and_training_fields():
    aggregate = _aggregate_summaries([_summary(), _summary(2)])
    assert aggregate["team_score_mean"] == 41
    assert aggregate["joint_attempt_success_rate_mean"] == 1.5
    assert aggregate["training_reward_mean"] == 44.5
    assert aggregate["training_shaping_mean"] == 2.5
    assert all(f"{key}_mean" in aggregate for key in SHARED_KITCHEN_METRICS)


def test_xp_and_mi_coefficients_warm_up_over_first_ten_percent():
    assert _warmup_scale(0, 1000) == 0
    assert _warmup_scale(4, 80) == 0.5
    assert _warmup_scale(50, 1000) == 0.5
    assert _warmup_scale(100, 1000) == 1
    assert _warmup_scale(500, 1000) == 1


def test_one_shared_kitchen_lipo_update_is_finite_and_changes_parameters(tmp_path):
    trainer = LipoTrainer(_trainer_args(tmp_path), torch.device("cpu"), tmp_path)
    sp, xp = _rollouts()
    original = trainer.members[0].actor[0].weight.detach().clone()
    metrics = trainer.update_member(0, sp, [xp], selected_xp_index=0)
    assert set(metrics) == {"loss", "sp_pg", "xp_pg", "sp_value", "xp_value", "mi", "entropy"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(original, trainer.members[0].actor[0].weight.detach())


def test_shared_kitchen_evaluation_aggregates_native_metrics(tmp_path):
    trainer = LipoTrainer(_trainer_args(tmp_path), torch.device("cpu"), tmp_path)

    class FakeKitchenBatch:
        def _observation(self):
            return {
                "vector": np.zeros((1, 3, 866), dtype=np.float32),
                "state": np.zeros((1, 863), dtype=np.float32),
                "spatial": np.zeros((1, 3, 26, 7, 7), dtype=np.float32),
            }

        def reset(self):
            return self._observation()

        def step(self, actions):
            assert np.asarray(actions).shape == (1, 3)
            return self._observation(), np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.bool_), [
                {"episode_summary": _summary()}
            ]

    validation = trainer.evaluate(FakeKitchenBatch(), eval_episodes=1)
    assert validation["mean_team_score"] == 40
    assert validation["mean_native_metrics"]["joint_attempt_success_rate"] == 0.5
    assert validation["mean_episode_summary"]["training_shaping"] == 1.5
    assert validation["mean_episode_summary"]["joint_attempt_steps"] == 3
    assert validation["members"][0]["training_reward_mean"] == 43.5


def test_run_warm_starts_before_rl_and_counts_demo_steps_once(tmp_path, monkeypatch):
    demo_path = tmp_path / "demos.npz"
    sample_count = 6
    vector = np.zeros((sample_count, 866), dtype=np.float32)
    vector[np.arange(sample_count), 863 + np.arange(sample_count) % 3] = 1.0
    actions = np.arange(sample_count, dtype=np.int64) % 6
    np.savez(
        demo_path,
        spatial=np.zeros((sample_count, 26, 7, 7), dtype=np.float32),
        vector=vector,
        actions=actions,
        validation=np.asarray([False, False, False, False, True, True]),
        env_steps=np.asarray(100),
    )

    class FakeKitchenBatch:
        def __init__(self, num_envs, **kwargs):
            self.num_envs = num_envs

        def _observation(self):
            return {
                "vector": np.zeros((self.num_envs, 3, 866), dtype=np.float32),
                "state": np.zeros((self.num_envs, 863), dtype=np.float32),
                "spatial": np.zeros((self.num_envs, 3, 26, 7, 7), dtype=np.float32),
            }

        def reset(self):
            return self._observation()

        def set_stage(self, stage):
            assert stage == "standard"

        def step(self, actions):
            assert np.asarray(actions).shape == (self.num_envs, 3)
            return (
                self._observation(),
                np.zeros(self.num_envs, dtype=np.float32),
                np.ones(self.num_envs, dtype=np.bool_),
                [{"episode_summary": _summary()} for _ in range(self.num_envs)],
            )

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    fake_envs = ModuleType("baseline_common.envs")
    fake_envs.KitchenBatch = FakeKitchenBatch
    fake_envs.curriculum = lambda progress: "standard"
    monkeypatch.setitem(sys.modules, "baseline_common.envs", fake_envs)
    args = _trainer_args(tmp_path / "run")
    args.demonstrations = demo_path
    args.steps = 104
    args.max_seconds = 30
    trainer = LipoTrainer(args, torch.device("cpu"), args.output)
    assert trainer.run() == "completed"

    initial = torch.load(args.output / "initial.pt", map_location="cpu", weights_only=False)
    bc_initial = torch.load(args.output / "bc_initial.pt", map_location="cpu", weights_only=False)
    assert initial["total_env_steps"] == 0
    assert bc_initial["demonstration_env_steps"] == 100
    assert bc_initial["sp_env_steps"] == bc_initial["xp_env_steps"] == 0
    assert bc_initial["total_env_steps"] == 100
    assert trainer.total_env_steps == 104
    assert trainer.sp_env_steps + trainer.xp_env_steps == 4
    metric_rows = [
        json.loads(line)
        for line in (args.output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_rows = [row for row in metric_rows if row.get("kind") == "train"]
    assert len(train_rows) == 1
    assert train_rows[0]["xp_coef"] == 0.0
    assert train_rows[0]["mi_coef"] == 0.0
    assert torch.equal(initial["member_states"][0]["model"]["actor.0.weight"],
                       bc_initial["member_states"][0]["model"]["actor.0.weight"]) is False


def test_loader_accepts_only_shared_kitchen_v2_checkpoints(tmp_path):
    config = ModelConfig(population=2)
    members = [LipoMember(config) for _ in range(2)]
    path = tmp_path / "shared.pt"
    torch.save({
        "format_version": 2,
        "environment": "SharedKitchen",
        "environment_commit": SHARED_KITCHEN_COMMIT,
        "model_config": config.__dict__,
        "member_states": [member.state_dict() for member in members],
    }, path)
    loaded = load_population(path)
    assert len(loaded.members) == 2

    old_path = tmp_path / "mixed.pt"
    torch.save({
        "format_version": 1,
        "model_config": config.__dict__,
        "member_states": [],
    }, old_path)
    with pytest.raises(ValueError, match="SharedKitchen LIPO v2"):
        load_population(old_path)

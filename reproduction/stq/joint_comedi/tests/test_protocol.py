from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch

import joint_comedi.algorithm as algorithm
from joint_comedi.model import (
    ACTOR_PARAMETER_COUNT,
    ComediPolicy,
    JointActor,
    build_actor,
    compute_gae,
    ppo_actor_loss,
)


class FakeKitchen:
    def __init__(self, num_envs=2):
        self.num_envs = int(num_envs)
        self.ticks = np.zeros(self.num_envs, dtype=np.int64)
        self.last_actions = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.closed = False

    def reset(self):
        self.ticks[:] = 0
        self.last_actions[:] = 0
        return self.observation()

    def observation(self):
        state = np.zeros((self.num_envs, 863), dtype=np.float32)
        state[:, 0] = self.ticks
        vector = np.zeros((self.num_envs, 3, 866), dtype=np.float32)
        vector[:, :, 0] = self.ticks[:, None]
        vector[:, :, 1] = self.last_actions
        spatial = np.zeros((self.num_envs, 3, 26, 7, 7), dtype=np.float32)
        for role in range(3):
            spatial[:, role, 12 + role * 4, 1 + role, 1] = 1.0
        return {"state": state, "vector": vector, "spatial": spatial}

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, 3)
        assert np.issubdtype(actions.dtype, np.integer)
        self.last_actions = actions.astype(np.float32)
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
                    "resource_conflicts": 0,
                    "movement_conflicts": 0,
                    "ticks": 4,
                    "meals_per_100_steps": 25.0,
                }
                infos.append({"episode_summary": {
                    **metrics, "episode": 0, "training_reward": 4.0, "training_shaping": 0.0,
                }})
                self.ticks[env_id] = 0
                self.last_actions[env_id] = 0
            else:
                infos.append({})
        return self.observation(), np.ones(self.num_envs, dtype=np.float32), dones, infos

    def close(self):
        self.closed = True


def test_actor_is_one_central_encoder_with_exact_three_head_shape_and_count():
    actor = JointActor()
    assert sum(parameter.numel() for parameter in actor.parameters()) == ACTOR_PARAMETER_COUNT
    seen = []
    hook = actor.conv.register_forward_pre_hook(lambda _module, inputs: seen.append(tuple(inputs[0].shape)))
    logits = actor(torch.zeros(5, 26, 7, 7), torch.zeros(5, 863))
    hook.remove()
    assert logits.shape == (5, 3, 6)
    assert seen == [(5, 26, 7, 7)]


def test_sampling_encodes_one_team_row_per_environment_not_three_ego_rows():
    actor = build_actor()
    env = FakeKitchen(4)
    seen = []
    hook = actor.conv.register_forward_pre_hook(lambda _module, inputs: seen.append(tuple(inputs[0].shape)))
    actions, log_probs = algorithm.sample_joint_actions(actor, env.reset(), "cpu")
    hook.remove()
    assert actions.shape == (4, 3)
    assert log_probs.shape == (4, 3)
    assert seen == [(4, 26, 7, 7)]


def test_self_play_rollout_keeps_joint_frames_and_single_actor_pass_per_step():
    env = FakeKitchen(2)
    policy = ComediPolicy(0, lr=1e-3)
    seen = []
    hook = policy.actor.conv.register_forward_pre_hook(lambda _module, inputs: seen.append(tuple(inputs[0].shape)))
    batch, _observation, info = algorithm.collect_self_play(
        env, policy, 3, device=torch.device("cpu"), gamma=0.99, gae_lambda=0.95
    )
    hook.remove()
    assert info["env_steps"] == 6
    assert batch["spatial"].shape == (6, 26, 7, 7)
    assert batch["state"].shape == (6, 863)
    assert batch["actions"].shape == (6, 3)
    assert batch["log_probs"].shape == (6, 3)
    assert batch["advantages"].shape == (6, 3)
    assert seen == [(2, 26, 7, 7)] * 3


def test_cross_play_uses_six_role_masks_and_role_local_validity():
    assert algorithm.CROSSPLAY_ROLE_MASKS.shape == (6, 3)
    assert len({tuple(mask.tolist()) for mask in algorithm.CROSSPLAY_ROLE_MASKS}) == 6
    assert sorted(algorithm.CROSSPLAY_ROLE_MASKS.sum(axis=1).tolist()) == [1, 1, 1, 2, 2, 2]

    env = FakeKitchen(2)
    current = ComediPolicy(1, lr=1e-3)
    prior = build_actor().cpu().eval()
    current_seen, prior_seen = [], []
    current_hook = current.actor.conv.register_forward_pre_hook(
        lambda _module, inputs: current_seen.append(tuple(inputs[0].shape))
    )
    prior_hook = prior.conv.register_forward_pre_hook(
        lambda _module, inputs: prior_seen.append(tuple(inputs[0].shape))
    )
    batch, _obs, info = algorithm.collect_cross_play(
        env, current, prior, 0, 1, device="cpu", gamma=0.99, gae_lambda=0.95,
        start_config=0,
    )
    current_hook.remove()
    prior_hook.remove()
    np.testing.assert_array_equal(batch["valid"], algorithm.CROSSPLAY_ROLE_MASKS[:2])
    assert info["env_steps"] == 2
    assert batch["actions"].shape == (2, 3)
    assert current_seen == [(2, 26, 7, 7)]
    assert prior_seen == [(2, 26, 7, 7)]


def test_mp_prefix_is_invalid_and_only_all_current_suffix_is_trainable():
    env = FakeKitchen(2)
    current = ComediPolicy(1, lr=1e-3)
    prior = build_actor().cpu().eval()
    batch, _obs, info = algorithm.collect_mixed_play(
        env, current, prior, 4, device="cpu", gamma=0.99, gae_lambda=0.95,
        rng=np.random.default_rng(2), prefix_max=64, current_probability=0.5,
    )
    assert batch["actions"].shape == (8, 3)
    assert batch["valid"].shape == (8, 3)
    assert info["prefix_env_steps"] > 0
    assert info["suffix_env_steps"] > 0
    assert np.all(batch["valid"] == batch["valid"][:, :1])
    assert np.all(batch["advantages"][~batch["valid"]] == 0.0)


def test_teacher_self_play_uses_each_teachers_own_joint_states_and_labels():
    class CodedActor:
        def __init__(self, code):
            self.code = int(code)

        def parameters(self):
            return iter(())

    env = FakeKitchen(3)
    teachers = [CodedActor(i + 1) for i in range(3)]
    seen = []

    def fake_sample(actor, observation, device, *, deterministic=False):
        count = len(observation["state"])
        actions = np.full((count, 3), actor.code, dtype=np.int64)
        seen.append((actor.code, count))
        return actions, np.zeros((count, 3), dtype=np.float32)

    original = algorithm.sample_joint_actions
    algorithm.sample_joint_actions = fake_sample
    try:
        batches, info = algorithm.collect_teacher_self_play(
            env, teachers, 2, device="cpu", start_teacher=0
        )
    finally:
        algorithm.sample_joint_actions = original
    assert set(batches) == {0, 1, 2}
    assert info["teacher_env_steps"] == {"0": 2, "1": 2, "2": 2}
    assert seen == [(1, 1), (2, 1), (3, 1)] * 2
    for teacher_id, batch in batches.items():
        assert batch["spatial"].shape == (2, 26, 7, 7)
        assert batch["actions"].shape == (2, 3)
        assert np.all(batch["actions"] == teacher_id + 1)
        assert batch["valid"].shape == (2,)


def test_gae_role_masks_cut_invalid_prefix_advantages():
    rewards = np.array([[1.0], [10.0]], dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    dones = np.array([[False], [False]])
    valid = np.array([[False], [True]])
    advantages, _returns = compute_gae(
        rewards, values, dones, np.zeros(1, dtype=np.float32),
        gamma=1.0, gae_lambda=1.0, valid=valid,
    )
    np.testing.assert_allclose(advantages[:, 0], [0.0, 10.0])


def test_xp_negative_sign_precedes_clipping_and_entropy_is_unsigned():
    loss = ppo_actor_loss(
        torch.tensor([np.log(1.5)]), torch.tensor([0.0]), torch.tensor([1.0]),
        clip=0.2, weight=-0.25,
    )
    assert loss.item() == pytest.approx(0.375)
    with_entropy = ppo_actor_loss(
        torch.tensor([np.log(1.5)]), torch.tensor([0.0]), torch.tensor([1.0]),
        clip=0.2, weight=-0.25, entropy=torch.tensor([1.0]), entropy_coef=0.1,
    )
    assert with_entropy.item() == pytest.approx(loss.item() - 0.1)


def _update_batch(frames=3, *, phase="sp"):
    batch = {
        "spatial": np.zeros((frames, 26, 7, 7), dtype=np.float32),
        "state": np.zeros((frames, 863), dtype=np.float32),
        "critic_spatial": np.zeros((frames, 3, 26, 7, 7), dtype=np.float32),
        "critic_vector": np.zeros((frames, 3, 866), dtype=np.float32),
        "actions": np.zeros((frames, 3), dtype=np.int64),
        "log_probs": np.full((frames, 3), -np.log(6.0), dtype=np.float32),
        "advantages": np.ones((frames, 3), dtype=np.float32),
        "returns": np.ones((frames, 3), dtype=np.float32),
        "valid": np.ones((frames, 3), dtype=bool),
    }
    if phase == "xp":
        batch["prior_id"] = 0
    return batch


def test_sp_xp_mp_update_accumulates_before_one_actor_step_and_updates_role_critics():
    policy = ComediPolicy(1, lr=1e-3)
    before = [parameter.detach().clone() for parameter in policy.actor.parameters()]
    metrics = algorithm.update_bundle(
        policy,
        (("sp", _update_batch(3), 1.0),
         ("xp", _update_batch(3, phase="xp"), -0.5),
         ("mp", _update_batch(3, phase="mp"), 1.0)),
        device="cpu", epochs=1, minibatches=1, clip=0.2, entropy_coef=0.001,
        value_coef=0.5, max_grad_norm=0.5, prior_count=1,
        rng=np.random.default_rng(3),
    )
    actor_step = next(iter(policy.actor_optimizer.state.values()))["step"]
    assert actor_step.item() == 1
    assert set(metrics["samples_by_phase"]) == {"sp", "xp", "mp"}
    assert all(metrics["samples_by_phase"][phase] == 9 for phase in ("sp", "xp", "mp"))
    assert set(metrics["value_loss_by_phase"]) == {"sp", "xp", "mp"}
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.actor.parameters()))


def test_teacher_bc_is_joint_role_balanced_and_does_not_train_critics():
    policy = ComediPolicy(0, lr=1e-3)
    batch = _update_batch(3)
    teacher = {
        "spatial": np.zeros((2, 26, 7, 7), dtype=np.float32),
        "state": np.zeros((2, 863), dtype=np.float32),
        "bc_actions": np.ones((2, 3), dtype=np.int64),
        "valid": np.ones(2, dtype=bool),
    }
    metrics = algorithm.update_bundle(
        policy,
        (("sp", batch, 1.0),),
        device="cpu", epochs=1, minibatches=1, clip=0.2, entropy_coef=0.0,
        value_coef=0.5, max_grad_norm=0.5, prior_count=0,
        bc_coef=0.5, bc_objectives=((7, teacher),), rng=np.random.default_rng(4),
    )
    assert metrics["bc_samples_by_teacher"] == {"7": 6}
    assert metrics["bc_loss"] > 0.0
    assert set(metrics["value_loss_by_phase"]) == {"sp"}

import numpy as np
import pytest
import torch

from comedi import algorithm
from comedi.algorithm import CROSSPLAY_ROLE_MASKS, update_bundle
from comedi.model import (
    ConvActor,
    ComediPolicy,
    compute_gae,
    joint_action_batch,
    masked_rows,
    ppo_actor_loss,
    select_most_compatible_prior,
)
from comedi.train import allocate_budgets, effective_args, parser


def test_actor_encoder_matches_shared_kitchen_schema():
    actor = ConvActor()
    assert actor.encoder.conv.in_channels == 26
    assert actor.encoder.conv.out_channels == 32
    assert actor.encoder.conv.kernel_size == (3, 3)
    assert actor.encoder.fc1.in_features == 32 * 5 * 5 + 866
    assert actor.encoder.fc1.out_features == 64
    assert actor.encoder.fc2.in_features == actor.encoder.fc2.out_features == 64
    logits = actor(torch.zeros((2, 26, 7, 7)), torch.zeros((2, 866)))
    assert logits.shape == (2, 6)


def test_demonstration_steps_are_reserved_once_from_total_budget():
    args = effective_args(parser().parse_args(["--output", "unused", "--num-envs", "16"]))
    budgets = allocate_budgets(args, demonstration_env_steps=9_600)
    assert budgets["requested_total"] == 16_000_000
    assert budgets["demonstration_env_steps"] == 9_600
    assert budgets["effective_rl_total"] == 15_990_400
    assert budgets["effective_total"] == 16_000_000


def test_sp_mp_and_each_xp_role_have_independent_critics():
    policy = ComediPolicy(num_priors=2)
    assert policy.critic("sp", 0) is not policy.critic("mp", 0)
    assert policy.critic("xp", 0, 1) is not policy.critic("xp", 1, 1)
    assert policy.critic("xp", 0, 0) is not policy.critic("xp", 0, 1)


def test_frozen_old_actor_has_no_trainable_parameters():
    actor = ComediPolicy(num_priors=0).actor
    before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    actor.requires_grad_(False)
    actor.eval()
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    after = actor.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_xp_selects_only_maximum_compatibility_prior():
    returns = {0: [2.0, 3.0], 1: [8.0], 2: [1.0, 2.0]}
    assert select_most_compatible_prior(returns) == 1
    with pytest.raises(ValueError):
        select_most_compatible_prior({0: []})


def test_three_role_xp_covers_both_team_directions():
    assert CROSSPLAY_ROLE_MASKS.shape == (6, 3)
    assert len({tuple(mask.tolist()) for mask in CROSSPLAY_ROLE_MASKS}) == 6
    assert sorted(CROSSPLAY_ROLE_MASKS.sum(axis=1).tolist()) == [1, 1, 1, 2, 2, 2]


def test_negative_xp_weight_scales_advantage_before_clipping_and_keeps_entropy():
    # Official CoMeDi PPO signs advantages before clipping: min(-.375, -.3)
    # gives a loss of +.375. Entropy remains an independent exploration bonus.
    loss = ppo_actor_loss(
        torch.tensor([np.log(1.5)]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        clip=0.2,
        weight=-0.25,
    )
    assert loss.item() == pytest.approx(0.375)
    entropy_loss = ppo_actor_loss(
        torch.tensor([np.log(1.5)]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        clip=0.2,
        weight=-0.25,
        entropy=torch.tensor([1.0]),
        entropy_coef=0.1,
    )
    assert entropy_loss.item() == pytest.approx(loss.item() - 0.1)

    # The XP critic still regresses toward positive returns; only the actor
    # objective receives the negative cross-play weight.
    value = torch.tensor([0.0], requires_grad=True)
    value_loss = torch.nn.functional.mse_loss(value, torch.tensor([2.0]))
    value_loss.backward()
    assert value.grad.item() < 0.0


def test_teacher_bc_uses_each_teachers_own_self_play_actions(monkeypatch):
    class Actor:
        def __init__(self, action):
            self.action = action

    class FakeEnv:
        num_envs = 2

        def __init__(self):
            self.last_actions = np.zeros((2, 3), dtype=np.float32)
            self.seen_actions = []

        def _observation(self):
            vector = np.zeros((2, 3, 866), dtype=np.float32)
            vector[:, :, 0] = self.last_actions
            return {
                "vector": vector,
                "spatial": np.zeros((2, 3, 26, 7, 7), dtype=np.float32),
                "state": np.zeros((2, 863), dtype=np.float32),
            }

        def reset(self):
            self.last_actions.fill(0)
            return self._observation()

        def step(self, actions):
            self.seen_actions.append(np.asarray(actions).copy())
            self.last_actions = np.asarray(actions, dtype=np.float32)
            return self._observation(), np.zeros(2), np.zeros(2, dtype=bool), [{}, {}]

    def sample_teacher_actions(actor, observation, device, *, deterministic=False):
        count = observation["vector"].shape[0]
        actions = np.full((count, 3), actor.action, dtype=np.int64)
        return actions, np.zeros((count, 3), dtype=np.float32)

    monkeypatch.setattr(algorithm, "sample_joint_actions", sample_teacher_actions)
    teachers = [type("Policy", (), {"actor": Actor(1)})(), type("Policy", (), {"actor": Actor(4)})()]
    env = FakeEnv()
    batches, info = algorithm.collect_teacher_self_play(
        env, teachers, 3, device=torch.device("cpu"), start_teacher=0
    )

    assert set(batches) == {0, 1}
    assert info["teacher_env_steps"] == {"0": 3, "1": 3}
    assert all(np.all(batch["bc_actions"] == teacher_id * 3 + 1)
               for teacher_id, batch in batches.items())
    assert all(np.all(batch["actions"] == batch["bc_actions"])
               for batch in batches.values())
    assert all(np.all(action[0] == 1) and np.all(action[1] == 4)
               for action in env.seen_actions)


def test_teacher_bc_updates_actor_without_using_teacher_data_as_ppo():
    policy = ComediPolicy(num_priors=0, lr=1e-3)

    def arrays(count):
        return {
            "spatial": np.zeros((count, 26, 7, 7), dtype=np.float32),
            "vector": np.zeros((count, 866), dtype=np.float32),
            "actions": np.zeros(count, dtype=np.int64),
            "log_probs": np.full(count, -np.log(6.0), dtype=np.float32),
            "advantages": np.ones(count, dtype=np.float32),
            "returns": np.ones(count, dtype=np.float32),
            "roles": np.arange(count, dtype=np.int64) % 3,
            "valid": np.ones(count, dtype=bool),
        }

    teacher = arrays(3)
    teacher["bc_actions"] = np.ones(3, dtype=np.int64)
    metrics = update_bundle(
        policy,
        (("sp", arrays(3), 1.0),),
        bc_objectives=((7, teacher),),
        bc_coef=0.5,
        device=torch.device("cpu"), epochs=1, minibatches=1,
        clip=0.2, entropy_coef=0.0, value_coef=0.5, max_grad_norm=1.0,
        prior_count=0, rng=np.random.default_rng(4),
    )
    assert metrics["samples_by_phase"] == {"sp": 3}
    assert metrics["bc_samples_by_teacher"] == {"7": 3}
    assert set(metrics["value_loss_by_phase"]) == {"sp"}
    assert metrics["bc_loss"] > 0.0


def test_mp_prefix_rows_are_excluded_from_actor_gradient():
    logits = torch.tensor([[0.2], [0.1]], requires_grad=True)
    mask = torch.tensor([False, True])
    selected = masked_rows(logits, mask)
    loss = ppo_actor_loss(
        selected,
        torch.zeros_like(selected),
        torch.ones_like(selected),
        clip=0.2,
    )
    loss.backward()
    assert logits.grad[0].item() == 0.0
    assert logits.grad[1].item() != 0.0


def test_joint_actions_require_three_integer_role_actions():
    assert joint_action_batch(np.zeros((4, 3), dtype=np.int64), 4).shape == (4, 3)
    with pytest.raises(ValueError):
        joint_action_batch(np.zeros((4, 2), dtype=np.int64), 4)
    with pytest.raises(TypeError):
        joint_action_batch(np.zeros((4, 3), dtype=np.float32), 4)


def test_gae_stops_at_episode_boundary_and_excludes_invalid_prefix():
    rewards = np.array([[1.0], [10.0]], dtype=np.float32)
    values = np.zeros_like(rewards)
    dones = np.array([[True], [False]])
    adv, returns = compute_gae(
        rewards, values, dones, np.zeros(1, dtype=np.float32),
        gamma=1.0, gae_lambda=1.0,
    )
    assert np.allclose(adv[:, 0], [1.0, 10.0])
    assert np.allclose(returns, adv)

    valid_adv, _ = compute_gae(
        rewards, values, dones, np.zeros(1, dtype=np.float32),
        gamma=1.0, gae_lambda=1.0, valid=np.array([[False], [True]]),
    )
    assert np.allclose(valid_adv[:, 0], [0.0, 10.0])


def test_joint_bundle_accumulates_sp_xp_mp_before_one_actor_step():
    policy = ComediPolicy(num_priors=1, lr=1e-3)

    def batch(phase):
        count = 3
        result = {
            "spatial": np.zeros((count, 26, 7, 7), dtype=np.float32),
            "vector": np.zeros((count, 866), dtype=np.float32),
            "actions": np.zeros(count, dtype=np.int64),
            "log_probs": np.full(count, -np.log(6.0), dtype=np.float32),
            "advantages": np.ones(count, dtype=np.float32),
            "returns": np.ones(count, dtype=np.float32),
            "roles": np.arange(3, dtype=np.int64),
            "valid": np.ones(count, dtype=bool),
        }
        if phase == "xp":
            result["priors"] = np.zeros(count, dtype=np.int64)
        return result

    metrics = update_bundle(
        policy,
        (("sp", batch("sp"), 1.0),
         ("xp", batch("xp"), -0.5),
         ("mp", batch("mp"), 1.0)),
        device=torch.device("cpu"), epochs=1, minibatches=1,
        clip=0.2, entropy_coef=0.0, value_coef=0.5, max_grad_norm=1.0,
        prior_count=1, rng=np.random.default_rng(3),
    )
    actor_step = next(iter(policy.actor_optimizer.state.values()))["step"]
    assert actor_step.item() == 1
    assert set(metrics["samples_by_phase"]) == {"sp", "xp", "mp"}
    assert all(metrics["samples_by_phase"][phase] > 0 for phase in ("sp", "xp", "mp"))

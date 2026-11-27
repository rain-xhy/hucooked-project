import copy

import numpy as np
import torch

from .common import OptionClock
from .behavior_cloning import warm_start_hierarchical
from .model import CentralFlatActorCritic, JointHiPTPolicy
from .population import PopulationTrainer, _population_objective, staged_population_step
from .train import _configure, parser
from .trainer import HierarchicalTrainer


def _small_flat(seed):
    torch.manual_seed(seed)
    return CentralFlatActorCritic(
        state_dim=5, channels=2, height=3, width=3, hidden_size=8, head_size=6,
    )


def _prime_adam(models, optimizers, spatial, state):
    for member, (model, optimizer) in enumerate(zip(models, optimizers)):
        optimizer.zero_grad(set_to_none=True)
        logits, value = model(spatial, state)
        loss = logits.square().mean() + (member + 1) * value.square().mean()
        loss.backward()
        optimizer.step()


def _resident_step(models, optimizers, batch):
    logits, values = zip(*(model(batch["spatial"], batch["state"]) for model in models))
    loss, _ = _population_objective(
        logits, values, batch["actions"], batch["old_logp"], batch["old_value"],
        batch["advantage"], batch["returns"], batch["sample_ids"],
        clip=.2, value_coef=.5, entropy_coef=.01, jsd_coef=.05,
    )
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for model, optimizer in zip(models, optimizers):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return float(loss.detach())


def _assert_tree_close(actual, expected, atol=2e-6, rtol=2e-6):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key], atol, rtol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_tree_close(left, right, atol, rtol)
    else:
        assert actual == expected


def test_staged_jsd_matches_resident_joint_adam_and_skips_unsampled_critic():
    # Member 2 has no PPO rows. Its actor must still learn from JSD, while its
    # primed critic Adam state and critic weights must remain untouched.
    resident = [_small_flat(i) for i in range(3)]
    staged = [copy.deepcopy(model) for model in resident]
    resident_opts = [torch.optim.Adam(model.parameters(), lr=2e-4, eps=1e-5)
                     for model in resident]
    staged_opts = [torch.optim.Adam(model.parameters(), lr=2e-4, eps=1e-5)
                   for model in staged]
    torch.manual_seed(91)
    prime_spatial, prime_state = torch.randn(4, 2, 3, 3), torch.randn(4, 5)
    _prime_adam(resident, resident_opts, prime_spatial, prime_state)
    _prime_adam(staged, staged_opts, prime_spatial, prime_state)
    # Prime member 2's critic with nonzero Adam momentum before the comparison.
    _assert_tree_close(
        [o.state_dict() for o in resident_opts], [o.state_dict() for o in staged_opts]
    )
    before_actor = [p.detach().clone() for p in staged[2].actor.parameters()]
    before_critic = [p.detach().clone() for p in staged[2].critic.parameters()]
    before_critic_state = copy.deepcopy(staged_opts[2].state_dict())

    torch.manual_seed(123)
    batch = {
        "spatial": torch.randn(5, 2, 3, 3),
        "state": torch.randn(5, 5),
        "actions": torch.randint(0, 6, (5, 3)),
        "old_logp": torch.randn(5, 3) * .1,
        "old_value": torch.randn(5, 3) * .1,
        "advantage": torch.randn(5, 3),
        "returns": torch.randn(5, 3),
        "sample_ids": torch.tensor([0, 1, 0, 1, 0]),
    }
    resident_loss = _resident_step(resident, resident_opts, batch)
    workspace = _small_flat(999)
    staged_metrics = staged_population_step(
        staged, staged_opts, workspace,
        spatial=batch["spatial"].numpy(), state=batch["state"].numpy(),
        actions=batch["actions"].numpy(), old_logp=batch["old_logp"].numpy(),
        old_value=batch["old_value"].numpy(), advantage=batch["advantage"].numpy(),
        returns=batch["returns"].numpy(), sample_ids=batch["sample_ids"].numpy(),
        device="cpu", clip=.2, value_coef=.5, entropy_coef=.01, jsd_coef=.05,
        max_grad_norm=1.,
    )
    assert abs(staged_metrics["loss"] - resident_loss) < 1e-7

    for left, right in zip(staged, resident):
        _assert_tree_close(left.state_dict(), right.state_dict())
    _assert_tree_close([o.state_dict() for o in staged_opts],
                       [o.state_dict() for o in resident_opts])
    assert any(not torch.equal(old, new.detach())
               for old, new in zip(before_actor, staged[2].actor.parameters()))
    assert all(torch.equal(old, new.detach())
               for old, new in zip(before_critic, staged[2].critic.parameters()))
    names = list(dict(staged[2].named_parameters()))
    critic_id = before_critic_state["param_groups"][0]["params"][names.index("critic.weight")]
    assert torch.equal(before_critic_state["state"][critic_id]["step"],
                       staged_opts[2].state_dict()["state"][critic_id]["step"])


def test_joint_model_runs_encoder_once_and_returns_three_role_heads():
    model = JointHiPTPolicy(z_dim=4)
    seen = []
    handle = model.central.spatial.register_forward_pre_hook(
        lambda _module, inputs: seen.append(tuple(inputs[0].shape))
    )
    hidden = model.initial_state(2)
    result = model.step(torch.randn(2, 26, 7, 7), torch.randn(2, 863), hidden)
    high_logits, high_values, low_logits, low_values, next_hidden = result
    assert seen == [(2, 26, 7, 7)]
    assert high_logits.shape == (2, 3, 4)
    assert high_values.shape == (2, 3)
    assert low_logits.shape == (2, 3, 4, 6)
    assert low_values.shape == (2, 3, 4)
    assert next_hidden[0].shape == (1, 2, 128)
    seen.clear()
    model.unroll(torch.randn(5, 2, 26, 7, 7), torch.randn(5, 2, 863),
                 hidden, torch.zeros(5, 2))
    assert seen == [(10, 26, 7, 7)]
    handle.remove()


def test_role_action_and_value_projections_have_independent_role_rows():
    model = JointHiPTPolicy(
        z_dim=4, state_dim=5, channels=2, height=3, width=3,
        hidden_size=8, head_size=6,
    )
    high_logits, high_values, low_logits, low_values = model.heads(torch.randn(2, 8))
    assert high_logits.shape == (2, 3, 4)
    assert high_values.shape == (2, 3)
    assert low_logits.shape == (2, 3, 4, 6)
    assert low_values.shape == (2, 3, 4)
    loss = (high_logits[:, 0].sum() + high_values[:, 0].sum()
            + low_logits[:, 0].sum() + low_values[:, 0].sum())
    loss.backward()
    z, actions = model.z_dim, model.action_dim
    assert torch.count_nonzero(model.heads.high_actor.weight.grad[z:]) == 0
    assert torch.count_nonzero(model.heads.high_critic.weight.grad[1:]) == 0
    low_actor_grad = model.heads.low_actor.weight.grad.reshape(
        3, actions, model.heads.low_actor.in_features
    )
    low_actor_bias_grad = model.heads.low_actor.bias.grad.reshape(3, actions)
    assert torch.count_nonzero(low_actor_grad[1:]) == 0
    assert torch.count_nonzero(low_actor_bias_grad[1:]) == 0
    low_critic_grad = model.heads.low_critic.weight.grad.reshape(3, -1)
    low_critic_bias_grad = model.heads.low_critic.bias.grad
    assert torch.count_nonzero(low_critic_grad[1:]) == 0
    assert torch.count_nonzero(low_critic_bias_grad[1:]) == 0
    assert not torch.equal(high_logits[:, 0], high_logits[:, 1])
    assert not torch.equal(low_logits[:, 0], low_logits[:, 1])


def test_fused_complete_episode_bptt_matches_stepwise_unroll_outputs_and_gradients():
    stepwise = JointHiPTPolicy(
        z_dim=4, state_dim=5, channels=2, height=3, width=3,
        hidden_size=8, head_size=6,
    )
    fused = copy.deepcopy(stepwise)
    torch.manual_seed(303)
    x = torch.randn(7, 2, 2, 3, 3)
    state = torch.randn(7, 2, 5)
    done = torch.zeros(7, 2)
    done[0] = 1.0
    x_step = x.clone().requires_grad_(True)
    s_step = state.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    s_fused = state.clone().requires_grad_(True)
    out_step = stepwise.unroll(
        x_step, s_step, stepwise.initial_state(2), done
    )
    out_fused = fused.unroll_complete_episodes(
        x_fused, s_fused, fused.initial_state(2)
    )
    for step, full in zip(out_step[:4] + out_step[5:], out_fused[:4] + out_fused[5:]):
        torch.testing.assert_close(step, full, atol=2e-5, rtol=2e-5)
    for step, full in zip(out_step[4], out_fused[4]):
        torch.testing.assert_close(step, full, atol=2e-5, rtol=2e-5)

    def objective(outputs):
        return sum(value.square().mean() for value in outputs[:4]) + .03 * outputs[5].square().mean()

    objective(out_step).backward()
    objective(out_fused).backward()
    torch.testing.assert_close(x_step.grad, x_fused.grad, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(s_step.grad, s_fused.grad, atol=3e-5, rtol=3e-4)
    for (_, step_param), (_, fused_param) in zip(stepwise.named_parameters(), fused.named_parameters()):
        torch.testing.assert_close(step_param.grad, fused_param.grad, atol=3e-5, rtol=3e-4)


def test_option_clock_advances_and_resets_each_role_independently():
    clock = OptionClock(1)
    clock.start(np.array([[True, True, True]]), np.array([[0, 1, 2]]),
                np.array([[1, 2, 3]]))
    assert np.array_equal(clock.advance(np.array([False])), [[True, False, False]])
    clock.start(np.array([[True, False, False]]), np.array([[3, 1, 2]]),
                np.array([[4, 2, 3]]))
    assert np.array_equal(clock.z, [[3, 1, 2]])
    clock.advance(np.array([True]))
    assert np.array_equal(clock.remaining, [[0, 0, 0]])
    assert np.array_equal(clock.z, [[0, 0, 0]])


class _TinyEnv:
    num_envs = 1

    def __init__(self):
        self.rng = np.random.default_rng(7)
        self.steps = 0
        self.actions = []

    def _obs(self):
        return {
            "spatial": self.rng.normal(size=(1, 3, 26, 7, 7)).astype(np.float32),
            "state": self.rng.normal(size=(1, 863)).astype(np.float32),
        }

    def reset(self):
        return self._obs()

    def step(self, actions):
        self.actions.append(np.asarray(actions).copy())
        self.steps += 1
        return self._obs(), np.asarray([1.], dtype=np.float32), np.asarray([False]), [{}]

    def close(self):
        pass


def test_central_hierarchical_trainer_keeps_three_option_records():
    flat = CentralFlatActorCritic()
    snapshot = {key: value.detach().clone() for key, value in flat.state_dict().items()}
    snapshots = {level: [snapshot] for level in ("initial", "mid", "final")}
    env = _TinyEnv()
    trainer = HierarchicalTrainer(
        env, snapshots, num_envs=1, population_size=1, device="cpu", seed=3,
        minibatches=1, selfplay_fraction=1., target_env_steps=41,
    )
    metrics = trainer.collect_and_update(41, influence_coef=1.)
    assert len(env.actions) == 41
    assert all(action.shape == (1, 3) for action in env.actions)
    assert metrics["high"]["transitions"] >= 3
    assert metrics["learner_role_steps"] == [41, 41, 41]
    assert trainer.clock.z.shape == (1, 3)


def test_hierarchical_bc_unrolls_joint_sequences_and_leaves_z_actor_unsupervised():
    model = JointHiPTPolicy(
        z_dim=4, state_dim=5, channels=2, height=3, width=3,
        hidden_size=8, head_size=6,
    )
    dataset = {
        "spatial": np.random.default_rng(4).normal(size=(4, 2, 3, 3)).astype(np.float32),
        "state": np.random.default_rng(5).normal(size=(4, 5)).astype(np.float32),
        "actions": np.random.default_rng(6).integers(0, 6, size=(4, 3), dtype=np.int64),
        "validation": np.array([False, False, True, True]),
        "episode_lengths": np.array([2, 2]),
        "env_steps": 2,
    }
    high_actor_before = model.heads.high_actor.weight.detach().clone()
    low_actor_before = model.heads.low_actor.weight.detach().clone()
    metrics = warm_start_hierarchical(model, dataset, device="cpu", seed=9,
                                      epochs=1, episodes_per_batch=1)
    assert metrics["sequence_mode"].startswith("complete episodes")
    assert metrics["high_level_z_supervised"] is False
    assert torch.equal(high_actor_before, model.heads.high_actor.weight.detach())
    assert not torch.equal(low_actor_before, model.heads.low_actor.weight.detach())


def test_rl_mode_never_reads_demo_and_bc_smoke_counts_demo_without_changing_epochs(tmp_path):
    rl = parser().parse_args([
        "--mode", "rl", "--device", "cpu", "--smoke",
        "--demonstrations", str(tmp_path / "must_not_be_read.npz"),
    ])
    demo, digest, *_ = _configure(rl)
    assert (demo, digest) == (0, None)

    frames = 4
    vector = np.zeros((frames, 3, 866), dtype=np.float32)
    vector[:, :, :863] = np.arange(863, dtype=np.float32)
    data_path = tmp_path / "joint_demo.npz"
    np.savez_compressed(
        data_path,
        vector=vector.reshape(-1, 866),
        spatial=np.zeros((frames, 3, 26, 7, 7), dtype=np.float32).reshape(-1, 26, 7, 7),
        actions=np.zeros(frames * 3, dtype=np.int64),
        validation=np.repeat([False, False, True, True], 3),
        episode_lengths=np.array([2, 2], dtype=np.int64),
        env_steps=np.array(2),
    )
    bc = parser().parse_args([
        "--mode", "bc_rl", "--device", "cpu", "--smoke",
        "--demonstrations", str(data_path), "--bc-epochs", "64",
    ])
    demo, _, updates, pool_updates, batch_size = _configure(bc)
    assert demo == 2
    assert bc.bc_epochs == 64
    assert bc.steps == demo + 2 * batch_size
    assert bc.pool_steps == demo + batch_size
    assert (updates, pool_updates) == (2, 1)


class _PopulationEnv:
    num_envs = 2

    def __init__(self):
        self.rng = np.random.default_rng(18)
        self.step_index = 0
        self.closed = False

    def _obs(self):
        return {
            "spatial": self.rng.normal(size=(2, 3, 26, 7, 7)).astype(np.float32),
            "state": self.rng.normal(size=(2, 863)).astype(np.float32),
        }

    def reset(self):
        return self._obs()

    def step(self, actions):
        assert np.asarray(actions).shape == (2, 3)
        self.step_index += 1
        done = np.array([self.step_index == 1, False])
        rewards = np.array([1., 0.5], dtype=np.float32)
        return self._obs(), rewards, done, [{}, {}]

    def close(self):
        self.closed = True


def test_population_collect_update_handles_time_env_role_gae_and_terminal():
    env = _PopulationEnv()
    trainer = PopulationTrainer(
        env, num_envs=2, population_size=3, device="cpu", seed=11,
        epochs=1, minibatches=2,
    )
    metrics = trainer.collect_and_update(2)
    assert metrics["update"] == 1
    assert metrics["stage_steps"] == 4
    assert metrics["jsd"] is not None
    assert metrics["active_population_members"] == 3
    assert env.step_index == 2
    assert np.all(trainer.agent_steps >= 0)
    trainer.close()
    assert env.closed

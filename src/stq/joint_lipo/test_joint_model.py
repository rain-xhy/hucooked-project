from __future__ import annotations

from types import SimpleNamespace
import copy
from pathlib import Path

import numpy as np
import torch

from .model import JointLipoActor, JointLipoConfig, JointLipoMember, actor_parameter_count
from .train import EPISODE_SUMMARY_FIELDS, LipoTrainer, _validate_args, build_parser, resolve_output


def test_joint_actor_encodes_team_once_and_emits_three_actions():
    config = JointLipoConfig()
    actor = JointLipoActor(config)
    state = torch.zeros(4, 863)
    z = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3]])
    calls = []
    handle = actor.trunk.register_forward_hook(lambda *_: calls.append(1))
    logits = actor(state, z)
    handle.remove()

    assert logits.shape == (4, 3, 6)
    assert len(calls) == 1
    assert actor_parameter_count(config) == 62_162
    assert sum(parameter.numel() for parameter in actor.parameters()) == 62_162


def test_member_cpu_bank_roundtrips_weights_and_adam_state(tmp_path):
    args = SimpleNamespace(
        population=2,
        xp_candidates=1,
        num_envs=2,
        seed=3,
        lr=5e-4,
        xp_coef=0.3,
        mi_coef=0.5,
        gamma=0.9985,
        gae_lambda=0.95,
        minibatch_size=2,
        epochs=1,
        clip_param=0.3,
        ent_coef=0.03,
        vf_coef=1.0,
        grad_clip=0.5,
        mode="rl",
        demonstrations=None,
        checkpoint_interval=250_000,
        output=tmp_path,
    )
    trainer = LipoTrainer(args, torch.device("cpu"), tmp_path)
    assert trainer.resident_trainable_policy_count == 1
    assert all(next(member.parameters()).device.type == "cpu" for member in trainer.members)

    member = trainer._activate(0, optimizer_state=True)
    optimizer = trainer.workspace_optimizers["actor"]
    state = torch.randn(2, 863)
    z = torch.tensor([[0, 1, 2], [3, 4, 5]])
    optimizer.zero_grad(set_to_none=True)
    member.actor(state, z).square().mean().backward()
    optimizer.step()
    trainer._store_active(optimizer_state=True)

    saved_state = trainer.optimizers[0]["actor"].state_dict()["state"]
    assert saved_state
    assert all(value.device.type == "cpu" for row in saved_state.values() for value in row.values()
               if torch.is_tensor(value))
    assert not trainer.workspace_optimizers["actor"].state
    assert trainer.active_member_id is None
    assert trainer.members[0].training is False
    assert all(not parameter.requires_grad for parameter in trainer.members[0].parameters())


def test_cli_gpu_defaults_and_seed_output_path():
    args = build_parser().parse_args(["--mode", "rl", "--seed", "9", "--demonstrations", "missing.npz"])
    _validate_args(args)
    assert args.physical_gpu == 1
    assert args.memory_fraction == 0.035
    assert resolve_output(args) == (
        Path.cwd() / "joint_lipo" / "rl" / "seed_9" / "results"
    )


def test_lipo_signed_ppo_objective_keeps_xp_sign_before_clipping():
    from .algorithm import ppo_policy_loss

    old = torch.zeros(1)
    new = torch.tensor([2.0]).log()
    advantage = torch.tensor([1.0])
    signed = ppo_policy_loss(new, old, advantage, 0.1, minimize_reward=True)
    assert torch.allclose(signed, torch.tensor(1.1))
    assert torch.allclose(ppo_policy_loss(new, old, advantage, 0.1), torch.tensor(-1.1))


class FakeKitchen:
    num_envs = 2

    def set_stage(self, _stage):
        pass

    def reset(self):
        self.tick = 0
        self.state = np.zeros((self.num_envs, 863), dtype=np.float32)
        return self._observation()

    def _observation(self):
        vector = np.zeros((self.num_envs, 3, 866), dtype=np.float32)
        vector[:, :, :863] = self.state[:, None, :]
        for role in range(3):
            vector[:, role, 863 + role] = 1.0
        return {
            "vector": vector,
            "state": self.state.copy(),
            "spatial": np.zeros((self.num_envs, 3, 26, 7, 7), dtype=np.float32),
        }

    def step(self, _actions):
        self.tick += 1
        self.state[:, 0] = self.tick / 10.0
        dones = np.asarray([self.tick % 2 == 0, self.tick % 2 == 0])
        summary = {field: 0.0 for field in EPISODE_SUMMARY_FIELDS}
        summary["team_score"] = 1.0
        infos = [{"episode_summary": dict(summary)}, {"episode_summary": dict(summary)}]
        return self._observation(), np.ones(self.num_envs, dtype=np.float32), dones, infos

    state = np.zeros((num_envs, 863), dtype=np.float32)


def test_joint_rollouts_and_one_update_keep_env_role_axes(tmp_path):
    args = SimpleNamespace(
        population=2, xp_candidates=1, num_envs=2, seed=5, lr=5e-4,
        xp_coef=0.3, mi_coef=0.5, gamma=0.9985, gae_lambda=0.95,
        minibatch_size=2, epochs=1, clip_param=0.3, ent_coef=0.03,
        vf_coef=1.0, grad_clip=0.5, mode="rl", demonstrations=None,
        checkpoint_interval=250_000, output=tmp_path,
    )
    trainer = LipoTrainer(args, torch.device("cpu"), tmp_path)
    env = FakeKitchen()
    sp_calls = []
    hook = trainer.workspace.actor.trunk.register_forward_hook(lambda *_: sp_calls.append(1))
    sp = trainer.collect_sp(env, 0, 2, "standard")
    hook.remove()
    assert len(sp_calls) == 2

    learner_calls, partner_calls = [], []
    learner_hook = trainer.workspace.actor.trunk.register_forward_hook(
        lambda *_: learner_calls.append(1)
    )
    partner_hook = trainer.members[1].actor.trunk.register_forward_hook(
        lambda *_: partner_calls.append(1)
    )
    xp = trainer.collect_xp(env, 0, 1, role=2, horizon=2, stage="standard")
    learner_hook.remove()
    partner_hook.remove()
    assert len(learner_calls) == len(partner_calls) == 2

    assert sp.actions.shape == (2, 2, 3)
    assert sp.old_log_probs.shape == (2, 2, 3)
    assert xp.actions.shape == (2, 2)
    assert xp.old_action_probs.shape == (2, 2, 6)
    metrics = trainer.update_member(0, sp, [xp], 0)

    assert set(metrics) == {"loss", "sp_pg", "xp_pg", "sp_value", "xp_value", "mi", "entropy"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert trainer.active_member_id is None


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_checkpoint_and_evaluation_preserve_committed_adam_state(tmp_path):
    args = SimpleNamespace(
        population=2, xp_candidates=1, num_envs=2, seed=11, lr=5e-4,
        xp_coef=0.3, mi_coef=0.5, gamma=0.9985, gae_lambda=0.95,
        minibatch_size=2, epochs=1, clip_param=0.3, ent_coef=0.03,
        vf_coef=1.0, grad_clip=0.5, mode="rl", demonstrations=None,
        checkpoint_interval=250_000, output=tmp_path,
    )
    trainer = LipoTrainer(args, torch.device("cpu"), tmp_path)
    env = FakeKitchen()
    sp = trainer.collect_sp(env, 0, 2, "standard")
    xp = trainer.collect_xp(env, 0, 1, role=1, horizon=2, stage="standard")
    trainer.update_member(0, sp, [xp], 0)

    adam_before = {}
    for key, optimizers in trainer.optimizers[0].items():
        if isinstance(optimizers, list):
            adam_before[key] = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        else:
            adam_before[key] = copy.deepcopy(optimizers.state_dict())
    weights_before = copy.deepcopy(trainer.members[0].state_dict())
    member1_adam_before = copy.deepcopy(trainer.optimizers[1]["actor"].state_dict())
    member1_weights_before = copy.deepcopy(trainer.members[1].state_dict())
    assert adam_before["actor"]["state"]
    assert adam_before["sp_critic"]["state"]
    assert adam_before["discriminator"]["state"]
    assert adam_before["xp_critics"][1]["state"]

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    trainer._checkpoint(path=checkpoint_a, reason="after_update")
    trainer.evaluate(env, 1)
    # Simulate inference on another CPU-bank partner, then release it.
    partner = trainer._activate(1, optimizer_state=False)
    partner.actor(torch.zeros(2, 863), torch.zeros(2, 3, dtype=torch.long))
    trainer._store_active(optimizer_state=False)
    trainer._checkpoint(path=checkpoint_b, reason="after_inference")

    for key, optimizers in trainer.optimizers[0].items():
        if isinstance(optimizers, list):
            current = [optimizer.state_dict() for optimizer in optimizers]
        else:
            current = optimizers.state_dict()
        _assert_nested_equal(adam_before[key], current)
    _assert_nested_equal(weights_before, trainer.members[0].state_dict())
    _assert_nested_equal(member1_adam_before, trainer.optimizers[1]["actor"].state_dict())
    _assert_nested_equal(member1_weights_before, trainer.members[1].state_dict())
    saved_a = torch.load(checkpoint_a, map_location="cpu", weights_only=False)
    saved_b = torch.load(checkpoint_b, map_location="cpu", weights_only=False)
    _assert_nested_equal(saved_a["member_states"][0]["optimizers"], adam_before)
    _assert_nested_equal(saved_b["member_states"][0]["optimizers"], adam_before)

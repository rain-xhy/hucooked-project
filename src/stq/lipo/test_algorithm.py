"""Focused checks for LIPO's defining update/selection mechanisms."""

import torch

from lipo.algorithm import ppo_policy_loss, role_for_xp, select_max_compatible
from lipo.models import LipoMember, ModelConfig


def test_xp_policy_gradient_reverses_sp_reward_sign():
    theta_sp = torch.tensor(0.0, requires_grad=True)
    logp_sp = torch.log_softmax(torch.stack((theta_sp, theta_sp.new_zeros(()))), dim=0)[0].reshape(1)
    old_logp = torch.log(torch.tensor([0.5]))
    advantages = torch.tensor([1.0])
    sp_loss = ppo_policy_loss(logp_sp, old_logp, advantages, 0.3)
    sp_gradient = torch.autograd.grad(sp_loss, theta_sp)[0]

    theta_xp = torch.tensor(0.0, requires_grad=True)
    logp_xp = torch.log_softmax(torch.stack((theta_xp, theta_xp.new_zeros(()))), dim=0)[0].reshape(1)
    xp_loss = ppo_policy_loss(logp_xp, old_logp, advantages, 0.3, minimize_reward=True)
    xp_gradient = torch.autograd.grad(xp_loss, theta_xp)[0]

    assert sp_gradient.item() < 0.0  # gradient descent raises selected-action probability
    assert xp_gradient.item() > 0.0  # reversed gradient lowers it on the incompatible pair


def test_max_compatible_selection_uses_highest_sampled_return():
    assert select_max_compatible([-0.4, 0.75, 0.1]) == 1
    assert select_max_compatible([2.0, 2.0, -1.0]) == 0


def test_latent_mi_loss_reaches_actor_and_discriminator():
    torch.manual_seed(4)
    member = LipoMember(ModelConfig(population=2))
    obs = torch.randn(12, 866)
    roles = torch.arange(12) % 3
    latent_ids = torch.arange(12) % 8
    logits = member.actor_logits(obs, roles, latent_ids)
    probs = logits.softmax(dim=-1)
    mi_logits = member.infer_latent(obs, roles, probs)
    loss = torch.nn.functional.cross_entropy(mi_logits, latent_ids)
    actor_grad, = torch.autograd.grad(loss, logits, retain_graph=True)
    loss.backward()

    assert torch.isfinite(actor_grad).all()
    assert actor_grad.abs().sum().item() > 0.0
    assert member.discriminator[0].weight.grad is not None
    assert torch.isfinite(member.discriminator[0].weight.grad).all()
    assert member.discriminator[0].weight.grad.abs().sum().item() > 0.0


def test_focal_xp_policy_rotates_through_all_three_roles():
    roles = [role_for_xp(iteration, candidate_slot=0, member_id=0) for iteration in range(3)]
    assert roles == [0, 1, 2]
    assert {role_for_xp(iteration, 1, 2) for iteration in range(3)} == {0, 1, 2}

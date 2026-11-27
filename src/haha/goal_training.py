"""Training helpers shared by the three-chef MixedKitchen and SharedKitchen policies."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, mse_loss


def mlp(input_dim, output_dim, *, width=128):
    return nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(), nn.Linear(width, output_dim))


def tensor(value, device="cpu"):
    return torch.as_tensor(value, device=device)


def imitation(policy, examples, epochs, device, rng):
    report = {}
    for name, rows in zip(("manager", "worker"), examples):
        module = getattr(policy, name)
        obs, masks, targets_ = [tensor(np.stack(values), device) for values in zip(*rows)]
        obs = obs.view(-1, obs.shape[-1])
        masks = masks.view(-1, masks.shape[-1])
        targets_ = targets_.flatten()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.0015)
        for _ in range(epochs):
            for indices in tensor(rng.permutation(len(obs)), device).split(1024):
                logits = policy.logits(module, obs[indices], masks[indices])
                loss = cross_entropy(logits, targets_[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            correct = sum((policy.logits(module, obs[i:i+4096], masks[i:i+4096]).argmax(-1)
                           == targets_[i:i+4096]).sum().item() for i in range(0, len(obs), 4096))
        report[name] = {"samples": len(obs), "accuracy": correct / len(obs)}
    return report


def ppo_update(policy, optimizer, rollout, device, anchor):
    records, final, _ = rollout
    common, obs, masks, learning, labels, goals, old_logp, rewards, dones = [
        tensor(np.asarray(column), device) for column in zip(*records)
    ]
    alive = (~dones).float()
    with torch.no_grad():
        values = policy.critic(common).squeeze(-1)
        tail = policy.critic(tensor(final, device)).squeeze(-1)
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(tail)
        for index in reversed(range(len(values))):
            delta = rewards[index].float() + 0.995 * tail * alive[index] - values[index]
            gae = delta + 0.995 * 0.97 * alive[index] * gae
            advantages[index] = gae
            tail = values[index]
        returns = (advantages + values).reshape(-1)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    chefs = obs.shape[-2]
    common = common.reshape(-1, common.shape[-1])
    obs = obs.reshape(-1, chefs, obs.shape[-1])
    masks = masks.reshape(-1, chefs, masks.shape[-1])
    learning = learning.reshape(-1, chefs)
    labels = labels.reshape(-1, chefs)
    goals = goals.reshape(-1, chefs)
    old_logp = old_logp.reshape(-1, chefs)
    advantages = advantages.reshape(-1)
    for _ in range(4):
        for indices in torch.randperm(len(common), device=device).split(512):
            eligible = learning[indices]
            logits = policy.logits(policy.manager, obs[indices], masks[indices])[eligible]
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(goals[indices][eligible]) - old_logp[indices][eligible]).exp()
            adv = advantages[indices, None].expand_as(eligible)[eligible]
            actor_loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            value_loss = mse_loss(policy.critic(common[indices]).squeeze(-1), returns[indices])
            anchor_loss = cross_entropy(logits, labels[indices][eligible])
            loss = actor_loss + 0.5 * value_loss + anchor * anchor_loss - 0.005 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

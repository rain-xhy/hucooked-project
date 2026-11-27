"""PACE/NestRL MLP policy adapted to SharedKitchen (OBS_DIM=376, 3 chefs, 6 actions)."""
import torch
from torch import nn

from .env import OBS_DIM

HIDDEN_DIM = 256
HISTORY_EPISODES = 5
HISTORY_STEPS = 40


def partner_action_ids(actions):
    """Encode the two partners' joint actions (0..35) for each of the 3 chefs [N, 3]."""
    a0, a1, a2 = actions[:, 0], actions[:, 1], actions[:, 2]
    return torch.stack((a1 * 6 + a2, a0 * 6 + a2, a0 * 6 + a1), dim=1)


def ego_reorder(obs):
    """Reorder the 3 chef blocks obs[..., 1:211] to [self, (self+1)%3, (self+2)%3] using self_id obs[..., 373:376]."""
    chefs = obs[..., 1:211].unflatten(-1, (3, 70))
    self_idx = obs[..., 373:376].argmax(dim=-1)
    offsets = torch.arange(3, device=obs.device, dtype=self_idx.dtype)
    order = (self_idx.unsqueeze(-1) + offsets) % 3
    ego_chefs = chefs.gather(-2, order.unsqueeze(-1).expand(*order.shape, 70)).flatten(-2)
    return torch.cat((obs[..., :1], ego_chefs, obs[..., 211:]), dim=-1)


def mlp(input_size, hidden_size=HIDDEN_DIM):
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, hidden_size), nn.ReLU())


def masked_mean(values, mask):
    valid = ~mask
    total = values.masked_fill(mask.unsqueeze(-1), 0).sum(-2)
    return total / valid.sum(-1, keepdim=True).clamp_min(1)


class HistoryFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("classes", torch.arange(36), persistent=False)

    def forward(self, observations, partner_ids):
        encoded = partner_ids.unsqueeze(-1) == self.classes
        return torch.cat((ego_reorder(observations), encoded), -1)


class EpisodeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = mlp(OBS_DIM + 36)

    def forward(self, history, step_mask):
        return masked_mean(self.steps(history.masked_fill(step_mask.unsqueeze(-1), 0)), step_mask)


class PeriodEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = mlp(HIDDEN_DIM)

    def forward(self, episodes, episode_mask, step_mask):
        mask = episode_mask | step_mask.all(-1)
        return self.transform(masked_mean(episodes, mask))


class Actor(nn.Module):
    def __init__(self, use_history=True):
        super().__init__()
        self.use_history = bool(use_history)
        self.base = mlp(OBS_DIM + HIDDEN_DIM if self.use_history else OBS_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 6)

    def forward(self, current, latent=None):
        x = torch.cat((ego_reorder(current), latent), -1) if self.use_history else ego_reorder(current)
        return self.output(self.base(x)).log_softmax(-1)


class Critic(nn.Module):
    def __init__(self, use_history=True):
        super().__init__()
        self.use_history = bool(use_history)
        self.base = mlp(OBS_DIM + HIDDEN_DIM if self.use_history else OBS_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, current, latent=None):
        x = torch.cat((ego_reorder(current), latent), -1) if self.use_history else ego_reorder(current)
        return self.output(self.base(x))


class V3Policy(nn.Module):
    def __init__(self, num_partner_classes=3, use_history=True):
        super().__init__()
        self.use_history = bool(use_history)
        if self.use_history:
            self.partner_head = nn.Linear(HIDDEN_DIM, num_partner_classes)
            self.history = HistoryFeatures()
            self.episode = EpisodeEncoder()
            self.period = PeriodEncoder()
        else:
            self.partner_head = None
            self.history = None
            self.episode = None
            self.period = None
        self.actor = Actor(use_history=self.use_history)
        self.critic = Critic(use_history=self.use_history)

    def forward(self, current, history=None, partner_ids=None, episode_mask=None, step_mask=None):
        if not self.use_history:
            return self.actor(current), self.critic(current), None
        if history.size(1) > HISTORY_EPISODES or history.size(2) > HISTORY_STEPS:
            valid_steps = ~step_mask
            valid_episodes = ~episode_mask & valid_steps.any(-1)
            ep_rank = valid_episodes.flip((1,)).cumsum(1).flip((1,))
            step_rank = valid_steps.flip((2,)).cumsum(2).flip((2,))
            episode_mask = episode_mask | (ep_rank > HISTORY_EPISODES)
            step_mask = step_mask | (step_rank > HISTORY_STEPS)
        steps = self.history(history, partner_ids)
        episodes = self.episode(steps, step_mask)
        latent = self.period(episodes, episode_mask, step_mask)
        return self.actor(current, latent), self.critic(current, latent), self.partner_head(latent)

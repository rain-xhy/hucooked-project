"""v3 PACE-style MLP policy. True mask entries denote padding."""
import torch
from torch import nn

from .features import OBS_DIM


HIDDEN_DIM = 256
HISTORY_EPISODES = 5
HISTORY_STEPS = 40


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
        return torch.cat((observations, encoded), -1)


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
    def __init__(self):
        super().__init__()
        self.base = mlp(OBS_DIM + HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 6)

    def forward(self, current, latent):
        return self.output(self.base(torch.cat((current, latent), -1))).log_softmax(-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = mlp(OBS_DIM + HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, current, latent):
        return self.output(self.base(torch.cat((current, latent), -1)))


class V3Policy(nn.Module):
    def __init__(self, num_partner_classes=3):
        super().__init__()
        if type(num_partner_classes) is not int or num_partner_classes < 1:
            raise ValueError('Expected a positive partner class count')
        self.partner_head = nn.Linear(HIDDEN_DIM, num_partner_classes)
        self.history = HistoryFeatures()
        self.episode = EpisodeEncoder()
        self.period = PeriodEncoder()
        self.actor = Actor()
        self.critic = Critic()

    def forward(self, current, history, partner_ids, episode_mask, step_mask):
        # Accept chronological, optionally padded rollout storage. Keep the newest
        # valid episodes and newest valid steps even when padding is on the right.
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

"""Train a NestRL-style MLP against a GPU prep chef on native cramped_room."""

import sys
from pathlib import Path

from standard_setup import ROOT, activate


def main():
    invocation_dir = Path.cwd()
    activate()
    import numpy as np
    import torch
    from rollout_env import HISTORY_SIZE
    from soup_kitchen import HORIZON, OBSERVATION_SIZE, SoupKitchen, SoupPrepPartner
    from torch_kitchen_vec import CudaKitchenTransition, TorchKitchenVecEnv
    from train_nestrl import main as train

    class SoupVecEnv(TorchKitchenVecEnv):
        layout_name = "cramped_room"
        kitchen_type = SoupKitchen
        horizon = HORIZON
        observation_size = OBSERVATION_SIZE
        recipe_names = ("ThreeOnionSoup",)

        def __init__(self, *args, defer_rollout_stats=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.defer_rollout_stats = defer_rollout_stats
            self.rollout_step = 0
            self.episode_step = 0
            self.completed_episodes = []

        def make_rule_partner(self):
            return SoupPrepPartner(self.kitchen, player=1 - self.player)

        @torch.no_grad()
        def step(self, actions):
            if not self.defer_rollout_stats:
                return super().step(actions)

            start = self.mark()
            partner_actions = self.rule_partner.act()
            after_partner = self.mark()
            joint_actions = torch.stack((actions.reshape(self.num_envs), partner_actions), dim=1)
            if self.player == 1:
                joint_actions = joint_actions.flip((1,))
            terminal, reward, done, _, recipe = self.kitchen.step(joint_actions)
            after_kitchen = self.mark()
            self.returns += reward
            self.rollout_step += 1
            self.episode_step += 1
            self.obs = terminal
            if self.episode_step == self.horizon:
                reset_start = self.mark()
                self.episode_step = 0
                # SoupKitchen terminates all rows together at the fixed horizon.
                self.completed_episodes.append((self.rollout_step, torch.stack(
                    (self.returns, recipe >= 0, self.meta_episode != HISTORY_SIZE - 1), dim=0,
                )))
                self.obs = self.kitchen.reset(mask=done)
                self.returns.masked_fill_(done, 0)
                self.meta_episode.add_(done).remainder_(HISTORY_SIZE)
                if self.profile:
                    self.spans.append(("reset", reset_start, self.mark()))
            if self.profile:
                self.spans.extend((("opponent", start, after_partner),
                                   ("kitchen", after_partner, after_kitchen)))
            transition = CudaKitchenTransition(partner_actions, terminal[:, self.player], done)
            return self.obs[:, self.player], reward[:, None], done, transition

        def finish_rollout(self):
            batches = []
            if self.completed_episodes:
                steps, values = zip(*self.completed_episodes)
                rewards, successes, contexts = torch.stack(values, dim=1).cpu().numpy()
                rewards = np.round(rewards, decimals=6)
                batches = list(zip(steps, rewards, successes))
                self.episodes += rewards.size
                self.episodes_with_context += int(contexts.sum())
                self.rounds += rewards.size - int(contexts.sum())
            self.completed_episodes.clear()
            self.rollout_step = 0
            return batches

    sys.argv[1:1] = [
        "l1", "--player", "1", "--device", "cuda", "--env-mode", "gpu", "--history-storage", "gpu",
        "--processes", "64", "--rollout-steps", "1024", "--updates", "1",
        "--ppo-mini-batch-size", "1024", "--learning-rate", "0.001", "--lr-warmup-fraction", "0.02",
        "--trackio-project", "mlp-standard-l1-cramped-room", "--trackio-sample-every", "40", "--profile",
    ]
    train(output_root=ROOT, recipes=("three-onion-prep",), gpu_vec_class=SoupVecEnv,
          invocation_dir=invocation_dir)


if __name__ == "__main__":
    main()

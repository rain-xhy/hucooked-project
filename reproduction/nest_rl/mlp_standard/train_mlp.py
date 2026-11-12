"""Train a NestRL-style MLP with GPU rule chefs on a native kitchen."""

import argparse
from functools import wraps
import sys
from pathlib import Path

from standard_setup import ROOT, activate


def main():
    layout_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    layout_parser.add_argument("--layout", choices=("cramped_room", "three_player_handoff"), default="cramped_room")
    options, training_args = layout_parser.parse_known_args()
    if "--help" in training_args or "-h" in training_args:
        layout_parser.print_help()
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

        def reset(self):
            self.episode_step = 0
            return super().reset()

        def make_rule_partner(self):
            return SoupPrepPartner(self.kitchen, player=1 - self.player)

        @torch.no_grad()
        def step(self, actions):
            if not self.defer_rollout_stats:
                return super().step(actions)

            start = self.mark()
            partner_actions = self.rule_partner.act()
            after_partner = self.mark()
            joint_actions, history_actions = self.combine_actions(actions.reshape(self.num_envs), partner_actions)
            terminal, reward, done, _, recipe = self.kitchen.step(joint_actions)
            after_kitchen = self.mark()
            self.returns += reward
            self.rollout_step += 1
            self.episode_step += 1
            self.obs = terminal
            if self.episode_step == self.horizon:
                reset_start = self.mark()
                self.episode_step = 0
                # These kitchens terminate all rows together at the fixed horizon.
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
            transition = CudaKitchenTransition(history_actions, terminal[:, self.player], done)
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

    vec_class = SoupVecEnv
    recipes = ("three-onion-prep",)
    project = "mlp-standard-l1-cramped-room"
    if options.layout == "three_player_handoff":
        from gym.spaces import Discrete
        from handoff_kitchen import COOK, HandoffKitchen, HandoffRuleTeam
        from handoff_kitchen import OBSERVATION_SIZE as HANDOFF_OBSERVATION_SIZE
        from learning.model import Encoder

        class HandoffVecEnv(SoupVecEnv):
            layout_name = "three_player_handoff"
            kitchen_type = HandoffKitchen
            observation_size = HANDOFF_OBSERVATION_SIZE
            learner_player = COOK

            def make_rule_partner(self):
                return HandoffRuleTeam(self.kitchen)

            def combine_actions(self, learner, partners):
                supply, serve = partners.unbind(dim=1)
                return torch.stack((supply, learner, serve), dim=1), supply * self.action_space.n + serve

        # Represent both teammates as one opponent team in NestRL's history.
        # The actor keeps six actions, while the encoder receives 6 x 6 joint actions.
        encoder_init = Encoder.__init__

        @wraps(encoder_init)
        def init_encoder(encoder, num_policies, obs_shape, action_space, *args, **kwargs):
            return encoder_init(encoder, num_policies, obs_shape, Discrete(action_space.n ** 2), *args, **kwargs)

        Encoder.__init__ = init_encoder
        vec_class = HandoffVecEnv
        recipes = ("onion-supply-and-serve",)
        project = "mlp-standard-l1-three-player-handoff"

    sys.argv[1:] = [
        "l1", "--player", "1", "--device", "cuda", "--env-mode", "gpu", "--history-storage", "gpu",
        "--processes", "64", "--rollout-steps", "1024", "--updates", "1",
        "--ppo-mini-batch-size", "1024", "--learning-rate", "0.001", "--lr-warmup-fraction", "0.02",
        "--trackio-project", project, "--trackio-sample-every", "40", "--profile", *training_args,
    ]
    try:
        train(output_root=ROOT, recipes=recipes, gpu_vec_class=vec_class, invocation_dir=invocation_dir)
    finally:
        if options.layout == "three_player_handoff":
            Encoder.__init__ = encoder_init


if __name__ == "__main__":
    main()

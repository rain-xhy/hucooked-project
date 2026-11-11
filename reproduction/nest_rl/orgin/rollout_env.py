"""Build private kitchen partners inside each vector environment."""

from pathlib import Path
from time import perf_counter

from overcooked_setup import rule_partner

HISTORY_SIZE = 5


def load_adaptive_policy(path, device, obs_size=73):
    import torch

    model = torch.load(path, map_location=device, weights_only=False)
    if not model.latent_training_mode or model.is_recurrent:
        raise ValueError(f"Expected a latent PPO checkpoint from this runner: {path}")
    if getattr(model.actor.cnn, "observation_size", 73) != obs_size:
        raise ValueError(f"Checkpoint observation mode does not match the kitchen: {path}")
    return model.eval().requires_grad_(False)


def load_adaptive_partner(path, obs_size=73):
    import torch
    from learning.storage_ import PeriodicHistoryStorage

    class AdaptivePartner:
        def __init__(self, path):
            self.path = path
            self.model = load_adaptive_policy(path, "cpu", obs_size)
            self.history = None
            self.episodes = 0
            self.episodes_with_context = 0
            self.rounds = 0
            self.indices = torch.arange(1)

        def set_id(self, agent_id):
            self.agent_id = agent_id

        def full_reset(self):
            self.rounds += 1
            if self.history is not None:
                self.history.clear()

        def reset(self):
            if self.history is not None and self.history.current_episode[0] > 0:
                self.episodes_with_context += 1

        @torch.no_grad()
        def __call__(self, obs):
            if self.history is None:
                self.history = PeriodicHistoryStorage(
                    num_processes=1, num_policies=1,
                    history_storage_size=HISTORY_SIZE, clear_period=HISTORY_SIZE,
                    max_samples_per_period=None, refresh_interval=1, sample_size=None,
                    has_rew_done=False, step_mode=False, use_episodes=True,
                    has_meta_time_step=False, include_current_episode=True,
                    obs_shape=obs.shape, act_shape=(), max_episode_length=self.horizon,
                    merge_encoder_computation=True, last_episode_only=False,
                    pop_oldest_episode=False,
                )
            self.last_obs = torch.as_tensor(obs).float().clone()
            indices = (self.indices,) + self.history.get_all_current_indices()
            _, action, _, _ = self.model.act(
                self.last_obs.unsqueeze(0), None, torch.ones(1, 1), self.indices,
                history=(self.history, indices), deterministic=False,
            )
            return action.item()

        def observe(self, peer_action, done, next_obs):
            self.history.add(0, self.last_obs, int(peer_action), None)
            if done:
                self.history.add(0, torch.as_tensor(next_obs).float(), None, None)
                self.history.finish_episode(0)
                self.episodes += 1

    return AdaptivePartner(path)


def kitchen_factory(make_env, profile, spatial_obs=False):
    """Install the runner's environment inside the thunk that SB3 spawns."""
    def make_kitchen(*args, **kwargs):
        build = make_env(*args, **kwargs)

        def build_kitchen():
            import torch
            from environment.overcooked import overcooked_single

            torch.set_num_threads(1)
            original = overcooked_single.Overcooked_SingleEnv
            if spatial_obs:
                from gym.spaces import Box
                from spatial_observation import GRID_CELLS, OBSERVATION_SIZE, pack_world, world_tiles

            class Kitchen(original):
                def __init__(self, env):
                    super().__init__(env)
                    self.profiling = profile
                    self.partner_history_seconds = 0.0
                    self.adaptive = False
                    if spatial_obs:
                        self.observation_space = Box(-1.0, GRID_CELLS - 1.0, shape=(OBSERVATION_SIZE,))
                        self.spatial_tiles = None

                def spatial_view(self, actions=(-1, -1)):
                    world = self.env.get_env_world()
                    if self.spatial_tiles is None:
                        self.spatial_tiles = world_tiles(world)
                    self.last_obs = dict(zip(self.players, pack_world(world, self.spatial_tiles, actions)))
                    return self.last_obs[f"player_{self._id}"]

                def reset(self):
                    obs = super().reset()
                    return self.spatial_view() if spatial_obs else obs

                def set_opponent(self, spec):
                    self.adaptive = isinstance(spec, Path)
                    if self.adaptive:
                        partner = load_adaptive_partner(spec, self.observation_space.shape[0])
                        partner.horizon = self.episode_length()
                    else:
                        partner = rule_partner(spec)
                    super().set_opponent(partner)

                def step(self, action):
                    obs, reward, done, info = super().step(action)
                    if spatial_obs:
                        joint = (int(action), int(info["opponent_act"]))
                        obs = self.spatial_view(joint if self._id == 0 else joint[::-1])
                    if self.adaptive:
                        start = perf_counter() if self.profiling else 0.0
                        partner = self.opponent
                        partner.observe(action, done, self.last_obs[f"player_{partner.agent_id}"])
                        if self.profiling:
                            self.partner_history_seconds += perf_counter() - start
                    return obs, reward, done, info

                def rollout_stats(self):
                    stats = {
                        "kitchen_seconds": self.env_time,
                        "opponent_seconds": self.op_inf_time,
                        "opponent_history_seconds": self.partner_history_seconds,
                    }
                    if self.adaptive:
                        partner = self.opponent
                        stats.update(
                            path=str(partner.path),
                            episodes=partner.episodes,
                            episodes_with_context=partner.episodes_with_context,
                            rounds=partner.rounds,
                        )
                    return stats

            overcooked_single.Overcooked_SingleEnv = Kitchen
            try:
                return build()
            finally:
                overcooked_single.Overcooked_SingleEnv = original

        return build_kitchen

    return make_kitchen

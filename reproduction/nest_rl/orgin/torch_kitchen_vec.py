"""CUDA rollout adapter for NestRL's two-chef kitchens."""

from collections import defaultdict
from functools import wraps

from gym.spaces import Box, Discrete
import numpy as np
import torch

from learning.model import AggregatedMLPEncoder
from cuda_history import CudaHistory
from rollout_env import HISTORY_SIZE, load_adaptive_policy
from torch_kitchen import HORIZON, TorchKitchen
from torch_kitchen_features import RECIPE_NAMES


class FrozenPartnerBatch:
    """Cache the frozen mean encoder's per-episode sums for independent histories."""

    def __init__(self, path, indices, device, obs_size, actions):
        self.path = path
        self.indices = indices
        self.model = load_adaptive_policy(path, device)
        encoder = self.model.encoder
        self.base = encoder.base
        if (
            not isinstance(self.base, AggregatedMLPEncoder) or self.base.agg_func != "mean"
            or not encoder.deterministic_latent or not encoder.discrete_action
            or encoder.action_dim != actions or encoder.cnn is not None
        ):
            raise ValueError(f"Expected a deterministic mean-MLP L1 encoder: {path}")
        self.action_ids = torch.arange(actions, device=device)
        self.episode_ids = torch.arange(HISTORY_SIZE, device=device)
        self.rows = torch.arange(len(indices), device=device)
        self.no_action = torch.zeros((len(indices), actions), device=device)
        with torch.no_grad():
            sample = self.base.pre_agg_mlp(torch.zeros((1, obs_size + actions), device=device))
        self.sums = sample.new_zeros((len(indices), HISTORY_SIZE, sample.shape[-1]))
        self.lengths = sample.new_zeros((len(indices), HISTORY_SIZE, 1))

    def clear(self):
        self.sums.zero_()
        self.lengths.zero_()

    def latents(self, episode):
        means = self.sums / self.lengths.clamp_min(1)
        # The CPU encoder includes the current episode even when it is empty.
        future = self.episode_ids[None, :] > episode[:, None]
        return self.base.period_forward(means, future).relu()

    def act(self, obs, episode, generator):
        distribution, _, _ = self.model.actor.get_action_dist(obs, None, None, self.latents(episode))
        return torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)

    def observe(self, obs, action, terminal_obs, done, episode):
        one_hot = action[:, None] == self.action_ids
        embedding = self.base.pre_agg_mlp(torch.cat((obs, one_hot), dim=-1))
        terminal = self.base.pre_agg_mlp(torch.cat((terminal_obs, self.no_action), dim=-1))
        self.sums[self.rows, episode] += embedding + terminal * done[:, None]
        self.lengths[self.rows, episode] += 1 + done[:, None]
        end_period = done & (episode == HISTORY_SIZE - 1)
        self.sums.masked_fill_(end_period[:, None, None], 0)
        self.lengths.masked_fill_(end_period[:, None, None], 0)


class KitchenInfos(list):
    """Expose the GPU transition tensors alongside the trainer's episode metadata."""

    def __init__(self, num_envs, opponent_actions, terminal_obs, done, finished):
        super().__init__({} for _ in range(num_envs))
        self.opponent_actions = opponent_actions
        self.terminal_obs = terminal_obs
        self.done = done
        self.finished = finished


class TorchKitchenVecEnv:
    def __init__(self, num_envs, device, seed, player, profile=False):
        self.kitchen = TorchKitchen(num_envs, device, seed)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.observation_space = Box(-1.0, 1.0, shape=(73,), dtype=np.float32)
        self.action_space = Discrete(6)
        self.player = player
        self.profile = profile
        self.spans = []
        self.opponent_specs = [None] * num_envs
        self.partners = []
        self.rule_partner = None
        self.rule_seed = seed + 2
        self.generator = torch.Generator(device=device).manual_seed(seed + 1)
        self.meta_episode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.episodes = 0
        self.episodes_with_context = 0
        self.rounds = 0

    def env_method(self, name, *args, indices=None):
        # This adapter implements the calls made by the pinned training loop.
        if name == "episode_length":
            return [HORIZON] if indices is not None else [HORIZON] * self.num_envs
        if name == "set_opponent":
            self.opponent_specs[indices] = args[0]
            return [None]
        if name == "set_id" and args[0] == self.player:
            return [None]
        if name == "full_reset":
            self.meta_episode.zero_()
            self.returns.zero_()
            self.rounds += self.num_envs
            for partner in self.partners:
                partner.clear()
            return [None] * self.num_envs
        if name == "rollout_stats":
            return [self.rollout_stats()]
        raise ValueError(f"Unsupported CUDA kitchen method: {name}")

    def reset(self):
        self.obs = self.kitchen.reset()
        if self.rule_partner is None and not self.partners:
            if all(isinstance(spec, str) for spec in self.opponent_specs):
                from cuda_l0 import CudaRulePartner

                self.rule_partner = CudaRulePartner(
                    self.kitchen, self.opponent_specs, player=1 - self.player, seed=self.rule_seed,
                )
            else:
                groups = defaultdict(list)
                for index, path in enumerate(self.opponent_specs):
                    groups[path].append(index)
                self.partners = [
                    FrozenPartnerBatch(
                        path, torch.tensor(indices, device=self.device), self.device,
                        self.observation_space.shape[0], self.action_space.n,
                    )
                    for path, indices in groups.items()
                ]
        return self.obs[:, self.player]

    def reset_rule_evaluation(self, seed):
        self.kitchen = TorchKitchen(self.num_envs, self.device, seed)
        self.rule_seed = seed + 2
        self.rule_partner = None
        self.env_method("full_reset")
        return self.reset()

    def mark(self):
        if self.profile:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

    @torch.no_grad()
    def step(self, actions):
        start = self.mark()
        peer = 1 - self.player
        actions = actions.reshape(self.num_envs)
        if self.rule_partner is not None:
            partner_actions = self.rule_partner.act()
        else:
            partner_actions = torch.empty_like(actions)
            for partner in self.partners:
                ids = partner.indices
                partner_actions[ids] = partner.act(
                    self.obs[ids, peer], self.meta_episode[ids], self.generator,
                )
        after_partner = self.mark()

        joint_actions = torch.stack((actions, partner_actions), dim=1)
        if self.player == 1:
            joint_actions = joint_actions.flip((1,))
        terminal, reward, done, truncated, recipe = self.kitchen.step(joint_actions)
        after_kitchen = self.mark()

        for partner in self.partners:
            ids = partner.indices
            partner.observe(
                self.obs[ids, peer], actions[ids], terminal[ids, peer],
                done[ids], self.meta_episode[ids],
            )
        after_history = self.mark()

        self.returns += reward
        # Transfer one packed metadata array. Simulation, actions and next obs stay on CUDA.
        flags = torch.stack(
            (done, truncated, recipe, self.kitchen.ticks, self.meta_episode), dim=1,
        ).cpu().numpy()
        done_cpu = flags[:, 0].astype(bool)
        finished = np.flatnonzero(done_cpu)
        infos = KitchenInfos(self.num_envs, partner_actions, terminal[:, self.player], done, finished)
        self.obs = terminal
        if finished.size:
            start_reset = self.mark()
            reset_obs = self.kitchen.reset(mask=done)
            self.obs = torch.where(done[:, None, None], reset_obs, terminal)
            if self.profile:
                self.spans.append(("reset", start_reset, self.mark()))
            terminal_obs = terminal[:, self.player].cpu().numpy()
            returns = self.returns.cpu().numpy()
            ends_period = int(np.count_nonzero(flags[finished, 4] == HISTORY_SIZE - 1))
            self.episodes += finished.size
            self.episodes_with_context += finished.size - ends_period
            self.rounds += ends_period
            for index in finished:
                _, timeout, recipe_index, ticks, episode = flags[index]
                reason = (f"{RECIPE_NAMES[recipe_index]} is completed"
                          if recipe_index >= 0 else f"passed {HORIZON} timesteps")
                info = {
                    "t": int(ticks),
                    "termination_info": f"Terminating because {reason}",
                    "episode": {"r": round(float(returns[index]), 6), "l": int(ticks)},
                    "episode_stats": {},
                    "terminal_observation": terminal_obs[index],
                }
                if timeout:
                    info["bad_transition"] = True
                if episode == HISTORY_SIZE - 1:
                    info["interaction_stats"] = {}
                infos[index] = info
            self.returns.masked_fill_(done, 0)
            self.meta_episode.add_(done).remainder_(HISTORY_SIZE)
        if self.profile:
            self.spans.extend((
                ("opponent", start, after_partner),
                ("kitchen", after_partner, after_kitchen),
                ("opponent_history", after_kitchen, after_history),
            ))
        return self.obs[:, self.player], reward[:, None], done_cpu, infos

    def rollout_stats(self):
        times = defaultdict(float)
        if self.spans:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.spans:
                times[name] += start.elapsed_time(end) / 1000
        return {
            "path": ", ".join(dict.fromkeys(map(str, self.opponent_specs))),
            "episodes": int(self.episodes),
            "episodes_with_context": int(self.episodes_with_context),
            "rounds": self.rounds,
            "kitchen_seconds": times["kitchen"] + times["reset"],
            "opponent_seconds": times["opponent"],
            "opponent_history_seconds": times["opponent_history"],
        }

    def close(self):
        self.partners.clear()
        self.rule_partner = None
        self.spans.clear()


@torch.no_grad()
def evaluate_cuda_rules(_args, pool, episodes, envs, _history, policy, _log_steps,
                        *, use_latent, update_history):
    """Evaluate one equally weighted set of episodes per recipe on CUDA."""
    assert use_latent and update_history and len(pool) == envs.num_envs
    obs = envs.reset_rule_evaluation(torch.initial_seed() + 1_000_000)
    history = CudaHistory(envs.num_envs, episodes * HORIZON, envs.observation_space.shape,
                          HORIZON, HISTORY_SIZE, envs.device)
    rows = torch.arange(envs.num_envs, device=envs.device)
    masks = torch.zeros((envs.num_envs, 1), device=envs.device)
    counts = np.zeros(envs.num_envs, dtype=np.int64)
    rewards = np.zeros(envs.num_envs)
    successes = np.zeros(envs.num_envs)
    while (counts < episodes).any():
        context = (history, (rows, *history.get_all_current_indices()))
        _, action, _, _ = policy.act(obs, None, masks, rows, context)
        next_obs, _, _, infos = envs.step(action)
        end_period = history.append(obs, infos.opponent_actions, infos.terminal_obs, infos.done, infos.finished)
        obs = next_obs
        masks = (~end_period)[:, None].float()
        for index in infos.finished:
            if counts[index] < episodes:
                info = infos[index]
                counts[index] += 1
                rewards[index] += info["episode"]["r"]
                successes[index] += info["termination_info"].endswith("completed")
    rates = successes / episodes
    print(f"All success rates, by opponent policy: {tuple(map(float, rates))}")
    return {"reward": float(rewards.mean() / episodes), "success_rate": float(rates.mean())}, None


def install_cuda_kitchen_history(storage_class):
    """Record the vector transition and preserve the trainer's meta-episode masks."""
    original_insert = storage_class.insert

    @wraps(original_insert)
    def insert(storage, obs, next_obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, infos, *args, **kwargs):
        if isinstance(infos, KitchenInfos):
            history = storage.history
            assert storage.self_obs_mode and not storage.self_action_mode and not storage.collect_peer_traj
            assert storage.use_meta_episode
            previous = storage.obs[storage.step]
            if isinstance(history, CudaHistory):
                end_period = history.append(
                    previous, infos.opponent_actions, infos.terminal_obs, infos.done, infos.finished,
                )
                masks = (~end_period).unsqueeze(1).to(storage.masks.dtype)
                bad_masks = torch.ones_like(masks)
            else:
                assert not history.has_rew_done and not history.has_meta_time_step
                for index, (view, action) in enumerate(zip(previous.cpu(), infos.opponent_actions.cpu())):
                    history.add(index, view, action, None)
                for index in infos.finished:
                    terminal = torch.from_numpy(infos[index]["terminal_observation"])
                    history.add(index, terminal, None, None)
                    masks[index] = float(not history.finish_episode(index))
                    bad_masks[index] = 1.0
            # The history and meta-episode masks above replace the upstream per-row insert.
            infos = ()
        return original_insert(storage, obs, next_obs, recurrent_hidden_states, actions,
                               action_log_probs, value_preds, rewards, masks, bad_masks, infos, *args, **kwargs)

    storage_class.insert = insert

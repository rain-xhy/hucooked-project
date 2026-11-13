"""Batched CUDA kitchen adapters for HA² Worker and Manager training."""

from functools import partial

import numpy as np
import torch
from gymnasium import spaces
from overcooked_ai_py.mdp.overcooked_mdp import Direction
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .artifacts import check_loaded_policy, checkpoint_info
from .encoding import make_planner
from .env import NoExecutableSubtask, proximity_bonus
from .subtasks import NAMES, UNKNOWN
from .tensor_features import TensorBCPartner, TensorFeatures, joint_step, manager_task_step
from .tensor_kitchen import INTERACT, POT, STAY, KitchenState, TensorKitchen


class TensorVecEnv(VecEnv):
    def __init__(self, args, *, masked):
        layouts = tuple(args.layouts[i % len(args.layouts)] for i in range(args.n_envs))
        self.kitchen = TensorKitchen(layouts, horizon=args.horizon)
        self.features = TensorFeatures(self.kitchen)
        if args.compile_kitchen:
            self.features.compile_kernels()
        weights = {name: args.run_dir / "bc" / name / "bc.pt" for name in args.layouts}
        self.partner = TensorBCPartner(self.features, weights)
        self.generator = torch.Generator(device=self.kitchen.device)
        self.base_steps = 0
        self.actions = None
        self.masked = masked
        observation = {"visual_obs": spaces.Box(0, 20, (27, 7, 7), dtype=np.float32)}
        if masked:
            observation["subtask_mask"] = spaces.MultiBinary(len(NAMES))
        super().__init__(args.n_envs, spaces.Dict(observation), spaces.Discrete(len(NAMES) if masked else 6))

    def _sample_players(self):
        return torch.randint(2, (self.num_envs,), device=self.kitchen.device, generator=self.generator)

    def _skill_mask(self):
        return self.features.skill_mask(
            self.state, self.players, self.previous[self.kitchen.rows, self.players],
        )

    @staticmethod
    def _numpy_observation(observation):
        return {key: value.to(torch.int8).cpu().numpy() if key == "subtask_mask" else value.cpu().numpy()
                for key, value in observation.items()}

    @torch.no_grad()
    def reset(self):
        if self._seeds[0] is not None:
            self.generator.manual_seed(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        self.state = self.kitchen.initial
        self.players = self._sample_players()
        self.previous = torch.full((self.num_envs, 2), UNKNOWN, device=self.kitchen.device)
        self.episode_returns = torch.zeros(self.num_envs, device=self.kitchen.device)
        self.actions = None
        self._reset_episodes()
        self.reset_infos = [dict(layout=name, player=player)
                            for name, player in zip(self.kitchen.layouts, self.players.tolist())]
        self.tensor_observations = self._observation()
        self.observations = self._numpy_observation(self.tensor_observations)
        return self.observations

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)

    @torch.no_grad()
    def step_wait(self):
        actions = self.actions
        self.actions = None
        if actions is None:
            raise RuntimeError("step_async() must precede step_wait()")
        if np.any(actions < 0) or np.any(actions >= self.action_space.n):
            raise ValueError("Action outside the action space")
        if self.masked and not self.observations["subtask_mask"][np.arange(self.num_envs), actions].all():
            raise ValueError("Invalid Manager action for current skill mask")
        observations, terminal, metadata = self.step_tensors(self.kitchen._tensor(actions), native=False)
        finished = metadata[:, 1].astype(bool)
        terminal = self._numpy_observation(terminal) if finished.any() else None
        base_steps = self.base_steps if self.masked else self.base_steps.tolist()
        infos = []
        for index, row in enumerate(metadata):
            info = self._event_info(index, row)
            info.update(low_action=int(row[4]), team_reward=float(row[6]),
                        base_steps=base_steps if self.masked else base_steps[index],
                        completed_subtasks=[None if value < 0 else int(value) for value in row[10:12]],
                        joint_action=[int(value) for value in row[12:14]],
                        **{"TimeLimit.truncated": bool(row[2])})
            if self.masked:
                info["reset_wait_steps"] = 0
            if finished[index]:
                info["terminal_observation"] = {key: value[index].copy() for key, value in terminal.items()}
                self.reset_infos[index] = dict(layout=info["layout"], player=int(row[9]))
            infos.append(info)
        self.observations = self._numpy_observation(observations)
        return self.observations, metadata[:, 0], finished, infos

    def _event_info(self, index, row):
        info = dict(layout=self.kitchen.layouts[index], player=int(row[3]), subtask=int(row[14]))
        if self.masked:
            info["worker_failed"] = bool(row[5])
        else:
            info.update(task_success=bool(row[15]), task_invalidated=bool(row[16]),
                        task_timeout=bool(row[17]), task_steps=int(row[8]))
        if row[1]:
            info["episode"] = {"r": float(row[7]), "l": int(row[8])}
        return info

    def rollout_events(self, metadata):
        for step, index in np.argwhere((metadata[:, :, 1] != 0) | (metadata[:, :, 5] != 0)):
            yield step, index, self._event_info(index, metadata[step, index])

    def close(self):
        self.actions = None

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        if attr_name == "action_masks" and self.masked:
            return [partial(self._action_mask, index) for index in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def _action_mask(self, index):
        return self.observations["subtask_mask"][index]

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks" and self.masked:
            return [self._action_mask(index) for index in self._get_indices(indices)]
        if method_name == "get_base_steps":
            steps = self.base_steps
            if isinstance(steps, torch.Tensor):
                counts = steps.tolist()
                return [counts[index] for index in self._get_indices(indices)]
            return [steps for _ in self._get_indices(indices)]
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class TensorManagerVecEnv(TensorVecEnv):
    def __init__(self, args):
        super().__init__(args, masked=True)
        worker = PPO.load(args.worker, device="cuda", n_envs=1)
        check_loaded_policy(worker, checkpoint_info(args.worker))
        self.worker = worker.policy.eval().requires_grad_(False)
        self.failure_limit = args.failure_limit
        self.unknown_penalty = args.unknown_penalty

    def _reset_episodes(self):
        self.failures = torch.zeros_like(self.players)

    def _observation(self):
        return {"visual_obs": self.kitchen.observation(self.state, self.players),
                "subtask_mask": self._skill_mask()}

    def step_tensors(self, tasks, *, native=True):
        kitchen = self.kitchen
        result, self.previous, joint = manager_task_step(
            self.features, self.partner, self.state, self.players, self.previous,
            tasks, self.worker, generator=self.generator,
        )
        low_actions = joint[kitchen.rows, self.players]
        interacted = low_actions == INTERACT
        failed = interacted & (result.completed[kitchen.rows, self.players] != tasks)
        self.failures = torch.where(interacted, torch.where(failed, self.failures + 1, 0), self.failures)
        terminated = (self.failures >= self.failure_limit) & (self.failure_limit > 0)
        dones = result.done | terminated
        rewards = result.reward.float() + (tasks == UNKNOWN).float() * self.unknown_penalty
        self.episode_returns += rewards
        self.base_steps += 1
        self.state = result.state
        next_players = torch.where(dones, self._sample_players(), self.players)
        metadata = torch.stack((
            rewards, dones, result.done & ~terminated, self.players, low_actions, failed,
            result.reward, self.episode_returns, self.state.timestep, next_players,
            result.completed[:, 0], result.completed[:, 1], joint[:, 0], joint[:, 1], tasks,
        ), dim=1)
        if not native:
            metadata = metadata.cpu().numpy()
        terminal = self._observation() if native or metadata[:, 1].any() else None
        self.state = kitchen.reset_where(self.state, dones)
        self.players = next_players
        self.previous = torch.where(dones[:, None], UNKNOWN, self.previous)
        self.failures = torch.where(dones, 0, self.failures)
        self.episode_returns = torch.where(dones, 0, self.episode_returns)
        self.tensor_observations = self._observation()
        return self.tensor_observations, terminal, metadata

class TensorWorkerVecEnv(TensorVecEnv):
    def __init__(self, args):
        super().__init__(args, masked=False)
        self.independent_sampling = args.vec_env == "tensor-native"
        self.task_timeout = args.task_timeout
        self.bonuses = None if args.no_worker_shaping else self._proximity_table()
        self.base_steps = torch.zeros_like(self.kitchen.rows)
        self.task_counts = torch.ones((self.num_envs, UNKNOWN), device=self.kitchen.device)
        self.wait_actions = torch.full_like(self.kitchen.rows, STAY)

    def _proximity_table(self):
        kitchen = self.kitchen
        bonuses = np.zeros((len(kitchen.mdps), kitchen.num_cells, 4, len(NAMES)), np.float32)
        for layout, mdp in enumerate(kitchen.mdps.values()):
            motion = make_planner(mdp).motion_planner
            pots, serving = mdp.get_pot_locations(), mdp.get_serving_locations()
            for position in mdp.get_valid_player_positions():
                cell = kitchen._cell(position)
                for direction, orientation in enumerate(Direction.ALL_DIRECTIONS):
                    for skill, targets in ((1, pots), (3, pots), (5, pots), (6, pots), (8, serving), (9, serving)):
                        bonuses[layout, cell, direction, skill] = proximity_bonus(
                            mdp, motion, position, orientation, targets, skill in (1, 5, 8),
                        )
        bonuses[..., 10] = 1
        return torch.as_tensor(bonuses, device=kitchen.device)

    def _observation(self):
        return {"visual_obs": self.kitchen.observation(self.state, self.players, self.goals)}

    def _reset_episodes(self):
        self.goals = torch.zeros_like(self.players)
        self.task_steps = torch.zeros_like(self.players)
        pending = torch.ones_like(self.players, dtype=torch.bool)
        if self.independent_sampling:
            self.wait_horizons = torch.zeros_like(self.players)
            self.exhausted = torch.zeros_like(pending)
            self.ready = ~self._choose_goals(pending)
        else:
            self._select_goals(pending)

    def _reset_games(self, dones):
        self.state = self.kitchen.reset_where(self.state, dones)
        self.players = torch.where(dones, self._sample_players(), self.players)
        self.previous = torch.where(dones[:, None], UNKNOWN, self.previous)

    def _choose_goals(self, pending):
        doable = self._skill_mask()[:, :UNKNOWN]
        waiting = pending & ~doable.any(dim=1)
        ready = pending & ~waiting
        # An exponential race samples from the same inverse-frequency weights.
        noise = torch.empty_like(self.task_counts).exponential_(generator=self.generator)
        tasks = (doable / (self.task_counts * noise)).argmax(dim=1)
        self.goals = torch.where(ready, tasks, self.goals)
        self.task_counts[self.kitchen.rows, tasks] += ready
        return waiting

    def _advance(self, actions, active):
        result, previous, joint = joint_step(
            self.features, self.partner, self.state, self.players, self.previous,
            actions, generator=self.generator,
        )
        self.base_steps += active
        self.state = KitchenState(*(torch.where(
            active.reshape(-1, *([1] * (after.ndim - 1))), after, before,
        ) for before, after in zip(self.state, result.state)))
        self.previous = torch.where(active[:, None], previous, self.previous)
        return result, joint

    def _select_goals(self, resetting):
        pending = resetting
        horizons = torch.zeros_like(self.players)
        waited = False
        while True:
            waiting = self._choose_goals(pending)
            if not bool(waiting.any()):
                break
            waited = True
            result, _ = self._advance(self.wait_actions, waiting)
            reached = waiting & result.done
            horizons += reached
            self._reset_games(reached)
            pending = waiting & (horizons < 8)
        if waited and bool((horizons >= 8).any()):
            raise NoExecutableSubtask("No executable Worker skill after eight kitchen horizons")

    def _skill_bonus(self):
        kitchen = self.kitchen
        position = self.state.position[kitchen.rows, self.players]
        direction = self.state.orientation[kitchen.rows, self.players]
        target = kitchen.front[position, direction]
        onions = self.state.objects[:, :, 1]
        others = (kitchen.terrain == POT) & (kitchen.cells != target[:, None])
        pot_bonus = ((onions[kitchen.rows, target] - 1 - torch.where(others, onions, 0).amax(dim=1))
                     .clamp_min(0).float() * .1)
        static_bonus = self.bonuses[kitchen.layout_ids, position, direction, self.goals]
        return torch.where(self.goals == 2, pot_bonus, static_bonus)

    def _transition(self, actions, active, collecting, *, native):
        kitchen = self.kitchen
        result, joint = self._advance(actions, active)
        self.task_steps += collecting
        interacted = actions == INTERACT
        success = result.completed[kitchen.rows, self.players] == self.goals
        invalidated = ~self._skill_mask()[kitchen.rows, self.goals] & ~interacted
        timeout = (self.task_steps >= self.task_timeout) & ~interacted
        terminated = interacted | invalidated | timeout
        game_done = active & result.done
        dones = collecting & (terminated | game_done)
        reward = torch.where(interacted & success, 1., torch.where(interacted | timeout, -1., -.01))
        if self.bonuses is not None:
            reward += torch.where(interacted & success, self._skill_bonus(), 0.)
        self.episode_returns += torch.where(collecting, reward, 0.)
        terminal = self._observation() if native or bool(dones.any()) else None
        metadata = torch.stack((
            reward, dones, game_done & ~terminated, self.players, actions, torch.zeros_like(dones),
            result.reward, self.episode_returns, self.task_steps, self.players,
            result.completed[:, 0], result.completed[:, 1], joint[:, 0], joint[:, 1], self.goals,
            success, invalidated, timeout,
        ), dim=1)
        self.task_steps = torch.where(dones, 0, self.task_steps)
        self.episode_returns = torch.where(dones, 0., self.episode_returns)
        self._reset_games(game_done)
        return terminal, metadata, game_done, dones

    def step_tensors(self, actions, *, native=True):
        active = torch.ones_like(self.players, dtype=torch.bool)
        terminal, metadata, _, dones = self._transition(actions, active, active, native=native)
        self._select_goals(dones)
        metadata[:, 9] = self.players
        if not native:
            metadata = metadata.cpu().numpy()
        self.tensor_observations = self._observation()
        return self.tensor_observations, terminal, metadata

    def step_worker_tick(self, actions, active):
        active = active & ~self.exhausted
        collecting = active & self.ready
        waiting = active & ~self.ready
        actions = torch.where(collecting, actions, STAY)
        terminal, metadata, game_done, dones = self._transition(actions, active, collecting, native=True)
        self.wait_horizons = torch.where(collecting, 0, self.wait_horizons + (waiting & game_done))
        self.exhausted |= self.wait_horizons >= 8
        pending = (waiting | dones) & ~self.exhausted
        still_waiting = self._choose_goals(pending)
        self.ready = (self.ready & ~dones) | (pending & ~still_waiting)
        self.tensor_observations = self._observation()
        return self.tensor_observations, terminal, metadata, collecting

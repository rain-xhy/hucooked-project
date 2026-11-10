"""Gymnasium adapters for the original HA² Worker and one-step Manager.

Task rewards are adapted from StephAO/HAHA a206f11 (LICENSE.upstream).
No evaluation-time random action replacement is performed.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction, OvercookedGridworld

from .encoding import egocentric, make_planner
from .subtasks import GOALS, NAMES, UNKNOWN, completed, doable, facing, held_name


class NoExecutableSubtask(RuntimeError):
    """Bounded reset exhausted; record unavailable skills rather than a zero score."""


class CookingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layout, partner, *, stage="manager", worker=None,
                 horizon=400, task_timeout=40, player=None, evaluation=False,
                 worker_shaping=True, unknown_penalty=-0.1, failure_limit=5,
                 deterministic=False):
        super().__init__()
        if stage not in {"worker", "manager"}:
            raise ValueError(f"Invalid stage: {stage}")
        if player not in {None, 0, 1}:
            raise ValueError("player must be 0, 1 or None")
        if stage == "manager" and worker is None:
            raise ValueError("A fixed Worker is required for Manager training")
        if worker is not None and (worker.action_space.n != 6 or
                                   set(worker.observation_space.spaces) != {"visual_obs"}):
            raise ValueError("Worker checkpoint must be the goal-conditioned six-action PPO")
        self.layout, self.partner, self.stage = layout, partner, stage
        self.worker, self.horizon, self.task_timeout = worker, horizon, task_timeout
        self.fixed_player, self.evaluation = player, evaluation
        self.deterministic = deterministic
        self.worker_shaping = worker_shaping
        self.unknown_penalty, self.failure_limit = unknown_penalty, failure_limit
        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.planner = make_planner(self.mdp)
        self.game = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        obs = {"visual_obs": spaces.Box(0, 20, (27, 7, 7), dtype=np.float32)}
        if stage == "manager":
            obs["subtask_mask"] = spaces.MultiBinary(len(NAMES))
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Discrete(6 if stage == "worker" else len(NAMES))
        self.needs_game_reset = True
        self.base_steps = 0
        self.reset_wait_steps = 0
        self.task_counts = np.ones(len(NAMES), dtype=np.float64)
        self.goal = UNKNOWN
        self.player = 0
        self.previous = [UNKNOWN, UNKNOWN]
        self.task_steps = self.failures = self.game_steps = 0

    @property
    def state(self):
        return self.game.state

    def _new_game(self):
        self.game.reset()
        self.player = (int(self.np_random.integers(2)) if self.fixed_player is None
                       else self.fixed_player)
        self.previous = [UNKNOWN, UNKNOWN]
        self.game_steps = self.failures = 0
        self.needs_game_reset = False
        self.valid_counters = [
            self.mdp.find_free_counters_valid_for_player(self.state, self.planner, i)
            for i in range(2)
        ]

    def action_masks(self):
        return doable(self.mdp, self.state, self.player, self.previous[self.player],
                      self.valid_counters)

    def observation(self, goal=None):
        obs = {"visual_obs": egocentric(
            self.mdp, self.state, self.player, self.horizon, goal
        )}
        if self.stage == "manager":
            obs["subtask_mask"] = self.action_masks().astype(np.int8)
        return obs

    def _advance(self, action):
        names = [held_name(p) for p in self.state.players]
        tiles = [facing(self.mdp, p) for p in self.state.players]
        teammate = 1 - self.player
        partner_action = self.partner.act(
            self.mdp, self.state, self.planner, teammate, self.np_random
        )
        joint = [Action.STAY, Action.STAY]
        joint[self.player] = Action.INDEX_TO_ACTION[int(action)]
        joint[teammate] = Action.INDEX_TO_ACTION[int(partner_action)]
        self.last_joint_action = [Action.ACTION_TO_INDEX[a] for a in joint]
        _, reward, done, _ = self.game.step(joint)
        self.base_steps += 1
        self.game_steps += 1
        done = bool(done or self.game_steps >= self.horizon)
        self.needs_game_reset = done
        events = [None, None]
        for i in range(2):
            if joint[i] == Action.INTERACT:
                events[i] = completed(names[i], held_name(self.state.players[i]), tiles[i])
                if events[i] is not None:
                    self.previous[i] = events[i]
        return float(reward), done, events, joint[self.player] == Action.INTERACT

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or (options or {}).get("new_game"):
            self.needs_game_reset = True
        if self.stage == "manager":
            self._new_game()
            self.goal = UNKNOWN
            return self.observation(), {"layout": self.layout, "player": self.player}
        # Keep the kitchen between skill episodes, as in the author implementation.
        # Bounded waiting prevents a weak BC partner from hanging reset forever.
        for _ in range(8):
            if self.needs_game_reset:
                self._new_game()
            for _ in range(self.horizon):
                mask = self.action_masks()
                mask[UNKNOWN] = False
                if mask.any():
                    probs = mask.astype(np.float64) / self.task_counts
                    self.goal = int(self.np_random.choice(len(NAMES), p=probs / probs.sum()))
                    self.task_counts[self.goal] += 1
                    self.task_steps = 0
                    return self.observation(GOALS[self.goal]), {
                        "layout": self.layout, "player": self.player, "subtask": self.goal
                    }
                _, done, _, _ = self._advance(Action.ACTION_TO_INDEX[Action.STAY])
                self.reset_wait_steps += 1
                if done:
                    break
        raise NoExecutableSubtask(
            f"No executable skill in {self.layout}, seat {self.player}; check the BC partner"
        )

    def _proximity(self, features, pickup=False):
        player = self.state.players[self.player]
        x, y = player.position
        dx, dy = player.orientation
        obj = (x + dx, y + dy)
        current, alternative = float("inf"), float("inf")
        walkable = set(self.mdp.get_valid_player_positions())
        for direction in Direction.ALL_DIRECTIONS:
            adjacent = (obj[0] + direction[0], obj[1] + direction[1])
            if adjacent not in walkable:
                continue
            orientation = player.orientation if adjacent == player.position else (
                -direction[0], -direction[1]
            )
            cost = self.planner.motion_planner.min_cost_to_feature(
                (adjacent, orientation), features
            )
            if adjacent == player.position:
                current = cost
            else:
                alternative = min(alternative, cost)
        if not np.isfinite(alternative):
            return 0.0 if pickup else -0.5
        if not np.isfinite(current):
            return 0.0 if pickup else float(np.clip(1 - alternative * .1, -.9, 2))
        delta = alternative - current if pickup else current - alternative
        return float(np.clip(delta * .1, -.9, 2))

    def _skill_bonus(self, task):
        pots = self.mdp.get_pot_locations()
        if task in {3, 6, 9}:
            targets = self.mdp.get_serving_locations() if task == 9 else pots
            return self._proximity(targets)
        if task in {1, 5, 8}:
            targets = self.mdp.get_serving_locations() if task == 8 else pots
            return self._proximity(targets, pickup=True)
        if task == 2:
            p = self.state.players[self.player]
            chosen = (p.position[0] + p.orientation[0], p.position[1] + p.orientation[1])
            counts = {loc: len(self.state.objects[loc].ingredients)
                      if loc in self.state.objects else 0 for loc in pots}
            others = [n for loc, n in counts.items() if loc != chosen]
            return max(0.0, (counts.get(chosen, 0) - 1 - max(others, default=0)) * .1)
        return 1.0 if task == 10 else 0.0

    def step(self, action):
        action = int(np.asarray(action).item())
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.needs_game_reset:
            raise RuntimeError("reset() is required before step()")
        if self.stage == "manager":
            if not self.action_masks()[action]:
                raise ValueError(f"Masked-out Manager task: {NAMES[action]}")
            self.goal = action
            if action == UNKNOWN:
                low_action = Action.ACTION_TO_INDEX[Action.STAY]
                self.previous[self.player] = UNKNOWN
            else:
                obs = {"visual_obs": egocentric(
                    self.mdp, self.state, self.player, self.horizon, GOALS[action]
                )}
                low_action = int(self.worker.predict(obs, deterministic=self.deterministic)[0])
            team_reward, game_done, events, interacted = self._advance(low_action)
            reward = team_reward
            terminated = False
            failed = interacted and events[self.player] != self.goal
            if not self.evaluation:
                reward += self.unknown_penalty if action == UNKNOWN else 0.0
                if interacted:
                    self.failures = self.failures + 1 if failed else 0
                    terminated = self.failure_limit > 0 and self.failures >= self.failure_limit
            info = self._info(team_reward, events, low_action)
            info["worker_failed"] = bool(failed)
            return self.observation(), reward, bool(terminated), game_done, info

        team_reward, game_done, events, interacted = self._advance(action)
        self.task_steps += 1
        success = events[self.player] == self.goal
        invalidated = not self.action_masks()[self.goal] and not interacted
        timeout = self.task_steps >= self.task_timeout
        terminated = interacted or invalidated or timeout
        reward = -0.01
        if interacted:
            reward = 1.0 if success else -1.0
            if success and self.worker_shaping:
                reward += self._skill_bonus(self.goal)
        elif timeout:
            reward = -1.0
        info = self._info(team_reward, events, action)
        info.update(task_success=bool(success), task_invalidated=bool(invalidated),
                    task_timeout=bool(timeout and not interacted), task_steps=self.task_steps)
        return self.observation(GOALS[self.goal]), reward, bool(terminated), game_done, info

    def _info(self, reward, events, low_action):
        return {
            "layout": self.layout, "player": self.player, "subtask": self.goal,
            "completed_subtasks": events, "low_action": int(low_action),
            "joint_action": self.last_joint_action,
            "team_reward": reward, "base_steps": self.base_steps,
            "reset_wait_steps": self.reset_wait_steps,
        }

    def get_base_steps(self):
        return self.base_steps

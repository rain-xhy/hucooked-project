"""Native reference and CUDA rollout adapters for the three-chef hierarchy."""

from functools import partial

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from torch.nn.functional import one_hot

from .artifacts import check_loaded_policy, checkpoint_info
from .bc import BCPartner
from .env import NoExecutableSubtask
from .handoff_planning import HANDOFF_PATIENCE, HANDOFF_RETRY_DELAY, HandoffPreparation
from .mixed_kitchen import HANDOFF_TILE, MixedStep, TensorMixedKitchen, counter_bonus_table, pack_native, select_state
from .mixed_tasks import (
    AUTHOR_TASKS, BUFFER_PUT, BUFFER_TAKE, CERTIFIED_SOUP, EGOCENTRIC_GOAL, EMPTY, GET_DISH, GET_ONION, GET_TOMATO, HANDOFF,
    INTERACT, LEGACY_GOAL, LOAD_POT, NAMES, PLATE_SOUP, PLANNING_NAMES, POOL_BC_FRACTION,
    POOL_NOISE_PROBABILITY, POOL_RANDOM_FRACTION, PREPARE_HANDOFF,
    RETURN_ITEM, SERVE, STAY, UNKNOWN,
    mixed_config, mixed_source, observation_space, skill_mask, worker_outcome,
)
from .tensor_ppo import actor_logits, sample_actions


class NativeMixedKitchen(TensorMixedKitchen):
    """Use the teammate's original simulator and native 187-value encoding."""

    def __init__(self, layouts, horizon):
        super().__init__(layouts, horizon, device="cpu")
        source = mixed_source()
        self.games = [source.MixedKitchen(mixed_config(name, horizon)) for name in layouts]

    def reset(self, generator, mask=None):
        if mask is None:
            seeds = torch.randint(2**31, (len(self.games),), generator=generator).tolist()
            self.views = [game.reset(seed)[0] for game, seed in zip(self.games, seeds)]
        else:
            if not bool(mask.any()):
                return self.last_state
            for index, (game, reset) in enumerate(zip(self.games, mask.tolist())):
                if reset:
                    self.views[index] = game.reset()[0]
        self.last_state = pack_native([game.state for game in self.games], self.device)
        return self.last_state

    def observations(self, state):
        return torch.from_numpy(np.asarray(self.views))

    def step(self, state, actions, active):
        event_types = {
            "ingredient_returned": RETURN_ITEM, "dish_returned": RETURN_ITEM,
            "pot_loaded": LOAD_POT, "soup_plated": PLATE_SOUP, "delivered": SERVE,
        }
        records = []
        for index, (game, joint, enabled) in enumerate(zip(self.games, actions.tolist(), active.tolist())):
            events, progress, aborted = [-1] * 3, [False] * 3, [False] * 3
            if not enabled:
                records.append((0., 0., False, events, progress, aborted))
                continue
            before = game.state
            targets = [
                (chef.position[0] + chef.orientation[0], chef.position[1] + chef.orientation[1])
                if a == INTERACT else None for chef, a in zip(before.chefs, joint)
            ]
            # Buffer moves and dish pickups have no native event. Read the
            # pre-step interaction, so simultaneous expiry cannot hide it.
            for i, target in enumerate(targets):
                if target is None or targets.count(target) != 1:
                    continue
                held = before.chefs[i].held
                if target == game.locations["D"][0] and held is None:
                    events[i] = GET_DISH
                elif target in game.buffer_positions:
                    item = before.buffers[game.buffer_positions.index(target)]
                    if held is None and item is not None:
                        events[i] = BUFFER_TAKE
                    elif held is not None and item is None:
                        events[i] = BUFFER_PUT
            self.views[index], reward, done, _, info = game.step(joint)
            for event in info["events"]:
                name = event["type"]
                if name == "ingredient_taken":
                    events[event["agent"]] = GET_ONION if event["order_id"] % 2 else GET_TOMATO
                elif name in event_types:
                    events[event["agent"]] = event_types[name]
                elif name == "joint_completed":
                    events[event["donor"]] = events[event["receiver"]] = HANDOFF
                elif name == "joint_aborted":
                    for i in before.joint.pair:
                        aborted[i] = True
            for i in game.state.joint.pair:
                progress[i] = True
            records.append((reward, info["raw_reward"], done, events, progress, aborted))
        self.last_state = pack_native([game.state for game in self.games], self.device)
        return MixedStep(self.last_state, *(torch.as_tensor(field) for field in zip(*records)))


class BatchedTeammates:
    """Evaluate the independent map policies together using batched linear layers."""

    def __init__(self, groups, num_envs):
        lanes = [rows.tolist() for rows, _ in groups]
        width = max(map(len, lanes))
        device = groups[0][0].device
        self.lanes = torch.tensor([row + [row[0]] * (width - len(row)) for row in lanes], device=device)
        inverse = [0] * num_envs
        for group, rows in enumerate(lanes):
            for index, row in enumerate(rows):
                inverse[row] = group * width + index
        self.inverse = torch.tensor(inverse, device=device)
        self.layers = [
            (torch.stack([network.layers[index].weight.T for _, network in groups]).contiguous(),
             torch.stack([network.layers[index].bias for _, network in groups])[:, None])
            for index in (0, 2, 4)
        ]

    def __call__(self, views):
        groups, width = self.lanes.shape
        values = views[self.lanes].flatten(1, 2)
        for index, (weights, bias) in enumerate(self.layers):
            values = torch.baddbmm(bias, values, weights)
            if index + 1 < len(self.layers):
                values = values.relu()
        return values.reshape(groups * width, views.shape[1], -1)[self.inverse]


class MixedVecEnv(VecEnv):
    worker_sync_interval = 32

    def __init__(self, args, *, stage=None, evaluation=False, player=None, role="bc", tensor_evaluation=False):
        stage = stage or args.stage
        layouts = tuple(args.layouts[i % len(args.layouts)] for i in range(args.n_envs))
        self.native_collector = tensor_evaluation or (args.vec_env == "tensor-native" and not evaluation)
        self.tensor_backend = tensor_evaluation or (args.vec_env in {"tensor", "tensor-native"} and not evaluation)
        self.tensor_policy = self.native_collector
        self.compile_training = self.native_collector and args.compile_kitchen
        self.kitchen = (TensorMixedKitchen(layouts, args.horizon,
                                           track_metrics=tensor_evaluation and stage == "manager",
                                           fixed_sync_steps=getattr(args, "mixed_sync_steps", None))
                        if self.tensor_backend else
                        NativeMixedKitchen(layouts, args.horizon))
        self.reset_observe = self._reset_observe
        self.worker_outcome = worker_outcome
        if self.tensor_backend and args.compile_kitchen:
            self.kitchen.step = torch.compile(self.kitchen.step, fullgraph=True)
            self.reset_observe = torch.compile(self.reset_observe, fullgraph=True)
            self.skill_mask = torch.compile(skill_mask, fullgraph=True)
            if stage == "worker":
                self.worker_outcome = torch.compile(worker_outcome, fullgraph=True)
        else:
            self.skill_mask = skill_mask
        self.device = self.kitchen.device
        self.generator = torch.Generator(device=self.device)
        self.fixed_player = self.kitchen._tensor(player) if isinstance(player, (tuple, list)) else player
        self.masked = stage == "manager"
        self.worker_goal = getattr(args, "mixed_worker_goal", LEGACY_GOAL)
        self.author_task_rules = getattr(args, "mixed_task_rules", None) == AUTHOR_TASKS
        self.handoff_targets = ((self.kitchen.front, self.kitchen.facilities[:, HANDOFF_TILE])
                                if getattr(args, "mixed_handoff_partner_ready", False) else None)
        planning = (getattr(args, "mixed_handoff_planning", None) or "baseline") if self.masked else "baseline"
        self.guarded_handoff = planning in ("guarded", "delivery")
        self.certified_delivery = planning == "delivery"
        self.track_manager = tensor_evaluation and self.masked
        self.handoff_commit = self.masked and getattr(args, "mixed_handoff_commit", False)
        self.training_pool = self.masked and not evaluation and getattr(args, "mixed_training_pool", False)
        if self.training_pool:
            self.pool_generator = torch.Generator(device=self.device)
        self.trace_joint_actions = self.track_manager and bool(
            getattr(args, "trajectories", 0) or getattr(args, "mixed_abort_events", False)
        )
        self.update_backoff = self._update_backoff
        self.preparation = None
        if planning != "baseline":
            self.preparation = HandoffPreparation(self.kitchen, include_eta=planning == "eta",
                                                  receive_early=self.certified_delivery)
            if self.tensor_backend and args.compile_kitchen:
                self.preparation.action = torch.compile(self.preparation.action, fullgraph=True)
                self.preparation.observe = torch.compile(self.preparation.observe, fullgraph=True)
                if self.guarded_handoff:
                    self.preparation.available_after_backoff = torch.compile(
                        self.preparation.available_after_backoff, fullgraph=True,
                    )
                    self.update_backoff = torch.compile(self._update_backoff, fullgraph=True)
                if self.certified_delivery:
                    self.preparation.serve_certified = torch.compile(self.preparation.serve_certified, fullgraph=True)
        self.names = NAMES if self.preparation is None else PLANNING_NAMES
        self.train_goals = self.kitchen._tensor([i for i in range(len(NAMES)) if i != UNKNOWN])
        self.counter_bonuses = None
        if not self.masked and not getattr(args, "no_worker_shaping", True):
            self.counter_bonuses = torch.stack([counter_bonus_table(layout) for layout in layouts]).to(self.device)
        self.evaluation = evaluation
        self.deterministic = evaluation and args.deterministic
        self.timeout = args.task_timeout
        self.unknown_penalty = args.unknown_penalty
        self.failure_limit = args.failure_limit
        self.base_steps = torch.zeros_like(self.kitchen.rows)
        self.partner_dir = args.run_dir / "bc"
        self.partner_roles = tuple(role.split("+"))
        self.partner_policies = {name: self.load_partner_policy(name) for name in self.partner_roles}
        # One condition is repeated for all replicas before the next seat.
        # Alternating replicas swaps the two partner policies across seats.
        self.partner_swap = self.kitchen.rows.remainder(2).bool()
        if self.masked:
            model = PPO.load(args.worker, device=self.device, n_envs=1)
            check_loaded_policy(model, checkpoint_info(args.worker))
            if model.observation_space != observation_space(False, self.worker_goal):
                raise ValueError("MixedKitchen Worker checkpoint has an incompatible goal observation")
            self.worker = model.policy.eval().requires_grad_(False)
        self.actions = None
        super().__init__(len(layouts), observation_space(self.masked, self.worker_goal, planning),
                         spaces.Discrete(len(self.names) if self.masked else 6))

    def _seats(self):
        if isinstance(self.fixed_player, torch.Tensor):
            return self.fixed_player
        if self.fixed_player is not None:
            return torch.full_like(self.kitchen.rows, self.fixed_player)
        return torch.randint(3, (self.num_envs,), device=self.device, generator=self.generator)

    def _view(self, observations, tasks=None):
        if self.worker_goal == EGOCENTRIC_GOAL:
            return self.kitchen.spatial_observation(
                self.state, self.players, tasks, allow_cooking=self.author_task_rules,
            )
        view = observations[self.kitchen.rows, self.players]
        if tasks is not None:
            goal = one_hot(tasks, len(NAMES))
            view = torch.cat((view, goal.to(view.dtype)), 1)
        return {"visual_obs": view}

    def _observation(self, observations, masks=None):
        if self.masked:
            obs = self._view(observations)
            mask = masks[self.kitchen.rows, self.players]
            if self.preparation is not None:
                available, eta = self.preparation.observe(self.state, self.players)
                if self.guarded_handoff:
                    available = available & self.preparation.available_after_backoff(
                        self.state, self.players, self.handoff_retry_at, self.handoff_ready_at_backoff,
                    )
                mask = torch.cat((mask, available[:, None]), dim=1)
                if eta is not None:
                    obs["handoff_eta"] = eta
            if self.certified_delivery:
                _, serve = self.preparation.serve_certified(self.state, self.players)
                only_serve = torch.arange(len(self.names), device=self.device) == SERVE
                mask = torch.where(serve[:, None], only_serve, mask)
            if self.handoff_commit and not self.evaluation:
                only_handoff = torch.arange(len(self.names), device=self.device) == HANDOFF
                mask = torch.where(self._committed(self.state)[:, None], only_handoff, mask)
            obs["subtask_mask"] = mask
            return obs
        return self._view(observations, self.goals)

    @staticmethod
    def _numpy_obs(observation):
        return {key: (value.to(torch.int8) if key == "subtask_mask" else value).cpu().numpy()
                for key, value in observation.items()}

    @torch.no_grad()
    def reset(self):
        self.observations = self._numpy_obs(self.reset_tensors())
        return self.observations

    @torch.no_grad()
    def reset_tensors(self):
        if self._seeds[0] is not None:
            self.generator.manual_seed(self._seeds[0])
            if self.training_pool:
                self.pool_generator.manual_seed(self._seeds[0] + 310_000)
        self._reset_seeds()
        self._reset_options()
        self.state = self.kitchen.reset(self.generator)
        self.previous = torch.full_like(self.state.held, UNKNOWN) if self.author_task_rules else None
        self.players = self._seats()
        if self.training_pool:
            self.partner_types = self._sample_pool()
            self.pool_episodes = one_hot(self.partner_types, 3).sum((0, 1))
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.failures = torch.zeros_like(self.players)
        if self.guarded_handoff:
            self.handoff_idle = torch.zeros_like(self.players)
            self.handoff_retry_at = torch.zeros_like(self.players)
            self.handoff_ready_at_backoff = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.task_steps = torch.zeros_like(self.players)
        self.wait_horizons = torch.zeros_like(self.players)
        self.exhausted = torch.zeros_like(self.players, dtype=torch.bool)
        self.goals = torch.full_like(self.players, UNKNOWN)
        self.task_counts = torch.ones((self.num_envs, self.train_goals.numel()), device=self.device)
        self.ready = torch.ones_like(self.exhausted)
        self.current_views = self.kitchen.observations(self.state)
        masks = self._available(self.state)
        if not self.masked:
            self.ready = self._assign_goals(self.ready, masks)
            if not self.native_collector:
                self._wait_for_goals()
        self.tensor_observations = self._observation(self.current_views, masks)
        self.actions = None
        return self.tensor_observations

    def _assign_goals(self, pending, masks):
        legal = masks[self.kitchen.rows, self.players][:, self.train_goals]
        ready = pending & legal.any(1)
        noise = torch.empty_like(self.task_counts).exponential_(generator=self.generator)
        slots = (legal / (self.task_counts * noise)).argmax(1)
        self.goals = torch.where(ready, self.train_goals[slots], self.goals)
        self.task_counts[self.kitchen.rows, slots] += ready
        return ready

    def _available(self, state):
        return self.skill_mask(state, self.previous, self.handoff_targets)

    def _reset_observe(self, state, done, seats, draws, previous):
        state = self.kitchen.reset_where(state, done, seats, draws)
        masks = skill_mask(state, previous, self.handoff_targets)
        return state, self.kitchen.observations(state), masks

    def _reset_games(self, done):
        if self.previous is not None:
            self.previous = torch.where(done[:, None], UNKNOWN, self.previous)
        if self.tensor_backend:
            self.state, self.current_views, masks = self.reset_observe(
                self.state, done, self.kitchen.sample_seats(self.generator),
                self.kitchen.draws(self.generator), self.previous,
            )
        else:
            self.state = self.kitchen.reset(self.generator, done)
            self.current_views = self.kitchen.observations(self.state)
            masks = self._available(self.state)
        self.players = torch.where(done, self._seats(), self.players)
        if self.training_pool:
            draws = self._sample_pool()
            self.pool_episodes += (one_hot(draws, 3) * done[:, None, None]).sum((0, 1))
            self.partner_types = torch.where(done[:, None], draws, self.partner_types)
        return masks

    def _sample_pool(self):
        draws = torch.rand((self.num_envs, 2), device=self.device, generator=self.pool_generator)
        return (draws >= POOL_BC_FRACTION).long() + (draws >= 1 - POOL_RANDOM_FRACTION).long()

    def load_partner_policy(self, role):
        layouts = self.kitchen.layouts
        groups = [] if role == "random" else [
            (self.kitchen._tensor([i for i, name in enumerate(layouts) if name == layout]),
             BCPartner.load(self.partner_dir / layout / f"{role}.pt", self.device).network)
            for layout in dict.fromkeys(layouts)
        ]
        batched = None
        if self.tensor_backend and groups and len({net.hidden for _, net in groups}) == 1:
            batched = BatchedTeammates(groups, len(layouts))
        return groups, batched

    def teammate_views(self, teammates):
        return self.current_views[self.kitchen.rows[:, None], teammates]

    @staticmethod
    def partner_logits(views, policy):
        groups, batched = policy
        if batched is not None:
            return batched(views)
        logits = views.new_zeros((*views.shape[:2], 6))
        for group, network in groups:
            logits[group] = network(views[group])
        return logits

    def _joint_actions(self, actions):
        rows = self.kitchen.rows
        teammates = self.kitchen.player_order[self.players, 1:]
        shape = (self.num_envs, 2)
        if self.partner_roles == ("random",):
            partner_actions = torch.randint(6, shape, device=self.device, generator=self.generator)
        else:
            views = self.teammate_views(teammates)
            logits = self.partner_logits(views, self.partner_policies[self.partner_roles[0]])
            if len(self.partner_roles) == 2:
                other = self.partner_logits(views, self.partner_policies[self.partner_roles[1]])
                swapped = torch.stack((self.partner_swap, ~self.partner_swap), 1)
                logits = torch.where(swapped[:, :, None], other, logits)
            partner_actions = sample_actions(logits, self.deterministic, self.generator)
            if self.deterministic and "random" in self.partner_roles:
                uniform_seat = swapped if self.partner_roles[1] == "random" else ~swapped
                random_actions = torch.randint(6, shape, device=self.device, generator=self.generator)
                partner_actions = torch.where(uniform_seat, random_actions, partner_actions)
        if self.training_pool:
            noise = torch.rand(shape, device=self.device, generator=self.pool_generator)
            replace = (self.partner_types == 2) | ((self.partner_types == 1) & (noise < POOL_NOISE_PROBABILITY))
            random_actions = torch.randint(6, shape, device=self.device, generator=self.pool_generator)
            partner_actions = torch.where(replace, random_actions, partner_actions)
        joint = torch.empty((self.num_envs, 3), dtype=torch.long, device=self.device)
        joint.scatter_(1, teammates, partner_actions)
        joint[rows, self.players] = actions
        return joint

    def _advance(self, actions, active, *, freeze_inactive=True):
        joint = self._joint_actions(actions)
        if self.trace_joint_actions:
            self.last_joint_actions = joint
        if self.tensor_backend:
            result = self.kitchen.step(self.state, joint, self.kitchen.draws(self.generator))
            self.state = select_state(active, result.state, self.state) if freeze_inactive else result.state
        else:
            result = self.kitchen.step(self.state, joint, active)
            self.state = result.state
        if self.previous is not None:
            completed = active[:, None] & (result.events >= 0)
            self.previous = torch.where(completed, result.events, self.previous)
        self.base_steps += active
        return result

    def _metadata(self, result, action, reward, done, task, failed, success, invalid, timeout, length):
        # Shared prefix used by TensorRolloutBuffer and the SB3 bridge.
        return torch.stack((reward, done, torch.zeros_like(done), self.players, action, failed,
                            result.raw, self.returns, length, task, success, invalid, timeout), 1)

    def step_worker_tick(self, actions, active, *, capture_terminal=False):
        active = active & ~self.exhausted
        collecting = active & self.ready
        waiting = active & ~self.ready
        rows, players = self.kitchen.rows, self.players
        before = self.state
        result = self._advance(torch.where(collecting, actions, STAY), active)
        self.task_steps += collecting
        legal = self._available(self.state)[rows, players, self.goals]
        reward, done, success, invalid, timeout = self.worker_outcome(
            actions, self.goals, result.events[rows, players], result.continuing[rows, players],
            result.aborted[rows, players], legal, self.task_steps, self.timeout,
        )
        if self.counter_bonuses is not None:
            bonus = self.kitchen.skill_bonus(
                before, players, self.goals, self.counter_bonuses,
            )
            reward = reward + success * bonus
        game_done = active & result.done
        invalid = invalid | (game_done & ~done)
        done = collecting & (done | game_done)
        self.returns += torch.where(collecting, reward, 0.)
        terminal = self._observation(self.kitchen.observations(self.state)) if capture_terminal else None
        metadata = self._metadata(result, actions, reward, done, self.goals, torch.zeros_like(done),
                                  success, invalid, timeout, self.task_steps)
        self.wait_horizons = torch.where(collecting, 0, self.wait_horizons + (waiting & game_done))
        self.exhausted |= self.wait_horizons >= 8
        masks = self._reset_games(game_done)
        pending = (waiting | done) & ~self.exhausted
        self.ready = (self.ready & ~done) | self._assign_goals(pending, masks)
        self.task_steps = torch.where(done, 0, self.task_steps)
        self.returns = torch.where(done, 0., self.returns)
        self.tensor_observations = self._observation(self.current_views)
        return self.tensor_observations, terminal, metadata, collecting

    def _wait_for_goals(self):
        stays = torch.full_like(self.players, STAY)
        while bool((~self.ready).any()):
            self.step_worker_tick(stays, ~self.ready)
            if bool(self.exhausted.any()):
                raise NoExecutableSubtask("MixedKitchen Worker could not find an available skill")

    def _update_backoff(self, before, result, active, preparing, tasks, done):
        rows, players = self.kitchen.rows, self.players
        after = result.state
        moved = (before.position[rows, players] != after.position[rows, players]) | (
            before.orientation[rows, players] != after.orientation[rows, players]
        )
        progressing = moved | result.continuing[rows, players] | (result.events[rows, players] == HANDOFF)
        attending = preparing & (tasks == HANDOFF) & active
        stagnant = attending & ~progressing
        idle = torch.where(attending & progressing, 0, self.handoff_idle + stagnant)
        backoff = stagnant & (idle >= HANDOFF_PATIENCE)
        new_order = active & ((before.batch != after.batch) | done)
        self.handoff_idle = torch.where(backoff | new_order, 0, idle)
        self.handoff_retry_at = torch.where(new_order, 0, torch.where(
            backoff, after.tick + HANDOFF_RETRY_DELAY, self.handoff_retry_at,
        ))
        ready = self.preparation.partner_ready(after, players)
        self.handoff_ready_at_backoff = torch.where(
            new_order[:, None], False, torch.where(backoff[:, None], ready, self.handoff_ready_at_backoff),
        )
        return backoff

    def _committed(self, state):
        remaining = state.sync_steps - state.progress
        due = state.sync_due.clamp_max(self.kitchen.config.horizon)
        return state.pair[self.kitchen.rows, self.players] & (state.tick + remaining <= due)

    def step_tensors(self, actions, *, native=True, active=None):
        if not self.masked:
            _, terminal, metadata, _ = self.step_worker_tick(
                actions, torch.ones_like(self.ready), capture_terminal=True,
            )
            self._wait_for_goals()
            return self.tensor_observations, terminal, metadata if native else metadata.cpu().numpy()
        rows, players = self.kitchen.rows, self.players
        tasks = actions
        if self.preparation is not None:
            preparing = actions == PREPARE_HANDOFF
            tasks = torch.where(preparing, UNKNOWN, actions)
        worker_obs = self._view(self.current_views, tasks)
        if self.tensor_policy:
            low = sample_actions(actor_logits(self.worker, worker_obs), self.deterministic,
                                 self.generator if self.evaluation else None)
        else:
            low = self.worker.get_distribution(worker_obs).get_actions(deterministic=self.deterministic)
        low = torch.where(tasks == UNKNOWN, STAY, low)
        if self.preparation is not None:
            preparation_action, preparation_task = self.preparation.action(self.state, players)
            low = torch.where(preparing, preparation_action, low)
            tasks = torch.where(preparing, preparation_task, tasks)
        if self.certified_delivery:
            serve_action, serve = self.preparation.serve_certified(self.state, players)
            low = torch.where(serve & (tasks == SERVE), serve_action, low)
        freeze_inactive = active is not None
        if active is None:
            active = torch.ones_like(self.ready)
        before = self.state
        if self.handoff_commit:
            committed = active & self._committed(before)
            if self.track_manager:
                self.handoff_commit_stats = torch.stack((committed, committed & (low != INTERACT)), 1)
            low = torch.where(committed, INTERACT, low)
            tasks = torch.where(committed, HANDOFF, tasks)
        result = self._advance(low, active, freeze_inactive=freeze_inactive)
        handoff = tasks == HANDOFF
        attempt = ((low == INTERACT) & ~(handoff & result.continuing[rows, players])) | (
            handoff & result.aborted[rows, players]
        )
        success = result.events[rows, players] == tasks
        attempt = attempt & active
        failed = attempt & ~success
        self.failures = torch.where(attempt, torch.where(failed, self.failures + 1, 0), self.failures)
        done = result.done
        reward = result.reward
        if not self.evaluation:
            done = done | ((self.failures >= self.failure_limit) & (self.failure_limit > 0))
            reward = reward + (actions == UNKNOWN) * self.unknown_penalty
        done = done & active
        backoff = (self.update_backoff(before, result, active, preparing, tasks, done) if self.guarded_handoff
                   else torch.zeros_like(done))
        if self.track_manager:
            own_seat = self.kitchen.eye[players]
            received = (result.events == HANDOFF) & (before.held == EMPTY)
            delivered = (result.events == SERVE) & (before.held == CERTIFIED_SOUP)
            self.handoff_diagnostics = torch.stack((
                (received & own_seat).any(1), (received & ~own_seat).any(1),
                (delivered & own_seat).any(1), (delivered & ~own_seat).any(1), backoff,
                ((result.events == HANDOFF) & ~received & own_seat).any(1),
            ), 1)
        reward = torch.where(active, reward, 0.)
        self.returns += reward
        terminal = None
        if not (native and self.native_collector):
            masks = self._available(self.state)
            terminal = self._observation(self.kitchen.observations(self.state), masks)
        empty = torch.zeros_like(done)
        metadata = self._metadata(result, low, reward, done, actions, failed, success, empty, empty, self.state.tick)
        if self.evaluation and not self.tensor_backend:
            self.game_metrics = [game.state.metrics for game in self.kitchen.games]
        self.step_metrics = result.metrics
        masks = self._reset_games(done)
        self.returns = torch.where(done, 0., self.returns)
        self.failures = torch.where(done, 0, self.failures)
        self.tensor_observations = self._observation(self.current_views, masks)
        return self.tensor_observations, terminal, metadata if native else metadata.cpu().numpy()

    def _event(self, index, row):
        info = dict(layout=self.kitchen.layouts[index], player=int(row[3]), subtask=int(row[9]),
                    low_action=int(row[4]), team_reward=float(row[6]))
        if self.masked:
            info["worker_failed"] = bool(row[5])
        else:
            info.update(task_success=bool(row[10]), task_invalidated=bool(row[11]),
                        task_timeout=bool(row[12]), task_steps=int(row[8]))
        if row[1]:
            info["episode"] = {"r": float(row[7]), "l": int(row[8])}
        return info

    def rollout_events(self, metadata):
        for step, index in np.argwhere((metadata[:, :, 1] != 0) | (metadata[:, :, 5] != 0)):
            yield step, index, self._event(index, metadata[step, index])

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)

    @torch.no_grad()
    def step_wait(self):
        actions, self.actions = self.actions, None
        if actions is None:
            raise RuntimeError("step_async must precede step_wait")
        if (np.any(actions < 0) or np.any(actions >= self.action_space.n) or
                self.masked and not self.observations["subtask_mask"][np.arange(self.num_envs), actions].all()):
            raise ValueError("Invalid MixedKitchen action")
        observation, terminal, metadata = self.step_tensors(self.kitchen._tensor(actions), native=False)
        infos = [self._event(i, row) for i, row in enumerate(metadata)]
        done = metadata[:, 1].astype(bool)
        if done.any():
            terminal = self._numpy_obs(terminal)
            for i in np.flatnonzero(done):
                infos[i]["terminal_observation"] = {key: value[i].copy() for key, value in terminal.items()}
                infos[i]["TimeLimit.truncated"] = False
                if self.evaluation and self.masked:
                    infos[i]["game_metrics"] = self.game_metrics[i]
        self.observations = self._numpy_obs(observation)
        return self.observations, metadata[:, 0], done, infos

    def action_masks(self, index):
        return self.observations["subtask_mask"][index]

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        if attr_name == "action_masks" and self.masked:
            return [partial(self.action_masks, i) for i in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks" and self.masked:
            return [self.action_masks(i) for i in self._get_indices(indices)]
        if method_name == "get_base_steps":
            counts = self.base_steps.tolist()
            return [counts[i] for i in self._get_indices(indices)]
        raise AttributeError(method_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(attr_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.actions = None

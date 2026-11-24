"""Compare immediate, delayed and ordinary actions with short GPU rollouts."""

from argparse import Namespace

import torch

from .mixed_env import MixedVecEnv
from .mixed_tasks import CERTIFIED_SOUP, HANDOFF, ONION_SOUP, PREPARE_HANDOFF
from .tensor_ppo import actor_logits

LOOKAHEAD_STEPS = 12
LOOKAHEAD_SAMPLES = 3
PREPARATION_DELAY = 3
WINDOW_CREDIT = 8.0


class HandoffLookahead:
    def __init__(self, args, env, policy, seed):
        self.policy = policy
        self.window = args.mixed_lookahead == "window"
        # The forecast always uses the training BC policy. The real evaluator
        # can use held-out partners, and never supplies their policy or role here.
        layouts = [layout for layout in env.kitchen.layouts for _ in range(3 * LOOKAHEAD_SAMPLES)]
        shadow_options = Namespace(**{**vars(args), "layouts": layouts, "n_envs": len(layouts),
                                      "trajectories": 0, "compile_kitchen": False, "deterministic": False,
                                      "mixed_handoff_commit": False, "mixed_abort_events": False})
        self.shadow = MixedVecEnv(shadow_options, stage="manager", evaluation=True, role="bc",
                                 tensor_evaluation=True)
        self.shadow.seed(seed)
        self.shadow.reset_tensors()
        self.source = torch.arange(env.num_envs, device=env.device).repeat_interleave(3 * LOOKAHEAD_SAMPLES)
        self.branch = torch.arange(3, device=env.device).repeat_interleave(LOOKAHEAD_SAMPLES).repeat(env.num_envs)
        indices = torch.arange(env.action_space.n, device=env.device)
        self.ordinary = (indices != HANDOFF) & (indices != PREPARE_HANDOFF)

    def choose(self, env, actions, logits, active):
        obs = env.tensor_observations
        eligible = active & obs["subtask_mask"][:, PREPARE_HANDOFF] & (env.state.held == ONION_SOUP).any(1)
        if not bool(eligible.any()):
            return actions, eligible

        ordinary = logits.masked_fill(~self.ordinary, -1e8).argmax(1)
        # Keep the sampled ordinary action if the Manager already selected one.
        ordinary = torch.where(self.ordinary[actions], actions, ordinary)
        sim, source = self.shadow, self.source
        sim.state = type(env.state)(*(tensor[source] for tensor in env.state))
        sim.players = env.players[source]
        if env.previous is not None:
            sim.previous = env.previous[source]
        sim.handoff_idle = env.handoff_idle[source]
        sim.handoff_retry_at = env.handoff_retry_at[source]
        sim.handoff_ready_at_backoff = env.handoff_ready_at_backoff[source]
        sim.returns.zero_()
        sim.failures.zero_()
        sim.current_views = sim.kitchen.observations(sim.state)
        current = sim._observation(sim.current_views, sim._available(sim.state))

        origin = sim.state.batch
        alive = eligible[source]
        returns = torch.zeros_like(alive, dtype=torch.float32)
        discount = 1.
        for depth in range(LOOKAHEAD_STEPS):
            mask = current["subtask_mask"]
            if depth == 0:
                fallback = ordinary[source]
            else:
                fallback = actor_logits(self.policy, current).masked_fill(~(mask & self.ordinary), -1e8).argmax(1)
            want_handoff = (self.branch == 0) | ((self.branch == 1) & (depth >= PREPARATION_DELAY))
            planned = torch.where(want_handoff & mask[:, PREPARE_HANDOFF], PREPARE_HANDOFF, fallback)
            current, _, meta = sim.step_tensors(planned, active=alive)
            returns += discount * meta[:, 6] * alive
            alive = alive & ~meta[:, 1].bool()
            discount *= .99

        terminal = self.policy.predict_values(current).flatten()
        returns += discount * terminal * alive
        state = sim.state
        delivery_time = sim.preparation.serve_distance(state)
        viable = (state.held == CERTIFIED_SOUP) & (
            state.tick[:, None] + delivery_time <= state.due[:, 1, None].clamp_max(sim.kitchen.config.horizon)
        )
        pending_window = alive & (state.batch == origin) & viable.any(1)
        score = returns.reshape(env.num_envs, 3, LOOKAHEAD_SAMPLES).mean(2)
        if self.window:
            # A certified, undelivered item is evidence of a completed
            # continuous same-pair window. Expired or already paid orders do
            # not receive this terminal heuristic.
            credit = pending_window.reshape(env.num_envs, 3, LOOKAHEAD_SAMPLES).float().mean(2)
            score = score + discount * WINDOW_CREDIT * credit
        prefer_prepare = score[:, 0] > score[:, 1:].amax(1)
        tie = score[:, 0] == score[:, 1:].amax(1)
        prefer_prepare |= tie & (actions == PREPARE_HANDOFF)
        chosen = torch.where(prefer_prepare, PREPARE_HANDOFF, ordinary)
        return torch.where(eligible, chosen, actions), eligible

    def close(self):
        self.shadow.close()

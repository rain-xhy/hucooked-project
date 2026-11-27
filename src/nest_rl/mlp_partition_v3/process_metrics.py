"""Transition-derived diagnostics; cross-wall means a pickup by the other side."""
import torch
from overcooked_ai_py.cooperative.partition.model import WALL, WIDTH, STAY, INTERACT

EXTRA_NAMES = ("dual_pots_active_steps", "dual_pots_progress_steps", "idle_with_outstanding_work_steps",
               "cross_wall_pickups", "cross_wall_raw_pickups", "cross_wall_cut_pickups", "cross_wall_meal_pickups",
               "counter_wait_raw_steps", "counter_raw_pickups", "counter_wait_cut_steps", "counter_cut_pickups",
               "counter_wait_meal_steps", "counter_meal_pickups", "idle_actionable_ingredient_steps")


class ProcessMetrics:
    def __init__(self, env):
        self.env = env
        self.depositor = torch.full_like(env.state["counter"], -1)

    def before(self, actions):
        self.old = {k: v.clone() for k, v in self.env.state.items()}
        self.actions = actions.clone()

    def after(self):
        s, p = self.env.state, self.old
        result = torch.zeros(self.env.num_envs, len(EXTRA_NAMES), dtype=torch.long, device=self.env.device)
        result[:, 0] = (s["mix"] != 0).all(-1)
        progress = (s["mix"] > p["mix"]) | (s["heat"] > p["heat"]) | ((p["heat"] == 4) & (s["mix"] == 0))
        result[:, 1] = progress.all(-1)
        # Opportunity proxy: a pending ticket or physical work remains in the kitchen.
        work = (p["ids"] >= 0).any(-1) | (p["held"] != 0).any(-1) | (p["mix"] != 0).any(-1)
        result[:, 2] = ((self.actions == STAY) & work[:, None]).sum(-1)
        target = p["pos"] + self.env.offsets[p["facing"]]
        side = (p["pos"] % WIDTH > 6).long()
        for j, station in enumerate(WALL):
            at = (self.actions == INTERACT) & (target == station)
            actor_side = (at.long() * side).sum(-1)
            put = (p["counter"][:, j] == 0) & (s["counter"][:, j] != 0)
            take = (p["counter"][:, j] != 0) & (s["counter"][:, j] == 0)
            cross = take & (self.depositor[:, j] >= 0) & (actor_side != self.depositor[:, j])
            item = p["counter"][:, j]
            wait = s["tick"][:, 0] - p["deposited"][:, j]
            result[:, 3] += cross
            for k, kind in enumerate(((item >= 1) & (item <= 4), (item >= 5) & (item <= 6), item >= 7)):
                result[:, 4 + k] += cross & kind
                result[:, 7 + 2*k] += wait * (take & kind)
                result[:, 8 + 2*k] += take & kind
            self.depositor[:, j] = torch.where(put, actor_side, torch.where(take, -1, self.depositor[:, j]))
        return result

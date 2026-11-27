"""Pure-PyTorch zero-sync GPU-vectorized two-pot parallel partner planner (parallel-gpu-v3)."""
import torch

from overcooked_ai_py.cooperative.partition.model import (
    BEEF, BOARDS, CUT_POTATO, CUT_TOMATO, DELIVERY_SIDE, EGG, EMPTY,
    FIRST_MEAL, INTERACT, OFFSETS, POTATO, POTS, RECIPES, SERVERS, SOURCES,
    STAY, TOMATO, WALL, WIDTH, cell,
)
from .legacy_partners import _routing
from .partners import _distances

_ITEMS = (CUT_TOMATO, CUT_POTATO, EGG, BEEF)
_RAW_ITEMS = (TOMATO, POTATO, EGG, BEEF)
_ITEM_BITS = (1, 2, 4, 8)
_RAW_SIDE = (0, 0, 1, 1)
_RAW_SOURCE = (SOURCES[TOMATO], SOURCES[POTATO], SOURCES[EGG], SOURCES[BEEF])
_BOARD_SIDE = (0, 0, 1, 1)
_PARK_UPPER = cell(11, 1)
_PARK_LOWER = cell(11, 5)
_BIG = 10_000
_STATE_KEYS = ("pos", "facing", "held", "ids", "due", "recipe",
               "board", "counter", "mix", "locked", "heat")


class GpuPartners:
    """Two-pot parallel planner running entirely on the environment's torch device.

    Uses branchless fixed-shape tensor ops with zero host-device synchronization
    in actions(), and automatically captures a CUDA Graph on CUDA devices to
    collapse per-step kernel launch overhead into a single graph replay.
    """

    version = "parallel-gpu-v3"

    def __init__(self, env, roles, styles, learner_mask=None, use_cuda_graph=True):
        if env.config.layout != "counter":
            raise ValueError("Scripted partners require the counter layout")
        self.env = env
        self.device = env.device
        self.use_cuda_graph = bool(use_cuda_graph and self.device.type == "cuda")
        n = env.num_envs
        self.roles = self._labels(roles)
        self.styles = self._labels(styles)
        if learner_mask is None:
            self.learner_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(learner_mask, dtype=torch.bool, device=self.device).clone()
            if mask.shape != (n,):
                raise ValueError("learner_mask must contain one boolean per kitchen")
            self.learner_mask = mask
        self.rows = torch.arange(n, dtype=torch.long, device=self.device)
        self.paths = torch.as_tensor(_routing(), dtype=torch.uint8, device=self.device)
        self.dist = torch.as_tensor(_distances(), dtype=torch.int32, device=self.device)
        self._recipe_bits = torch.tensor(RECIPES, dtype=torch.long, device=self.device)
        self._recipe_counts = torch.tensor([int(b).bit_count() for b in RECIPES], dtype=torch.long, device=self.device)
        self._delivery_side = torch.tensor(DELIVERY_SIDE, dtype=torch.long, device=self.device)
        self._pots = torch.tensor(POTS, dtype=torch.long, device=self.device)
        self._boards = torch.tensor(BOARDS, dtype=torch.long, device=self.device)
        self._board_side = torch.tensor(_BOARD_SIDE, dtype=torch.long, device=self.device)
        self._wall = torch.tensor(WALL, dtype=torch.long, device=self.device)
        self._servers = torch.tensor(SERVERS, dtype=torch.long, device=self.device)
        self._offsets = torch.tensor((*OFFSETS, 0, 0), dtype=torch.long, device=self.device)
        self._slot_idx = torch.arange(3, dtype=torch.long, device=self.device)[None, :].expand(n, 3).contiguous()
        self._one_hot_2 = torch.eye(2, dtype=torch.bool, device=self.device)
        self._one_hot_3 = torch.eye(3, dtype=torch.bool, device=self.device)
        self._one_hot_4 = torch.eye(4, dtype=torch.bool, device=self.device)
        self._one_hot_5 = torch.eye(5, dtype=torch.bool, device=self.device)
        self._bits_t = torch.tensor(_ITEM_BITS, dtype=torch.long, device=self.device)[None, None, :]
        self._pot_arange = torch.arange(2, dtype=torch.long, device=self.device)[None, :]
        self._wall_cols = [torch.full((n,), WALL[k], dtype=torch.long, device=self.device) for k in range(5)]
        self._pot_cols = [torch.full((n,), POTS[p], dtype=torch.long, device=self.device) for p in range(2)]
        self._side_cols = [torch.full((n,), s, dtype=torch.long, device=self.device) for s in range(2)]
        self._src_cols = [torch.full((n,), _RAW_SOURCE[m], dtype=torch.long, device=self.device) for m in range(4)]

        item_to_idx = [-1] * (FIRST_MEAL + 3)
        item_to_idx[TOMATO] = 0
        item_to_idx[CUT_TOMATO] = 0
        item_to_idx[POTATO] = 1
        item_to_idx[CUT_POTATO] = 1
        item_to_idx[EGG] = 2
        item_to_idx[BEEF] = 3
        self._item_to_idx = torch.tensor(item_to_idx, dtype=torch.long, device=self.device)

        base = torch.zeros((2, 3), dtype=torch.int32, device=self.device)
        for p in range(2):
            for r in range(3):
                c = int(self.dist[POTS[p], SERVERS[DELIVERY_SIDE[r]]].item())
                for m, bit in enumerate(_ITEM_BITS):
                    if RECIPES[r] & bit:
                        if m < 2:
                            opts = [int(self.dist[_RAW_SOURCE[m], b].item() + self.dist[b, POTS[p]].item()) + 3
                                    for b in BOARDS]
                            c += min(opts)
                        else:
                            c += int(self.dist[_RAW_SOURCE[m], POTS[p]].item())
                base[p, r] = c
        self._base_pot_cost = base

        # Pre-allocate persistent buffers (never reallocated, so CUDAGraph pointers stay valid).
        self.chef_ids = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        self.learner_ids = torch.zeros((n,), dtype=torch.long, device=self.device)
        self.ticket_ids = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.job_ticket_ids = torch.full((n, 2), -1, dtype=torch.long, device=self.device)
        self.job_recipes = torch.full((n, 2), -1, dtype=torch.long, device=self.device)
        self.previous_mix = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.destinations = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        self.prev_task_type = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        self.prev_task_pot = torch.full((n, 3), -1, dtype=torch.long, device=self.device)
        self.eligible_work = torch.zeros((n, 3), dtype=torch.bool, device=self.device)
        self.idle_actionable = torch.zeros((n, 3), dtype=torch.bool, device=self.device)
        self._moves_out = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        self._state_buf = {k: torch.empty_like(env.state[k], dtype=torch.long, device=self.device) for k in _STATE_KEYS}
        self._graph = None
        self.reset()

    def _labels(self, value):
        t = torch.as_tensor(value, dtype=torch.long, device=self.device).clone()
        if t.shape != (self.env.num_envs,) or bool(((t < 0) | (t > 2)).any()):
            raise ValueError("Expected one integer label 0..2 per kitchen")
        return t

    def reset(self):
        pos = self.env.state["pos"]
        self.chef_ids.copy_(torch.where(pos % WIDTH < 6, -1, pos).argsort(dim=1))
        self._refresh_learner_ids()
        self.ticket_ids.fill_(-1)
        self.job_ticket_ids.fill_(-1)
        self.job_recipes.fill_(-1)
        self.previous_mix.zero_()
        self.destinations.zero_()
        self.prev_task_type.zero_()
        self.prev_task_pot.fill_(-1)
        self.eligible_work.zero_()
        self.idle_actionable.zero_()

    def _refresh_learner_ids(self):
        self.learner_ids.copy_(self.chef_ids[self.rows, self.roles])

    def state_dict(self):
        keys = ("roles", "styles", "chef_ids", "ticket_ids", "learner_mask",
                "job_ticket_ids", "job_recipes", "previous_mix", "destinations",
                "prev_task_type", "prev_task_pot", "eligible_work", "idle_actionable")
        result = {k: getattr(self, k).detach().cpu().clone() for k in keys}
        result["version"] = self.version
        return result

    def load_state_dict(self, state):
        if state.get("version") != self.version:
            raise ValueError("Partner checkpoint version mismatch")
        for key in ("roles", "styles", "learner_mask"):
            if not torch.equal(getattr(self, key).cpu(), state[key].detach().cpu()):
                raise ValueError("Partner checkpoint configuration mismatch: " + key)
        for key in ("chef_ids", "ticket_ids", "job_ticket_ids", "job_recipes",
                    "previous_mix", "destinations", "prev_task_type", "prev_task_pot",
                    "eligible_work", "idle_actionable"):
            incoming = state[key].to(self.device)
            target = getattr(self, key)
            if incoming.shape != target.shape:
                raise ValueError("Partner checkpoint shape mismatch: " + key)
            target.copy_(incoming)
        self._refresh_learner_ids()

    def diagnostics(self):
        return dict(version=self.version,
                    job_ticket_ids=self.job_ticket_ids.detach().cpu().numpy(),
                    job_recipes=self.job_recipes.detach().cpu().numpy(),
                    destinations=self.destinations.detach().cpu().numpy(),
                    idle_actionable=self.idle_actionable.detach().cpu().numpy(),
                    eligible_work=self.eligible_work.detach().cpu().numpy())

    def _nearest_drop(self, pos_i, counter, claimed_counter, used_dest):
        avail = (counter == EMPTY) & (~claimed_counter) & (~used_dest[:, self._wall])
        d = torch.where(avail, self.dist[pos_i[:, None], self._wall[None, :]], _BIG)
        best_d, best_k = d.min(dim=1)
        ok = best_d < _BIG
        return ok, best_k, torch.where(ok, self._wall[best_k], 0)

    def _nearest_board(self, pos_i, side_i, board, claimed_board, used_dest):
        avail = ((board == EMPTY) & (~claimed_board) & (~used_dest[:, self._boards]) &
                 (self._board_side[None, :] == side_i[:, None]))
        d = torch.where(avail, self.dist[pos_i[:, None], self._boards[None, :]], _BIG)
        best_d, best_k = d.min(dim=1)
        ok = best_d < _BIG
        return ok, best_k, torch.where(ok, self._boards[best_k], 0)

    def _pick_chef(self, target_cell, work_side, pos, side, hand, busy, controlled):
        cand = controlled & (~busy) & (hand == EMPTY) & (side == work_side[:, None])
        d = torch.where(cand, self.dist[pos, target_cell[:, None]], _BIG)
        best_d, best_i = d.min(dim=1)
        return best_d < _BIG, best_i

    def _copy_state_in(self):
        state = self.env.state
        buf = self._state_buf
        for k in _STATE_KEYS:
            buf[k].copy_(state[k])

    def _snapshot_mutable(self):
        return (self.ticket_ids.clone(), self.job_ticket_ids.clone(), self.job_recipes.clone(),
                self.previous_mix.clone(), self.destinations.clone(), self.prev_task_type.clone(),
                self.prev_task_pot.clone(), self.eligible_work.clone(), self.idle_actionable.clone())

    def _restore_mutable(self, snap):
        (self.ticket_ids.copy_(snap[0]), self.job_ticket_ids.copy_(snap[1]),
         self.job_recipes.copy_(snap[2]), self.previous_mix.copy_(snap[3]),
         self.destinations.copy_(snap[4]), self.prev_task_type.copy_(snap[5]),
         self.prev_task_pot.copy_(snap[6]), self.eligible_work.copy_(snap[7]),
         self.idle_actionable.copy_(snap[8]))

    def _capture_graph(self):
        snap = self._snapshot_mutable()
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._actions_impl()
                self._restore_mutable(snap)
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._actions_impl()
        self._restore_mutable(snap)
        self._graph = g

    @torch.no_grad()
    def actions(self):
        self._copy_state_in()
        if self.use_cuda_graph:
            if self._graph is None:
                self._capture_graph()
            self._graph.replay()
        else:
            self._actions_impl()
        return self._moves_out.clone()

    def _actions_impl(self):
        s = self._state_buf
        n = self.env.num_envs
        rows = self.rows
        pos = s["pos"]
        facing = s["facing"]
        hand = s["held"]
        ids = s["ids"]
        due = s["due"]
        recipe = s["recipe"]
        board = s["board"]
        counter = s["counter"]
        mix = s["mix"]
        locked = s["locked"]
        heat = s["heat"]

        side = (pos % WIDTH > 6).long()
        controlled = ~(self._one_hot_3[self.learner_ids] & self.learner_mask[:, None])

        # 1. Rank the 3 live tickets according to partner style.
        counts = self._recipe_counts[recipe]
        key = torch.where(self.styles[:, None] == 1, due,
                          torch.where(self.styles[:, None] == 2, counts, self._slot_idx))
        rank = (key * 4 + self._slot_idx).argsort(dim=1)
        ranked_ids = ids.gather(1, rank)
        ranked_rec = recipe.gather(1, rank)
        self.ticket_ids.copy_(ranked_ids[:, 0])

        # 2. Reserve tickets already covered by in-transit meals (hands + counters).
        used = torch.zeros((n, 3), dtype=torch.bool, device=self.device)
        in_transit = torch.cat((hand, counter), dim=1)
        for slot in range(8):
            item = in_transit[:, slot]
            is_meal = item >= FIRST_MEAL
            meal_r = (item - FIRST_MEAL).clamp(0, 2)
            match = is_meal[:, None] & (~used) & (ranked_rec == meal_r[:, None])
            has = match.any(dim=1)
            first = match.long().argmax(dim=1)
            used |= self._one_hot_3[first] & has[:, None]

        # 3. Reconcile existing pot jobs.
        cleared = (self.previous_mix != 0) & (mix == 0)
        jobs = torch.where(cleared, -1, self.job_ticket_ids)
        jrec = torch.where(cleared, -1, self.job_recipes)
        rbits_ranked = self._recipe_bits[ranked_rec]

        for pot in range(2):
            pmix = mix[:, pot]
            plock = locked[:, pot]
            valid = (~used) & ((plock[:, None] < 0) | (ranked_rec == plock[:, None])) & ((pmix[:, None] & rbits_ranked) == pmix[:, None])
            keep_m = valid & (ranked_ids == jobs[:, pot : pot + 1])
            keep = keep_m.any(dim=1)
            keep_slot = keep_m.long().argmax(dim=1)
            used |= self._one_hot_3[keep_slot] & keep[:, None]

            need_rebind = (~keep) & (pmix != 0)
            has_v = need_rebind & valid.any(dim=1)
            v_slot = valid.long().argmax(dim=1)
            used |= self._one_hot_3[v_slot] & has_v[:, None]

            compat = (pmix[:, None] & self._recipe_bits[None, :]) == pmix[:, None]
            fallback_r = torch.where(plock >= 0, plock, compat.long().argmax(dim=1))

            jobs[:, pot] = torch.where(keep, jobs[:, pot], torch.where(has_v, ranked_ids[rows, v_slot], -1))
            jrec[:, pot] = torch.where(keep, jrec[:, pot],
                                       torch.where(has_v, ranked_rec[rows, v_slot],
                                                   torch.where(need_rebind, fallback_r, -1)))

        # 4. Jointly assign top-2 unused tickets to free pots.
        free0 = jrec[:, 0] < 0
        free1 = jrec[:, 1] < 0
        avail = ~used
        has0 = avail.any(dim=1)
        s0 = avail.long().argmax(dim=1)
        avail2 = avail & (~(self._one_hot_3[s0] & has0[:, None]))
        has1 = avail2.any(dim=1)
        s1 = avail2.long().argmax(dim=1)

        d_p0 = torch.where(controlled & (side == 0), self.dist[pos, POTS[0]], _BIG).min(dim=1).values
        d_p1 = torch.where(controlled & (side == 1), self.dist[pos, POTS[1]], _BIG).min(dim=1).values
        wd = torch.stack((torch.where(d_p0 < _BIG, d_p0, 20), torch.where(d_p1 < _BIG, d_p1, 20)), dim=1)
        pot_cost = self._base_pot_cost[None, :, :] + wd[:, :, None]

        r0, r1 = ranked_rec[rows, s0], ranked_rec[rows, s1]
        id0, id1 = ranked_ids[rows, s0], ranked_ids[rows, s1]

        both_free = free0 & free1 & has0
        both_two = both_free & has1
        swap = (pot_cost[rows, 1, r0] + pot_cost[rows, 0, r1]) < (pot_cost[rows, 0, r0] + pot_cost[rows, 1, r1])
        jobs[:, 0] = torch.where(both_two, torch.where(swap, id1, id0), jobs[:, 0])
        jrec[:, 0] = torch.where(both_two, torch.where(swap, r1, r0), jrec[:, 0])
        jobs[:, 1] = torch.where(both_two, torch.where(swap, id0, id1), jobs[:, 1])
        jrec[:, 1] = torch.where(both_two, torch.where(swap, r0, r1), jrec[:, 1])

        both_one = both_free & (~has1)
        prefer_p1 = pot_cost[rows, 1, r0] < pot_cost[rows, 0, r0]
        jobs[:, 0] = torch.where(both_one & (~prefer_p1), id0, jobs[:, 0])
        jrec[:, 0] = torch.where(both_one & (~prefer_p1), r0, jrec[:, 0])
        jobs[:, 1] = torch.where(both_one & prefer_p1, id0, jobs[:, 1])
        jrec[:, 1] = torch.where(both_one & prefer_p1, r0, jrec[:, 1])

        only0 = free0 & (~free1) & has0
        only1 = (~free0) & free1 & has0
        jobs[:, 0] = torch.where(only0, id0, jobs[:, 0])
        jrec[:, 0] = torch.where(only0, r0, jrec[:, 0])
        jobs[:, 1] = torch.where(only1, id0, jobs[:, 1])
        jrec[:, 1] = torch.where(only1, r0, jrec[:, 1])

        self.job_ticket_ids.copy_(jobs)
        self.job_recipes.copy_(jrec)
        self.previous_mix.copy_(mix)

        # Compute missing ingredients per (pot, item_idx 0..3) [N, 2, 4].
        open_pot = (jrec >= 0) & (locked < 0)
        rb_pots = self._recipe_bits[jrec.clamp(0, 2)]
        need = open_pot[:, :, None] & ((rb_pots[:, :, None] & self._bits_t) != 0) & ((mix[:, :, None] & self._bits_t) == 0)

        # 5. Station & task reservations (all branchless).
        dest = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        busy = ~controlled
        claimed_board = torch.zeros((n, 4), dtype=torch.bool, device=self.device)
        claimed_counter = torch.zeros((n, 5), dtype=torch.bool, device=self.device)
        used_dest = torch.zeros((n, 91), dtype=torch.bool, device=self.device)
        task_type = torch.zeros((n, 3), dtype=torch.long, device=self.device)
        task_pot = torch.full((n, 3), -1, dtype=torch.long, device=self.device)

        # 5.1 Lock chefs continuing cutting or pot-tending, and account for their board's ingredient.
        for i in range(3):
            avail_i = (~busy[:, i]) & (hand[:, i] == EMPTY)
            b_match = (avail_i[:, None] & (self.prev_task_type[:, i : i + 1] == 1) &
                       (self.destinations[:, i : i + 1] == self._boards[None, :]) & (board != EMPTY))
            has_cut = b_match.any(dim=1)
            bk = b_match.long().argmax(dim=1)
            bcell = self._boards[bk]
            dest[:, i] = torch.where(has_cut, bcell, dest[:, i])
            busy[:, i] |= has_cut
            claimed_board |= self._one_hot_4[bk] & has_cut[:, None]
            used_dest.scatter_(1, bcell[:, None], used_dest.gather(1, bcell[:, None]) | has_cut[:, None])
            task_type[:, i] = torch.where(has_cut, 1, task_type[:, i])
            bitem = board[rows, bk]
            bm = self._item_to_idx[bitem.clamp(0, FIRST_MEAL + 2)].clamp(0, 3)
            bpot = self.prev_task_pot[:, i].clamp(0, 1)
            n0 = need[rows, 0, bm]
            n1 = need[rows, 1, bm]
            clear_p = torch.where((self.prev_task_pot[:, i] >= 0) & need[rows, bpot, bm], bpot,
                                  torch.where(n1 & ((~n0) | (side[:, i] == 1)), 1, 0))
            do_clear = has_cut & (n0 | n1)
            task_pot[:, i] = torch.where(has_cut, clear_p, task_pot[:, i])
            clear_mask = self._one_hot_2[clear_p][:, :, None] & self._one_hot_4[bm][:, None, :] & do_clear[:, None, None]
            need = need & (~clear_mask)

            avail_i = (~busy[:, i]) & (hand[:, i] == EMPTY)
            p_match = (avail_i[:, None] & (self.prev_task_type[:, i : i + 1] == 2) &
                       (self.prev_task_pot[:, i : i + 1] == self._pot_arange) &
                       (mix != 0) & (heat > 0))
            has_cook = p_match.any(dim=1)
            pk = p_match.long().argmax(dim=1)
            pcell = self._pots[pk]
            dest[:, i] = torch.where(has_cook, pcell, dest[:, i])
            busy[:, i] |= has_cook
            used_dest.scatter_(1, pcell[:, None], used_dest.gather(1, pcell[:, None]) | has_cook[:, None])
            task_type[:, i] = torch.where(has_cook, 2, task_type[:, i])
            task_pot[:, i] = torch.where(has_cook, pk, task_pot[:, i])

        # 5.2 Deliver or transfer ready meals in hands.
        for i in range(3):
            has_meal = (~busy[:, i]) & (hand[:, i] >= FIRST_MEAL)
            mr = (hand[:, i] - FIRST_MEAL).clamp(0, 2)
            live = (recipe == mr[:, None]).any(dim=1)
            deliver = has_meal & live & (side[:, i] == self._delivery_side[mr])
            serve_cell = self._servers[side[:, i]]
            ok_d, k_d, drop_cell = self._nearest_drop(pos[:, i], counter, claimed_counter, used_dest)
            transfer = has_meal & (~deliver) & ok_d
            assigned = deliver | transfer
            target = torch.where(deliver, serve_cell, drop_cell)
            dest[:, i] = torch.where(assigned, target, dest[:, i])
            busy[:, i] |= assigned
            claimed_counter |= self._one_hot_5[k_d] & transfer[:, None]
            used_dest.scatter_(1, target[:, None], used_dest.gather(1, target[:, None]) | assigned[:, None])

        # 5.3 Pick up ready meals waiting on the counter.
        for k in range(5):
            item = counter[:, k]
            mr = (item - FIRST_MEAL).clamp(0, 2)
            live = (item >= FIRST_MEAL) & (~claimed_counter[:, k]) & (~used_dest[:, WALL[k]]) & (recipe == mr[:, None]).any(dim=1)
            dside = self._delivery_side[mr]
            ok, ci = self._pick_chef(self._wall_cols[k], dside, pos, side, hand, busy, controlled)
            take = live & ok
            cmask = self._one_hot_3[ci] & take[:, None]
            dest = torch.where(cmask, WALL[k], dest)
            busy |= cmask
            claimed_counter[:, k] |= take
            used_dest[:, WALL[k]] |= take

        # 5.4 Tend/pick up complete pots.
        for pot in range(2):
            rec_p = jrec[:, pot]
            complete = (rec_p >= 0) & (mix[:, pot] == self._recipe_bits[rec_p.clamp(0, 2)]) & (~used_dest[:, POTS[pot]])
            ok, ci = self._pick_chef(self._pot_cols[pot], self._side_cols[pot], pos, side, hand, busy, controlled)
            take = complete & ok
            cmask = self._one_hot_3[ci] & take[:, None]
            dest = torch.where(cmask, POTS[pot], dest)
            busy |= cmask
            used_dest[:, POTS[pot]] |= take
            task_type = torch.where(cmask, 2, task_type)
            task_pot = torch.where(cmask, pot, task_pot)

        # 5.5a Ingredients already held by controlled chefs.
        for i in range(3):
            held_i = hand[:, i]
            has_ing = (~busy[:, i]) & (held_i > 0) & (held_i < FIRST_MEAL)
            m_idx = self._item_to_idx[held_i.clamp(0, FIRST_MEAL + 2)].clamp(0, 3)
            need0 = has_ing & need[rows, 0, m_idx]
            need1 = has_ing & need[rows, 1, m_idx]
            any_need = need0 | need1
            tpot = (need1 & ((~need0) | (side[:, i] == 1))).long()
            same_side = side[:, i] == tpot
            is_raw_veg = (held_i == TOMATO) | (held_i == POTATO)

            ok_b, k_b, b_cell = self._nearest_board(pos[:, i], side[:, i], board, claimed_board, used_dest)
            ok_d, k_d, d_cell = self._nearest_drop(pos[:, i], counter, claimed_counter, used_dest)
            pot_cell = self._pots[tpot]
            pot_free = ~used_dest.gather(1, pot_cell[:, None])[:, 0]

            go_drop_cross = any_need & (~same_side) & ok_d
            go_board = any_need & same_side & is_raw_veg & ok_b
            go_pot = any_need & same_side & (~is_raw_veg) & pot_free
            assigned = go_drop_cross | go_board | go_pot

            target = torch.where(go_pot, pot_cell, torch.where(go_board, b_cell, d_cell))
            dest[:, i] = torch.where(assigned, target, dest[:, i])
            busy[:, i] |= assigned
            used_dest.scatter_(1, target[:, None], used_dest.gather(1, target[:, None]) | assigned[:, None])
            claimed_counter |= self._one_hot_5[k_d] & go_drop_cross[:, None]
            claimed_board |= self._one_hot_4[k_b] & go_board[:, None]
            task_type[:, i] = torch.where(go_board, 1, task_type[:, i])
            task_pot[:, i] = torch.where(go_board, tpot, task_pot[:, i])
            clear_mask = self._one_hot_2[tpot][:, :, None] & self._one_hot_4[m_idx][:, None, :] & assigned[:, None, None]
            need = need & (~clear_mask)

        # 5.5b Remaining missing ingredients: counter -> board (cut or raw) -> source.
        for pot in range(2):
            pside_t = self._side_cols[pot]
            for m in range(4):
                rem = need[:, pot, m]
                cut_item = _ITEMS[m]
                raw_item = _RAW_ITEMS[m]

                # Priority 1: counter (cut preferred over raw, nearest available counter to pot).
                c_avail = (~claimed_counter) & (~used_dest[:, self._wall])
                if m < 2:
                    c_score = torch.where(c_avail & (counter == cut_item), self.dist[self._wall, POTS[pot]][None, :],
                                          torch.where(c_avail & (counter == raw_item), self.dist[self._wall, POTS[pot]][None, :] + 100, _BIG))
                else:
                    c_score = torch.where(c_avail & (counter == cut_item), self.dist[self._wall, POTS[pot]][None, :], _BIG)
                best_cd, ck = c_score.min(dim=1)
                has_c = rem & (best_cd < _BIG)
                c_cell = self._wall[ck]
                ok_c, ci_c = self._pick_chef(c_cell, pside_t, pos, side, hand, busy, controlled)
                take_c = has_c & ok_c
                cmask_c = self._one_hot_3[ci_c] & take_c[:, None]
                dest = torch.where(cmask_c, c_cell[:, None], dest)
                busy |= cmask_c
                claimed_counter |= self._one_hot_5[ck] & has_c[:, None]
                used_dest.scatter_(1, c_cell[:, None], used_dest.gather(1, c_cell[:, None]) | take_c[:, None])
                rem = rem & (~has_c)

                # Priority 2: board (both cut_item and raw_item; same-side preferred).
                if m < 2:
                    b_avail = (~claimed_board) & (~used_dest[:, self._boards]) & ((board == cut_item) | (board == raw_item))
                    same_b_pen = torch.where(self._board_side[None, :] == pot, 0, 50)
                    cut_bonus = torch.where(board == cut_item, 0, 10)
                    b_score = torch.where(b_avail, same_b_pen + cut_bonus + self.dist[self._boards, POTS[pot]][None, :], _BIG)
                    best_bd, bk = b_score.min(dim=1)
                    has_b = rem & (best_bd < _BIG)
                    b_cell = self._boards[bk]
                    b_side = self._board_side[bk]
                    ok_b, ci_b = self._pick_chef(b_cell, b_side, pos, side, hand, busy, controlled)
                    take_b = has_b & ok_b
                    cmask_b = self._one_hot_3[ci_b] & take_b[:, None]
                    dest = torch.where(cmask_b, b_cell[:, None], dest)
                    busy |= cmask_b
                    claimed_board |= self._one_hot_4[bk] & has_b[:, None]
                    used_dest.scatter_(1, b_cell[:, None], used_dest.gather(1, b_cell[:, None]) | take_b[:, None])
                    task_type = torch.where(cmask_b, 1, task_type)
                    task_pot = torch.where(cmask_b, pot, task_pot)
                    rem = rem & (~has_b)

                # Priority 3: raw source (only when pot has a live ticket).
                src_cell = _RAW_SOURCE[m]
                ok_s, ci_s = self._pick_chef(self._src_cols[m], self._side_cols[_RAW_SIDE[m]], pos, side, hand, busy, controlled)
                take_s = rem & (jobs[:, pot] >= 0) & (~used_dest[:, src_cell]) & ok_s
                cmask_s = self._one_hot_3[ci_s] & take_s[:, None]
                dest = torch.where(cmask_s, src_cell, dest)
                busy |= cmask_s
                used_dest[:, src_cell] |= take_s
                rem = rem & (~take_s)

                need[:, pot, m] = rem

        # 5.6 Stash surplus held items; park idle right-side chefs out of the way.
        for i in range(3):
            surplus = (~busy[:, i]) & (hand[:, i] != EMPTY)
            is_raw_veg = (hand[:, i] == TOMATO) | (hand[:, i] == POTATO)
            ok_b, k_b, b_cell = self._nearest_board(pos[:, i], side[:, i], board, claimed_board, used_dest)
            ok_d, k_d, d_cell = self._nearest_drop(pos[:, i], counter, claimed_counter, used_dest)
            use_b = surplus & is_raw_veg & ok_b
            use_d = surplus & (~use_b) & ok_d
            assigned = use_b | use_d
            target = torch.where(use_b, b_cell, d_cell)
            dest[:, i] = torch.where(assigned, target, dest[:, i])
            busy[:, i] |= assigned
            used_dest.scatter_(1, target[:, None], used_dest.gather(1, target[:, None]) | assigned[:, None])
            claimed_board |= self._one_hot_4[k_b] & use_b[:, None]
            claimed_counter |= self._one_hot_5[k_d] & use_d[:, None]

            park = (~busy[:, i]) & (hand[:, i] == EMPTY) & (side[:, i] == 1)
            park_cell = torch.where(self.chef_ids[:, 1] == i, _PARK_UPPER, _PARK_LOWER)
            dest[:, i] = torch.where(park, park_cell, dest[:, i])

        # 6. Vectorized routing lookup + right-side teammate collision avoidance.
        blocked = torch.zeros_like(pos)
        a, b = self.chef_ids[:, 1], self.chef_ids[:, 2]
        blocked[rows, a] = pos[rows, b]
        blocked[rows, b] = pos[rows, a]
        moves = self.paths[dest, blocked, pos].long()
        moves = torch.where(moves >= 6, torch.where(facing == moves - 6, INTERACT, moves - 6), moves)
        moves = torch.where(self._one_hot_3[self.learner_ids] & self.learner_mask[:, None], STAY, moves)
        collision = (pos[rows, a] + self._offsets[moves[rows, a]] ==
                     pos[rows, b] + self._offsets[moves[rows, b]])
        moves[rows, b] = torch.where(collision, STAY, moves[rows, b])

        side_has_need = torch.stack((need[:, :, :2].any(dim=(1, 2)), need[:, :, 2:].any(dim=(1, 2))), dim=1)
        self.eligible_work.copy_(controlled & (~busy) & (hand == EMPTY) & side_has_need.gather(1, side))
        self.idle_actionable.copy_(self.eligible_work & (moves == STAY))
        self.destinations.copy_(dest)
        self.prev_task_type.copy_(task_type)
        self.prev_task_pot.copy_(task_pot)
        self._moves_out.copy_(moves)

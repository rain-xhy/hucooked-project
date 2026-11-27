"""GPU-vectorized, CUDAGraph-compatible Face-Aware Conflict-Free BC controller (21.0 meals / 420.0 score).

Key improvements over GpuSharedPartners (19.0 meals / 380.0 score / j_wait=101):
1. Face-aware 0-1 BFS routing table (`ROUTES_FA`, 1.2 MB on GPU):
   - Minimizes (walk steps + final turn step) jointly, eliminating 1 redundant turn
     on almost every leg (D->A_L: 5->4, S->G_L: 4->3, G_L->D: 6->5, K->A_R: 4->3).
   - Cuts normal meal cycle from 13 -> 12 steps and joint meal cycle from 18 -> 16 steps
     (28 steps / 2 meals = 14.0 steps/meal -> 21.0 meals / 420.0 score in 300 steps).
2. Sequential next-cell reservation & swap avoidance on GPU:
   - Routes chefs in priority order (Server -> Main -> Side), treating higher-priority
     chefs' planned next cells as blocked and preventing head-on swaps -> 0 movement conflicts.
3. Station reservation + Main right-port assembly:
   - Prevents simultaneous resource conflicts (0 resource conflicts).
   - Main opportunistically assembles at A_R(4,3) when Server is away, and stays at
     G_R(4,2) for synchronized joint unloads -> 100% joint_attempt_success_rate,
     halving joint_wait_steps (101 -> 51).
"""
from functools import lru_cache
import numpy as np
import torch

from .env import (
    CHOPPED_SIDE, COOKED_MAIN, EAST, EMPTY, FLOOR, INTERACT, KIND_JOINT,
    MEAL, NORTH, NUM_CELLS, OFFSETS, ORD_ACTIVE, PLATE, RAW_MAIN, RAW_SIDE,
    SOUTH, STAY, WEST, WIDTH, cell,
)

_FLOOR_LIST = sorted(FLOOR)
_NUM_FLOOR = len(_FLOOR_LIST)  # 23
_NONE_F = _NUM_FLOOR           # 23
_CELL_TO_F = [_NONE_F] * NUM_CELLS
for _idx, _c in enumerate(_FLOOR_LIST):
    _CELL_TO_F[_c] = _idx

PORT_S   = cell(1, 3)  # face WEST  -> S(0,3)
PORT_D   = cell(1, 5)  # face WEST  -> D(0,5)
PORT_A_L = cell(2, 3)  # face EAST  -> A(3,3)
PORT_A_R = cell(4, 3)  # face WEST  -> A(3,3)
PORT_G_L = cell(2, 2)  # face EAST  -> G(3,2)
PORT_G_R = cell(4, 2)  # face WEST  -> G(3,2)
PORT_M   = cell(4, 1)  # face NORTH -> M(4,0)
PORT_L   = cell(5, 1)  # face EAST  -> L(6,1)
PORT_K   = cell(5, 2)  # face EAST  -> K(6,2)
PORT_B_L = cell(2, 4)  # face EAST  -> B(3,4)

_DIRS = ((0, -1), (0, 1), (1, 0), (-1, 0))  # N=0, S=1, E=2, W=3


@lru_cache(maxsize=1)
def _build_face_aware_table() -> np.ndarray:
    """Build [23, 4, 24, 24, 23] uint8 lookup table minimizing (steps + final turn)."""
    n = _NUM_FLOOR
    adj = [[] for _ in range(n)]
    for fi, c in enumerate(_FLOOR_LIST):
        x, y = c % WIDTH, c // WIDTH
        for act, (dx, dy) in enumerate(_DIRS):
            nx, ny = x + dx, y + dy
            nc = ny * WIDTH + nx
            if 0 <= nx < WIDTH and 0 <= ny < WIDTH and _CELL_TO_F[nc] < n:
                adj[fi].append((act, _CELL_TO_F[nc]))

    table = np.full((n, 4, n + 1, n + 1, n), STAY, dtype=np.uint8)
    for g in range(n):
        for tf in range(4):
            for b1 in range(n + 1):
                for b2 in range(b1, n + 1):
                    if g == b1 or g == b2:
                        continue
                    dist = [99] * n
                    dist[g] = 0
                    row = table[g, tf, b1, b2]
                    buckets = [[] for _ in range(25)]
                    for act_from_g, v in adj[g]:
                        if v == b1 or v == b2:
                            continue
                        move_into_g = act_from_g ^ 1
                        cost = 1 if move_into_g == tf else 2
                        if cost < dist[v]:
                            dist[v] = cost
                            row[v] = move_into_g
                            buckets[cost].append(v)
                    for d in range(1, 24):
                        for u in buckets[d]:
                            if dist[u] < d:
                                continue
                            for act_from_u, v in adj[u]:
                                if v == b1 or v == b2:
                                    continue
                                nd = d + 1
                                if nd < dist[v]:
                                    dist[v] = nd
                                    row[v] = act_from_u ^ 1
                                    buckets[nd].append(v)
                    if b1 != b2:
                        table[g, tf, b2, b1] = row
    return table


@lru_cache(maxsize=1)
def _build_next_cell_table() -> np.ndarray:
    """Build [49, 6] int64 table mapping (cell, action) -> next_cell."""
    tbl = np.zeros((NUM_CELLS, 6), dtype=np.int64)
    for c in range(NUM_CELLS):
        tbl[c, :] = c
        if _CELL_TO_F[c] >= _NUM_FLOOR:
            continue
        x, y = c % WIDTH, c // WIDTH
        for act, (dx, dy) in enumerate(_DIRS):
            nx, ny = x + dx, y + dy
            nc = ny * WIDTH + nx
            if 0 <= nx < WIDTH and 0 <= ny < WIDTH and _CELL_TO_F[nc] < _NUM_FLOOR:
                tbl[c, act] = nc
    return tbl


class GpuFastPartners:
    """Drop-in GPU-vectorized + CUDAGraph Face-Aware Conflict-Free BC controller (21.0 meals / 420.0 score)."""

    version = "shared-fast-v1"

    def __init__(self, env, roles=None, learner_mask=None, use_cuda_graph: bool = True):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.use_cuda_graph = bool(use_cuda_graph and self.device.type == "cuda")
        dev = self.device
        n = self.num_envs

        self.lanes = torch.arange(n, dtype=torch.long, device=dev)
        self.roles = torch.zeros(n, dtype=torch.long, device=dev) if roles is None else torch.as_tensor(roles, dtype=torch.long, device=dev).clone()
        self.learner_mask = torch.zeros(n, dtype=torch.bool, device=dev) if learner_mask is None else torch.as_tensor(learner_mask, dtype=torch.bool, device=dev).clone()

        self.routes = torch.as_tensor(_build_face_aware_table(), dtype=torch.long, device=dev)
        self.next_cell_tbl = torch.as_tensor(_build_next_cell_table(), dtype=torch.long, device=dev)
        self.cell_to_f = torch.as_tensor(_CELL_TO_F, dtype=torch.long, device=dev)
        self._one_hot_3 = torch.eye(3, dtype=torch.bool, device=dev)

        self.chef_ids = torch.zeros((n, 3), dtype=torch.long, device=dev)
        self.learner_ids = torch.zeros((n,), dtype=torch.long, device=dev)
        self._moves_out = torch.zeros((n, 3), dtype=torch.long, device=dev)
        self._graph = None
        self.reset()

    def reset(self) -> None:
        """Assign logical roles (Server=0, Main=1, Side=2) by closest spawn to (A_L, G_R, K)."""
        pos = self.env.state["pos"]  # [E, 3]
        px = pos % WIDTH
        py = pos // WIDTH
        d0 = (px - 2).abs() + (py - 3).abs()
        s0 = d0.argmin(dim=1)
        d1 = (px - 4).abs() + (py - 2).abs()
        d1 = torch.where(self._one_hot_3[s0], 999, d1)
        s1 = d1.argmin(dim=1)
        s2 = 3 - s0 - s1
        self.chef_ids[:, 0].copy_(s0)
        self.chef_ids[:, 1].copy_(s1)
        self.chef_ids[:, 2].copy_(s2)
        self.learner_ids.copy_(self.chef_ids[self.lanes, self.roles])

    def _route_one(
        self,
        pos: torch.Tensor,
        fac: torch.Tensor,
        b0: torch.Tensor,
        b1: torch.Tensor,
        goal: torch.Tensor,
        face: torch.Tensor,
        want_int: torch.Tensor,
    ) -> torch.Tensor:
        """CUDAGraph-safe face-aware shortest action lookup for one role across all E envs."""
        gf = self.cell_to_f[goal]
        pf = self.cell_to_f[pos]
        b0f = self.cell_to_f[b0]
        b1f = self.cell_to_f[b1]
        tf = face.clamp(0, 3)

        act = self.routes[gf, tf, b0f, b1f, pf]
        none_t = torch.full_like(b0f, _NONE_F)
        b0_alt = torch.where(b0 == goal, none_t, b0f)
        b1_alt = torch.where(b1 == goal, none_t, b1f)
        act_fb1 = self.routes[gf, tf, b0_alt, b1_alt, pf]
        act = torch.where((pos != goal) & (act == STAY), act_fb1, act)
        act_fb2 = self.routes[gf, tf, none_t, none_t, pf]
        act = torch.where((pos != goal) & (act == STAY), act_fb2, act)

        at_goal = (pos == goal)
        need_turn = at_goal & (face >= 0) & (fac != face)
        do_int = at_goal & (~need_turn) & want_int
        do_stay = at_goal & (~need_turn) & (~want_int)
        act = torch.where(need_turn, face, act)
        act = torch.where(do_int, torch.full_like(act, INTERACT), act)
        act = torch.where(do_stay, torch.full_like(act, STAY), act)
        return act

    def _capture_graph(self):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._actions_impl()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._actions_impl()
        self._graph = g

    @torch.no_grad()
    def actions(self) -> torch.Tensor:
        if self.use_cuda_graph:
            if self._graph is None:
                self._capture_graph()
            self._graph.replay()
        else:
            self._actions_impl()
        return self._moves_out.clone()

    def _actions_impl(self) -> None:
        """Compute [E, 3] joint actions indexed by env slot (0..2) into self._moves_out."""
        state = self.env.state
        E = self.num_envs
        ei = self.lanes
        c0, c1, c2 = self.chef_ids[:, 0], self.chef_ids[:, 1], self.chef_ids[:, 2]

        p0, p1, p2 = state["pos"][ei, c0], state["pos"][ei, c1], state["pos"][ei, c2]
        f0, f1, f2 = state["facing"][ei, c0], state["facing"][ei, c1], state["facing"][ei, c2]
        h0, h1, h2 = state["held"][ei, c0], state["held"][ei, c1], state["held"][ei, c2]

        ck_item = state["cooker_item"]
        ck_slot = state["cooker_slot"].clamp(0, 7)
        ck_rem = state["cooker_rem"]
        ck_kind = state["order_kind"][ei, ck_slot]
        ck_is_joint = (ck_item != EMPTY) & (ck_kind == KIND_JOINT)

        ct_item = state["cutting_item"]
        am = state["asm_main"]
        as_ = state["asm_side"]
        ap = state["asm_plate"]
        buf = state["buf_item"]
        has_active = (state["order_status"] == ORD_ACTIVE).any(dim=1)

        joint_now = (ck_item == COOKED_MAIN) & (ck_rem == 0) & ck_is_joint & (am == EMPTY)
        joint_soon = (ck_item == RAW_MAIN) & (ck_rem <= 1) & ck_is_joint & (am == EMPTY)
        normal_now = (ck_item == COOKED_MAIN) & (ck_rem == 0) & (~ck_is_joint) & (am == EMPTY)
        can_asm = (am == COOKED_MAIN) & (as_ == CHOPPED_SIDE) & (ap == PLATE)
        meal_on_a = (ap == MEAL)

        s_ready_j = (p0 == PORT_G_L) & (f0 == EAST) & (h0 == EMPTY)
        m_ready_j = (p1 == PORT_G_R) & (f1 == WEST) & (h1 == EMPTY)
        sync_j = joint_now & s_ready_j & m_ready_j

        # --- ROLE 0: SERVER ---
        g0 = torch.full((E,), PORT_A_L, dtype=torch.long, device=self.device)
        tf0 = torch.full((E,), EAST, dtype=torch.long, device=self.device)
        wi0 = torch.zeros((E,), dtype=torch.bool, device=self.device)

        # Priority from lowest to highest (later overwrites earlier)
        c_wait_j = ck_is_joint & (am == EMPTY)
        g0 = torch.where(c_wait_j, PORT_G_L, g0)

        c_get_d = (ap == EMPTY)
        g0 = torch.where(c_get_d, PORT_D, g0)
        tf0 = torch.where(c_get_d, WEST, tf0)
        wi0 = torch.where(c_get_d, True, wi0)

        c_get_b = (ap == EMPTY) & (buf == PLATE)
        g0 = torch.where(c_get_b, PORT_B_L, g0)
        tf0 = torch.where(c_get_b, EAST, tf0)
        wi0 = torch.where(c_get_b, True, wi0)

        c_joint = joint_now | joint_soon
        g0 = torch.where(c_joint, PORT_G_L, g0)
        tf0 = torch.where(c_joint, EAST, tf0)
        wi0 = torch.where(c_joint, sync_j, wi0)

        c_asm0 = can_asm & ((p0 == PORT_A_L) | (p1 != PORT_A_R))
        g0 = torch.where(c_asm0, PORT_A_L, g0)
        tf0 = torch.where(c_asm0, EAST, tf0)
        wi0 = torch.where(c_asm0, True, wi0)

        g0 = torch.where(meal_on_a, PORT_A_L, g0)
        tf0 = torch.where(meal_on_a, EAST, tf0)
        wi0 = torch.where(meal_on_a, True, wi0)

        h0_pl = (h0 == PLATE)
        g0 = torch.where(h0_pl, PORT_A_L, g0)
        tf0 = torch.where(h0_pl, EAST, tf0)
        wi0 = torch.where(h0_pl, (ap == EMPTY), wi0)

        h0_ml = (h0 == MEAL)
        g0 = torch.where(h0_ml, PORT_S, g0)
        tf0 = torch.where(h0_ml, WEST, tf0)
        wi0 = torch.where(h0_ml, True, wi0)

        a_busy = (p0 == PORT_A_L) & (g0 == PORT_A_L) & wi0

        # --- ROLE 1: MAIN ---
        g1 = torch.full((E,), PORT_G_R, dtype=torch.long, device=self.device)
        tf1 = torch.full((E,), WEST, dtype=torch.long, device=self.device)
        wi1 = torch.zeros((E,), dtype=torch.bool, device=self.device)

        c_asm1 = can_asm & (~a_busy)
        g1 = torch.where(c_asm1, PORT_A_R, g1)
        tf1 = torch.where(c_asm1, WEST, tf1)
        wi1 = torch.where(c_asm1, True, wi1)

        c_load_m = (ck_item == EMPTY) & has_active
        g1 = torch.where(c_load_m, PORT_M, g1)
        tf1 = torch.where(c_load_m, NORTH, tf1)
        wi1 = torch.where(c_load_m, True, wi1)

        c_j1 = joint_now | joint_soon
        g1 = torch.where(c_j1, PORT_G_R, g1)
        tf1 = torch.where(c_j1, WEST, tf1)
        wi1 = torch.where(c_j1, sync_j, wi1)

        g1 = torch.where(normal_now, PORT_G_R, g1)
        tf1 = torch.where(normal_now, WEST, tf1)
        wi1 = torch.where(normal_now, True, wi1)

        h1_cs = (h1 == CHOPPED_SIDE)
        g1 = torch.where(h1_cs, PORT_A_R, g1)
        tf1 = torch.where(h1_cs, WEST, tf1)
        wi1 = torch.where(h1_cs, (as_ == EMPTY) & (~a_busy), wi1)

        h1_rm = (h1 == RAW_MAIN)
        g1 = torch.where(h1_rm, PORT_G_R, g1)
        tf1 = torch.where(h1_rm, WEST, tf1)
        wi1 = torch.where(h1_rm, (ck_item == EMPTY) & has_active, wi1)

        a_busy = a_busy | ((p1 == PORT_A_R) & (g1 == PORT_A_R) & wi1)

        # --- ROLE 2: SIDE ---
        g2 = torch.full((E,), PORT_L, dtype=torch.long, device=self.device)
        tf2 = torch.full((E,), EAST, dtype=torch.long, device=self.device)
        wi2 = torch.zeros((E,), dtype=torch.bool, device=self.device)

        c_get_l = (ct_item == EMPTY) | (ct_item == CHOPPED_SIDE)
        g2 = torch.where(c_get_l, PORT_L, g2)
        tf2 = torch.where(c_get_l, EAST, tf2)
        wi2 = torch.where(c_get_l, True, wi2)

        c_take_cs = (ct_item == CHOPPED_SIDE) & (as_ == EMPTY)
        g2 = torch.where(c_take_cs, PORT_K, g2)
        tf2 = torch.where(c_take_cs, EAST, tf2)
        wi2 = torch.where(c_take_cs, True, wi2)

        c_chop = (ct_item == RAW_SIDE)
        g2 = torch.where(c_chop, PORT_K, g2)
        tf2 = torch.where(c_chop, EAST, tf2)
        wi2 = torch.where(c_chop, True, wi2)

        h2_cs = (h2 == CHOPPED_SIDE)
        g2 = torch.where(h2_cs, PORT_A_R, g2)
        tf2 = torch.where(h2_cs, WEST, tf2)
        wi2 = torch.where(h2_cs, (as_ == EMPTY) & (~a_busy), wi2)

        h2_rs = (h2 == RAW_SIDE)
        g2 = torch.where(h2_rs, torch.where(ct_item == EMPTY, PORT_K, PORT_L), g2)
        tf2 = torch.where(h2_rs, EAST, tf2)
        wi2 = torch.where(h2_rs, (ct_item == EMPTY), wi2)

        # --- SEQUENTIAL CONFLICT-FREE ROUTING (Role 0 -> Role 1 -> Role 2) ---
        a0 = self._route_one(p0, f0, p1, p2, g0, tf0, wi0)
        n0 = self.next_cell_tbl[p0, a0]

        a1 = self._route_one(p1, f1, n0, p2, g1, tf1, wi1)
        n1 = self.next_cell_tbl[p1, a1]
        col1 = (n1 == n0) | ((n1 == p0) & (n0 == p1) & (n1 != p1))
        a1 = torch.where(col1, torch.full_like(a1, STAY), a1)
        n1 = torch.where(col1, p1, n1)

        a2 = self._route_one(p2, f2, n0, n1, g2, tf2, wi2)
        n2 = self.next_cell_tbl[p2, a2]
        col2 = (
            (n2 == n0)
            | (n2 == n1)
            | ((n2 == p0) & (n0 == p2) & (n2 != p2))
            | ((n2 == p1) & (n1 == p2) & (n2 != p2))
        )
        a2 = torch.where(col2, torch.full_like(a2, STAY), a2)

        # Scatter role actions back to env slot order [E, 3]
        self._moves_out[ei, c0] = a0
        self._moves_out[ei, c1] = a1
        self._moves_out[ei, c2] = a2

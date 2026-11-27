"""Pure-PyTorch GPU CUDA + CUDAGraph vectorized SharedKitchen (origin/hy de37e08 parity)."""
from dataclasses import dataclass
import math
import numpy as np
import torch

WIDTH = 7
HEIGHT = 7
NUM_CELLS = WIDTH * HEIGHT  # 49
ORDER_SLOTS = 8

NORTH, SOUTH, EAST, WEST, STAY, INTERACT = range(6)
DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
OFFSETS = (-7, 7, 1, -1)

LAYOUT = (
    "XXXXMXX",
    "X     L",
    "X  G  K",
    "S  A  X",
    "X  B  X",
    "D     X",
    "XXXXXXX",
)

def cell(x, y):
    return y * WIDTH + x

STATIONS = {c: cell(x, y) for y, row in enumerate(LAYOUT) for x, c in enumerate(row) if c not in "X "}
FLOOR = frozenset(cell(x, y) for y, row in enumerate(LAYOUT) for x, c in enumerate(row) if c == " ")
COOKER_PORTS = (cell(2, 2), cell(4, 2))
ASSEMBLY_PORTS = (cell(2, 3), cell(4, 3))
SPAWNS = (cell(2, 3), cell(4, 2), cell(5, 2))

CELL_M = STATIONS["M"]  # (4, 0) = 4
CELL_L = STATIONS["L"]  # (6, 1) = 13
CELL_G = STATIONS["G"]  # (3, 2) = 17
CELL_K = STATIONS["K"]  # (6, 2) = 20
CELL_S = STATIONS["S"]  # (0, 3) = 21
CELL_A = STATIONS["A"]  # (3, 3) = 24
CELL_B = STATIONS["B"]  # (3, 4) = 31
CELL_D = STATIONS["D"]  # (0, 5) = 35

# Item kinds: 0..6
EMPTY = 0
RAW_MAIN = 1
RAW_SIDE = 2
CHOPPED_SIDE = 3
COOKED_MAIN = 4
PLATE = 5
MEAL = 6
NUM_ITEMS = 7

# Order statuses: 0..3
ORD_QUEUED = 0
ORD_ACTIVE = 1
ORD_STARTED = 2
ORD_SERVED = 3

# Order kinds: 0..1
KIND_NORMAL = 0
KIND_JOINT = 1

# Dynamic centralized state dim (excluding 490 static terrain one-hot):
# tick(1) + 3 chefs * (1 active + 49 pos + 4 dir + 7 kind + 9 slot = 70) = 210
# + 6 station items * (7 kind + 9 slot = 16) = 96
# + 2 timers (cooker_rem, cutting_prog) = 2
# + 8 order slots * (4 status + 2 kind + 2 times = 8) = 64
# = 373 centralized + 3 self one-hot = 376.
CENTRAL_DYNAMIC_DIM = 1 + 3 * (1 + NUM_CELLS + 4 + NUM_ITEMS + (ORDER_SLOTS + 1)) + 6 * (NUM_ITEMS + (ORDER_SLOTS + 1)) + 2 + ORDER_SLOTS * 8
OBS_DIM = CENTRAL_DYNAMIC_DIM + 3  # 376


@dataclass(frozen=True)
class SharedKitchenConfig:
    horizon: int = 300
    normal_orders: int | None = None
    joint_orders: int | None = None
    active_limit: int = 2
    cook_steps: int = 4
    chop_steps: int = 2
    normal_reward: float = 20.0
    joint_reward: float = 20.0
    shuffle_players: bool = True
    shaping_scale: float = 0.0
    shaping_gamma: float = 0.99

    def __post_init__(self):
        for name in ("horizon", "active_limit", "cook_steps", "chop_steps"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise ValueError(name + " must be a positive integer")
        if not self.continuous_orders:
            for name in ("normal_orders", "joint_orders"):
                v = getattr(self, name)
                if v is None:
                    object.__setattr__(self, name, 4)
                elif type(v) is not int or v < 0:
                    raise ValueError(name + " must be a nonnegative integer")
            if not 1 <= self.normal_orders + self.joint_orders <= ORDER_SLOTS:
                raise ValueError(f"Fixed orders must be in 1..{ORDER_SLOTS} for GPU env")

    @property
    def continuous_orders(self):
        return self.normal_orders is None and self.joint_orders is None


class TorchSharedKitchen:
    """Vectorized CUDA/CPU SharedKitchen with optional CUDAGraph step+obs replay."""

    def __init__(self, config=None, *, num_envs=64, device="cuda", use_cuda_graph=True):
        self.config = config or SharedKitchenConfig()
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph and self.device.type == "cuda")
        n = self.num_envs
        dev = self.device

        self.lanes = torch.arange(n, dtype=torch.long, device=dev)
        self._offsets = torch.tensor((*OFFSETS, 0, 0), dtype=torch.long, device=dev)
        floor_mask = torch.zeros(NUM_CELLS, dtype=torch.bool, device=dev)
        for c in FLOOR:
            floor_mask[c] = True
        self._floor_mask = floor_mask
        cooker_port_mask = torch.zeros(NUM_CELLS, dtype=torch.bool, device=dev)
        for c in COOKER_PORTS:
            cooker_port_mask[c] = True
        self._cooker_port_mask = cooker_port_mask
        asm_port_mask = torch.zeros(NUM_CELLS, dtype=torch.bool, device=dev)
        for c in ASSEMBLY_PORTS:
            asm_port_mask[c] = True
        self._asm_port_mask = asm_port_mask
        self._spawns = torch.tensor(SPAWNS, dtype=torch.long, device=dev)
        self._chef_bits = torch.tensor([1, 2, 4], dtype=torch.long, device=dev)
        self._one_hot_3 = torch.eye(3, dtype=torch.bool, device=dev)
        self._one_hot_3_f = torch.eye(3, dtype=torch.float32, device=dev)
        self._one_hot_4_f = torch.eye(4, dtype=torch.float32, device=dev)
        self._one_hot_2_f = torch.eye(2, dtype=torch.float32, device=dev)
        self._one_hot_7_f = torch.eye(NUM_ITEMS, dtype=torch.float32, device=dev)
        self._one_hot_9_f = torch.eye(ORDER_SLOTS + 1, dtype=torch.float32, device=dev)
        self._one_hot_49_f = torch.eye(NUM_CELLS, dtype=torch.float32, device=dev)
        self._one_hot_8 = torch.eye(ORDER_SLOTS, dtype=torch.bool, device=dev)
        self._slot_arange = torch.arange(ORDER_SLOTS, dtype=torch.long, device=dev)
        self._ones_active = torch.ones((n, 1), dtype=torch.float32, device=dev)
        self._no_slot = torch.full((n,), -1, dtype=torch.long, device=dev)

        # Persistent state tensors (never reallocated, CUDAGraph safe).
        self.state = dict(
            pos=torch.zeros((n, 3), dtype=torch.long, device=dev),
            facing=torch.zeros((n, 3), dtype=torch.long, device=dev),
            held=torch.zeros((n, 3), dtype=torch.long, device=dev),
            held_slot=torch.full((n, 3), -1, dtype=torch.long, device=dev),
            held_contrib=torch.zeros((n, 3), dtype=torch.long, device=dev),
            cooker_item=torch.zeros((n,), dtype=torch.long, device=dev),
            cooker_slot=torch.full((n,), -1, dtype=torch.long, device=dev),
            cooker_rem=torch.zeros((n,), dtype=torch.long, device=dev),
            cooker_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            cutting_item=torch.zeros((n,), dtype=torch.long, device=dev),
            cutting_prog=torch.zeros((n,), dtype=torch.long, device=dev),
            cutting_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_main=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_main_slot=torch.full((n,), -1, dtype=torch.long, device=dev),
            asm_main_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_side=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_side_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_plate=torch.zeros((n,), dtype=torch.long, device=dev),
            asm_plate_slot=torch.full((n,), -1, dtype=torch.long, device=dev),
            asm_plate_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            buf_item=torch.zeros((n,), dtype=torch.long, device=dev),
            buf_slot=torch.full((n,), -1, dtype=torch.long, device=dev),
            buf_contrib=torch.zeros((n,), dtype=torch.long, device=dev),
            order_id=torch.zeros((n, ORDER_SLOTS), dtype=torch.long, device=dev),
            order_kind=torch.zeros((n, ORDER_SLOTS), dtype=torch.long, device=dev),
            order_status=torch.zeros((n, ORDER_SLOTS), dtype=torch.long, device=dev),
            order_released=torch.full((n, ORDER_SLOTS), -1, dtype=torch.long, device=dev),
            order_started=torch.full((n, ORDER_SLOTS), -1, dtype=torch.long, device=dev),
            order_completed=torch.full((n, ORDER_SLOTS), -1, dtype=torch.long, device=dev),
            next_order_id=torch.zeros((n,), dtype=torch.long, device=dev),
            tick=torch.zeros((n,), dtype=torch.long, device=dev),
            score=torch.zeros((n,), dtype=torch.float32, device=dev),
            normal_completed=torch.zeros((n,), dtype=torch.long, device=dev),
            joint_completed=torch.zeros((n,), dtype=torch.long, device=dev),
            joint_ready_steps=torch.zeros((n,), dtype=torch.long, device=dev),
            joint_attempt_steps=torch.zeros((n,), dtype=torch.long, device=dev),
            joint_successes=torch.zeros((n,), dtype=torch.long, device=dev),
            unmatched_attempt_steps=torch.zeros((n,), dtype=torch.long, device=dev),
            resource_conflicts=torch.zeros((n,), dtype=torch.long, device=dev),
            movement_conflicts=torch.zeros((n,), dtype=torch.long, device=dev),
            three_contributor_meals=torch.zeros((n,), dtype=torch.long, device=dev),
        )
        self._actions_in = torch.zeros((n, 3), dtype=torch.long, device=dev)
        self._obs_out = torch.zeros((n, 3, OBS_DIM), dtype=torch.float32, device=dev)
        self._reward_out = torch.zeros((n,), dtype=torch.float32, device=dev)
        self._delivered_normal_out = torch.zeros((n,), dtype=torch.bool, device=dev)
        self._delivered_joint_out = torch.zeros((n,), dtype=torch.bool, device=dev)
        self._step_graph = None
        self._rng = np.random.default_rng(0)
        self.reset(seed=0)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        s = self.state
        n = self.num_envs
        if self.config.shuffle_players:
            perms = np.stack([self._rng.permutation(3) for _ in range(n)], axis=0)
            sp = np.asarray(SPAWNS, dtype=np.int64)[perms]
            s["pos"].copy_(torch.as_tensor(sp, dtype=torch.long, device=self.device))
        else:
            s["pos"].copy_(self._spawns[None, :].expand(n, 3))
        s["facing"].zero_()  # NORTH = (0, -1)
        s["held"].zero_()
        s["held_slot"].fill_(-1)
        s["held_contrib"].zero_()
        for k in ("cooker_item", "cooker_rem", "cooker_contrib",
                  "cutting_item", "cutting_prog", "cutting_contrib",
                  "asm_main", "asm_main_contrib", "asm_side", "asm_side_contrib",
                  "asm_plate", "asm_plate_contrib", "buf_item", "buf_contrib",
                  "tick", "normal_completed", "joint_completed", "joint_ready_steps",
                  "joint_attempt_steps", "joint_successes", "unmatched_attempt_steps",
                  "resource_conflicts", "movement_conflicts", "three_contributor_meals"):
            s[k].zero_()
        for k in ("cooker_slot", "asm_main_slot", "asm_plate_slot", "buf_slot",
                  "order_released", "order_started", "order_completed"):
            s[k].fill_(-1)
        s["score"].zero_()

        # Initialize 8 order slots (matches origin/hy de37e08 _observed_orders).
        s["order_id"].copy_(self._slot_arange[None, :].expand(n, ORDER_SLOTS))
        if self.config.continuous_orders:
            kinds = (self._slot_arange % 2)[None, :].expand(n, ORDER_SLOTS)
            s["order_kind"].copy_(kinds)
            s["order_status"].fill_(ORD_QUEUED)
            al = min(self.config.active_limit, ORDER_SLOTS)
            s["order_status"][:, :al] = ORD_ACTIVE
            s["order_released"][:, :al] = 0
            s["next_order_id"].fill_(al)
        else:
            kinds_list = []
            for idx in range(max(self.config.normal_orders, self.config.joint_orders)):
                if idx < self.config.normal_orders:
                    kinds_list.append(KIND_NORMAL)
                if idx < self.config.joint_orders:
                    kinds_list.append(KIND_JOINT)
            total = len(kinds_list)
            pad_kinds = kinds_list + [0] * (ORDER_SLOTS - total)
            s["order_kind"].copy_(torch.tensor(pad_kinds, dtype=torch.long, device=self.device)[None, :].expand(n, ORDER_SLOTS))
            s["order_status"].fill_(ORD_SERVED)
            s["order_status"][:, :total] = ORD_QUEUED
            al = min(self.config.active_limit, total)
            s["order_status"][:, :al] = ORD_ACTIVE
            s["order_released"][:, :al] = 0
            s["next_order_id"].fill_(total)

        self._write_observations()
        return self._obs_out.clone(), self.episode_metrics()

    def _snapshot_state(self):
        return {k: v.clone() for k, v in self.state.items()}

    def _restore_state(self, snap):
        for k, v in snap.items():
            self.state[k].copy_(v)

    def _capture_step_graph(self):
        snap = self._snapshot_state()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._step_impl()
                self._restore_state(snap)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_impl()
        self._restore_state(snap)
        self._step_graph = g

    @torch.no_grad()
    def step(self, actions):
        self._actions_in.copy_(actions)
        if self.use_cuda_graph:
            if self._step_graph is None:
                self._capture_step_graph()
            self._step_graph.replay()
        else:
            self._step_impl()
        info = dict(
            delivered_normal=self._delivered_normal_out,
            delivered_joint=self._delivered_joint_out,
        )
        return self._obs_out, self._reward_out, False, False, info

    def _potential(self):
        """Potential-based shaping Phi(s) over pipeline milestones (zero reward-hacking loops)."""
        s = self.state
        held = s["held"]
        has_rm = (held == RAW_MAIN).any(dim=1).float()
        has_rs = (held == RAW_SIDE).any(dim=1).float()
        has_cs = ((held == CHOPPED_SIDE).any(dim=1) | (s["buf_item"] == CHOPPED_SIDE)).float()
        has_pl = ((held == PLATE).any(dim=1) | (s["buf_item"] == PLATE)).float()
        has_ml = ((held == MEAL).any(dim=1) | (s["buf_item"] == MEAL)).float()

        ck_rm = (s["cooker_item"] == RAW_MAIN).float()
        ck_prog = (self.config.cook_steps - s["cooker_rem"].clamp(0, self.config.cook_steps)).float() / float(self.config.cook_steps)
        ck_cm = (s["cooker_item"] == COOKED_MAIN).float()
        am_cm = (s["asm_main"] == COOKED_MAIN).float()

        ct_rs = (s["cutting_item"] == RAW_SIDE).float()
        ct_prog = s["cutting_prog"].clamp(0, self.config.chop_steps).float() / float(self.config.chop_steps)
        ct_cs = (s["cutting_item"] == CHOPPED_SIDE).float()
        as_cs = (s["asm_side"] == CHOPPED_SIDE).float()

        ap_pl = (s["asm_plate"] == PLATE).float()
        ap_ml = (s["asm_plate"] == MEAL).float()

        main_phi = 0.15 * has_rm + ck_rm * (0.35 + 0.25 * ck_prog) + 0.65 * ck_cm + 1.00 * am_cm
        side_phi = 0.15 * has_rs + ct_rs * (0.30 + 0.30 * ct_prog) + 0.65 * ct_cs + 0.75 * has_cs + 1.00 * as_cs
        meal_phi = 0.25 * has_pl + 0.60 * ap_pl + 2.80 * ap_ml + 3.20 * has_ml
        return main_phi + side_phi + meal_phi

    def _step_impl(self):
        s = self.state
        n = self.num_envs
        lanes = self.lanes
        actions = self._actions_in  # [N, 3]
        old_phi = self._potential() if self.config.shaping_scale != 0.0 else None

        # Snapshot pre-step read-only fields.
        old_pos = s["pos"].clone()
        old_dir = s["facing"].clone()
        old_held = s["held"].clone()
        old_hslot = s["held_slot"].clone()
        old_hcon = s["held_contrib"].clone()
        old_ck_item = s["cooker_item"].clone()
        old_ck_slot = s["cooker_slot"].clone()
        old_ck_rem = s["cooker_rem"].clone()
        old_ck_con = s["cooker_contrib"].clone()
        old_ct_item = s["cutting_item"].clone()
        old_ct_prog = s["cutting_prog"].clone()
        old_ct_con = s["cutting_contrib"].clone()
        old_am = s["asm_main"].clone()
        old_am_slot = s["asm_main_slot"].clone()
        old_am_con = s["asm_main_contrib"].clone()
        old_as = s["asm_side"].clone()
        old_as_con = s["asm_side_contrib"].clone()
        old_ap = s["asm_plate"].clone()
        old_ap_slot = s["asm_plate_slot"].clone()
        old_ap_con = s["asm_plate_contrib"].clone()
        old_buf = s["buf_item"].clone()
        old_buf_slot = s["buf_slot"].clone()
        old_buf_con = s["buf_contrib"].clone()
        old_tick = s["tick"].clone()
        old_score = s["score"].clone()

        # Interaction target cell per chef [N, 3].
        is_inter = actions == INTERACT
        target = old_pos + self._offsets[old_dir]
        target = torch.where(is_inter, target, -1)

        in_ck_port = self._cooker_port_mask[old_pos]
        in_asm_port = self._asm_port_mask[old_pos]
        at_G = (target == CELL_G) & in_ck_port
        at_A = (target == CELL_A) & in_asm_port
        at_K = target == CELL_K
        at_B = target == CELL_B
        at_M = target == CELL_M
        at_L = target == CELL_L
        at_D = target == CELL_D
        at_S = target == CELL_S

        # --- Request generation (matching kitchen.py _requests) ---
        # 1. Cooker unload (atomic 2-chef for joint, >=1 for normal).
        ck_ready = (old_ck_item == COOKED_MAIN) & (old_ck_rem == 0) & (old_am == EMPTY)
        ck_ord_kind = s["order_kind"][lanes, old_ck_slot.clamp(0, ORDER_SLOTS - 1)]
        ck_is_joint = ck_ready & (ck_ord_kind == KIND_JOINT)
        ck_empty_actors = at_G & (old_held == EMPTY)
        ck_empty_cnt = ck_empty_actors.sum(dim=1)

        s["joint_ready_steps"] += ck_is_joint.long()
        s["joint_attempt_steps"] += (ck_is_joint & (ck_empty_cnt >= 1)).long()
        s["unmatched_attempt_steps"] += (ck_is_joint & (ck_empty_cnt == 1)).long()

        req_unload = (ck_is_joint & (ck_empty_cnt == 2)) | (ck_ready & (~ck_is_joint) & (ck_empty_cnt >= 1))
        unload_con = (ck_empty_actors.long() * self._chef_bits[None, :]).sum(dim=1)

        # 2. Cooker load: pick lowest order_id among active slots.
        is_active = s["order_status"] == ORD_ACTIVE
        has_active = is_active.any(dim=1)
        act_id_key = torch.where(is_active, s["order_id"], 1_000_000_000)
        pending_slot = act_id_key.argmin(dim=1)
        req_load = at_G & (old_ck_item[:, None] == EMPTY) & (old_held == RAW_MAIN) & has_active[:, None]

        # 3. Cutting board requests.
        req_place_cut = at_K & (old_ct_item[:, None] == EMPTY) & (old_held == RAW_SIDE)
        req_chop = at_K & (old_held == EMPTY) & (old_ct_item[:, None] == RAW_SIDE)
        req_take_cut = at_K & (old_held == EMPTY) & (old_ct_item[:, None] == CHOPPED_SIDE)

        # 4. Buffer requests.
        req_place_buf = at_B & (old_held != EMPTY) & (old_buf[:, None] == EMPTY)
        req_take_buf = at_B & (old_held == EMPTY) & (old_buf[:, None] != EMPTY)

        # 5. Assembly requests.
        req_place_as = at_A & (old_held == CHOPPED_SIDE) & (old_as[:, None] == EMPTY)
        req_place_ap = at_A & ((old_held == PLATE) | (old_held == MEAL)) & (old_ap[:, None] == EMPTY)
        req_take_meal = at_A & (old_held == EMPTY) & (old_ap[:, None] == MEAL)
        req_assemble = (at_A & (old_held == EMPTY) & (old_ap[:, None] != MEAL) &
                        (old_am[:, None] != EMPTY) & (old_as[:, None] != EMPTY) & (old_ap[:, None] == PLATE))

        # 6. Non-conflicting source & serve requests.
        req_take_main = at_M & (old_held == EMPTY)
        req_ret_main = at_M & (old_held == RAW_MAIN)
        req_take_side = at_L & (old_held == EMPTY)
        req_ret_side = at_L & ((old_held == RAW_SIDE) | (old_held == CHOPPED_SIDE))
        req_take_plate = at_D & (old_held == EMPTY)
        req_ret_plate = at_D & (old_held == PLATE)
        req_serve = at_S & (old_held == MEAL)

        # --- Resource conflict detection (6 shared resources) ---
        use_cooker = req_unload.long() + req_load.long().sum(dim=1)
        use_cutting = (req_place_cut | req_chop | req_take_cut).long().sum(dim=1)
        use_buffer = (req_place_buf | req_take_buf).long().sum(dim=1)
        use_am = req_unload.long() + req_assemble.long().sum(dim=1)
        use_as = req_place_as.long().sum(dim=1) + req_assemble.long().sum(dim=1)
        use_ap = (req_place_ap | req_take_meal | req_assemble).long().sum(dim=1)

        conf_cooker = use_cooker > 1
        conf_cutting = use_cutting > 1
        conf_buffer = use_buffer > 1
        conf_am = use_am > 1
        conf_as = use_as > 1
        conf_ap = use_ap > 1

        s["resource_conflicts"] += (conf_cooker.long() + conf_cutting.long() + conf_buffer.long() +
                                    conf_am.long() + conf_as.long() + conf_ap.long())

        ok_unload = req_unload & (~conf_cooker) & (~conf_am)
        ok_load = req_load & (~conf_cooker)[:, None]
        ok_place_cut = req_place_cut & (~conf_cutting)[:, None]
        ok_chop = req_chop & (~conf_cutting)[:, None]
        ok_take_cut = req_take_cut & (~conf_cutting)[:, None]
        ok_place_buf = req_place_buf & (~conf_buffer)[:, None]
        ok_take_buf = req_take_buf & (~conf_buffer)[:, None]
        ok_place_as = req_place_as & (~conf_as)[:, None]
        ok_place_ap = req_place_ap & (~conf_ap)[:, None]
        ok_take_meal = req_take_meal & (~conf_ap)[:, None]
        ok_assemble = req_assemble & (~(conf_am | conf_as | conf_ap))[:, None]

        # --- Apply approved operations ---
        # Sources (take/return)
        take_any = req_take_main | req_take_side | req_take_plate
        take_kind = torch.where(req_take_main, RAW_MAIN, torch.where(req_take_side, RAW_SIDE, PLATE))
        s["held"].copy_(torch.where(take_any, take_kind, s["held"]))
        s["held_slot"].copy_(torch.where(take_any, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(take_any, self._chef_bits[None, :], s["held_contrib"]))

        ret_any = req_ret_main | req_ret_side | req_ret_plate
        s["held"].copy_(torch.where(ret_any, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ret_any, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ret_any, 0, s["held_contrib"]))

        # Cooker load
        do_load = ok_load.any(dim=1)
        load_i = ok_load.long().argmax(dim=1)
        load_con = old_hcon[lanes, load_i] | self._chef_bits[load_i]
        s["cooker_item"].copy_(torch.where(do_load, RAW_MAIN, s["cooker_item"]))
        s["cooker_slot"].copy_(torch.where(do_load, pending_slot, s["cooker_slot"]))
        s["cooker_rem"].copy_(torch.where(do_load, self.config.cook_steps, s["cooker_rem"]))
        s["cooker_contrib"].copy_(torch.where(do_load, load_con, s["cooker_contrib"]))
        s["held"].copy_(torch.where(ok_load, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ok_load, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_load, 0, s["held_contrib"]))
        load_smask = self._one_hot_8[pending_slot] & do_load[:, None]
        s["order_status"].copy_(torch.where(load_smask, ORD_STARTED, s["order_status"]))
        s["order_started"].copy_(torch.where(load_smask, (old_tick + 1)[:, None], s["order_started"]))

        # Cooker unload -> assembly.main
        s["asm_main"].copy_(torch.where(ok_unload, COOKED_MAIN, s["asm_main"]))
        s["asm_main_slot"].copy_(torch.where(ok_unload, old_ck_slot, s["asm_main_slot"]))
        s["asm_main_contrib"].copy_(torch.where(ok_unload, old_ck_con | unload_con, s["asm_main_contrib"]))
        s["cooker_item"].copy_(torch.where(ok_unload, EMPTY, s["cooker_item"]))
        s["cooker_slot"].copy_(torch.where(ok_unload, -1, s["cooker_slot"]))
        s["cooker_rem"].copy_(torch.where(ok_unload, 0, s["cooker_rem"]))
        s["cooker_contrib"].copy_(torch.where(ok_unload, 0, s["cooker_contrib"]))
        s["joint_successes"] += (ok_unload & ck_is_joint).long()

        # Cutting board: place / chop / take
        do_pcut = ok_place_cut.any(dim=1)
        pcut_i = ok_place_cut.long().argmax(dim=1)
        s["cutting_item"].copy_(torch.where(do_pcut, RAW_SIDE, s["cutting_item"]))
        s["cutting_prog"].copy_(torch.where(do_pcut, 0, s["cutting_prog"]))
        s["cutting_contrib"].copy_(torch.where(do_pcut, old_hcon[lanes, pcut_i] | self._chef_bits[pcut_i], s["cutting_contrib"]))
        s["held"].copy_(torch.where(ok_place_cut, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ok_place_cut, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_place_cut, 0, s["held_contrib"]))

        do_chop = ok_chop.any(dim=1)
        chop_i = ok_chop.long().argmax(dim=1)
        new_prog = old_ct_prog + 1
        finished_chop = do_chop & (new_prog >= self.config.chop_steps)
        s["cutting_prog"].copy_(torch.where(do_chop, new_prog, s["cutting_prog"]))
        s["cutting_item"].copy_(torch.where(finished_chop, CHOPPED_SIDE, s["cutting_item"]))
        s["cutting_contrib"].copy_(torch.where(do_chop, old_ct_con | self._chef_bits[chop_i], s["cutting_contrib"]))

        do_tcut = ok_take_cut.any(dim=1)
        tcut_i = ok_take_cut.long().argmax(dim=1)
        s["held"].copy_(torch.where(ok_take_cut, CHOPPED_SIDE, s["held"]))
        s["held_slot"].copy_(torch.where(ok_take_cut, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_take_cut, old_ct_con[:, None] | self._chef_bits[None, :], s["held_contrib"]))
        s["cutting_item"].copy_(torch.where(do_tcut, EMPTY, s["cutting_item"]))
        s["cutting_prog"].copy_(torch.where(do_tcut, 0, s["cutting_prog"]))
        s["cutting_contrib"].copy_(torch.where(do_tcut, 0, s["cutting_contrib"]))

        # Buffer: place / take
        do_pbuf = ok_place_buf.any(dim=1)
        pbuf_i = ok_place_buf.long().argmax(dim=1)
        s["buf_item"].copy_(torch.where(do_pbuf, old_held[lanes, pbuf_i], s["buf_item"]))
        s["buf_slot"].copy_(torch.where(do_pbuf, old_hslot[lanes, pbuf_i], s["buf_slot"]))
        s["buf_contrib"].copy_(torch.where(do_pbuf, old_hcon[lanes, pbuf_i] | self._chef_bits[pbuf_i], s["buf_contrib"]))
        s["held"].copy_(torch.where(ok_place_buf, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ok_place_buf, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_place_buf, 0, s["held_contrib"]))

        do_tbuf = ok_take_buf.any(dim=1)
        s["held"].copy_(torch.where(ok_take_buf, old_buf[:, None], s["held"]))
        s["held_slot"].copy_(torch.where(ok_take_buf, old_buf_slot[:, None], s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_take_buf, old_buf_con[:, None] | self._chef_bits[None, :], s["held_contrib"]))
        s["buf_item"].copy_(torch.where(do_tbuf, EMPTY, s["buf_item"]))
        s["buf_slot"].copy_(torch.where(do_tbuf, -1, s["buf_slot"]))
        s["buf_contrib"].copy_(torch.where(do_tbuf, 0, s["buf_contrib"]))

        # Assembly: place side / place plate or meal / assemble / take meal
        do_pas = ok_place_as.any(dim=1)
        pas_i = ok_place_as.long().argmax(dim=1)
        s["asm_side"].copy_(torch.where(do_pas, CHOPPED_SIDE, s["asm_side"]))
        s["asm_side_contrib"].copy_(torch.where(do_pas, old_hcon[lanes, pas_i] | self._chef_bits[pas_i], s["asm_side_contrib"]))
        s["held"].copy_(torch.where(ok_place_as, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ok_place_as, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_place_as, 0, s["held_contrib"]))

        do_pap = ok_place_ap.any(dim=1)
        pap_i = ok_place_ap.long().argmax(dim=1)
        s["asm_plate"].copy_(torch.where(do_pap, old_held[lanes, pap_i], s["asm_plate"]))
        s["asm_plate_slot"].copy_(torch.where(do_pap, old_hslot[lanes, pap_i], s["asm_plate_slot"]))
        s["asm_plate_contrib"].copy_(torch.where(do_pap, old_hcon[lanes, pap_i] | self._chef_bits[pap_i], s["asm_plate_contrib"]))
        s["held"].copy_(torch.where(ok_place_ap, EMPTY, s["held"]))
        s["held_slot"].copy_(torch.where(ok_place_ap, -1, s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_place_ap, 0, s["held_contrib"]))

        do_asm = ok_assemble.any(dim=1)
        asm_i = ok_assemble.long().argmax(dim=1)
        asm_con = old_am_con | old_as_con | old_ap_con | self._chef_bits[asm_i]
        s["asm_plate"].copy_(torch.where(do_asm, MEAL, s["asm_plate"]))
        s["asm_plate_slot"].copy_(torch.where(do_asm, old_am_slot, s["asm_plate_slot"]))
        s["asm_plate_contrib"].copy_(torch.where(do_asm, asm_con, s["asm_plate_contrib"]))
        s["asm_main"].copy_(torch.where(do_asm, EMPTY, s["asm_main"]))
        s["asm_main_slot"].copy_(torch.where(do_asm, -1, s["asm_main_slot"]))
        s["asm_main_contrib"].copy_(torch.where(do_asm, 0, s["asm_main_contrib"]))
        s["asm_side"].copy_(torch.where(do_asm, EMPTY, s["asm_side"]))
        s["asm_side_contrib"].copy_(torch.where(do_asm, 0, s["asm_side_contrib"]))

        do_tm = ok_take_meal.any(dim=1)
        s["held"].copy_(torch.where(ok_take_meal, MEAL, s["held"]))
        s["held_slot"].copy_(torch.where(ok_take_meal, old_ap_slot[:, None], s["held_slot"]))
        s["held_contrib"].copy_(torch.where(ok_take_meal, old_ap_con[:, None] | self._chef_bits[None, :], s["held_contrib"]))
        s["asm_plate"].copy_(torch.where(do_tm, EMPTY, s["asm_plate"]))
        s["asm_plate_slot"].copy_(torch.where(do_tm, -1, s["asm_plate_slot"]))
        s["asm_plate_contrib"].copy_(torch.where(do_tm, 0, s["asm_plate_contrib"]))

        # Serve (at most 1 chef per kitchen can hold a given meal; handle all 3 slots safely)
        deliv_norm = torch.zeros((n,), dtype=torch.bool, device=self.device)
        deliv_joint = torch.zeros((n,), dtype=torch.bool, device=self.device)
        for c in range(3):
            sv = req_serve[:, c]
            sl = old_hslot[:, c].clamp(0, ORDER_SLOTS - 1)
            okind = s["order_kind"][lanes, sl]
            is_j = okind == KIND_JOINT
            rew = torch.where(is_j, self.config.joint_reward, self.config.normal_reward)
            s["score"] += torch.where(sv, rew, 0.0)
            s["normal_completed"] += (sv & (~is_j)).long()
            s["joint_completed"] += (sv & is_j).long()
            deliv_norm |= sv & (~is_j)
            deliv_joint |= sv & is_j
            final_con = old_hcon[:, c] | self._chef_bits[c]
            s["three_contributor_meals"] += (sv & (final_con == 7)).long()
            smask = self._one_hot_8[sl] & sv[:, None]
            s["order_status"].copy_(torch.where(smask, ORD_SERVED, s["order_status"]))
            s["order_completed"].copy_(torch.where(smask, (old_tick + 1)[:, None], s["order_completed"]))
            s["held"][:, c] = torch.where(sv, EMPTY, s["held"][:, c])
            s["held_slot"][:, c] = torch.where(sv, -1, s["held_slot"][:, c])
            s["held_contrib"][:, c] = torch.where(sv, 0, s["held_contrib"][:, c])

        # --- Movement & collision resolution (exact match to kitchen.py _move) ---
        is_move = actions < STAY
        s["facing"].copy_(torch.where(is_move, actions, old_dir))
        cand_pos = old_pos + self._offsets[actions]
        in_bounds = (cand_pos >= 0) & (cand_pos < NUM_CELLS)
        # Prevent wrap-around on horizontal moves (all outer ring tiles are X walls, so floor_mask suffices).
        valid_step = is_move & in_bounds & self._floor_mask[cand_pos.clamp(0, NUM_CELLS - 1)]
        desired = torch.where(valid_step, cand_pos, old_pos)

        # Pairwise target collision or head-on swap.
        same_01 = (desired[:, 0] == desired[:, 1]) | ((desired[:, 0] == old_pos[:, 1]) & (desired[:, 1] == old_pos[:, 0]))
        same_02 = (desired[:, 0] == desired[:, 2]) | ((desired[:, 0] == old_pos[:, 2]) & (desired[:, 2] == old_pos[:, 0]))
        same_12 = (desired[:, 1] == desired[:, 2]) | ((desired[:, 1] == old_pos[:, 2]) & (desired[:, 2] == old_pos[:, 1]))
        blocked = torch.stack((same_01 | same_02, same_01 | same_12, same_02 | same_12), dim=1)

        # Propagate blocking into stationary/blocked chefs (at most 2 iterations for 3 chefs).
        for _ in range(2):
            stay_or_blk = blocked | (desired == old_pos)
            b0 = blocked[:, 0] | ((desired[:, 0] == old_pos[:, 1]) & stay_or_blk[:, 1]) | ((desired[:, 0] == old_pos[:, 2]) & stay_or_blk[:, 2])
            b1 = blocked[:, 1] | ((desired[:, 1] == old_pos[:, 0]) & stay_or_blk[:, 0]) | ((desired[:, 1] == old_pos[:, 2]) & stay_or_blk[:, 2])
            b2 = blocked[:, 2] | ((desired[:, 2] == old_pos[:, 0]) & stay_or_blk[:, 0]) | ((desired[:, 2] == old_pos[:, 1]) & stay_or_blk[:, 1])
            blocked = torch.stack((b0, b1, b2), dim=1)

        s["movement_conflicts"] += (blocked & (desired != old_pos)).long().sum(dim=1)
        s["pos"].copy_(torch.where(blocked, old_pos, desired))

        # --- Cooker automatic countdown ---
        still_cooking = (old_ck_item == RAW_MAIN) & (old_ck_rem > 0) & (s["cooker_item"] == RAW_MAIN) & (~do_load)
        new_rem = old_ck_rem - 1
        s["cooker_rem"].copy_(torch.where(still_cooking, new_rem, s["cooker_rem"]))
        s["cooker_item"].copy_(torch.where(still_cooking & (new_rem == 0), COOKED_MAIN, s["cooker_item"]))

        # --- Advance tick & release new orders ---
        new_tick = old_tick + 1
        s["tick"].copy_(new_tick)
        not_end = new_tick < self.config.horizon
        for _ in range(self.config.active_limit):
            live_cnt = ((s["order_status"] == ORD_ACTIVE) | (s["order_status"] == ORD_STARTED)).sum(dim=1)
            need_rel = not_end & (live_cnt < self.config.active_limit)
            if self.config.continuous_orders:
                # Prefer queued (unused initial slots < 8), else reuse served slot with smallest order_id.
                is_q = s["order_status"] == ORD_QUEUED
                is_reusable = (s["order_status"] == ORD_SERVED) & (s["order_completed"] <= new_tick[:, None])
                q_key = torch.where(is_q, self._slot_arange[None, :], 1_000_000_000)
                r_key = torch.where(is_reusable, s["order_id"], 1_000_000_000)
                has_q = is_q.any(dim=1)
                slot = torch.where(has_q, q_key.argmin(dim=1), r_key.argmin(dim=1))
                nid = s["next_order_id"]
                smask = self._one_hot_8[slot] & need_rel[:, None]
                s["order_id"].copy_(torch.where(smask, nid[:, None], s["order_id"]))
                s["order_kind"].copy_(torch.where(smask, (nid % 2)[:, None], s["order_kind"]))
                s["order_status"].copy_(torch.where(smask, ORD_ACTIVE, s["order_status"]))
                s["order_released"].copy_(torch.where(smask, new_tick[:, None], s["order_released"]))
                s["order_started"].copy_(torch.where(smask, -1, s["order_started"]))
                s["order_completed"].copy_(torch.where(smask, -1, s["order_completed"]))
                s["next_order_id"].copy_(torch.where(need_rel, nid + 1, nid))
            else:
                is_q = s["order_status"] == ORD_QUEUED
                has_q = is_q.any(dim=1)
                rel = need_rel & has_q
                q_key = torch.where(is_q, self._slot_arange[None, :], 1_000_000_000)
                slot = q_key.argmin(dim=1)
                smask = self._one_hot_8[slot] & rel[:, None]
                s["order_status"].copy_(torch.where(smask, ORD_ACTIVE, s["order_status"]))
                s["order_released"].copy_(torch.where(smask, new_tick[:, None], s["order_released"]))

        raw_rew = s["score"] - old_score
        if self.config.shaping_scale != 0.0:
            new_phi = torch.where(new_tick >= self.config.horizon, 0.0, self._potential())
            shaping = self.config.shaping_scale * (self.config.shaping_gamma * new_phi - old_phi)
            self._reward_out.copy_(raw_rew + shaping)
        else:
            self._reward_out.copy_(raw_rew)
        self._delivered_normal_out.copy_(deliv_norm)
        self._delivered_joint_out.copy_(deliv_joint)
        self._write_observations()

    def _item_feats(self, kind, slot):
        # kind [N] in 0..6, slot [N] in -1..7 -> [N, 16]
        return torch.cat((self._one_hot_7_f[kind], self._one_hot_9_f[slot + 1]), dim=-1)

    def _write_observations(self):
        s = self.state
        n = self.num_envs
        h = float(self.config.horizon)
        parts = [(s["tick"].float() / h)[:, None]]
        for c in range(3):
            parts.append(self._ones_active)
            parts.append(self._one_hot_49_f[s["pos"][:, c]])
            parts.append(self._one_hot_4_f[s["facing"][:, c]])
            parts.append(self._item_feats(s["held"][:, c], s["held_slot"][:, c]))
        no_slot = self._no_slot
        parts.append(self._item_feats(s["cooker_item"], s["cooker_slot"]))
        parts.append(self._item_feats(s["cutting_item"], no_slot))
        parts.append(self._item_feats(s["asm_main"], s["asm_main_slot"]))
        parts.append(self._item_feats(s["asm_side"], no_slot))
        parts.append(self._item_feats(s["asm_plate"], s["asm_plate_slot"]))
        parts.append(self._item_feats(s["buf_item"], s["buf_slot"]))
        parts.append((s["cooker_rem"].float() / float(self.config.cook_steps))[:, None])
        parts.append((s["cutting_prog"].float() / float(self.config.chop_steps))[:, None])
        ord_stat = self._one_hot_4_f[s["order_status"]]  # [N, 8, 4]
        ord_kind = self._one_hot_2_f[s["order_kind"]]    # [N, 8, 2]
        ord_rel = (s["order_released"].clamp_min(0).float() / h)[:, :, None]
        ord_st = (s["order_started"].clamp_min(0).float() / h)[:, :, None]
        parts.append(torch.cat((ord_stat, ord_kind, ord_rel, ord_st), dim=-1).reshape(n, ORDER_SLOTS * 8))
        common = torch.cat(parts, dim=-1)  # [N, 373]
        for c in range(3):
            self._obs_out[:, c, :CENTRAL_DYNAMIC_DIM].copy_(common)
            self._obs_out[:, c, CENTRAL_DYNAMIC_DIM:].copy_(self._one_hot_3_f[c][None, :])

    def episode_metrics(self):
        """Aggregate SharedKitchen episode_metrics across all lanes (origin/hy a00dafd compatible)."""
        s = self.state
        norm_c = s["normal_completed"].float()
        joint_c = s["joint_completed"].float()
        meals = norm_c + joint_c
        ticks = s["tick"].float().clamp_min(1.0)
        attempts = s["joint_attempt_steps"].float()
        successes = s["joint_successes"].float()
        rate = torch.where(attempts > 0, successes / attempts.clamp_min(1.0), 0.0)
        wait_steps = (s["joint_ready_steps"] - s["joint_successes"]).float()
        out = dict(
            team_score=float(s["score"].mean()),
            meals_completed=float(meals.mean()),
            normal_completed=float(norm_c.mean()),
            joint_completed=float(joint_c.mean()),
            ticks=float(s["tick"].float().mean()),
            joint_attempt_success_rate=float(rate.mean()),
            joint_wait_steps=float(wait_steps.mean()),
            unmatched_attempt_steps=float(s["unmatched_attempt_steps"].float().mean()),
            three_contributor_meals=float(s["three_contributor_meals"].float().mean()),
            resource_conflicts=float(s["resource_conflicts"].float().mean()),
            movement_conflicts=float(s["movement_conflicts"].float().mean()),
        )
        if self.config.continuous_orders:
            out["meals_per_100_steps"] = float((100.0 * meals / ticks).mean())
        else:
            total = self.config.normal_orders + self.config.joint_orders
            out["orders_total"] = total
            out["completion_rate"] = float((meals / total).mean())
        return out

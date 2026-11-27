"""Three trainable chefs in the native continuous-order SharedKitchen."""
import multiprocessing as mp
import traceback
import numpy as np
from overcooked_ai_py.cooperative.shared import SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.layouts import LAYOUT, STATIONS
from overcooked_ai_py.cooperative.shared.kitchen import ITEM_KINDS

OBS_DIM, STATE_DIM, ACTION_DIM, NUM_AGENTS = 866, 863, 6, 3
SPATIAL_CHANNELS, HEIGHT, WIDTH = 26, 7, 7
ENVIRONMENT_COMMIT = "de37e0818c2e76311e105ab8689e5162bd84133f"
SHAPING_SCALE = .2


def task_config(stage="standard", gamma=.9985, train=True):
    if stage != "standard":
        raise ValueError("SharedKitchen supports the native standard task")
    return SharedKitchenConfig()


def curriculum(progress):
    return "standard"


def encode(core):
    vector = np.asarray(core.observations(), dtype=np.float32)
    state = np.asarray(core.centralized_state(), dtype=np.float32)
    if vector.shape != (3, OBS_DIM) or state.shape != (STATE_DIM,):
        raise ValueError("SharedKitchen observation schema changed")
    spatial = np.zeros((3, SPATIAL_CHANNELS, HEIGHT, WIDTH), np.float32)
    for y, row in enumerate(LAYOUT):
        for x, tile in enumerate(row):
            spatial[:, "X MLKGABDS".index(tile), y, x] = 1.
    s = core.state
    x, y = STATIONS["G"]
    spatial[:, 10, y, x] = s.cooker.remaining / core.config.cook_steps
    x, y = STATIONS["K"]
    spatial[:, 11, y, x] = s.cutting.progress / core.config.chop_steps
    for ego in range(3):
        order = [ego] + [j for j in range(3) if j != ego]
        for relative, slot in enumerate(order):
            chef = s.chefs[slot]
            x, y = chef.position
            spatial[ego, 12 + relative*4:16 + relative*4, y, x] = (
                1., *chef.orientation, ITEM_KINDS.index(chef.held.kind if chef.held else None)/6.)
    x, y = STATIONS["A"]
    spatial[:, 24, y, x] = sum(item is not None for item in (
        s.assembly.main, s.assembly.side, s.assembly.plate))/3.
    x, y = STATIONS["B"]
    spatial[:, 25, y, x] = ITEM_KINDS.index(s.buffer.kind if s.buffer else None)/6.
    return {"vector": vector, "spatial": spatial, "state": state}


def potential(core):
    # A bounded state potential gives progress feedback. With a zero terminal
    # potential, discounted shaping telescopes and adds no repeat-action bonus.
    weights = {"raw_main": 1., "raw_side": 1., "chopped_side": 2.,
               "cooked_main": 3., "plate": 1., "meal": 6.}
    s = core.state
    items = [c.held for c in s.chefs] + [
        s.cooker.item, s.cutting.item, s.assembly.main,
        s.assembly.side, s.assembly.plate, s.buffer]
    value = sum(weights[item.kind] for item in items if item is not None)
    if s.cooker.item and s.cooker.remaining:
        value += 1. - s.cooker.remaining/core.config.cook_steps
    if s.cutting.item and s.cutting.item.kind == "raw_side":
        value += s.cutting.progress/core.config.chop_steps
    return value


class KitchenSlot:
    def __init__(self, seed, index, stage, train, gamma):
        self.seed, self.index, self.stage = int(seed), int(index), stage
        self.train, self.gamma, self.episode = train, gamma, -1
        self.core = SharedKitchen(task_config(stage, gamma, train))

    def reset(self):
        self.episode += 1
        seed = int(np.random.SeedSequence(
            [self.seed, self.index, self.episode]).generate_state(1)[0])
        self.core.reset(seed=seed)
        self.shaping_total = 0.
        self.reward_total = 0.
        return encode(self.core)

    def step(self, actions):
        old = potential(self.core) if self.train else 0.
        _, raw, terminated, truncated, info = self.core.step(tuple(int(a) for a in actions))
        done = bool(terminated or truncated)
        shaping = SHAPING_SCALE*(self.gamma*(0. if done else potential(self.core))-old) if self.train else 0.
        self.shaping_total += shaping
        self.reward_total += float(raw) + shaping
        info = dict(info)
        info["training_shaping"] = shaping
        if done:
            summary = dict(info["metrics"])
            summary.update(episode=self.episode, training_reward=self.reward_total,
                           training_shaping=self.shaping_total)
            info["episode_summary"] = summary
            obs = self.reset()
        else:
            obs = encode(self.core)
        return obs, float(raw + shaping), done, info


def _stack(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _worker(conn, indices, seed, stage, train, gamma):
    slots = [KitchenSlot(seed, i, stage, train, gamma) for i in indices]
    try:
        while True:
            command, payload = conn.recv()
            if command == 'close':
                break
            if command == 'reset':
                conn.send(('ok', [s.reset() for s in slots]))
            elif command == 'stage':
                for s in slots: s.stage = payload
                conn.send(('ok', None))
            elif command == 'step':
                conn.send(('ok', [s.step(a) for s, a in zip(slots, payload)]))
            else:
                raise ValueError(command)
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


class KitchenBatch:
    def __init__(self, num_envs=8, seed=1, stage='standard', workers=0, train=True, gamma=.9985):
        self.num_envs, self.stage = int(num_envs), stage
        self.workers, self.slots = [], []
        self.closed = False
        if workers:
            ctx = mp.get_context('spawn')
            for ids in np.array_split(np.arange(self.num_envs), min(workers, self.num_envs)):
                ids = ids.tolist()
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, ids, seed, stage, train, gamma), daemon=True)
                p.start(); child.close()
                self.workers.append((ids, parent, p))
        else:
            self.slots = [KitchenSlot(seed, i, stage, train, gamma) for i in range(self.num_envs)]

    def _receive(self, conn):
        if not conn.poll(180):
            raise TimeoutError('Kitchen worker did not reply in 180 seconds')
        status, data = conn.recv()
        if status != 'ok':
            raise RuntimeError(data)
        return data

    def reset(self):
        if self.workers:
            for _, conn, _ in self.workers: conn.send(('reset', None))
            rows = [None] * self.num_envs
            for ids, conn, _ in self.workers:
                for i, obs in zip(ids, self._receive(conn)): rows[i] = obs
        else:
            rows = [s.reset() for s in self.slots]
        return _stack(rows)

    def set_stage(self, stage):
        task_config(stage, .9985)  # validate before mutating workers
        self.stage = stage
        if self.workers:
            for _, conn, _ in self.workers: conn.send(('stage', stage))
            for _, conn, _ in self.workers: self._receive(conn)
        else:
            for s in self.slots: s.stage = stage

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 3) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError('Expected integer actions [num_envs, 3]')
        if np.any(actions < 0) or np.any(actions > 5): raise ValueError('Action out of range')
        if self.workers:
            for ids, conn, _ in self.workers: conn.send(('step', actions[ids]))
            rows = [None] * self.num_envs
            for ids, conn, _ in self.workers:
                for i, row in zip(ids, self._receive(conn)): rows[i] = row
        else:
            rows = [s.step(a) for s, a in zip(self.slots, actions)]
        obs, reward, done, info = zip(*rows)
        return _stack(obs), np.asarray(reward, np.float32), np.asarray(done, bool), list(info)

    def close(self):
        if self.closed: return
        self.closed = True
        for _, conn, p in self.workers:
            try: conn.send(('close', None))
            except (OSError, EOFError): pass
        for _, conn, p in self.workers:
            p.join(3)
            if p.is_alive(): p.terminate(); p.join(3)
            conn.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

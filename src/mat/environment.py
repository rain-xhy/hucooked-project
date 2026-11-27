"""Use HuCooked's original configuration and submit one joint action per tick."""

from dataclasses import asdict, dataclass
import multiprocessing as mp
from pathlib import Path
import sys
import traceback

import numpy as np


SCENE_COMMIT = "10aba7415fb0ce8340b2cc99025ca1f0627a92e0"
SCENE_ENTRY = Path("src/overcooked_ai_py/cooperative/shared/parallel.py")


def find_scene(explicit):
    if explicit is not None:
        candidates = [explicit]
    else:
        root = Path(__file__).resolve().parents[2]
        candidates = [root / "overcooked_env"]
    for candidate in candidates:
        if (candidate / SCENE_ENTRY).is_file():
            return candidate.resolve()
    raise FileNotFoundError("Pass --env-root pointing to the SharedKitchen source directory.")


@dataclass
class Transition:
    observation: np.ndarray
    reward: float
    done: bool
    episode: dict | None


class Kitchen:
    def __init__(self, scene, seed, seed_stride=1):
        source = str(scene / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from overcooked_ai_py.cooperative.shared.parallel import ParallelSharedKitchen

        self.env = ParallelSharedKitchen()
        self.agents = self.env.possible_agents
        self.next_seed = seed
        self.seed_stride = seed_stride
        self.seed = None
        self.spawns = None

    def describe(self):
        config = self.env.core.config
        return {
            "config": asdict(config),
            "fingerprint": self.env.core.fingerprint(),
            "score_cap": (
                None if getattr(config, "continuous_orders", False)
                else config.normal_orders * config.normal_reward + config.joint_orders * config.joint_reward
            ),
            "observation_size": self.env.observation_space(self.agents[0]).shape[0],
            "agents": len(self.agents),
            "actions": int(self.env.action_space(self.agents[0]).n),
        }

    def reset(self):
        self.seed = self.next_seed
        self.next_seed += self.seed_stride
        observations, _ = self.env.reset(seed=self.seed)
        self.spawns = tuple(chef.position for chef in self.env.core.state.chefs)
        return np.stack([observations[agent] for agent in self.agents])

    def step(self, actions, restart=True):
        joint_action = dict(zip(self.agents, (int(action) for action in actions), strict=True))
        obs, rewards, terminated, truncated, info = self.env.step(joint_action)
        first = self.agents[0]
        done = terminated[first] or truncated[first]
        observation = np.stack([obs[agent] for agent in self.agents])
        episode = None
        if done:
            episode = {
                **info[first]["metrics"],
                "seed": self.seed,
                "spawns": self.spawns,
                "terminated": terminated[first],
                "truncated": truncated[first],
            }
            if restart:
                observation = self.reset()
        return Transition(observation, float(rewards[first]), done, episode)

    def snapshot(self):
        return {
            "scene": self.env.core.snapshot(),
            "seed": self.seed,
            "next_seed": self.next_seed,
            "spawns": self.spawns,
        }

    def restore(self, checkpoint):
        self.env.core.restore(checkpoint["scene"])
        self.env.agents = self.agents.copy()
        self.seed = checkpoint["seed"]
        self.next_seed = checkpoint["next_seed"]
        self.spawns = checkpoint["spawns"]
        return np.stack(self.env.core.observations())


def kitchen_worker(connection, scene, seed, seed_stride):
    try:
        kitchen = Kitchen(scene, seed, seed_stride)
        while True:
            method, args = connection.recv()
            if method == "close":
                return
            connection.send((True, getattr(kitchen, method)(*args)))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class KitchenBatch:
    def __init__(self, scene, count, seed, subprocesses):
        self.count = count
        self.connections = []
        self.processes = []
        self.local = []
        if subprocesses:
            context = mp.get_context("spawn")
            try:
                for index in range(count):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=kitchen_worker, args=(child, scene, seed + index, count), daemon=True
                    )
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
            except Exception:
                self.close()
                raise
        else:
            self.local = [Kitchen(scene, seed + index, count) for index in range(count)]

    def _call(self, method, args):
        if self.local:
            return [getattr(env, method)(*arg) for env, arg in zip(self.local, args, strict=True)]
        for connection, arg in zip(self.connections, args, strict=True):
            connection.send((method, arg))
        results = []
        for connection in self.connections:
            ok, value = connection.recv()
            if not ok:
                raise RuntimeError(value)
            results.append(value)
        return results

    def reset(self):
        return np.stack(self._call("reset", [()] * self.count))

    def step(self, actions):
        return self._call("step", [(row,) for row in actions])

    def snapshots(self):
        return self._call("snapshot", [()] * self.count)

    def restore(self, snapshots):
        return np.stack(self._call("restore", [(item,) for item in snapshots]))

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", ()))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
        self.connections.clear()
        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

"""Run several independent Overcooked kitchens in each SB3 subprocess."""

from functools import partial
from itertools import chain

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class KitchenGroup:
    def __init__(self, env_fns, first_rank):
        self.kitchens = DummyVecEnv(env_fns)
        self.first_rank = first_rank
        self.observation_space = self.kitchens.observation_space
        self.action_space = self.kitchens.action_space

    def reset(self):
        return self.kitchens.reset()

    def step(self, actions):
        obs, rewards, dones, infos = self.kitchens.step(actions)
        # The outer worker sees one environment. Individual resets happen inside DummyVecEnv.
        return obs, 0.0, False, (rewards, dones, infos)

    def seed(self, seed):
        return self.kitchens.seed(seed + self.first_rank)

    def close(self):
        self.kitchens.close()

    def dispatch(self, method, args, kwargs, indices=None):
        return getattr(self.kitchens, method)(*args, indices=indices, **kwargs)


class GroupedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, workers):
        self.workers = workers
        self.group_size = len(env_fns) // workers
        groups = [
            partial(KitchenGroup, env_fns[start:start + self.group_size], start)
            for start in range(0, len(env_fns), self.group_size)
        ]
        super().__init__(groups)
        self.num_envs = len(env_fns)

    def reset(self):
        return np.concatenate(super().reset(), axis=0)

    def step_async(self, actions):
        super().step_async(actions.reshape(self.workers, self.group_size, *actions.shape[1:]))

    def step_wait(self):
        obs, _, _, results = super().step_wait()
        rewards, dones, infos = zip(*results)
        return np.concatenate(obs), np.concatenate(rewards), np.concatenate(dones), list(chain.from_iterable(infos))

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, 2**32 - 1))
        for remote in self.remotes:
            remote.send(("seed", seed))
        return list(chain.from_iterable(remote.recv() for remote in self.remotes))

    def _call_kitchens(self, method, args, kwargs, indices):
        if indices is None:
            results = super().env_method("dispatch", method, args, kwargs, indices=range(self.workers))
            return list(chain.from_iterable(results)) if method != "set_attr" else None
        results = []
        for index in self._get_indices(indices):
            worker, local = divmod(index, self.group_size)
            result, = super().env_method("dispatch", method, args, kwargs, local, indices=worker)
            if method != "set_attr":
                results.extend(result)
        return results if method != "set_attr" else None

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self._call_kitchens("env_method", (method_name, *method_args), method_kwargs, indices)

    def get_attr(self, attr_name, indices=None):
        return self._call_kitchens("get_attr", (attr_name,), {}, indices)

    def set_attr(self, attr_name, value, indices=None):
        self._call_kitchens("set_attr", (attr_name, value), {}, indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._call_kitchens("env_is_wrapped", (wrapper_class,), {}, indices)

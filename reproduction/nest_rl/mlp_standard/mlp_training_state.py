"""Save and restore the standard kitchen at a PPO rollout boundary."""

import os
import random

import numpy as np
import torch


KITCHEN_FIELDS = (
    "agent_pos", "orientation", "held", "counter_items", "onions", "cook_tick",
    "ticks", "delivered", "last_sparse",
)
ENV_FIELDS = ("obs", "meta_episode", "returns", "episodes", "episodes_with_context", "rounds")
HISTORY_FIELDS = ("obs", "actions", "starts", "lengths", "cursor", "episode_id", "first_period", "current_episode")
ROLLOUT_FIELDS = ("obs", "masks", "bad_masks", "recurrent_hidden_states")


def select(obj, fields):
    return {field: getattr(obj, field) for field in fields}


def restore(obj, state):
    for field, value in state.items():
        target = getattr(obj, field)
        if isinstance(target, torch.Tensor):
            target.copy_(value)
        else:
            setattr(obj, field, value)


class TrainingState:
    def __init__(self, options, output, rollout_size):
        self.directory = output / "training-state"
        self.interval = options.checkpoint_every_steps
        self.rollout_size = rollout_size
        self.total_updates = options.updates
        self.config = {key: value for key, value in vars(options).items()
                       if key not in ("resume", "run_name", "defer_rollout_stats", "profile_cuda_stages")}
        self.snapshot = None
        self.completed_updates = 0
        if options.resume is not None:
            self.snapshot = torch.load(options.resume, map_location="cpu", weights_only=False)
            if self.snapshot["version"] != 1 or self.snapshot["config"] != self.config:
                raise ValueError("The checkpoint requires the original training options, except for the new run name.")
            self.completed_updates = self.snapshot["completed_updates"]
            if not 0 < self.completed_updates < self.total_updates:
                raise ValueError("The checkpoint must precede the requested training endpoint.")

    def restore(self, policy, agent, env, rollouts):
        if self.snapshot is None:
            return 0
        state = self.snapshot
        policy.load_state_dict(state["policy"])
        agent.optimizer.load_state_dict(state["optimizer"])
        restore(env.kitchen, state["kitchen"])
        restore(env, state["environment"])
        env.episode_step = int(state["kitchen"]["ticks"][0])
        restore(rollouts.history, state["history"])
        for field, value in state["rollout_head"].items():
            rollouts_field = getattr(rollouts, field)
            rollouts_field[0].copy_(value)
        env.generator.set_state(state["env_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], env.device)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.snapshot = None
        print(f"Resumed at {self.completed_updates * self.rollout_size:,} env steps", flush=True)
        return self.completed_updates

    def save(self, completed_updates, policy, agent, env, rollouts):
        steps = completed_updates * self.rollout_size
        if steps % self.interval and completed_updates != self.total_updates:
            return
        if rollouts.step != 0:
            raise RuntimeError("A training checkpoint must be written at a rollout boundary.")
        state = {
            "version": 1,
            "config": self.config,
            "completed_updates": completed_updates,
            "policy": policy.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "kitchen": select(env.kitchen, KITCHEN_FIELDS),
            "environment": select(env, ENV_FIELDS),
            "history": select(rollouts.history, HISTORY_FIELDS),
            "rollout_head": {
                field: tensor[0].clone() for field, tensor in select(rollouts, ROLLOUT_FIELDS).items()
                if tensor is not None
            },
            "env_rng": env.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(env.device),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        self.directory.mkdir(exist_ok=True)
        path = self.directory / f"{steps}.pt"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        print(f"Training state saved: {path}", flush=True)

"""Reference-controller demonstrations for the new environment's BC partners."""

import importlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .artifacts import atomic_path
from .mixed_kitchen import select_state
from .mixed_tasks import MIXED_COMMIT, mixed_config, mixed_source


def expert_trajectory(job):
    layout, horizon, seed = job
    source = mixed_source()
    controller = importlib.import_module(f"{source.__name__}.controller").ReferenceController
    env = source.MixedKitchen(mixed_config(layout, horizon))
    obs, _ = env.reset(seed=seed)
    expert = controller()
    observations, actions = [], []
    for _ in range(horizon):
        joint = expert(env)
        observations.extend(obs)
        actions.extend(joint)
        obs, _, done, truncated, _ = env.step(joint)
        if done or truncated:
            break
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def prepare_mixed(args):
    pending = []
    for index, layout in enumerate(args.layouts):
        path = args.run_dir / "data" / f"{layout}.npz"
        metadata = {"environment_commit": MIXED_COMMIT, "layout": layout, "horizon": args.horizon,
                    "seed": args.seed, "episodes": args.expert_episodes, "kind": "reference_controller"}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data["source"])) != metadata:
                    raise FileExistsError(f"Different demonstrations already exist at {path}")
            continue
        seed = args.seed + index * args.expert_episodes
        pending.append((path, metadata, [(layout, args.horizon, seed + episode)
                                        for episode in range(args.expert_episodes)]))
    if not pending:
        return
    if args.expert_workers == 1:
        _write_trajectories(pending, map)
    else:
        with ProcessPoolExecutor(max_workers=args.expert_workers, mp_context=mp.get_context("spawn")) as pool:
            _write_trajectories(pending, pool.map)


def _write_trajectories(pending, apply):
    # Ordered results preserve seed-to-trajectory assignment and dataset splits.
    trajectories = iter(apply(expert_trajectory, [job for _, _, jobs in pending for job in jobs]))
    for path, metadata, jobs in pending:
        obs, actions = zip(*(next(trajectories) for _ in jobs))
        groups = np.repeat(np.arange(len(jobs), dtype=np.int64), [len(a) for a in actions])
        with atomic_path(path) as temporary:
            np.savez_compressed(temporary, observations=np.concatenate(obs), actions=np.concatenate(actions),
                                groups=groups, source=json.dumps(metadata))


@torch.no_grad()
def mixed_selfplay_score(network, kitchen, seed):
    device = kitchen.device
    episodes = len(kitchen.layouts)
    generator = torch.Generator(device=device).manual_seed(seed)
    state = kitchen.reset(generator)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    returns = torch.zeros(episodes, device=device)
    for _ in range(kitchen.config.horizon):
        logits = network(kitchen.observations(state))
        noise = torch.empty_like(logits).exponential_(generator=generator)
        joint = (logits - noise.log()).argmax(2)
        result = kitchen.step(state, joint, kitchen.draws(generator))
        returns += torch.where(active, result.raw, 0.)
        state = select_state(active, result.state, state)
        active = active & ~result.done
    return float(returns.mean())

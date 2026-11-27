"""Native, unshaped evaluation for central HiPT policies."""

from __future__ import annotations

import numpy as np
import torch

from .common import OptionClock, sample_durations
from .model import SPATIAL_CHANNELS, SPATIAL_HEIGHT, SPATIAL_WIDTH

NATIVE_METRICS = (
    "team_score", "meals_completed", "normal_completed", "joint_completed",
    "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
    "three_contributor_meals", "resource_conflicts", "movement_conflicts",
    "ticks", "meals_per_100_steps",
)
EPISODE_METRICS = NATIVE_METRICS + ("training_reward", "training_shaping")


def _summarize(records):
    means, stds = {}, {}
    for key in EPISODE_METRICS:
        values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
        means[key] = float(values.mean()) if len(values) else 0.0
        stds[key] = float(values.std()) if len(values) else 0.0
    return {"episodes": len(records), "metric_means": means,
            "metric_stds": stds, "records": records}


@torch.no_grad()
def evaluate(model, *, episodes=12, num_envs=4, workers=0, seed=10001,
             device="cpu", gamma=0.9985, p_min=20, p_max=40):
    from joint_common.envs import KitchenBatch

    if episodes <= 0 or num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    env = KitchenBatch(num_envs=num_envs, seed=seed, stage="standard", workers=workers,
                       train=False, gamma=gamma, reward_profile="legacy")
    target, records = int(episodes), []
    was_training = model.training
    model.eval()
    try:
        obs = env.reset()
        hidden = model.initial_state(num_envs, device=device)
        previous_done = np.zeros(num_envs, dtype=bool)
        clock = OptionClock(num_envs)
        while len(records) < target:
            done_t = torch.as_tensor(previous_done, device=device, dtype=torch.float32)
            high_logits, _, low_logits, _, hidden = model.step(
                torch.as_tensor(obs["spatial"][:, 0], device=device, dtype=torch.float32),
                torch.as_tensor(obs["state"], device=device, dtype=torch.float32),
                hidden, done_t,
            )
            chosen_z = high_logits.argmax(-1).cpu().numpy()
            due = clock.due
            durations = sample_durations(rng, (num_envs, 3), p_min, p_max)
            clock.start(due, chosen_z, durations)
            z = torch.as_tensor(clock.z, device=device, dtype=torch.long)
            selected = low_logits.gather(
                2, z[..., None, None].expand(num_envs, 3, 1, model.action_dim)
            ).squeeze(2)
            actions = selected.argmax(-1).cpu().numpy()
            nxt, _, done, infos = env.step(actions)
            clock.advance(done)
            for ended, info in zip(done, infos):
                if ended and "episode_summary" in info and len(records) < target:
                    records.append(info["episode_summary"])
            obs, previous_done = nxt, done.copy()
    finally:
        env.close()
        model.train(was_training)
    return _summarize(records)


@torch.no_grad()
def evaluate_population(models, *, episodes_per_member=4, workers=0, seed=700001,
                        gamma=0.9985):
    """Greedy matched-seed native evaluation of every central pool member."""
    from joint_common.envs import KitchenBatch

    population = len(models)
    if population == 0 or episodes_per_member <= 0:
        raise ValueError("Population and episodes_per_member must be positive")
    total = population * int(episodes_per_member)
    env = KitchenBatch(num_envs=total, seed=seed, stage="standard", workers=workers,
                       train=False, gamma=gamma, reward_profile="legacy")
    training = [model.training for model in models]
    for model in models:
        model.eval()
    metrics = {key: np.full(total, np.nan, dtype=np.float64) for key in EPISODE_METRICS}
    captured = np.zeros(total, dtype=bool)
    try:
        obs = env.reset()
        while not captured.all():
            actions = np.empty((total, 3), dtype=np.int64)
            for member, model in enumerate(models):
                start = member * episodes_per_member
                stop = start + episodes_per_member
                logits, _ = model(
                    torch.as_tensor(obs["spatial"][start:stop, 0], dtype=torch.float32),
                    torch.as_tensor(obs["state"][start:stop], dtype=torch.float32),
                )
                actions[start:stop] = logits.argmax(-1).numpy()
            nxt, _, done, infos = env.step(actions)
            for i, (ended, info) in enumerate(zip(done, infos)):
                if ended and not captured[i]:
                    summary = info.get("episode_summary")
                    if summary is None:
                        raise RuntimeError("Native episode ended without a summary")
                    for key in EPISODE_METRICS:
                        metrics[key][i] = float(summary[key])
                    captured[i] = True
            obs = nxt
    finally:
        env.close()
        for model, mode in zip(models, training):
            model.train(mode)
    if not np.isfinite(metrics["team_score"]).all():
        raise FloatingPointError("Nonfinite native pool team_score")
    return {key: values.reshape(population, episodes_per_member).mean(axis=1)
            for key, values in metrics.items()}

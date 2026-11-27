"""Score completed episodes with the simulator's unscaled team score."""

import json

import numpy as np
import torch

from .environment import Kitchen


def summarize(episodes, score_cap):
    scores = [episode["team_score"] for episode in episodes]
    count = len(scores)
    return {
        "episodes": count,
        "score_cap": score_cap,
        "mean_score": sum(scores) / count if count else None,
        "max_score": max(scores) if count else None,
        "full_score_rate": sum(score == score_cap for score in scores) / count if count else None,
        "mean_ticks": sum(episode["ticks"] for episode in episodes) / count if count else None,
        "mean_joint_orders": sum(episode["joint_completed"] for episode in episodes) / count if count else None,
        "spawn_assignments": len({tuple(map(tuple, episode["spawns"])) for episode in episodes}),
    }


def score_policy(scene, seed, count, device, model=None, scripted=False):
    games = [Kitchen(scene, seed + index) for index in range(count)]
    observations = [game.reset() for game in games]
    description = games[0].describe()
    controllers = []
    if scripted:
        from overcooked_ai_py.cooperative.shared.controller import ScriptedController

        controllers = [ScriptedController(game.env.core) for game in games]
    rng = np.random.default_rng(seed)
    episodes = []
    active = list(range(count))
    if model is not None:
        model.eval()
    while active:
        if model is not None:
            batch = torch.as_tensor(np.stack([observations[i] for i in active]), device=device)
            actions = model.act(batch, greedy=True)[0].cpu().numpy()
        elif scripted:
            actions = [controllers[i].actions(games[i].env.core) for i in active]
        else:
            actions = rng.integers(description["actions"], size=(len(active), description["agents"]))
        remaining = []
        for i, joint in zip(active, actions, strict=True):
            result = games[i].step(joint, restart=False)
            observations[i] = result.observation
            if result.done:
                episodes.append(result.episode)
            else:
                remaining.append(i)
        active = remaining
    episodes.sort(key=lambda episode: episode["seed"])
    summary = summarize(episodes, description["score_cap"])
    return {"summary": summary, "episodes": episodes}


def demonstrations(scene, seed):
    game = Kitchen(scene, seed)
    from overcooked_ai_py.cooperative.shared.controller import ScriptedController

    score_cap = game.describe()["score_cap"]
    spawns = set()
    observations = []
    actions = []
    returns = []
    for _ in range(6 * 100):
        observation = game.reset()
        if game.spawns in spawns:
            continue
        spawns.add(game.spawns)
        controller = ScriptedController(game.env.core)
        rewards = []
        while True:
            joint = controller.actions(game.env.core)
            observations.append(observation)
            actions.append(joint)
            result = game.step(joint, restart=False)
            rewards.append(result.reward)
            observation = result.observation
            if result.done:
                if result.episode["team_score"] != score_cap:
                    raise RuntimeError(f"Reference trajectory at seed {game.seed} did not reach the score cap.")
                remaining_score = np.cumsum(np.asarray(rewards[::-1], dtype=np.float32))[::-1]
                returns.extend(remaining_score / score_cap)
                break
        if len(spawns) == 6:
            return np.stack(observations), np.asarray(actions, dtype=np.int64), np.asarray(returns, dtype=np.float32)
    raise RuntimeError("Could not collect reference trajectories for all six player assignments.")


def write_evaluation(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

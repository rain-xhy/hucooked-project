"""Native SharedKitchen teacher demonstrations and supervised warm starts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def demonstration_steps(path):
    with np.load(path) as data:
        return int(data["env_steps"])


def generate(path, episodes=32, validation_episodes=8, seed=1,
             perturbation_probability=.1):
    from .envs import encode
    from overcooked_ai_py.cooperative.shared import SharedKitchen
    from overcooked_ai_py.cooperative.shared.controller import ScriptedController
    rows = {"spatial": [], "vector": [], "actions": [], "validation": []}
    summaries = []
    episode_lengths = []
    perturbation_probabilities = []
    if not 0. <= perturbation_probability <= 1.:
        raise ValueError("perturbation_probability must be in [0, 1]")
    train_steps = 0
    for i in range(episodes + validation_episodes):
        validation = i >= episodes
        split_index = i - episodes if validation else i
        split_size = validation_episodes if validation else episodes
        perturbed = split_index >= (split_size + 1) // 2
        probability = perturbation_probability if perturbed else 0.
        perturb_rng = np.random.default_rng(seed + 500000 + i)
        env = SharedKitchen()
        env.reset(seed=seed+i if not validation else 100000+seed+i)
        controller = ScriptedController(env)
        episode_steps = 0
        while not (env.terminated or env.truncated):
            obs = encode(env)
            actions = np.asarray(controller.actions(env), dtype=np.int64)
            rows["spatial"].append(obs["spatial"])
            rows["vector"].append(obs["vector"])
            rows["actions"].append(actions)
            rows["validation"].extend([validation]*3)
            behavior = actions.copy()
            if probability:
                deviated = perturb_rng.random(3) < probability
                # Labels remain the expert correction for the observed state.
                # Movement/wait deviations expose recovery and timing states
                # without injecting random ingredient interactions.
                behavior[deviated] = perturb_rng.integers(0, 5, int(deviated.sum()))
            _, _, _, _, info = env.step(behavior)
            train_steps += int(not validation)
            episode_steps += 1
        episode_lengths.append(episode_steps)
        perturbation_probabilities.append(probability)
        summaries.append({"split": "validation" if validation else "train",
                          "perturbed": perturbed, **info["metrics"]})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, spatial=np.concatenate(rows["spatial"]),
        vector=np.concatenate(rows["vector"]), actions=np.concatenate(rows["actions"]),
        validation=np.asarray(rows["validation"]), env_steps=np.asarray(train_steps),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        perturbation_probabilities=np.asarray(perturbation_probabilities),
        expert_correction_labels=np.asarray(True))
    perturbed_scores = [s["team_score"] for s in summaries if s["perturbed"]]
    print(json.dumps({"dataset": str(target), "demo_env_steps": train_steps,
                      "episodes": episodes, "validation_episodes": validation_episodes,
                      "perturbation_probability": perturbation_probability,
                      "mean_behavior_score": float(np.mean([s["team_score"] for s in summaries])),
                      "mean_behavior_meals": float(np.mean([s["meals_completed"] for s in summaries])),
                      "mean_perturbed_score": float(np.mean(perturbed_scores)) if perturbed_scores else None}), flush=True)


def warm_start(model, forward_logits, *, dataset_path, device, seed,
               epochs=8, batch_size=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with np.load(dataset_path) as data:
        spatial = data["spatial"]
        vector = data["vector"]
        actions = data["actions"]
        validation = data["validation"]
        steps = int(data["env_steps"])
    train_ids = np.flatnonzero(~validation)
    valid_ids = np.flatnonzero(validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for _ in range(epochs):
        for chunk in np.array_split(rng.permutation(train_ids), max(1, int(np.ceil(len(train_ids)/batch_size)))):
            logits = forward_logits(torch.as_tensor(spatial[chunk], device=device),
                                    torch.as_tensor(vector[chunk], device=device))
            labels = torch.as_tensor(actions[chunk], device=device)
            if logits.ndim == 3:
                labels = labels[:, None].expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(valid_ids), batch_size):
            ids = valid_ids[start:start+batch_size]
            logits = forward_logits(torch.as_tensor(spatial[ids], device=device),
                                    torch.as_tensor(vector[ids], device=device))
            labels = torch.as_tensor(actions[ids], device=device)
            if logits.ndim == 3:
                labels = labels[:, None].expand(-1, logits.shape[1])
            correct += int((logits.argmax(-1) == labels).sum().cpu())
            total += labels.numel()
    model.train()
    return {"demo_env_steps": steps, "epochs": epochs,
            "train_ce": float(np.mean(losses[-max(1,len(train_ids)//batch_size):])),
            "validation_accuracy": correct/max(1,total)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--validation-episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--perturbation-probability", type=float, default=.1)
    a = p.parse_args()
    generate(a.output, a.episodes, a.validation_episodes, a.seed,
             a.perturbation_probability)

"""Train HA² on the official two-map PartitionKitchen benchmark."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, mse_loss

from overcooked_ai_py.cooperative.partition import PartitionConfig, TorchPartitionKitchen
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES
from .schedule import Dispatcher, FACILITIES, LEGACY_GOALS, path_table


ENVIRONMENT_COMMIT = "2d0ef3716706206eaecdc61296beccbdd23c31e5"
LAYOUTS = ("sync", "counter")


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 96), nn.Tanh(), nn.Linear(96, 96),
                         nn.Tanh(), nn.Linear(96, outputs))


class Policy(nn.Module):
    def __init__(self, action_count, goal_count):
        super().__init__()
        self.manager = mlp(48, action_count)
        self.worker = mlp(46 + goal_count + 3, 6)
        self.critic = mlp(48, 1)


@torch.no_grad()
def rollout(args, distances, model, layout, lanes, seed, *, sample=False, collect=False, trace=False):
    env = TorchPartitionKitchen(PartitionConfig(layout=layout), device=args.device, num_envs=lanes)
    obs, _ = env.reset(seed=seed)
    dispatch = Dispatcher(env, distances, parallel=not args.serial,
                          deadline_aware=args.deadline_aware, prestage=args.prestage, gate_choice=args.gate_choice,
                          protect_supply=args.protect_supply)
    worker_data, manager_data, records, history = [], [], [], []
    recipes = torch.arange(3, device=env.device)
    totals = torch.zeros((lanes, 3, 2), dtype=torch.long, device=env.device)
    team_score = torch.zeros(lanes, dtype=torch.float64, device=env.device)
    decisions = torch.zeros((lanes, dispatch.action_count), dtype=torch.long, device=env.device)
    for _ in range(env.config.horizon):
        state, mask, slack = dispatch.manager_state(obs)
        active = dispatch.chosen < 0
        expert = dispatch.teacher(mask, slack)
        if model is None:
            action = expert
            value = log_prob = torch.zeros(lanes, device=env.device)
        else:
            dist = Categorical(logits=model.manager(state).masked_fill(~mask, -1e9))
            action = dist.sample() if sample else dist.logits.argmax(dim=1)
            value = model.critic(state).squeeze(1)
            log_prob = dist.log_prob(action)
        if collect and active.any().item():
            manager_data.append((state[active], mask[active], expert[active]))
        decisions += (action[:, None] == torch.arange(dispatch.action_count, device=env.device)) & active[:, None]
        dispatch.start(action)
        moves = dispatch.propose(obs, None if model is None else model.worker, worker_data if collect else None)
        task = dispatch.chosen.clone() if trace else None
        phase = dispatch.phase.clone() if trace else None
        gate = dispatch.variant.clone() if trace else None
        obs, reward, _, _, info = env.step(moves)
        dispatch.advance(moves, info)
        team_score += reward
        kinds = info["recipe"][:, :, None] == recipes
        totals[:, :, 0] += (kinds & info["delivered"][:, :, None]).sum(dim=1)
        totals[:, :, 1] += (kinds & info["expired"][:, :, None]).sum(dim=1)
        if sample:
            learning = active & (mask.sum(dim=1) > 1) if args.gate_choice else active
            records.append((state, mask, learning, action, log_prob, value, reward.float() / env.config.delivery_reward))
        if trace:
            history.append((moves, task, phase, gate, reward, info["recipe"], info["order_ids"],
                            info["delivered"], info["expired"], info["age"], info["sync_successes"], info["pickups"]))
    scores = team_score.cpu().tolist()
    counts = totals.cpu().tolist()
    assignments = decisions.cpu().tolist()
    info_keys = ("actions", "task", "phase", "gate", "reward", "recipe", "order_ids", "delivered",
                 "expired", "age", "sync_successes", "pickups")
    replay = None
    if trace:
        matrices = {key: torch.stack(values).transpose(0, 1).cpu().tolist()
                    for key, values in zip(info_keys, zip(*history))}
        replay = [{key: values[i] for key, values in matrices.items()} for i in range(lanes)]
    rows = [
        {"layout": layout, "seed": seed + i, "score": scores[i],
         "recipes": {name: {"delivered": a, "expired": b} for name, (a, b) in zip(RECIPE_NAMES, counts[i])},
         "manager_decisions": assignments[i],
         **({"trajectory": replay[i]} if trace else {})}
        for i in range(lanes)
    ]
    return rows, records, (manager_data, worker_data)


def report(rows):
    scores = np.asarray([r["score"] for r in rows])
    return {
        "episodes": len(rows), "mean": float(scores.mean()), "std": float(scores.std()),
        "median": float(np.median(scores)), "minimum": float(scores.min()), "maximum": float(scores.max()),
        "recipes": {name: {event: sum(r["recipes"][name][event] for r in rows)
                            for event in ("delivered", "expired")} for name in RECIPE_NAMES},
    }


def imitation(policy, examples, epochs):
    metrics = {}
    for name, batches in zip(("manager", "worker"), examples):
        network = getattr(policy, name)
        inputs, masks, labels = [torch.cat(values) for values in zip(*batches)]
        optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
        for _ in range(epochs):
            for ids in torch.randperm(len(inputs), device=inputs.device).split(2048):
                if name == "worker":
                    # Only tied shortest-path actions require a learned choice.
                    ids = ids[masks[ids].sum(dim=1) > 1]
                    if ids.numel() == 0:
                        continue
                loss = cross_entropy(network(inputs[ids]).masked_fill(~masks[ids], -1e9), labels[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            match = network(inputs).masked_fill(~masks, -1e9).argmax(dim=1) == labels
            nontrivial = masks.sum(dim=1) > 1
            metrics[name] = {"examples": len(inputs), "nontrivial": int(nontrivial.sum()),
                             "agreement": float(match.float().mean()),
                             "tie_agreement": float(match[nontrivial].float().mean()) if nontrivial.any() else None}
    return metrics


def advantages(records, discount=0.997, lam=0.95):
    states, masks, acting, actions, logp, value, rewards = [torch.stack(x) for x in zip(*records)]
    accumulated = torch.zeros_like(value[0])
    beta = torch.ones_like(accumulated)
    next_value = torch.zeros_like(accumulated)
    next_advantage = torch.zeros_like(accumulated)
    advantage = torch.zeros_like(value)
    for t in reversed(range(len(states))):
        accumulated = rewards[t] + discount * accumulated
        beta = beta * discount
        active = acting[t]
        delta = accumulated + beta * next_value - value[t]
        targets = delta + beta * lam * next_advantage
        advantage[t] = targets
        next_value = torch.where(active, value[t], next_value)
        next_advantage = torch.where(active, targets, next_advantage)
        accumulated = torch.where(active, 0.0, accumulated)
        beta = torch.where(active, 1.0, beta)
    return states[acting], masks[acting], actions[acting], logp[acting], advantage[acting], (value + advantage)[acting]


def train_update(policy, optimizer, buffers, *, actor=None):
    actor = policy.manager if actor is None else actor
    features, masks, actions, old_logp, adv, returns = [torch.cat(x) for x in zip(*buffers)]
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    losses = []
    for _ in range(5):
        for ids in torch.randperm(len(features), device=features.device).split(512):
            scores = actor(features[ids]).masked_fill(~masks[ids], -1e9)
            dist = Categorical(logits=scores)
            ratio = (dist.log_prob(actions[ids]) - old_logp[ids]).exp()
            loss_pi = -torch.minimum(ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]).mean()
            loss_v = mse_loss(policy.critic(features[ids]).squeeze(1), returns[ids])
            loss = loss_pi + 0.5 * loss_v - 0.012 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(torch.stack((loss_pi.detach(), loss_v.detach())))
    return torch.stack(losses).mean(dim=0).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "teacher", "evaluate"), default="train")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--deadline-aware", action="store_true")
    parser.add_argument("--prestage", action="store_true")
    parser.add_argument("--gate-choice", action="store_true")
    parser.add_argument("--protect-supply", action="store_true")
    parser.add_argument("--reset-manager", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lanes", type=int, default=144)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=18)
    parser.add_argument("--validation-every", type=int, default=3)
    parser.add_argument("--demonstrations", type=int, default=96)
    args = parser.parse_args()
    if min(args.lanes, args.validation_every, args.demonstrations, args.test_episodes) < 1:
        parser.error("Batch sizes and validation interval must be positive")
    if args.reset_manager and (not args.initialize or args.mode != "train"):
        parser.error("Resetting the Manager requires training with a Worker checkpoint")
    if args.mode == "evaluate" and not args.initialize:
        parser.error("Evaluation requires a checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = monotonic()

    def log(stage, **values):
        entry = dict(stage=stage, seconds=round(monotonic()-started, 2), **values)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    log("config", commit=ENVIRONMENT_COMMIT, **vars(args) | {
        "output": str(args.output), "initialize": str(args.initialize) if args.initialize else None,
    })
    paths = torch.as_tensor(path_table(), device=args.device)
    action_count = (12 if args.gate_choice else 6) + int(args.deadline_aware)
    policy = None if args.mode == "teacher" else Policy(action_count, len(FACILITIES) if args.gate_choice else LEGACY_GOALS).to(args.device)
    if args.initialize:
        weights = torch.load(args.initialize, map_location=args.device, weights_only=True)
        if args.reset_manager:
            policy.worker.load_state_dict({k.removeprefix("worker."): v for k, v in weights.items()
                                           if k.startswith("worker.")})
        else:
            policy.load_state_dict(weights)

    def score(model, stage, seed, episodes, *, trace=False):
        tables = {layout: rollout(args, paths, model, layout, episodes, seed, trace=trace)[0]
                  for layout in LAYOUTS}
        summaries = {layout: report(rows) for layout, rows in tables.items()}
        if trace:
            (args.output / "evaluation.json").write_text(json.dumps(tables), encoding="utf-8")
        log(stage, results=summaries)
        return float(np.mean([r["mean"] for r in summaries.values()]))

    if args.mode != "train":
        score(policy, "evaluation", args.test_seed, args.test_episodes, trace=True)
        return
    rng = np.random.default_rng(args.seed)
    if args.initialize is None:
        demonstrations = [[], []]
        for layout in LAYOUTS:
            data = rollout(args, paths, None, layout, args.demonstrations,
                           int(rng.integers(1_000_000, 100_000_000)), collect=True)[2]
            for stored, batches in zip(demonstrations, data):
                stored.extend(batches)
        log("imitation", **imitation(policy, demonstrations, 18))
        del demonstrations
    best = score(policy, "after_imitation" if not args.initialize else "initialized", 20031, 40)
    best_stage = 0
    torch.save(policy.state_dict(), args.output / "best.pt")
    optimizer = torch.optim.Adam([
        {"params": policy.manager.parameters(), "lr": 0.00035},
        {"params": policy.critic.parameters(), "lr": 0.001},
    ])
    for step in range(1, args.updates + 1):
        buffers, stats = [], {}
        for layout in LAYOUTS:
            seed = int(rng.integers(100_000_000, 200_000_000))
            rows, data, _ = rollout(args, paths, policy, layout, args.lanes, seed, sample=True)
            buffers.append(advantages(data))
            stats[layout] = report(rows)
        losses = train_update(policy, optimizer, buffers)
        log("update", update=step, environment_steps=step*2*args.lanes*300, manager_loss=losses[0],
            critic_loss=losses[1], training=stats)
        if step % args.validation_every == 0 or step == args.updates:
            value = score(policy, "validation", 20031, 40)
            if value > best:
                best, best_stage = value, step
                torch.save(policy.state_dict(), args.output / "best.pt")
    torch.save(policy.state_dict(), args.output / "last.pt")
    policy.load_state_dict(torch.load(args.output / "best.pt", map_location=args.device, weights_only=True))
    log("selected", validation_mean=best, update=best_stage)
    score(policy, "evaluation", args.test_seed, args.test_episodes, trace=True)


if __name__ == "__main__":
    main()

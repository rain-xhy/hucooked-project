"""V3 evaluation and PPO training for one chef with two fixed rule teammates."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from overcooked_ai_py.cooperative.partition import PartitionConfig, TorchPartitionKitchen
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES, OFFSETS, STAY

from .team import ENVIRONMENT_COMMIT, LAYOUTS, advantages, mlp, report, train_update
from .schedule import CELLS, COOK, FACILITIES, HELPER, LEFT, Dispatcher, path_table


SEATS = ("left", "right_north", "right_south")
FEATURE_COUNT = 46 + len(FACILITIES) + 3 + 6
HANDOFF_FEATURES = 46 + 3 + 3 + 2


class ChefPolicy(nn.Module):
    def __init__(self, prior_strength, *, handoff=False):
        super().__init__()
        self.actions = 2 if handoff else 6
        features = HANDOFF_FEATURES if handoff else FEATURE_COUNT
        self.worker = mlp(features, self.actions)
        self.critic = mlp(features, 1)
        self.prior_strength = prior_strength
        nn.init.zeros_(self.worker[-1].weight)
        nn.init.zeros_(self.worker[-1].bias)

    def logits(self, features):
        return self.worker(features) + self.prior_strength * features[:, -self.actions:]


class PretrainedWorker(nn.Module):
    def __init__(self, input_width):
        super().__init__()
        if input_width not in (FEATURE_COUNT - 6, FEATURE_COUNT - 7):
            raise ValueError("Incompatible V3 Worker input width")
        self.legacy_gates = input_width == FEATURE_COUNT - 7
        self.worker = mlp(input_width, 6)

    def logits(self, features):
        return self.worker(features[:, :-6])


@torch.no_grad()
def rollout(args, paths, policy, layout, lanes, seed, *, seat=None, sample=False, trace=False):
    env = TorchPartitionKitchen(PartitionConfig(layout=layout), device=args.device, num_envs=lanes)
    obs, _ = env.reset(seed=seed)
    legacy_gates = getattr(policy, "legacy_gates", False)
    dispatch = Dispatcher(env, paths, deadline_aware=True, prestage=True,
                          gate_choice=not legacy_gates, protect_supply=legacy_gates)
    device = env.device
    rows = torch.arange(lanes, device=device)
    seats = (rows + seed) % 3 if seat is None else torch.full_like(rows, seat)
    controlled = dispatch.canonical()[rows, seats]
    route_offsets = torch.tensor(OFFSETS, device=device)
    action_eye = torch.eye(6, device=device)
    recipe_codes = torch.arange(3, device=device)
    recipe_eye = torch.eye(3, device=device)
    gate_eye = torch.eye(2, device=device)
    handoff_control = args.control == "handoff"
    records, trajectory = [], []
    scores = torch.zeros(lanes, device=device)
    orders = torch.zeros((lanes, 3, 2), dtype=torch.long, device=device)
    disagreements = torch.zeros_like(rows)
    active_choices = torch.zeros_like(rows)

    for _ in range(env.config.horizon):
        _, feasible, slack = dispatch.manager_state(obs)
        teacher = dispatch.teacher(feasible, slack)
        decision = teacher
        if policy is not None and handoff_control:
            recipe = teacher // 4
            swap = teacher % 2 == 1
            initial = dispatch.canonical()
            team = torch.stack((initial[:, LEFT],
                    torch.where(swap, initial[:, HELPER], initial[:, COOK]),
                    torch.where(swap, initial[:, COOK], initial[:, HELPER])), dim=1)
            role = (team == controlled[:, None]).long().argmax(dim=1)
            proposal = (teacher // 2) % 2
            active = (dispatch.chosen < 0) & (
                ((recipe == 0) & (role != HELPER)) | ((recipe == 2) & (role != COOK))
            )
            features = torch.cat((obs[rows, controlled], recipe_eye[recipe.clamp_max(2)],
                                  recipe_eye[role], gate_eye[proposal]), dim=1)
            mask = torch.ones((lanes, 2), dtype=torch.bool, device=device)
            dist = Categorical(logits=policy.logits(features))
            action = dist.sample() if sample else dist.logits.argmax(dim=1)
            decision = torch.where(active, teacher + 2 * (action - proposal), teacher)
            disagreements += active & (action != proposal)
            active_choices += active
        dispatch.start(decision)
        contexts = [] if policy is not None and not handoff_control else None
        rule_actions = dispatch.propose(obs, examples=contexts)
        moves = rule_actions
        if policy is not None and not handoff_control:
            role_order = dispatch.assignment[:, (COOK, HELPER, LEFT)]
            slot = (role_order == controlled[:, None]).long().argmax(dim=1)
            features, strict, proposal = [torch.stack(column, dim=1)[rows, slot]
                                          for column in zip(*contexts)]
            position = env.state["pos"]
            neighbor = position[rows, controlled, None] + route_offsets
            floor_moves = env.floor[neighbor]
            moving = (strict[:, :4] & floor_moves).any(dim=1)
            occupied = (neighbor[:, :, None] == position[:, None, :]).any(dim=2)
            next_position = position + route_offsets[rule_actions.clamp_max(3)]
            other_move = (torch.arange(3, device=device)[None, :] != controlled[:, None])
            other_move &= rule_actions < STAY
            other_move &= env.floor[next_position]
            reserved = ((neighbor[:, :, None] == next_position[:, None, :])
                        & other_move[:, None, :]).any(dim=2)
            mask = strict.clone()
            mask[:, :4] &= ~(floor_moves & reserved)
            if args.navigation == "flexible":
                mask[:, :4] |= moving[:, None] & floor_moves & ~occupied & ~reserved
            # Partner actions are computed before the learned action and are never overwritten.
            features = torch.cat((features, action_eye[proposal]), dim=1)
            dist = Categorical(logits=policy.logits(features).masked_fill(~mask, -1e9))
            action = dist.sample() if sample else dist.logits.argmax(dim=1)
            moves = rule_actions.clone()
            moves[rows, controlled] = action
            active = mask.sum(dim=1) > 1
            disagreements += action != proposal
            active_choices += active
        obs, reward, _, _, info = env.step(moves)
        dispatch.advance(moves, info)
        scores += reward.float()
        kind = info["recipe"][:, :, None] == recipe_codes
        orders[:, :, 0] += (kind & info["delivered"][:, :, None]).sum(dim=1)
        orders[:, :, 1] += (kind & info["expired"][:, :, None]).sum(dim=1)
        if sample:
            records.append((features, mask, active, action, dist.log_prob(action),
                            policy.critic(features).squeeze(1), reward.float() / env.config.delivery_reward))
        if trace:
            trajectory.append((moves, rule_actions, dispatch.variant.clone(), teacher, decision, reward, info["recipe"], info["order_ids"],
                               info["delivered"], info["expired"], info["sync_successes"], info["pickups"]))

    replays = None
    if trace:
        keys = ("actions", "rule_actions", "gate", "rule_dispatch", "dispatch", "reward", "recipe", "order_ids", "delivered",
                "expired", "sync_successes", "pickups")
        columns = [torch.stack(col).transpose(0, 1).cpu().tolist() for col in zip(*trajectory)]
        replays = [{key: value[i] for key, value in zip(keys, columns)} for i in range(lanes)]
    outputs = []
    scalars = zip(scores.cpu().tolist(), orders.cpu().tolist(), controlled.cpu().tolist(),
                  seats.cpu().tolist(), disagreements.cpu().tolist(), active_choices.cpu().tolist())
    for i, (score, counts, chef, position, changes, choices) in enumerate(scalars):
        outputs.append({
            "layout": layout, "seed": seed + i, "score": score, "learned_chef": chef if policy is not None else None,
            "seat": SEATS[position], "deviations": changes, "legal_choices": choices,
            "recipes": {name: {"delivered": a, "expired": b} for name, (a, b) in zip(RECIPE_NAMES, counts)},
            **({"trajectory": replays[i]} if trace else {}),
        })
    return outputs, records


def details(rows):
    return report(rows) | {
        "deviations_per_episode": float(np.mean([r["deviations"] for r in rows])),
        "decision_steps_per_episode": float(np.mean([r["legal_choices"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "rules", "evaluate"), default="train")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--reuse-worker", type=Path)
    parser.add_argument("--control", choices=("movement", "handoff"), default="movement")
    parser.add_argument("--navigation", choices=("shortest", "flexible"), default="flexible")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=708)
    parser.add_argument("--prior-strength", type=float, default=3)
    parser.add_argument("--lanes", type=int, default=120)
    parser.add_argument("--updates", type=int, default=18)
    parser.add_argument("--validation-every", type=int, default=3)
    parser.add_argument("--test-seed", type=int, default=75031)
    parser.add_argument("--test-episodes", type=int, default=80)
    args = parser.parse_args()
    if min(args.lanes, args.validation_every, args.test_episodes) < 1:
        parser.error("Lane count, validation interval, and test episodes must be positive")
    if args.mode == "evaluate" and not (args.initialize or args.reuse_worker):
        parser.error("Evaluation requires --initialize or --reuse-worker")
    if args.reuse_worker and (args.initialize or args.mode != "evaluate" or args.navigation != "shortest" or args.control != "movement"):
        parser.error("--reuse-worker requires evaluation with shortest-path actions")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    start = monotonic()

    def log(name, **values):
        row = {"stage": name, "seconds": round(monotonic() - start, 2), **values}
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    log("config", commit=ENVIRONMENT_COMMIT, **(vars(args) | {
        "output": str(args.output), "initialize": str(args.initialize) if args.initialize else None,
        "reuse_worker": str(args.reuse_worker) if args.reuse_worker else None,
    }))
    paths = torch.as_tensor(path_table(), device=args.device)
    policy = None if args.mode == "rules" else ChefPolicy(args.prior_strength, handoff=args.control == "handoff").to(args.device)
    if args.reuse_worker:
        weight = torch.load(args.reuse_worker, map_location=args.device, weights_only=True)
        policy = PretrainedWorker(weight["worker.0.weight"].shape[1]).to(args.device)
        policy.worker.load_state_dict({k.removeprefix("worker."): v for k, v in weight.items()
                                       if k.startswith("worker.")})
    if args.initialize:
        state = torch.load(args.initialize, map_location=args.device, weights_only=True)
        policy.load_state_dict(state)

    def evaluate(stage, seed, count, trace=False):
        if trace and policy is not None:
            evaluated = {
                layout: {name: rollout(args, paths, policy, layout, count, seed, seat=i, trace=True)[0]
                         for i, name in enumerate(SEATS)} for layout in LAYOUTS
            }
        else:
            evaluated = {layout: {"rotating": rollout(args, paths, policy, layout, count, seed, trace=trace)[0]}
                         for layout in LAYOUTS}
        values = {}
        for layout, group in evaluated.items():
            values[layout] = {"overall": details([r for episodes in group.values() for r in episodes]),
                              "per_seat": {key: details(episodes) for key, episodes in group.items()}}
        if trace:
            (args.output / "evaluation.json").write_text(json.dumps(evaluated), encoding="utf-8")
        log(stage, results=values)
        return float(np.mean([result["overall"]["mean"] for result in values.values()]))

    if args.mode != "train":
        evaluate("evaluation", args.test_seed, args.test_episodes, trace=True)
        return
    optimizer = torch.optim.Adam([
        {"params": policy.worker.parameters(), "lr": 0.00025},
        {"params": policy.critic.parameters(), "lr": 0.001},
    ])
    best = evaluate("initial", 32117, 48)
    chosen = 0
    torch.save(policy.state_dict(), args.output / "best.pt")
    for update in range(1, args.updates + 1):
        buffers, totals = [], {}
        for layout in LAYOUTS:
            seed = int(rng.integers(1_000_000, 50_000_000))
            rows, records = rollout(args, paths, policy, layout, args.lanes, seed, sample=True)
            buffers.append(advantages(records))
            totals[layout] = details(rows)
        losses = train_update(policy, optimizer, buffers, actor=policy.logits)
        log("update", update=update, env_steps=2*update*args.lanes*300,
            action_loss=losses[0], critic_loss=losses[1], training=totals)
        if update % args.validation_every == 0 or update == args.updates:
            value = evaluate("validation", 32117, 48)
            if value > best:
                best, chosen = value, update
                torch.save(policy.state_dict(), args.output / "best.pt")
    torch.save(policy.state_dict(), args.output / "last.pt")
    log("selected", update=chosen, validation_mean=best)
    policy.load_state_dict(torch.load(args.output / "best.pt", map_location=args.device, weights_only=True))
    evaluate("evaluation", args.test_seed, args.test_episodes, trace=True)


if __name__ == "__main__":
    main()

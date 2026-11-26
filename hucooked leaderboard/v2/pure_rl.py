"""Train the SharedKitchen hierarchy with PPO and environment rewards only."""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

from overcooked_ai_py.cooperative.shared import SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.kitchen import DIRECTIONS, INTERACT, STAY
from overcooked_ai_py.cooperative.shared.layouts import FLOOR, STATIONS

from haha.artifacts import portable_path
from haha.goal_training import mlp, tensor
from .policy import (
    CUT, DROP_SIDE, FEATURE_DIM, GET_SIDE, GOALS, WAIT_SIDE, SharedPolicy, ordered_chefs, reserve,
    reserved_features, valid_actions, worker_inputs,
)
from .train import CONTINUOUS_ENVIRONMENT_COMMIT, Execution, Trial, encode, evaluate, rank, summary


def distances():
    """Distance to each target through the kitchen's walkable tiles."""
    result = []
    for goal, _ in GOALS:
        distance = {goal: 0}
        pending = deque([goal])
        while pending:
            x, y = pending.popleft()
            for dx, dy in DIRECTIONS:
                tile = x + dx, y + dy
                if tile in FLOOR and tile not in distance:
                    distance[tile] = distance[x, y] + 1
                    pending.append(tile)
        result.append(distance)
    return result


class SharedBatch:
    def __init__(self, seeds, execution, rng, *, curriculum=()):
        self.execution, self.rng = execution, rng
        self.curriculum = curriculum
        self.trials = [self.trial(seed) for seed in seeds]
        self.skill = np.zeros((len(seeds), 3), dtype=np.int64)
        self.skill_left = np.zeros_like(self.skill)
        self.distances = distances()

    def trial(self, seed):
        if self.curriculum and self.rng.random() < .5:
            return deepcopy(self.curriculum[int(self.rng.integers(len(self.curriculum)))])
        return Trial(seed, "multi", self.execution)

    def skill_targets(self, masks):
        # Sample a legal subtask to practice. The environment provides the
        # transitions and rewards for every primitive action.
        expired = (self.skill_left <= 0) | ~np.take_along_axis(masks, self.skill[..., None], axis=-1)[..., 0]
        labels = np.argmax(np.where(masks, self.rng.random(masks.shape), -1.0), axis=-1)
        self.skill[expired] = labels[expired]
        self.skill_left[expired] = 16
        return self.skill.copy()

    def potential(self, goals):
        values = []
        for trial, selected in zip(self.trials, goals):
            costs = []
            for chef_id, goal_id in zip(ordered_chefs(trial.controller), selected):
                chef = trial.env.state.chefs[chef_id]
                goal, station = GOALS[goal_id]
                cost = self.distances[goal_id][chef.position]
                if station is not None and chef.position == goal:
                    x, y = STATIONS[station]
                    cost += 0.5 * (chef.orientation != (x - goal[0], y - goal[1]))
                costs.append(cost)
            values.append(costs)
        return np.asarray(values, dtype=np.float32)

    @torch.no_grad()
    def actions(self, policy, obs, goals, *, stochastic=False, temperature=1.0, sample_role=None):
        joint = np.full(goals.shape, STAY, dtype=np.int64)
        reservations = [set() for _ in self.trials]
        records = []
        for role in range(3):
            chefs = [ordered_chefs(trial.controller)[role] for trial in self.trials]
            ids = goals[:, role]
            reserved = np.stack([reserved_features(s) for s in reservations])
            x = worker_inputs(obs[:, role], ids, reserved)
            mask = np.asarray([valid_actions(trial.env, chef, goal)
                               for trial, chef, goal in zip(self.trials, chefs, ids)])
            dist = Categorical(logits=policy.logits(policy.worker, tensor(x), tensor(mask)) / temperature)
            items = dist.sample() if stochastic or role == sample_role else dist.logits.argmax(-1)
            for index, (trial, chef, action) in enumerate(zip(self.trials, chefs, items.tolist())):
                joint[index, chef] = action
                reserve(trial.env, chef, action, reservations[index])
            records.append((x, mask, items.numpy(), dist.log_prob(items).numpy()))
        return joint, [np.stack(values, axis=1) for values in zip(*records)]

    def advance(self, actions, *, reset=True):
        rewards, ended, infos, scores = [], [], [], []
        for i, (trial, joint) in enumerate(zip(self.trials, actions)):
            reward, done, info = trial.step(joint)
            rewards.append(reward)
            ended.append(done)
            infos.append(info)
            if done:
                scores.append(trial.reward)
                if reset:
                    self.trials[i] = self.trial(int(self.rng.integers(2**31)))
        return np.asarray(rewards, dtype=np.float32), np.asarray(ended), infos, scores


class PPO:
    def __init__(self, actor, critic, rate, device, *, worker=None, temperature=1.0):
        self.actor, self.critic, self.device = actor, critic, device
        self.worker, self.temperature = worker, temperature
        groups = [{"params": list(actor.parameters()), "lr": rate},
                  {"params": list(critic.parameters()), "lr": .001}]
        if worker is not None:
            groups.append({"params": list(worker.parameters()), "lr": rate})
        self.parameters = [p for group in groups for p in group["params"]]
        self.optimizer = torch.optim.Adam(groups)

    def update(self, records, final):
        obs, masks, actions, old_logp, critic_x, rewards, done, *worker_data = [
            tensor(np.stack(column), self.device) for column in zip(*records)
        ]
        alive = (~done).float()
        with torch.no_grad():
            values = self.critic(critic_x).squeeze(-1)
            next_value = self.critic(tensor(final, self.device)).squeeze(-1)
            gae = torch.zeros_like(next_value)
            advantages = torch.empty_like(values)
            for t in reversed(range(len(records))):
                delta = rewards[t] + .995 * next_value * alive[t] - values[t]
                gae = delta + .995 * .97 * gae * alive[t]
                advantages[t] = gae
                next_value = values[t]
            returns = (advantages + values).flatten()
            advantages = advantages.flatten()
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        obs = obs.flatten(0, -2)
        masks = masks.flatten(0, -2)
        critic_x = critic_x.flatten(0, -2)
        actions, old_logp = actions.flatten(), old_logp.flatten()
        if self.worker is not None:
            worker_x, worker_mask, worker_action = worker_data
            worker_x = worker_x.flatten(0, -2)
            worker_mask = worker_mask.flatten(0, -2)
            worker_action = worker_action.flatten()
        for _ in range(4):
            for ids in torch.randperm(len(obs), device=self.device).split(1024):
                logits = self.actor(obs[ids]).masked_fill(~masks[ids], -1e9)
                dist = Categorical(logits=logits / self.temperature)
                logp = dist.log_prob(actions[ids])
                entropy = dist.entropy()
                if self.worker is not None:
                    logits = self.worker(worker_x[ids]).masked_fill(~worker_mask[ids], -1e9)
                    motion = Categorical(logits=logits / self.temperature)
                    logp = logp + motion.log_prob(worker_action[ids])
                    entropy = entropy + motion.entropy()
                ratio = (logp - old_logp[ids]).exp()
                adv = advantages[ids]
                actor_loss = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
                value_loss = mse_loss(self.critic(critic_x[ids]).squeeze(-1), returns[ids])
                loss = actor_loss + .5 * value_loss - .01 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, .5)
                self.optimizer.step()


def collect_worker(batch, policy, steps, *, stochastic=True):
    records, hits = [], 0
    _, obs, masks, _ = encode(batch.trials)
    for _ in range(steps):
        goals = batch.skill_targets(masks)
        before = batch.potential(goals)
        joint, actor = batch.actions(policy, obs, goals, stochastic=stochastic)
        chosen = actor[2]
        waits = np.asarray([station is None for _, station in GOALS])[goals]
        hits_here = (chosen == INTERACT) | (waits & (before == 0) & (chosen == STAY))
        _, ended, _, _ = batch.advance(joint)
        after = batch.potential(goals)
        after[ended] = before[ended]
        rewards = hits_here.astype(np.float32) + .15 * (before - after) - .01
        batch.skill_left -= 1
        _, obs, masks, _ = encode(batch.trials)
        invalidated = ~np.take_along_axis(masks, goals[..., None], axis=-1)[..., 0]
        done = hits_here | (batch.skill_left <= 0) | invalidated | ended[:, None]
        batch.skill_left[done] = 0
        records.append((*actor, actor[0], rewards, done))
        hits += int(hits_here.sum())
    _, tail = batch.actions(policy, obs, batch.skill_targets(masks))
    return records, tail[0], hits


def order_progress(batch):
    values = []
    for trial in batch.trials:
        state = trial.env.state
        progress = sum(
            .30 if order.assembled_at >= 0 else .20 if order.unloaded_at >= 0 else
            .10 if order.cooked_at >= 0 else .05
            for order in state.orders if order.status == "started"
        )
        progress += .04 * (state.assembly.plate is not None)
        progress += .04 * (state.assembly.side is not None)
        progress += .02 * (state.cutting.item is not None)
        values.append(progress)
    return np.asarray(values, dtype=np.float32)


def production_delay(batch):
    values = []
    for trial in batch.trials:
        state = trial.env.state
        cooker, assembly = state.cooker, state.assembly
        ready_pot = cooker.item is not None and cooker.remaining == 0
        incomplete_assembly = assembly.main is not None and (assembly.side is None or assembly.plate is None)
        values.append(.012 * ready_pot + .012 * incomplete_assembly)
    return np.asarray(values, dtype=np.float32)


def prep_progress(batch):
    """Credit the real state of side ingredients and meals awaiting delivery."""
    values = []
    for trial in batch.trials:
        state = trial.env.state
        server, _, side = (state.chefs[i] for i in ordered_chefs(trial.controller))
        table, cutting = state.assembly, state.cutting
        pending = sum(order.status in ("active", "started") and order.assembled_at < 0
                      for order in state.orders)
        stages = [1.] if table.side is not None else []
        if side.held is not None:
            stages.append({"raw_side": .20, "chopped_side": .85}.get(side.held.kind, 0.))
        if cutting.item is not None:
            stages.append(.35 + .35 * cutting.progress / trial.env.config.chop_steps)
        meal_in_flight = ((table.plate is not None and table.plate.kind == "meal") or
                          (server.held is not None and server.held.kind == "meal"))
        values.append(.12 * sum(sorted(stages, reverse=True)[:pending]) + .12 * meal_in_flight)
    return np.asarray(values, dtype=np.float32)


def isolate_side_worker(worker):
    """Train existing side-specific input columns, leaving the other roles' logits intact."""
    worker.requires_grad_(False)
    weight = worker[0].weight.requires_grad_(True)
    mask = torch.zeros_like(weight)
    mask[:, FEATURE_DIM + 2] = 1
    for goal in (GET_SIDE, CUT, DROP_SIDE, WAIT_SIDE):
        mask[:, FEATURE_DIM + 3 + goal] = 1
    weight.register_hook(lambda grad: grad * mask)


@torch.no_grad()
def collect_side_worker(batch, policy, steps, *, temperature):
    records, scores = [], []
    for _ in range(steps):
        _, obs, masks, _ = encode(batch.trials)
        goals = policy.logits(policy.manager, tensor(obs), tensor(masks)).argmax(-1).numpy()
        joint, worker = batch.actions(policy, obs, goals, sample_role=2, temperature=temperature)
        before = batch.potential(goals)[:, 2]
        sides = [trial.controller.side for trial in batch.trials]
        native, ended, infos, completed = batch.advance(joint)
        after = batch.potential(goals)[:, 2]
        after[ended] = before[ended]
        progress = np.asarray([
            sum(
                (1 if event["type"] in ("take_source", "place_cutting", "chop", "take_cutting")
                 or (event["type"] == "place_assembly" and event.get("slot") == "side")
                 else -1 if event["type"] == "return_source" else 0)
                for event in info["events"] if side in event["actors"]
            )
            for side, info in zip(sides, infos)
        ], dtype=np.float32)
        rewards = native / 20 + .06 * progress + .04 * (before - after)
        inputs, mask, actions, logp = (field[:, 2] for field in worker)
        records.append((inputs, mask, actions, logp, inputs, rewards, ended))
        scores.extend(completed)
    _, obs, masks, _ = encode(batch.trials)
    goals = policy.logits(policy.manager, tensor(obs), tensor(masks)).argmax(-1).numpy()
    _, worker = batch.actions(policy, obs, goals)
    return records, worker[0][:, 2], scores


@torch.no_grad()
def handoff_states(policy, execution):
    """Keep native states from the starting policy's own training-seed rollouts."""
    rng = np.random.default_rng(30001)
    batch = SharedBatch(range(30001, 30009), execution, rng)
    snapshots = []
    for _ in range(240):
        _, obs, masks, _ = encode(batch.trials)
        goals = policy.logits(policy.manager, tensor(obs), tensor(masks)).argmax(-1).numpy()
        actions, _ = batch.actions(policy, obs, goals)
        for trial in batch.trials:
            state = trial.env.state
            main = state.assembly.main
            if main is None or state.tick < 30 or state.assembly.side is not None:
                continue
            order = trial.env.order(main.order_id)
            if order.kind == "joint" and state.tick - order.unloaded_at in (2, 4):
                trial.env.validate_state()
                snapshots.append(deepcopy(trial))
        batch.advance(actions, reset=False)
    if not snapshots:
        raise RuntimeError("The starting policy produced no joint-order handoff states")
    return snapshots, 240 * len(batch.trials)


def delivery_ticks(rows):
    return float(np.mean([
        sum(order["completed_at"] for order in row["orders"] if order["completed_at"] >= 0)
        for row in rows
    ]))


def last_deliveries(rows):
    deliveries = [sorted(order["completed_at"] for order in row["orders"] if order["completed_at"] >= 0)
                  for row in rows]
    result = {}
    for position in (19, 20, 21):
        times = [ticks[position - 1] for ticks in deliveries if len(ticks) >= position]
        result[str(position)] = {"completed_episodes": len(times), "mean_tick": float(np.mean(times)) if times else None}
    return result


def handoff_metrics(rows):
    intervals = {"normal": [], "joint": []}
    prep_lags = []
    for row in rows:
        take_ticks = {unit: event["tick"] for step in row["trajectory"] for event in step["events"]
                      if event["type"] == "take_source" for unit in event["units"]}
        for previous, order in zip(row["orders"], row["orders"][1:]):
            prior_tick, tick = previous["completed_at"], order["completed_at"]
            if min(prior_tick, tick) < 0:
                continue
            intervals[order["kind"]].append(tick - prior_tick)
            if order["kind"] == "normal":
                # Native meal units are main, side, then plate (each uses one source unit).
                prep_lags.append(take_ticks[order["delivered_units"][1]] - prior_tick)
    return {
        "mean_normal_interval": float(np.mean(intervals["normal"])) if intervals["normal"] else None,
        "mean_joint_interval": float(np.mean(intervals["joint"])) if intervals["joint"] else None,
        "mean_side_pickup_lag": float(np.mean(prep_lags)) if prep_lags else None,
        "side_pickup_before_prior_delivery_rate": float(np.mean(np.asarray(prep_lags) < 0)) if prep_lags else None,
    }


@torch.no_grad()
def widen_manager(manager, width):
    old_width = manager[0].out_features
    # Preserve the first and second hidden activations of all existing units.
    # The extra output columns start at zero and can learn from PPO gradients.
    with torch.random.fork_rng(devices=[]):
        expanded = mlp(manager[0].in_features, manager[-1].out_features, width=width)
    expanded = expanded.to(manager[0].weight.device)
    expanded[0].weight[:old_width].copy_(manager[0].weight)
    expanded[0].bias[:old_width].copy_(manager[0].bias)
    expanded[2].weight[:old_width, :old_width].copy_(manager[2].weight)
    expanded[2].weight[:old_width, old_width:].zero_()
    expanded[2].bias[:old_width].copy_(manager[2].bias)
    expanded[-1].weight[:, :old_width].copy_(manager[-1].weight)
    expanded[-1].weight[:, old_width:].zero_()
    expanded[-1].bias.copy_(manager[-1].bias)
    return expanded


@torch.no_grad()
def policy_directions(host, seed, execution):
    batch = SharedBatch([seed], execution, np.random.default_rng(seed))
    trials = batch.trials
    candidates = {}
    for _ in range(execution.horizon):
        _, obs, masks, _ = encode(trials)
        manager_logits = host.logits(host.manager, tensor(obs), tensor(masks))
        goals = manager_logits.argmax(-1).numpy()
        joint, motion = batch.actions(host, obs, goals)
        state = trials[0].env.state
        ready = state.cooker.item is not None and state.cooker.remaining == 0
        missing_plate = state.assembly.main is not None and state.assembly.plate is None
        missing_side = state.assembly.main is not None and state.assembly.side is None
        for role, chef_id in enumerate(ordered_chefs(trials[0].controller)):
            focused = (ready and role < 2) or (missing_plate and role == 0) or (missing_side and role == 2)
            if not focused:
                continue
            chef = state.chefs[chef_id]
            choices = (
                ("manager", obs[0, role], masks[0, role], goals[0, role]),
                ("worker", motion[0][0, role], motion[1][0, role], motion[2][0, role]),
            )
            for name, inputs, mask, selected in choices:
                net = getattr(host, name)
                features = net[:-1](tensor(inputs))
                logits = net[-1](features)
                alternatives = sorted((i for i in np.flatnonzero(mask) if i != selected),
                                      key=lambda i: float(logits[i]), reverse=True)
                for alternative in alternatives[:2]:
                    gap = float(logits[selected] - logits[alternative])
                    key = (name, role, int(goals[0, role]), chef.position, chef.orientation,
                           ready, missing_plate, missing_side, int(selected), int(alternative))
                    previous = candidates.get(key)
                    if previous is None or gap < previous[-1]:
                        candidates[key] = (name, features, int(alternative), gap)
        batch.advance(joint, reset=False)
    return sorted(candidates.values(), key=lambda entry: entry[-1])


def search_policy(host, seeds, execution, trials, start_steps, best, log, *, focused=False):
    """Improve output-layer parameters using only returns from native rollouts."""
    screen = seeds[:3]

    def score(chosen_seeds):
        rows = evaluate(host, "multi", chosen_seeds, execution)
        stats = summary(rows)
        return (*rank(stats), -delivery_ticks(rows))

    incumbent = deepcopy(host.state_dict())
    best_screen = score(screen)
    steps = (len(seeds) + len(screen)) * execution.horizon
    selected = 0
    best_steps = start_steps
    directions = []
    cursor = 0
    for candidate in range(1, trials + 1):
        host.load_state_dict(incumbent)
        if focused:
            if not directions:
                directions = policy_directions(host, screen[0][0], execution)
                steps += execution.horizon
                cursor = 0
            if not directions or cursor >= len(directions):
                break
            name, features, alternative, gap = directions[cursor]
            cursor += 1
            layer = getattr(host, name)[-1]
            scale = (gap + .08) / (1.0 + float(features.square().sum()))
            with torch.no_grad():
                layer.weight[alternative].add_(features * scale)
                layer.bias[alternative].add_(scale)
        else:
            scale = (.025, .065, .14, .27)[(candidate - 1) % 4]
            layers = (host.manager[-1], host.worker[-1])
            chosen_layers = layers if candidate % 3 == 0 else (layers[candidate % 3 - 1],)
            with torch.no_grad():
                for layer in chosen_layers:
                    layer.bias.add_(torch.randn_like(layer.bias) * scale)
                    layer.weight.add_(torch.randn_like(layer.weight) * (scale / 5))
        candidate_screen = score(screen)
        steps += len(screen) * execution.horizon
        accepted = False
        if candidate_screen > best_screen:
            candidate_score = score(seeds)
            steps += len(seeds) * execution.horizon
            if candidate_score > best:
                best, best_screen = candidate_score, candidate_screen
                selected, best_steps = candidate, start_steps + steps
                incumbent = deepcopy(host.state_dict())
                accepted = True
                directions = []
        log("search_candidate", update=candidate, environment_steps=start_steps + steps,
            search_steps=steps, screening_score=candidate_screen[0],
            validation_score=best[0], accepted=accepted)
        if best[0] >= 420:
            break
    host.load_state_dict(incumbent)
    return selected, best_steps, steps


@torch.no_grad()
def collect_manager(batch, policy, steps, *, joint_worker=False, temperature=1.0, throughput=False,
                    prep_shaping=False):
    records, scores = [], []
    shaped_reward = joint_worker or throughput or prep_shaping
    for _ in range(steps):
        common, obs, masks, _ = encode(batch.trials)
        logits = policy.logits(policy.manager, tensor(obs), tensor(masks))
        dist = Categorical(logits=logits / temperature)
        choices = dist.sample()
        joint, worker = batch.actions(policy, obs, choices.numpy(), stochastic=joint_worker, temperature=temperature)
        before = order_progress(batch) if shaped_reward else None
        prep_before = prep_progress(batch) if prep_shaping else None
        delay = production_delay(batch) if throughput else None
        rewards, ended, _, completed = batch.advance(joint)
        critic_x = np.broadcast_to(common[:, None, :], (*choices.shape, FEATURE_DIM))
        team = rewards / 20
        if shaped_reward:
            team += .995 * order_progress(batch) * ~ended - before
        if prep_shaping:
            team += .995 * prep_progress(batch) * ~ended - prep_before
        if throughput:
            team -= delay
        team = np.broadcast_to(team[:, None], choices.shape)
        done = np.broadcast_to(ended[:, None], choices.shape)
        old_logp = dist.log_prob(choices).numpy()
        if joint_worker:
            old_logp = old_logp + worker[3]
        records.append((obs, masks, choices.numpy(), old_logp, critic_x, team, done,
                        *(worker[:3] if joint_worker else ())))
        scores.extend(completed)
    common = encode(batch.trials)[0]
    return records, np.broadcast_to(common[:, None, :], (*masks.shape[:2], FEATURE_DIM)).copy(), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=488)
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--worker-updates", type=int, default=36)
    parser.add_argument("--updates", type=int, default=36)
    parser.add_argument("--resume", type=Path, help="Pure RL training directory for joint fine-tuning")
    parser.add_argument("--resume-last", action="store_true", help="Continue the final checkpoint of the referenced run")
    parser.add_argument("--throughput", action="store_true", help="Penalize waiting at ready pots and incomplete assembly")
    parser.add_argument("--manager-only", action="store_true", help="Freeze the Worker during resumed PPO")
    parser.add_argument("--manager-width", type=int, help="Widen the Manager while preserving its initial logits")
    parser.add_argument("--side-worker", action="store_true", help="Fine-tune only the side role's Worker")
    parser.add_argument("--prep-shaping", action="store_true", help="Train the Manager with ingredient progress shaping")
    parser.add_argument("--handoff-curriculum", action="store_true", help="Start some rollouts at native handoff states")
    parser.add_argument("--fixed-budget", action="store_true", help="Run the full PPO budget even after reaching 420")
    parser.add_argument("--search-trials", type=int, default=0, help="Search actor parameters using rollout returns")
    parser.add_argument("--focused-search", action="store_true", help="Focus parameter perturbations on production delays")
    parser.add_argument("--test-seed-base", type=int, default=84031)
    parser.add_argument("--test-episodes", type=int, default=24)
    args = parser.parse_args()
    handoff_experiment = args.side_worker or args.prep_shaping or args.handoff_curriculum
    if min(args.lanes, args.steps, args.worker_updates, args.updates, args.test_episodes) < 1:
        parser.error("Lane count, step count, updates, and episode count must be positive")
    if (args.resume_last or args.throughput or args.manager_only or args.search_trials or
            args.manager_width is not None or args.fixed_budget or handoff_experiment) and args.resume is None:
        parser.error("Fine-tuning options require --resume")
    if args.side_worker and (args.manager_only or args.throughput or args.search_trials or args.prep_shaping):
        parser.error("--side-worker runs PPO on the side role alone")
    if args.prep_shaping and (not args.manager_only or args.search_trials):
        parser.error("--prep-shaping requires Manager-only PPO")
    if args.handoff_curriculum and not (args.side_worker or args.prep_shaping):
        parser.error("--handoff-curriculum requires --side-worker or --prep-shaping")
    if args.manager_width is not None and (not args.manager_only or args.manager_width < 1):
        parser.error("--manager-width requires --manager-only and a positive width")
    if args.search_trials < 0:
        parser.error("--search-trials must be nonnegative")
    if args.focused_search and not args.search_trials:
        parser.error("--focused-search requires --search-trials")
    if not getattr(SharedKitchenConfig(), "continuous_orders", False):
        parser.error("The latest continuous-order SharedKitchen is required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir()
    started = monotonic()
    execution = Execution(profile="continuous")
    policy = SharedPolicy().to(args.device)
    host = SharedPolicy()

    def log(stage, **data):
        record = dict(stage=stage, seconds=round(monotonic() - started, 2), **data)
        line = json.dumps(record)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as handle:
            handle.write(line + "\n")

    def new_batch(curriculum=()):
        return SharedBatch(rng.integers(2**31, size=args.lanes).tolist(), execution, rng, curriculum=curriculum)

    def checkpoint_copy():
        host.load_state_dict(policy.state_dict())

    def evaluate_team(stage, update, steps, *, final=False, reference=False):
        seeds = ([(args.test_seed_base + i, None) for i in range(args.test_episodes)] if final
                 else [(2001 + i, None) for i in range(12)])
        rows = evaluate(None if reference else host, "reference" if reference else "multi",
                        seeds, execution, trace=(final or handoff_experiment) and not reference)
        stats = summary(rows)
        rank_by_pace = args.throughput or args.search_trials or handoff_experiment
        if rank_by_pace:
            stats["mean_delivery_tick_sum"] = delivery_ticks(rows)
        if args.manager_width is not None or handoff_experiment:
            stats["late_deliveries"] = last_deliveries(rows)
        if handoff_experiment and not reference:
            stats.update(handoff_metrics(rows))
        if final:
            name = "official-reference" if reference else "best-evaluation"
            (args.output / f"{name}.json").write_text(json.dumps(rows))
        log(stage, update=update, environment_steps=steps, **stats)
        if rank_by_pace:
            return (*rank(stats), -stats["mean_delivery_tick_sum"])
        return rank(stats)

    log("config", **(vars(args) | {"output": portable_path(args.output), "environment": CONTINUOUS_ENVIRONMENT_COMMIT,
                                   "resume": portable_path(args.resume),
                                   "initialize": None, "imitation": False, "demonstrations": 0,
                                   "execution": asdict(execution),
                                   "worker_reward": (
                                       "native team delivery plus actual ingredient events and goal-distance progress"
                                       if args.side_worker else
                                       "local interaction or wait at goal + shortest-distance progress - step cost"
                                   ),
                                   "manager_reward": "native team delivery reward divided by 20",
                                   "search_objective": "native team score; earlier deliveries break ties" if args.search_trials else None,
                                   "joint_reward": (
                                       None if args.search_trials or args.side_worker else
                                       "team delivery plus order progress and side preparation potentials"
                                       if args.prep_shaping else
                                       "team delivery reward with order-progress potential and production-delay costs"
                                       if args.throughput else
                                       "team delivery reward with order-progress potential shaping" if args.resume else None
                                   )}))
    if args.resume:
        previous = [json.loads(line) for line in (args.resume / "progress.jsonl").read_text().splitlines()]
        if previous[0].get("imitation") is not False or previous[0].get("environment") != CONTINUOUS_ENVIRONMENT_COMMIT:
            parser.error("Joint fine-tuning requires a pure RL checkpoint from this Shared revision")
        source_stage = "ppo_update" if args.resume_last else "best-evaluation"
        base_steps = next(row["environment_steps"] for row in reversed(previous) if row["stage"] == source_stage)
        filename = "last.pt" if args.resume_last else "best.pt"
        weights = torch.load(args.resume / filename, map_location="cpu", weights_only=True)
        source_width = weights["manager.0.weight"].shape[0]
        width = args.manager_width if args.manager_width is not None else source_width
        if width < source_width:
            parser.error("Manager width must be at least the checkpoint width")
        if source_width != policy.manager[0].out_features:
            policy.manager = mlp(policy.manager[0].in_features, policy.manager[-1].out_features,
                                 width=source_width).to(args.device)
        policy.load_state_dict(weights)
        max_difference = 0.0
        if width > source_width:
            old_manager = policy.manager
            policy.manager = widen_manager(old_manager, width)
            initial_trials = [Trial(2001 + i, "multi", execution) for i in range(12)]
            initial_obs = tensor(encode(initial_trials)[1], args.device)
            with torch.no_grad():
                original = old_manager(initial_obs)
                widened = policy.manager(initial_obs)
                torch.testing.assert_close(widened, original, rtol=1e-5, atol=1e-5)
                max_difference = float((widened - original).abs().max())
        if host.manager[0].out_features != width:
            host = deepcopy(policy).cpu()
        log("model_parameters", manager_width=width, source_manager_width=source_width,
            parameters={name: sum(p.numel() for p in getattr(policy, name).parameters())
                        for name in ("manager", "worker", "critic")}, initial_max_logit_difference=max_difference)
        checkpoint_copy()
        best = evaluate_team("initialized", 0, base_steps)
        if args.search_trials:
            validation = [(2001 + i, None) for i in range(12)]
            chosen, selected_steps, used_steps = search_policy(host, validation, execution,
                                                               args.search_trials, base_steps, best, log,
                                                               focused=args.focused_search)
            torch.save(host.state_dict(), args.output / "best.pt")
            torch.save(host.state_dict(), args.output / "last.pt")
            log("selected", update=chosen, source_steps=base_steps, search_steps=used_steps,
                environment_steps=selected_steps)
            evaluate_team("best-evaluation", chosen, selected_steps, final=True)
            evaluate_team("official-reference", 0, 0, final=True, reference=True)
            return
        torch.save(policy.state_dict(), args.output / "best.pt")
        if args.side_worker:
            isolate_side_worker(policy.worker)
            critic = mlp(policy.worker[0].in_features, 1).to(args.device)
            joint = PPO(policy.worker, critic, .00032, args.device, temperature=.70)
        elif args.manager_only:
            rate, temperature = .00032, .9
        else:
            rate, temperature = (.00012, .55) if args.throughput else (.0002, .7)
        if not args.side_worker:
            worker_actor = None if args.manager_only else policy.worker
            joint = PPO(policy.manager, policy.critic, rate, args.device, worker=worker_actor, temperature=temperature)
        chosen, step_count = 0, args.lanes * args.steps
        curriculum = ()
        if args.handoff_curriculum:
            curriculum, collection_steps = handoff_states(host, execution)
            log("curriculum_states", snapshots=len(curriculum), collection_steps=collection_steps,
                training_seed_range=[30001, 30008], restart_probability=.5)
        batch = new_batch(curriculum)
        for update in range(1, args.updates + 1):
            checkpoint_copy()
            if args.side_worker:
                records, final, scores = collect_side_worker(batch, host, args.steps, temperature=joint.temperature)
            else:
                records, final, scores = collect_manager(batch, host, args.steps, joint_worker=not args.manager_only,
                                                         temperature=joint.temperature, throughput=args.throughput,
                                                         prep_shaping=args.prep_shaping)
            joint.update(records, final)
            steps = base_steps + update * step_count
            log("ppo_update", update=update, environment_steps=steps, joint_steps=update * step_count,
                sampled_mean=float(np.mean(scores)) if scores else None)
            if update % 6 == 0 or update == args.updates:
                checkpoint_copy()
                current = evaluate_team("validation", update, steps)
                if current > best:
                    best, chosen = current, update
                    torch.save(policy.state_dict(), args.output / "best.pt")
                if best[0] >= 420 and not args.fixed_budget:
                    break
        torch.save(policy.state_dict(), args.output / "last.pt")
        host.load_state_dict(torch.load(args.output / "best.pt", map_location="cpu", weights_only=True))
        log("selected", update=chosen, source_steps=base_steps, joint_steps=chosen * step_count)
        evaluate_team("best-evaluation", chosen, base_steps + chosen * step_count, final=True)
        evaluate_team("official-reference", 0, 0, final=True, reference=True)
        return
    worker_critic = mlp(policy.worker[0].in_features, 1).to(args.device)
    worker = PPO(policy.worker, worker_critic, .0005, args.device)
    batch = new_batch()
    step_count = args.lanes * args.steps
    for update in range(1, args.worker_updates + 1):
        checkpoint_copy()
        records, final, hits = collect_worker(batch, host, args.steps)
        worker.update(records, final)
        log("worker_ppo_update", update=update, environment_steps=update * step_count,
            skill_completions=hits, skill_decisions=step_count * 3)
    torch.save(policy.worker.state_dict(), args.output / "worker.pt")
    del worker, worker_critic

    checkpoint_copy()
    worker_steps = args.worker_updates * step_count
    evaluate_team("initialized", 0, worker_steps)
    manager = PPO(policy.manager, policy.critic, .00035, args.device)
    batch = new_batch()
    best, chosen = None, None
    for update in range(1, args.updates + 1):
        checkpoint_copy()
        records, final, scores = collect_manager(batch, host, args.steps)
        manager.update(records, final)
        steps = worker_steps + update * step_count
        log("ppo_update", update=update, environment_steps=steps, manager_steps=update * step_count,
            sampled_mean=float(np.mean(scores)) if scores else None)
        if update % 6 == 0 or update == args.updates:
            checkpoint_copy()
            current = evaluate_team("validation", update, steps)
            if best is None or current > best:
                best, chosen = current, update
                torch.save(policy.state_dict(), args.output / "best.pt")
    torch.save(policy.state_dict(), args.output / "last.pt")
    host.load_state_dict(torch.load(args.output / "best.pt", map_location="cpu", weights_only=True))
    log("selected", update=chosen, worker_steps=worker_steps, manager_steps=chosen * step_count)
    evaluate_team("best-evaluation", chosen, worker_steps + chosen * step_count, final=True)
    evaluate_team("official-reference", 0, 0, final=True, reference=True)


if __name__ == "__main__":
    main()

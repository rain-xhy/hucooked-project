"""All-model 3-chef self-play PPO training starting from a trained V3Policy checkpoint."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES
from .bc_experiment import load_policy
from .experiment import read, save
from .features import partner_action_ids
from .history import OnlineHistory, RolloutHistory
from .kitchen import V3PolicyKitchen
from .policy import V3Policy
from . import shaping as progress_shaping
from .tracking import COOPERATION_PROJECT, RUNS, open_run, recorded_steps
from .train import atomic_save, scheduled_lr

VALIDATION_SEED = 10_000_000


@dataclass(frozen=True)
class SelfPlaySettings:
    lr: float = 3e-5
    seed: int = 1
    updates: int = 200
    num_envs: int = 1344
    rollout_steps: int = 200
    horizon: int = 400
    epochs: int = 4
    minibatch_size: int = 3201  # Divisible by 3 (1067 per chef)
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.015
    target_kl: float = 0.02
    max_grad_norm: float = 2.0
    warmup_fraction: float = 0.05
    warmup_start_factor: float = 0.1
    shaping_start: float = 1.0
    shaping_end: float = 0.0
    shaping_decay_updates: int = 80
    stagger_history: bool = True
    recipe_rewards: tuple = (20., 10., 50.)


class SelfPlayTrainer:
    """Shared-policy 3-chef self-play PPO with per-chef causal histories."""

    def __init__(self, cfg: SelfPlaySettings, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env = V3PolicyKitchen(
            PartitionConfig(layout="counter", horizon=cfg.horizon, recipe_rewards=cfg.recipe_rewards),
            num_envs=cfg.num_envs, device=device)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.histories = [OnlineHistory(cfg.num_envs, device, stagger=cfg.stagger_history) for _ in range(3)]
        self.model = V3Policy(3).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        self.tick = 0
        self.update = 0
        self.optimizer_step_count = 0
        self.episode_totals = torch.zeros(2, dtype=torch.float64, device=device)
        self.episode_totals_valid = True

    def current_shaping_coef(self):
        cfg = self.cfg
        if cfg.shaping_decay_updates <= 0:
            return cfg.shaping_end
        frac = min(1.0, self.update / cfg.shaping_decay_updates)
        return cfg.shaping_start + (cfg.shaping_end - cfg.shaping_start) * frac

    def state_dict(self):
        return dict(
            config=asdict(self.cfg),
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            update=self.update,
            tick=self.tick,
            optimizer_step_count=self.optimizer_step_count,
            episode_totals=self.episode_totals,
            episode_totals_valid=self.episode_totals_valid,
            env=self.env.state,
            draws=self.env.draws,
            lane_rng=[[r.bit_generator.state for r in pair] for pair in self.env.rng],
            histories=[h.state_dict() for h in self.histories],
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        )

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update = state["update"]
        self.tick = state["tick"]
        self.optimizer_step_count = state["optimizer_step_count"]
        self.episode_totals.copy_(state["episode_totals"])
        self.episode_totals_valid = state["episode_totals_valid"]
        self.env.state = state["env"]
        self.env.draws.copy_(state["draws"])
        for generators, saved in zip(self.env.rng, state["lane_rng"]):
            for generator, source in zip(generators, saved):
                generator.bit_generator.state = source
        for h, h_state in zip(self.histories, state["histories"]):
            h.load_state_dict(h_state)
        self.obs = self.env.observations()
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])

    @torch.no_grad()
    def collect(self):
        cfg = self.cfg
        shaping_coef = self.current_shaping_coef()
        banks = [RolloutHistory(h, cfg.rollout_steps) for h in self.histories]
        for h in self.histories:
            h.refresh(self.model)
        actions = [[], [], []]
        logprobs = [[], [], []]
        values = [[], [], []]
        rewards = []
        terminals = []
        totals = self.obs.new_zeros(3)
        shaping_total = self.obs.new_zeros(())
        recipe_deliveries = torch.zeros(3, dtype=torch.long, device=self.device)
        episodes = 0
        scored_episodes = 0
        completed_totals = torch.zeros_like(self.episode_totals)
        before = None

        for _ in range(cfg.rollout_steps):
            step_acts = []
            for c in range(3):
                cur = self.obs[:, c]
                logp, val, _ = self.histories[c].forward(self.model, cur)
                choice = logp.exp().multinomial(1)
                step_acts.append(choice[:, 0])
                actions[c].append(choice[:, 0])
                logprobs[c].append(logp.gather(1, choice)[:, 0])
                values[c].append(val[:, 0])
            joint = torch.stack(step_acts, dim=1)
            if shaping_coef > 0 and before is None:
                before = progress_shaping.potential(self.env.state)
            prev_obs = self.obs
            self.obs, reward, _, _, info = self.env.step(joint)
            pids = partner_action_ids(joint)
            for c in range(3):
                banks[c].record(prev_obs[:, c], pids[:, c])
                self.histories[c].append(prev_obs[:, c], pids[:, c], self.model)
            self.tick += 1
            done = self.tick == cfg.horizon
            if shaping_coef > 0:
                after = torch.zeros_like(before) if done else progress_shaping.potential(self.env.state)
                shaped = shaping_coef * (cfg.gamma * after - before)
                before = after
            else:
                shaped = torch.zeros_like(reward)
            rewards.append(reward.float() + shaped.float())
            shaping_total += shaped.sum()
            self.episode_totals += torch.stack((reward.sum(), shaped.sum()))
            terminals.append(done)
            totals += torch.stack((reward.sum(), info["delivered"].sum(), info["expired"].sum()))
            recipe_deliveries.scatter_add_(0, info["recipe"].flatten(), info["delivered"].long().flatten())
            if done:
                if self.episode_totals_valid:
                    completed_totals += self.episode_totals
                    scored_episodes += cfg.num_envs
                self.episode_totals.zero_()
                self.episode_totals_valid = True
                for c in range(3):
                    self.histories[c].finish_episode()
                    banks[c].finish_episode()
                self.obs, _ = self.env.reset()
                self.tick = 0
                before = None
                episodes += cfg.num_envs

        for b in banks:
            b.seal()

        # Compute GAE independently for each chef's value stream, then normalize advantages globally.
        per_chef = []
        all_adv = []
        for c in range(3):
            next_v = self.histories[c].forward(self.model, self.obs[:, c])[1][:, 0]
            vals = torch.stack(values[c])
            advs = torch.zeros_like(vals)
            carry = torch.zeros_like(next_v)
            for t in reversed(range(cfg.rollout_steps)):
                discount = 0.0 if terminals[t] else cfg.gamma
                delta = rewards[t] + discount * next_v - vals[t]
                carry = delta + discount * cfg.gae_lambda * carry
                advs[t] = carry
                next_v = vals[t]
            rets = (advs + vals).flatten()
            adv_flat = advs.flatten()
            all_adv.append(adv_flat)
            per_chef.append(dict(
                action=torch.cat(actions[c]),
                logp=torch.cat(logprobs[c]),
                returns=rets,
                old_values=vals.flatten(),
                adv=adv_flat,
            ))
        cat_adv = torch.cat(all_adv)
        mean_adv, std_adv = cat_adv.mean(), cat_adv.std(unbiased=False) + 1e-8
        for d in per_chef:
            d["adv"] = (d["adv"] - mean_adv) / std_adv

        reward_sum, deliveries, expired = totals.tolist()
        metrics = dict(
            episodes=episodes,
            reward=reward_sum,
            deliveries=deliveries,
            expirations=expired,
            total_score=reward_sum,
            reward_per_lane=reward_sum / cfg.num_envs,
            shaping_reward=shaping_total.item(),
            shaping_coef=shaping_coef,
            score_episodes=scored_episodes,
        )
        if scored_episodes:
            completed_score, completed_shaping = completed_totals.tolist()
            metrics["mean_score"] = completed_score / scored_episodes
            metrics["mean_shaping_reward"] = completed_shaping / scored_episodes
        for recipe, count, points in zip(RECIPE_NAMES, recipe_deliveries.tolist(), cfg.recipe_rewards):
            metrics[f"per_recipe/{recipe}/score"] = count * points
        return banks, per_chef, metrics

    def optimize(self, banks, per_chef):
        cfg = self.cfg
        n_per_chef = len(per_chef[0]["action"])
        mb_per_chef = max(1, cfg.minibatch_size // 3)
        steps_per_update = cfg.epochs * math.ceil(n_per_chef / mb_per_chef)
        total_steps = cfg.updates * steps_per_update
        metrics = []
        stopped = False

        for _ in range(cfg.epochs):
            perms = [torch.randperm(n_per_chef, device=self.device).split(mb_per_chef) for _ in range(3)]
            for idx_triplet in zip(*perms):
                lr = scheduled_lr(cfg.lr, self.optimizer_step_count, total_steps,
                                  cfg.warmup_fraction, cfg.warmup_start_factor)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                logp_list, val_list, act_list, old_lp_list, adv_list, old_v_list, ret_list = [], [], [], [], [], [], []
                for c in range(3):
                    idx = idx_triplet[c]
                    lp, v, _, _ = banks[c].replay(self.model, idx)
                    logp_list.append(lp)
                    val_list.append(v[:, 0])
                    act_list.append(per_chef[c]["action"][idx])
                    old_lp_list.append(per_chef[c]["logp"][idx])
                    adv_list.append(per_chef[c]["adv"][idx])
                    old_v_list.append(per_chef[c]["old_values"][idx])
                    ret_list.append(per_chef[c]["returns"][idx])

                logp = torch.cat(logp_list)
                value = torch.cat(val_list)
                action = torch.cat(act_list)
                old_logp = torch.cat(old_lp_list)
                advantage = torch.cat(adv_list)
                old_value = torch.cat(old_v_list)
                target = torch.cat(ret_list)

                selected = logp.gather(1, action[:, None])[:, 0]
                logratio = selected - old_logp
                ratio = logratio.exp()
                with torch.no_grad():
                    current_kl = (ratio - 1 - logratio).mean()
                    if not torch.isfinite(current_kl):
                        raise FloatingPointError("Non-finite PPO likelihood ratio / KL")
                if cfg.target_kl is not None and metrics and current_kl > 1.5 * cfg.target_kl:
                    stopped = True
                    break

                clipped = ratio.clamp(0.8, 1.2)
                actor = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                clipped_value = old_value + (value - old_value).clamp(-0.2, 0.2)
                critic = 0.5 * torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
                entropy = -(logp.exp() * logp).sum(-1).mean()
                loss = actor + 0.5 * critic - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_step_count += 1

                with torch.no_grad():
                    kl = (ratio - 1 - logratio).mean()
                    clip_fraction = (ratio != clipped).float().mean()
                    metrics.append(torch.stack((loss, actor, critic, entropy, kl, clip_fraction, norm)).detach())
            if stopped:
                break

        names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
        result = dict(zip(names, torch.stack(metrics).mean(0).tolist()))
        result.update(optimizer_steps=len(metrics), kl_early_stop=int(stopped))
        return result

    def step(self):
        self.model.train()
        start = time.perf_counter()
        banks, per_chef, metrics = self.collect()
        metrics.update(self.optimize(banks, per_chef))
        self.update += 1
        env_step_count = self.cfg.num_envs * self.cfg.rollout_steps
        agent_step_count = env_step_count * 3
        elapsed = time.perf_counter() - start
        metrics.update(
            env_steps=self.update * env_step_count,
            agent_steps=self.update * agent_step_count,
            lr=self.optimizer.param_groups[0]["lr"],
            steps_per_second=agent_step_count / elapsed,
            elapsed_seconds=elapsed,
        )
        return metrics


@torch.inference_mode()
def evaluate_selfplay(model, device="cuda", seed=VALIDATION_SEED, num_lanes=9, horizon=400, recipe_rewards=(20., 10., 50.)):
    """45-episode (9 lanes x 5 consecutive episodes) greedy all-model evaluation."""
    training = model.training
    model.eval()
    try:
        env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=horizon, recipe_rewards=recipe_rewards),
                              num_envs=num_lanes, device=device)
        histories = [OnlineHistory(num_lanes, device) for _ in range(3)]
        scores = torch.zeros(5, num_lanes, device=device, dtype=torch.float64)
        orders = torch.zeros(5, num_lanes, 3, 4, device=device, dtype=torch.long)
        for ep in range(5):
            obs, _ = env.reset(seed=seed + ep * 1000)
            for _ in range(horizon):
                acts = []
                for c in range(3):
                    logp, _, _ = histories[c].forward(model, obs[:, c])
                    acts.append(logp.argmax(-1))
                joint = torch.stack(acts, dim=1)
                pids = partner_action_ids(joint)
                for c in range(3):
                    histories[c].append(obs[:, c], pids[:, c], model)
                obs, reward, _, _, info = env.step(joint)
                scores[ep] += reward
                for col, val in ((0, info["delivered"]), (1, info["expired"]), (3, info["age"] * info["delivered"])):
                    orders[ep, :, :, col].scatter_add_(1, info["recipe"], val.long())
            pending = env.state["recipe"]
            orders[ep, :, :, 2].scatter_add_(1, pending, torch.ones_like(pending))
            for c in range(3):
                histories[c].finish_episode()
        o_sum = orders.sum(dim=(0, 1)).cpu().tolist()
        per_recipe = {}
        for r_idx, r_name in enumerate(RECIPE_NAMES):
            deliv, exp, unres, age_sum = o_sum[r_idx]
            denom = deliv + exp
            per_recipe[r_name] = dict(
                delivered=deliv, expired=exp, unresolved=unres,
                delivery_rate=(deliv / denom) if denom else 0.0,
                mean_delivery_steps=(age_sum / deliv) if deliv else None,
            )
        return dict(
            mode="all_policy_greedy",
            seed=int(seed),
            episodes=5 * num_lanes,
            mean_score=float(scores.mean()),
            std_score=float(scores.std()),
            min_score=float(scores.min()),
            max_score=float(scores.max()),
            per_episode_mean=[float(scores[ep].mean()) for ep in range(5)],
            per_recipe=per_recipe,
        )
    finally:
        model.train(training)


def run_selfplay(root, init_policy, *, device="cuda", cfg: SelfPlaySettings = SelfPlaySettings(), eval_every=2):
    root = Path(root)
    init_policy = Path(init_policy)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(
        version="all-policy-selfplay-v1",
        **asdict(cfg),
        initial_policy=str(init_policy.resolve()),
        initial_sha256=hashlib.sha256(init_policy.read_bytes()).hexdigest(),
        eval_every_updates=eval_every,
        validation_seed=VALIDATION_SEED,
    )
    manifest = root / "experiment.json"
    if manifest.exists() and read(manifest) != config:
        raise ValueError("Self-play configuration changed; use a new campaign")
    save(manifest, config)

    trainer = SelfPlayTrainer(cfg, device)
    latest = root / "latest.pt"
    if latest.exists():
        trainer.load(latest)
    else:
        trainer.model.load_state_dict(torch.load(init_policy, map_location=device, weights_only=False)["model"])

    journal = root / "train.jsonl"
    if journal.exists():
        retained = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break
            if rec["update"] <= trainer.update:
                retained.append(line)
        journal.write_text("".join(line + "\n" for line in retained), encoding="utf-8")

    run_name = f"{root.name}-selfplay-seed{cfg.seed}"
    published = recorded_steps(run_name, project=COOPERATION_PROJECT)
    run = open_run(run_name, "all-policy-selfplay", config, True, project=COOPERATION_PROJECT)
    agent_samples = cfg.num_envs * cfg.rollout_steps * 3

    def publish(update, metrics=None, val=None):
        step = update * agent_samples
        if step in published:
            return
        values = {}
        if metrics is not None:
            if "mean_score" in metrics:
                values["train/score_per_episode"] = metrics["mean_score"]
            if "mean_shaping_reward" in metrics:
                values["train/process_reward_per_episode"] = metrics["mean_shaping_reward"]
        if val is not None:
            values["eval/score_per_episode"] = val["mean_score"]
            for r_name, r_val in val["per_recipe"].items():
                values[f"delivery_rate/{r_name}"] = r_val["delivery_rate"]
                if r_val["mean_delivery_steps"] is not None:
                    values[f"delivery_steps/{r_name}"] = r_val["mean_delivery_steps"]
        if values:
            run.log(values, step=step)
            published.add(step)

    try:
        init_val_path = root / "validation-initial.json"
        if not init_val_path.exists():
            init_val = evaluate_selfplay(trainer.model, device=device, seed=VALIDATION_SEED, horizon=cfg.horizon)
            save(init_val_path, init_val)
        else:
            init_val = read(init_val_path)
        publish(0, val=init_val)
        print(f"Self-play init eval (45 eps): mean={init_val['mean_score']:.2f} min={init_val['min_score']:.0f} max={init_val['max_score']:.0f}", flush=True)

        while trainer.update < cfg.updates:
            metrics = trainer.step()
            u = trainer.update
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(update=u, **metrics), allow_nan=False) + "\n")
            val = None
            if u % eval_every == 0 or u == cfg.updates:
                np_s, py_s = np.random.get_state(), random.getstate()
                cuda_devs = [trainer.device] if trainer.device.type == "cuda" else []
                try:
                    with torch.random.fork_rng(devices=cuda_devs):
                        val = evaluate_selfplay(trainer.model, device=device, seed=VALIDATION_SEED, horizon=cfg.horizon)
                finally:
                    np.random.set_state(np_s)
                    random.setstate(py_s)
                save(root / "validation" / f"{u:03d}.json", val)
                atomic_save(dict(model=trainer.model.state_dict(), config=asdict(cfg), update=u),
                            root / f"policy_{u:06d}.pt")
                atomic_save(trainer.state_dict(), latest)
            publish(u, metrics, val)
            save(root / "status.json", dict(
                update=u, total_updates=cfg.updates, status="complete" if u == cfg.updates else "running",
                shaping_coef=metrics["shaping_coef"], train_score=metrics.get("mean_score"),
                eval_score=val["mean_score"] if val else None,
            ))
            val_str = f" eval={val['mean_score']:.2f}" if val else ""
            sc_str = f"{metrics['mean_score']:.2f}" if "mean_score" in metrics else "None"
            print(f"SelfPlay {u}/{cfg.updates} score={sc_str}{val_str} kl={metrics['approx_kl']:.5f} gn={metrics['grad_norm']:.2f} ent={metrics['entropy']:.3f} shp={metrics['shaping_coef']:.2f} sps={metrics['steps_per_second']:,.0f}", flush=True)
    finally:
        run.finish()


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--init-policy", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=1344)
    p.add_argument("--rollout-steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=3201)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--entropy-coef", type=float, default=0.015)
    p.add_argument("--max-grad-norm", type=float, default=2.0)
    p.add_argument("--shaping-start", type=float, default=1.0)
    p.add_argument("--shaping-end", type=float, default=0.0)
    p.add_argument("--shaping-decay-updates", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=2)
    args = p.parse_args()
    cfg = SelfPlaySettings(
        lr=args.lr, seed=args.seed, updates=args.updates, num_envs=args.num_envs,
        rollout_steps=args.rollout_steps, horizon=args.horizon, epochs=args.epochs,
        minibatch_size=args.minibatch_size, gamma=args.gamma, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm, shaping_start=args.shaping_start,
        shaping_end=args.shaping_end, shaping_decay_updates=args.shaping_decay_updates,
    )
    run_selfplay(RUNS / args.campaign, args.init_policy, device=args.device, cfg=cfg, eval_every=args.eval_every)


if __name__ == "__main__":
    main()

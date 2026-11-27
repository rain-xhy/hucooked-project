"""NestRL L2 PPO training: majority greedy self-play + diverse early/mid/late frozen L1 partner pool on GPU SharedKitchen."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import trackio

from .env import SharedKitchenConfig, TorchSharedKitchen
from .evaluation import ROLE_NAMES, evaluate_shared, role_slot_map
from .history import OnlineHistory, RolloutHistory
from .policy import V3Policy, partner_action_ids
from .pretrain import atomic_save
from .train_l1 import PROJECT, RUNS, scheduled_lr


@dataclass(frozen=True)
class L2Settings:
    seed: int = 1
    num_envs: int = 2016
    rollout_steps: int = 300
    horizon: int = 300
    updates: int = 200
    epochs: int = 4
    minibatch_size: int = 3024
    lr: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.008
    target_kl: float | None = 0.02
    max_grad_norm: float = 2.0
    warmup_fraction: float = 0.05
    warmup_start_factor: float = 0.1
    shaping_scale: float = 0.05
    stagger_history: bool = True
    eval_every: int = 5
    critic_warmup_updates: int = 2
    use_history: bool = True
    selfplay_frac: float = 0.65
    partner_type: str = "fast"


@torch.no_grad()
def evaluate_vs_frozen_partners(learner_model, partner_model, num_envs=45, episodes=5, horizon=300, seed=10_000_000, device="cuda"):
    """Evaluate 1 learner paired with 2 frozen L1 neural partners (unshaped, greedy)."""
    learner_model.eval()
    partner_model.eval()
    env = TorchSharedKitchen(SharedKitchenConfig(horizon=horizon), num_envs=num_envs, device=device, use_cuda_graph=True)
    roles = torch.arange(num_envs, dtype=torch.long, device=device) % 3
    peer_slots = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long, device=device)[roles]
    lanes = env.lanes
    h_learner = OnlineHistory(num_envs, device, stagger=False)
    h_peers = [OnlineHistory(num_envs, device, stagger=False) for _ in range(2)]
    agg = {}
    role_scores = {r: 0.0 for r in ROLE_NAMES}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep * 1000)
        r2s = role_slot_map(env)
        l_slot = r2s[lanes, roles]
        p0_slot = r2s[lanes, peer_slots[:, 0]]
        p1_slot = r2s[lanes, peer_slots[:, 1]]
        h_learner.refresh(learner_model)
        for hp in h_peers:
            hp.refresh(partner_model)
        for _ in range(horizon):
            prev_obs = obs.clone()
            act_l = h_learner.forward(learner_model, prev_obs[lanes, l_slot])[0].argmax(-1)
            act_p0 = h_peers[0].forward(partner_model, prev_obs[lanes, p0_slot])[0].argmax(-1)
            act_p1 = h_peers[1].forward(partner_model, prev_obs[lanes, p1_slot])[0].argmax(-1)
            joint = torch.empty((num_envs, 3), dtype=torch.long, device=device)
            joint[lanes, l_slot] = act_l
            joint[lanes, p0_slot] = act_p0
            joint[lanes, p1_slot] = act_p1
            obs, _, _, _, _ = env.step(joint)
            pids = partner_action_ids(joint)
            h_learner.append(prev_obs[lanes, l_slot], pids[lanes, l_slot], learner_model)
            h_peers[0].append(prev_obs[lanes, p0_slot], pids[lanes, p0_slot], partner_model)
            h_peers[1].append(prev_obs[lanes, p1_slot], pids[lanes, p1_slot], partner_model)
        h_learner.finish_episode()
        for hp in h_peers:
            hp.finish_episode()
        m = env.episode_metrics()
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + float(v)
        sc = env.state["score"]
        for r_idx, r_name in enumerate(ROLE_NAMES):
            role_scores[r_name] += sc[roles == r_idx].mean().item()

    res = {k: v / episodes for k, v in agg.items()}
    res["per_role_score"] = {r: v / episodes for r, v in role_scores.items()}
    return res


class AuthenticL2Trainer:
    """NestRL L2: majority greedy self-play + diverse frozen L1 partner pool."""

    def __init__(self, cfg: L2Settings, partner_pool: list[V3Policy], device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env = TorchSharedKitchen(
            SharedKitchenConfig(
                horizon=cfg.horizon,
                shaping_scale=cfg.shaping_scale,
                shaping_gamma=cfg.gamma,
            ),
            num_envs=cfg.num_envs, device=device, use_cuda_graph=True)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self.roles = torch.arange(cfg.num_envs, dtype=torch.long, device=self.device) % 3
        self.peer_roles = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long, device=self.device)[self.roles]
        self._refresh_role_slots()

        # Partition envs into: [pool_0 | pool_1 | ... | self_play] (each slice is a multiple of 3 for role balance)
        self.partner_pool = []
        for pm in partner_pool:
            pm = pm.to(device).eval()
            for p in pm.parameters():
                p.requires_grad_(False)
            self.partner_pool.append(pm)

        n_sp_raw = int(round(cfg.num_envs * cfg.selfplay_frac))
        self.n_sp = (n_sp_raw // 3) * 3
        self.n_pool_total = cfg.num_envs - self.n_sp
        k = len(self.partner_pool)
        self.pool_slices: list[tuple[int, int]] = []
        if k > 0 and self.n_pool_total > 0:
            base = (self.n_pool_total // k // 3) * 3
            cur = 0
            for i in range(k):
                nxt = self.n_pool_total if i == k - 1 else cur + base
                self.pool_slices.append((cur, nxt))
                cur = nxt
        self.sp_slice = (self.n_pool_total, cfg.num_envs)

        self.learner_history = OnlineHistory(cfg.num_envs, device, stagger=cfg.stagger_history)
        self.peer_histories = [OnlineHistory(cfg.num_envs, device, stagger=cfg.stagger_history) for _ in range(2)]
        self.model = V3Policy(3, use_history=cfg.use_history).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        self.tick = 0
        self.update = 0
        self.optimizer_step_count = 0

    def _refresh_role_slots(self):
        r2s = role_slot_map(self.env)
        lanes = self.env.lanes
        self.learner_slot = r2s[lanes, self.roles]
        self.p0_slot = r2s[lanes, self.peer_roles[:, 0]]
        self.p1_slot = r2s[lanes, self.peer_roles[:, 1]]

    def state_dict(self):
        return dict(
            config=asdict(self.cfg),
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            update=self.update,
            tick=self.tick,
            optimizer_step_count=self.optimizer_step_count,
            learner_history=self.learner_history.state_dict(),
        )

    @torch.no_grad()
    def _peer_greedy_actions(self, prev_obs: torch.Tensor, slot_vec: torch.Tensor, peer_idx: int) -> torch.Tensor:
        """Compute greedy (argmax) actions for one peer across all pool slices + self-play slice."""
        out = torch.empty((self.cfg.num_envs,), dtype=torch.long, device=self.device)
        lanes = self.env.lanes
        cur_all = prev_obs[lanes, slot_vec]
        if not self.cfg.use_history:
            for (s, e), pm in zip(self.pool_slices, self.partner_pool):
                if e > s:
                    out[s:e] = pm.actor(cur_all[s:e]).argmax(-1)
            s, e = self.sp_slice
            if e > s:
                out[s:e] = self.model.actor(cur_all[s:e]).argmax(-1)
            return out
        # Fallback if use_history=True
        for (s, e), pm in zip(self.pool_slices, self.partner_pool):
            if e > s:
                out[s:e] = self.peer_histories[peer_idx].forward(pm, cur_all)[0][s:e].argmax(-1)
        s, e = self.sp_slice
        if e > s:
            out[s:e] = self.peer_histories[peer_idx].forward(self.model, cur_all)[0][s:e].argmax(-1)
        return out

    @torch.no_grad()
    def collect(self):
        cfg = self.cfg
        lanes = self.env.lanes
        bank = RolloutHistory(self.learner_history, cfg.rollout_steps)
        self.learner_history.refresh(self.model)

        actions, logprobs, values, rewards, terminals = [], [], [], [], []
        completed_score = 0.0
        completed_sp_score = 0.0
        completed_pool_score = 0.0
        completed_meals = 0.0
        completed_eps = 0

        for _ in range(cfg.rollout_steps):
            l_sl, p0, p1 = self.learner_slot, self.p0_slot, self.p1_slot
            prev_obs = self.obs.clone()
            cur_l = prev_obs[lanes, l_sl]
            logp, val, _ = self.learner_history.forward(self.model, cur_l)
            choice = logp.exp().multinomial(1)

            # Both frozen pool partners AND self-play peers act greedily (argmax)
            act_p0 = self._peer_greedy_actions(prev_obs, p0, 0)
            act_p1 = self._peer_greedy_actions(prev_obs, p1, 1)

            joint = torch.empty((cfg.num_envs, 3), dtype=torch.long, device=self.device)
            joint[lanes, l_sl] = choice[:, 0]
            joint[lanes, p0] = act_p0
            joint[lanes, p1] = act_p1

            self.obs, reward, _, _, _ = self.env.step(joint)
            pids = partner_action_ids(joint)
            bank.record(cur_l, pids[lanes, l_sl])
            self.learner_history.append(cur_l, pids[lanes, l_sl], self.model)

            actions.append(choice[:, 0])
            logprobs.append(logp.gather(1, choice)[:, 0])
            values.append(val[:, 0])
            rewards.append(reward.clone())
            self.tick += 1
            done = self.tick == cfg.horizon
            terminals.append(done)
            if done:
                m = self.env.episode_metrics()
                completed_score += m["team_score"]
                completed_meals += m["meals_per_100_steps"]
                sc = self.env.state["score"]
                if self.n_sp > 0:
                    completed_sp_score += float(sc[self.sp_slice[0]:self.sp_slice[1]].mean())
                if self.n_pool_total > 0:
                    completed_pool_score += float(sc[:self.n_pool_total].mean())
                completed_eps += 1
                self.learner_history.finish_episode()
                bank.finish_episode()
                self.obs, _ = self.env.reset()
                self._refresh_role_slots()
                self.tick = 0

        bank.seal()
        next_v = self.learner_history.forward(self.model, self.obs[lanes, self.learner_slot])[1][:, 0]
        vals = torch.stack(values)
        rew_t = torch.stack(rewards)
        advs = torch.zeros_like(vals)
        carry = torch.zeros_like(next_v)
        for t in reversed(range(cfg.rollout_steps)):
            disc = 0.0 if terminals[t] else cfg.gamma
            delta = rew_t[t] + disc * next_v - vals[t]
            carry = delta + disc * cfg.gae_lambda * carry
            advs[t] = carry
            next_v = vals[t]
        rets = (advs + vals).flatten()
        # Per-role advantage normalization
        adv_norm = torch.empty_like(advs)
        for r in range(3):
            rmask = (self.roles == r)
            sub = advs[:, rmask]
            adv_norm[:, rmask] = (sub - sub.mean()) / (sub.std(unbiased=False) + 1e-8)
        data = dict(
            action=torch.cat(actions),
            logp=torch.cat(logprobs),
            returns=rets,
            old_values=vals.flatten(),
            adv=adv_norm.flatten(),
        )
        metrics = dict(
            completed_batches=completed_eps,
            mean_step_reward=rew_t.mean().item(),
            mean_return=rets.mean().item(),
            mean_value=vals.mean().item(),
        )
        if completed_eps:
            metrics["mean_score"] = completed_score / completed_eps
            metrics["meals_per_100_steps"] = completed_meals / completed_eps
            if self.n_sp > 0:
                metrics["selfplay_score"] = completed_sp_score / completed_eps
            if self.n_pool_total > 0:
                metrics["pool_score"] = completed_pool_score / completed_eps
        return bank, data, metrics

    def warmup_critic(self):
        cfg = self.cfg
        if cfg.critic_warmup_updates <= 0:
            return
        c_opt = torch.optim.Adam(self.model.critic.parameters(), lr=3e-4, eps=1e-8)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.critic.parameters():
            p.requires_grad_(True)
        try:
            for w in range(cfg.critic_warmup_updates):
                self.model.train()
                bank, data, m = self.collect()
                batch = len(data["action"])
                vlosses = []
                for _ in range(cfg.epochs):
                    for index in torch.randperm(batch, device=self.device).split(cfg.minibatch_size):
                        _, value, _, _ = bank.replay(self.model, index)
                        target = data["returns"][index]
                        vloss = 0.5 * (value[:, 0] - target).square().mean()
                        c_opt.zero_grad(set_to_none=True)
                        vloss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), cfg.max_grad_norm)
                        c_opt.step()
                        vlosses.append(vloss.detach())
                print(f"L2 Critic warmup {w + 1}/{cfg.critic_warmup_updates} | vloss={torch.stack(vlosses).mean().item():.2f} mean_ret={m['mean_return']:.1f} mean_v={m['mean_value']:.1f}", flush=True)
        finally:
            for p in self.model.parameters():
                p.requires_grad_(True)

    def optimize(self, bank, data):
        cfg = self.cfg
        batch = len(data["action"])
        metrics = []
        stopped = False
        lr = scheduled_lr(cfg.lr, self.update, cfg.updates, cfg.warmup_fraction, cfg.warmup_start_factor)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        for _ in range(cfg.epochs):
            for index in torch.randperm(batch, device=self.device).split(cfg.minibatch_size):
                logp, value, _, _ = bank.replay(self.model, index)
                selected = logp.gather(1, data["action"][index, None])[:, 0]
                logratio = selected - data["logp"][index]
                ratio = logratio.exp()
                with torch.no_grad():
                    current_kl = (ratio - 1 - logratio).mean()
                    if not torch.isfinite(current_kl):
                        raise FloatingPointError("Non-finite KL")
                if cfg.target_kl is not None and metrics and current_kl > 1.5 * cfg.target_kl:
                    stopped = True
                    break
                clipped = ratio.clamp(0.8, 1.2)
                adv = data["adv"][index]
                actor = -torch.minimum(ratio * adv, clipped * adv).mean()
                old_v = data["old_values"][index]
                clipped_v = old_v + (value[:, 0] - old_v).clamp(-0.2, 0.2)
                target = data["returns"][index]
                critic = 0.5 * torch.maximum((value[:, 0] - target).square(), (clipped_v - target).square()).mean()
                entropy = -(logp.exp() * logp).sum(-1).mean()
                loss = actor + 0.5 * critic - cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_step_count += 1
                with torch.no_grad():
                    kl = (ratio - 1 - logratio).mean()
                    cf = (ratio != clipped).float().mean()
                    metrics.append(torch.stack((loss, actor, critic, entropy, kl, cf, norm)).detach())
            if stopped:
                break

        names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
        res = dict(zip(names, torch.stack(metrics).mean(0).tolist()))
        res.update(optimizer_steps=len(metrics), kl_early_stop=int(stopped))
        return res

    def step(self):
        self.model.train()
        t0 = time.perf_counter()
        bank, data, metrics = self.collect()
        metrics.update(self.optimize(bank, data))
        self.update += 1
        count = self.cfg.num_envs * self.cfg.rollout_steps
        elapsed = time.perf_counter() - t0
        metrics.update(env_steps=self.update * count, lr=self.optimizer.param_groups[0]["lr"],
                       steps_per_second=count / elapsed, elapsed_seconds=elapsed)
        return metrics


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", default="no-hist-l2-sp65-pool-seed1")
    p.add_argument("--partner-pool", type=Path, nargs="+", required=True, help="One or more frozen L1 checkpoints for the partner pool.")
    p.add_argument("--init-policy", type=Path, required=True, help="Learner init checkpoint (or 'scratch').")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=2016)
    p.add_argument("--rollout-steps", type=int, default=300)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=3024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--entropy-coef", type=float, default=0.008)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--shaping-scale", type=float, default=0.05)
    p.add_argument("--selfplay-frac", type=float, default=0.65)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--partner-type", choices=("shared", "fast"), default="fast")
    args = p.parse_args()

    root = RUNS / args.campaign
    ppo_dir = root / "l2"
    ppo_dir.mkdir(parents=True, exist_ok=True)

    init_from_scratch = (str(args.init_policy) == "scratch")
    init_path = None if init_from_scratch else args.init_policy
    use_history = not args.no_history

    cfg = L2Settings(
        seed=args.seed, num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        updates=args.updates, epochs=args.epochs, minibatch_size=args.minibatch_size,
        lr=args.lr, gamma=args.gamma, entropy_coef=args.entropy_coef,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        shaping_scale=args.shaping_scale, eval_every=args.eval_every,
        critic_warmup_updates=0 if init_from_scratch else 2,
        use_history=use_history, selfplay_frac=args.selfplay_frac,
        partner_type=args.partner_type,
    )
    meta_cfg = asdict(cfg)
    meta_cfg["partner_pool"] = [str(x) for x in args.partner_pool]
    meta_cfg["init_policy"] = str(args.init_policy)
    (ppo_dir / "experiment.json").write_text(json.dumps(meta_cfg, indent=2), encoding="utf-8")

    run = trackio.init(project=PROJECT, name=f"{args.campaign}-l2-seed{args.seed}",
                       group="shared-l2-pool-selfplay", config=meta_cfg, resume="allow")

    pool_models = []
    for pp in args.partner_pool:
        pm = V3Policy(3, use_history=use_history)
        pm.load_state_dict(torch.load(pp, map_location=args.device, weights_only=False)["model"])
        pool_models.append(pm)
    eval_strong_partner = pool_models[-1]

    trainer = AuthenticL2Trainer(cfg, partner_pool=pool_models, device=args.device)
    print(f"L2 Partner pool ({len(pool_models)} ckpts, pool_envs={trainer.n_pool_total}, sp_envs={trainer.n_sp}): {[p.name for p in args.partner_pool]}", flush=True)
    if init_path is not None:
        print(f"Initializing learner from: {init_path}", flush=True)
        trainer.model.load_state_dict(torch.load(init_path, map_location=args.device, weights_only=False)["model"])
        trainer.warmup_critic()
    else:
        print("Initializing learner from scratch (random weights)", flush=True)

    # Initial eval (step 0)
    ev0_l2 = evaluate_vs_frozen_partners(trainer.model, eval_strong_partner, num_envs=45, episodes=5, seed=10_000_000, device=args.device)
    ev0_all = evaluate_shared(trainer.model, mode="all_policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
    ev0_sin = evaluate_shared(trainer.model, mode="policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
    (ppo_dir / "validation-init.json").write_text(json.dumps(dict(l2_vs_strong_l1=ev0_l2, all_policy=ev0_all, single_vs_script=ev0_sin), indent=2), encoding="utf-8")
    print(f"L2 init | all_policy={ev0_all['team_score']:.1f} | vs_strongL1={ev0_l2['team_score']:.1f} (S={ev0_l2['per_role_score']['server']:.0f}/M={ev0_l2['per_role_score']['main']:.0f}/D={ev0_l2['per_role_score']['side']:.0f}) | vs_fastRule={ev0_sin['team_score']:.1f}", flush=True)
    run.log({
        "eval/all_policy_score": ev0_all["team_score"],
        "eval/l2_vs_frozen_l1_score": ev0_l2["team_score"],
        "eval/l2_role_server": ev0_l2["per_role_score"]["server"],
        "eval/l2_role_main": ev0_l2["per_role_score"]["main"],
        "eval/l2_role_side": ev0_l2["per_role_score"]["side"],
        "eval/l2_joint_attempt_success_rate": ev0_l2["joint_attempt_success_rate"],
        "eval/l2_joint_wait_steps": ev0_l2["joint_wait_steps"],
        "eval/score_per_episode": ev0_sin["team_score"],
    }, step=0)

    journal = ppo_dir / "train.jsonl"
    vdir = ppo_dir / "validation"
    vdir.mkdir(exist_ok=True)
    try:
        for _ in range(cfg.updates):
            m = trainer.step()
            u = trainer.update
            with journal.open("a", encoding="utf-8") as h:
                h.write(json.dumps(dict(update=u, **m), allow_nan=False) + "\n")
            log_dict = {
                "opt/loss": m["loss"],
                "opt/policy_loss": m["policy_loss"],
                "opt/value_loss": m["value_loss"],
                "opt/clip_fraction": m["clip_fraction"],
                "opt/approx_kl": m["approx_kl"],
                "opt/grad_norm": m["grad_norm"],
                "opt/entropy": m["entropy"],
                "opt/lr": m["lr"],
                "train/mean_step_reward": m["mean_step_reward"],
                "train/mean_return": m["mean_return"],
                "train/mean_value": m["mean_value"],
            }
            if "mean_score" in m:
                log_dict["train/score_per_episode"] = m["mean_score"]
                log_dict["train/meals_per_100_steps"] = m["meals_per_100_steps"]
            if "selfplay_score" in m:
                log_dict["train/selfplay_score"] = m["selfplay_score"]
            if "pool_score" in m:
                log_dict["train/pool_score"] = m["pool_score"]
            ev_str = ""
            if u % cfg.eval_every == 0 or u == cfg.updates:
                ev_l2 = evaluate_vs_frozen_partners(trainer.model, eval_strong_partner, num_envs=45, episodes=5, seed=10_000_000, device=args.device)
                ev_all = evaluate_shared(trainer.model, mode="all_policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
                ev_sin = evaluate_shared(trainer.model, mode="policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
                (vdir / f"{u:03d}.json").write_text(json.dumps(dict(l2_vs_strong_l1=ev_l2, all_policy=ev_all, single_vs_script=ev_sin), indent=2), encoding="utf-8")
                atomic_save(dict(model=trainer.model.state_dict(), config=meta_cfg, update=u), ppo_dir / f"policy_{u:06d}.pt")
                atomic_save(trainer.state_dict(), ppo_dir / "latest.pt")
                log_dict.update({
                    "eval/all_policy_score": ev_all["team_score"],
                    "eval/l2_vs_frozen_l1_score": ev_l2["team_score"],
                    "eval/l2_role_server": ev_l2["per_role_score"]["server"],
                    "eval/l2_role_main": ev_l2["per_role_score"]["main"],
                    "eval/l2_role_side": ev_l2["per_role_score"]["side"],
                    "eval/l2_joint_attempt_success_rate": ev_l2["joint_attempt_success_rate"],
                    "eval/l2_joint_wait_steps": ev_l2["joint_wait_steps"],
                    "eval/score_per_episode": ev_sin["team_score"],
                })
                ev_str = f" | eval all={ev_all['team_score']:.1f} | vsL1={ev_l2['team_score']:.1f} (S={ev_l2['per_role_score']['server']:.0f}/M={ev_l2['per_role_score']['main']:.0f}/D={ev_l2['per_role_score']['side']:.0f}) | vsRule={ev_sin['team_score']:.1f}"
            run.log(log_dict, step=m["env_steps"])
            (ppo_dir / "status.json").write_text(json.dumps(dict(
                update=u, total_updates=cfg.updates, status="complete" if u == cfg.updates else "running",
                env_steps=m["env_steps"], train_score=m.get("mean_score"),
            ), indent=2), encoding="utf-8")
            sc_str = f"{m['mean_score']:.1f}" if "mean_score" in m else "None"
            sp_str = f" sp={m['selfplay_score']:.1f}" if "selfplay_score" in m else ""
            pl_str = f" pool={m['pool_score']:.1f}" if "pool_score" in m else ""
            print(f"L2 {u:3d}/{cfg.updates} score={sc_str}{sp_str}{pl_str} kl={m['approx_kl']:.5f} gn={m['grad_norm']:.2f} ent={m['entropy']:.3f} os={m['optimizer_steps']} sps={m['steps_per_second']:,.0f}{ev_str}", flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

"""End-to-end BC -> L1 PPO training for GPU SharedKitchen with Trackio logging."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
import trackio

from .env import FLOOR, SharedKitchenConfig, TorchSharedKitchen
from .evaluation import ROLE_NAMES, evaluate_shared, role_slot_map
from .fast_partners import GpuFastPartners
from .gpu_partners import GpuSharedPartners
from .history import OnlineHistory, RolloutHistory
from .policy import V3Policy, partner_action_ids
from .pretrain import SharedBCSettings, atomic_save, collect_custom_expert, run_pretraining

PROJECT = "hucooked-shared-l1"
RUNS = Path(__file__).resolve().parent / "runs"


def scheduled_lr(base, step, total, warmup_frac=0.05, start_factor=0.05, min_factor=0.05):
    warmup = max(1, math.ceil(warmup_frac * total)) if warmup_frac > 0 else 0
    if warmup > 1 and step < warmup:
        return base * (start_factor + (1.0 - start_factor) * (step / (warmup - 1)))
    remain = max(1, total - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / remain))
    cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base * (min_factor + (1.0 - min_factor) * cosine)


@dataclass(frozen=True)
class L1Settings:
    seed: int = 1
    num_envs: int = 4032
    rollout_steps: int = 150
    horizon: int = 300
    updates: int = 60
    epochs: int = 4
    minibatch_size: int = 3024
    lr: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coef: float = 0.015
    target_kl: float = 0.02
    max_grad_norm: float = 2.0
    warmup_fraction: float = 0.1
    warmup_start_factor: float = 0.05
    bc_start: float = 0.05
    bc_end: float = 0.0
    bc_decay_updates: int = 30
    bc_batch_size: int = 256
    stagger_history: bool = True
    eval_every: int = 2
    critic_warmup_updates: int = 2
    shaping_scale: float = 0.0
    selfplay_frac: float = 0.0
    partner_noise: float = 0.0
    partner_noise_end: float | None = None
    random_spawn_prob: float = 0.0
    use_history: bool = True
    partner_type: str = "shared"


class SharedL1Trainer:
    def __init__(self, cfg: L1Settings, device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env = TorchSharedKitchen(
            SharedKitchenConfig(horizon=cfg.horizon, shaping_scale=cfg.shaping_scale, shaping_gamma=cfg.gamma),
            num_envs=cfg.num_envs, device=device, use_cuda_graph=True)
        self._floor_cells = torch.tensor(sorted(FLOOR), dtype=torch.long, device=self.device)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        self._maybe_randomize_spawns()
        self.roles = torch.arange(cfg.num_envs, dtype=torch.long, device=self.device) % 3
        partner_cls = GpuFastPartners if cfg.partner_type == "fast" else GpuSharedPartners
        self.partners = partner_cls(
            self.env, roles=self.roles,
            learner_mask=torch.ones(cfg.num_envs, dtype=torch.bool, device=self.device),
            use_cuda_graph=True)
        self.partners.reset()
        self.n_sp = int(round(cfg.num_envs * cfg.selfplay_frac))
        self.n_scr = cfg.num_envs - self.n_sp
        self.sp_lanes = torch.arange(self.n_scr, cfg.num_envs, dtype=torch.long, device=self.device)
        if self.n_sp > 0:
            self.sp_peer_roles = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long, device=self.device)[self.roles[self.sp_lanes]]
            self.sp_peer_histories = [OnlineHistory(self.n_sp, device, stagger=cfg.stagger_history) for _ in range(2)]
            self._refresh_sp_slots()
        else:
            self.sp_peer_histories = []
        self.history = OnlineHistory(cfg.num_envs, device, stagger=cfg.stagger_history)
        self.model = V3Policy(3, use_history=cfg.use_history).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        self.tick = 0
        self.update = 0
        self.optimizer_step_count = 0
        self.demonstrations = None
        self.episode_score_acc = torch.zeros((), dtype=torch.float64, device=self.device)

    def _maybe_randomize_spawns(self):
        if self.cfg.random_spawn_prob <= 0.0:
            return
        n = self.cfg.num_envs
        mask = torch.rand((n,), device=self.device) < self.cfg.random_spawn_prob
        if not bool(mask.any()):
            return
        # Sample 3 distinct random floor tiles per selected env (vectorized on GPU).
        rand_idx = torch.rand((n, len(self._floor_cells)), device=self.device).argsort(dim=1)[:, :3]
        rand_pos = self._floor_cells[rand_idx]
        self.env.state["pos"].copy_(torch.where(mask[:, None], rand_pos, self.env.state["pos"]))
        self.env._write_observations()
        self.obs = self.env._obs_out.clone()

    def _refresh_sp_slots(self):
        if self.n_sp <= 0:
            return
        # Use partners.chef_ids ([N, 3] = [server, main, side]), which works for both standard and random spawns.
        r2s = self.partners.chef_ids
        self.sp_p0_slot = r2s[self.sp_lanes, self.sp_peer_roles[:, 0]]
        self.sp_p1_slot = r2s[self.sp_lanes, self.sp_peer_roles[:, 1]]

    def current_partner_noise(self):
        cfg = self.cfg
        end = cfg.partner_noise if cfg.partner_noise_end is None else cfg.partner_noise_end
        if cfg.updates <= 1:
            return end
        frac = min(1.0, self.update / max(1, cfg.updates - 1))
        return cfg.partner_noise + (end - cfg.partner_noise) * frac

    def current_bc_coef(self):
        cfg = self.cfg
        if cfg.bc_decay_updates <= 0:
            return cfg.bc_end
        frac = min(1.0, self.update / cfg.bc_decay_updates)
        return cfg.bc_start + (cfg.bc_end - cfg.bc_start) * frac

    def state_dict(self):
        return dict(
            config=asdict(self.cfg),
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            update=self.update,
            tick=self.tick,
            optimizer_step_count=self.optimizer_step_count,
            history=self.history.state_dict(),
        )

    @torch.no_grad()
    def collect(self):
        cfg = self.cfg
        bank = RolloutHistory(self.history, cfg.rollout_steps)
        self.history.refresh(self.model)
        for hp in self.sp_peer_histories:
            hp.refresh(self.model)
        lanes = self.env.lanes
        actions, logprobs, values, rewards, terminals = [], [], [], [], []
        completed_score = 0.0
        completed_scr_score = 0.0
        completed_sp_score = 0.0
        completed_meals = 0.0
        completed_eps = 0

        for _ in range(cfg.rollout_steps):
            prev_obs = self.obs.clone()
            learner = self.partners.learner_ids
            cur = prev_obs[lanes, learner]
            logp, val, _ = self.history.forward(self.model, cur)
            choice = logp.exp().multinomial(1)
            joint = self.partners.actions()
            joint[lanes, learner] = choice[:, 0]
            if self.n_sp > 0:
                cur_p0 = prev_obs[self.sp_lanes, self.sp_p0_slot]
                cur_p1 = prev_obs[self.sp_lanes, self.sp_p1_slot]
                # Use greedy peers in self-play kitchens; only the learner samples multinomial,
                # while partner stochasticity is explicitly controlled by `partner_noise`.
                act_p0 = self.sp_peer_histories[0].forward(self.model, cur_p0)[0].argmax(-1)
                act_p1 = self.sp_peer_histories[1].forward(self.model, cur_p1)[0].argmax(-1)
                joint[self.sp_lanes, self.sp_p0_slot] = act_p0
                joint[self.sp_lanes, self.sp_p1_slot] = act_p1
            p_noise = self.current_partner_noise()
            if p_noise > 0.0:
                # Perturb partner chefs (never the learner) so the learner learns OOD recovery & coordination.
                noise_mask = (torch.rand((cfg.num_envs, 3), device=self.device) < p_noise)
                noise_mask[lanes, learner] = False
                rand_acts = torch.randint(0, 6, (cfg.num_envs, 3), device=self.device)
                joint = torch.where(noise_mask, rand_acts, joint)
            self.obs, reward, _, _, _ = self.env.step(joint)
            pids = partner_action_ids(joint)
            peers = pids[lanes, learner]
            bank.record(cur, peers)
            self.history.append(cur, peers, self.model)
            if self.n_sp > 0:
                self.sp_peer_histories[0].append(cur_p0, pids[self.sp_lanes, self.sp_p0_slot], self.model)
                self.sp_peer_histories[1].append(cur_p1, pids[self.sp_lanes, self.sp_p1_slot], self.model)
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
                if self.n_sp > 0:
                    sc = self.env.state["score"]
                    completed_scr_score += sc[:self.n_scr].mean().item() if self.n_scr > 0 else 0.0
                    completed_sp_score += sc[self.n_scr:].mean().item()
                completed_meals += m["meals_per_100_steps"]
                completed_eps += 1
                self.history.finish_episode()
                bank.finish_episode()
                for hp in self.sp_peer_histories:
                    hp.finish_episode()
                self.obs, _ = self.env.reset()
                self._maybe_randomize_spawns()
                self.partners.reset()
                self._refresh_sp_slots()
                self.tick = 0

        bank.seal()
        cur = self.obs[lanes, self.partners.learner_ids]
        next_v = self.history.forward(self.model, cur)[1][:, 0]
        vals = torch.stack(values)
        advs = torch.zeros_like(vals)
        carry = torch.zeros_like(next_v)
        for t in reversed(range(cfg.rollout_steps)):
            disc = 0.0 if terminals[t] else cfg.gamma
            delta = rewards[t] + disc * next_v - vals[t]
            carry = delta + disc * cfg.gae_lambda * carry
            advs[t] = carry
            next_v = vals[t]
        rets = (advs + vals).flatten()
        # Per-role advantage normalization so a fast-learning role (e.g. main) never drowns out other roles.
        norm_advs = torch.empty_like(advs)
        for r in range(3):
            rmask = self.roles == r
            if bool(rmask.any()):
                sub = advs[:, rmask]
                norm_advs[:, rmask] = (sub - sub.mean()) / (sub.std(unbiased=False) + 1e-8)
        data = dict(
            action=torch.cat(actions),
            logp=torch.cat(logprobs),
            returns=rets,
            old_values=vals.flatten(),
            adv=norm_advs.flatten(),
        )
        metrics = dict(
            completed_batches=completed_eps,
            mean_step_reward=torch.stack(rewards).mean().item(),
            mean_return=rets.mean().item(),
            mean_value=vals.mean().item(),
            partner_noise=self.current_partner_noise(),
        )
        if completed_eps:
            metrics["mean_score"] = completed_score / completed_eps
            metrics["meals_per_100_steps"] = completed_meals / completed_eps
            if self.n_sp > 0:
                metrics["script_score"] = completed_scr_score / completed_eps
                metrics["selfplay_score"] = completed_sp_score / completed_eps
        return bank, data, metrics

    def warmup_critic(self):
        """Fit Critic head on BC rollouts before PPO so initial value gradients don't wreck shared encoders."""
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
                print(f"Critic warmup {w + 1}/{cfg.critic_warmup_updates} | vloss={torch.stack(vlosses).mean().item():.2f} mean_ret={m['mean_return']:.1f} mean_v={m['mean_value']:.1f}", flush=True)
        finally:
            for p in self.model.parameters():
                p.requires_grad_(True)

    def optimize(self, bank, data):
        cfg = self.cfg
        batch = len(data["action"])
        bc_coef = self.current_bc_coef()
        steps_per_update = cfg.epochs * math.ceil(batch / cfg.minibatch_size)
        total_steps = cfg.updates * steps_per_update
        metrics = []
        bc_vals = []
        stopped = False

        lr = scheduled_lr(cfg.lr, self.update, cfg.updates,
                          cfg.warmup_fraction, cfg.warmup_start_factor)
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
                bc_loss = loss.new_zeros(())
                if bc_coef > 0 and self.demonstrations is not None:
                    ebank, edata = self.demonstrations
                    eidx = torch.randint(len(edata["action"]), (cfg.bc_batch_size,), device=self.device)
                    elogp, *_ = ebank.replay(self.model, eidx)
                    bc_loss = F.nll_loss(elogp, edata["action"][eidx])
                    loss = loss + bc_coef * bc_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_step_count += 1
                with torch.no_grad():
                    kl = (ratio - 1 - logratio).mean()
                    cf = (ratio != clipped).float().mean()
                    metrics.append(torch.stack((loss, actor, critic, entropy, kl, cf, norm)).detach())
                    bc_vals.append(bc_loss.detach())
            if stopped:
                break

        names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
        res = dict(zip(names, torch.stack(metrics).mean(0).tolist()))
        res.update(bc_loss=torch.stack(bc_vals).mean().item(), bc_coef=bc_coef,
                   optimizer_steps=len(metrics), kl_early_stop=int(stopped))
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


@torch.no_grad()
def evaluate_noisy_all_policy(model, noise=0.05, num_envs=45, episodes=3, horizon=300, seed=10_000_000, device="cuda"):
    """Stress-test 3-chef all-policy when 1 partner has `noise` random action perturbation."""
    was_training = model.training
    model.eval()
    env = TorchSharedKitchen(SharedKitchenConfig(horizon=horizon, shuffle_players=True),
                             num_envs=num_envs, device=device, use_cuda_graph=True)
    hs = [OnlineHistory(num_envs, device, stagger=False) for _ in range(3)]
    g = torch.Generator(device=device).manual_seed(seed)
    scores = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep * 1000)
            r2s = role_slot_map(env)
            for h in hs:
                h.refresh(model)
            for _ in range(horizon):
                joint = torch.empty((num_envs, 3), dtype=torch.long, device=device)
                cur_by_role = []
                for r in range(3):
                    sl = r2s[:, r]
                    cur_r = obs[env.lanes, sl]
                    cur_by_role.append((sl, cur_r))
                    act = hs[r].forward(model, cur_r)[0].argmax(-1)
                    if noise > 0 and r == 2:
                        flip = torch.rand(num_envs, device=device, generator=g) < noise
                        rand_a = torch.randint(0, 6, (num_envs,), device=device, generator=g)
                        act = torch.where(flip, rand_a, act)
                    joint[env.lanes, sl] = act
                pids = partner_action_ids(joint)
                for r in range(3):
                    sl, cur_r = cur_by_role[r]
                    hs[r].append(cur_r, pids[env.lanes, sl], model)
                obs, *_ = env.step(joint)
            scores.append(env.episode_metrics()["team_score"])
            for h in hs:
                h.finish_episode()
    finally:
        model.train(was_training)
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_random_spawn_all_policy(model, num_envs=45, episodes=3, horizon=300, seed=10_000_000, device="cuda"):
    """Evaluate 3-chef all-policy when all 3 chefs spawn at random distinct floor cells."""
    was_training = model.training
    model.eval()
    env = TorchSharedKitchen(SharedKitchenConfig(horizon=horizon, shuffle_players=True),
                             num_envs=num_envs, device=device, use_cuda_graph=True)
    floor_cells = torch.tensor(sorted(FLOOR), dtype=torch.long, device=device)
    hs = [OnlineHistory(num_envs, device, stagger=False) for _ in range(3)]
    g = torch.Generator(device=device).manual_seed(seed)
    scores = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep * 1000)
            rand_idx = torch.rand((num_envs, len(floor_cells)), device=device, generator=g).argsort(dim=1)[:, :3]
            env.state["pos"].copy_(floor_cells[rand_idx])
            env._write_observations()
            obs = env._obs_out.clone()
            for h in hs:
                h.refresh(model)
            for _ in range(horizon):
                prev = obs.clone()
                joint = torch.stack([hs[c].forward(model, prev[:, c])[0].argmax(-1) for c in range(3)], dim=1)
                obs, *_ = env.step(joint)
                pids = partner_action_ids(joint)
                for c in range(3):
                    hs[c].append(prev[:, c], pids[:, c], model)
            scores.append(env.episode_metrics()["team_score"])
            for h in hs:
                h.finish_episode()
    finally:
        model.train(was_training)
    return sum(scores) / len(scores)


def main():
    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", default="shared-gpu-bc-l1-4032x150-seed1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=4032)
    p.add_argument("--rollout-steps", type=int, default=150)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=3024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--target-kl", type=float, default=0.02, help="KL early stop threshold; <=0 disables early stop.")
    p.add_argument("--shaping-scale", type=float, default=0.0, help="Potential-based reward shaping scale (0=pure sparse).")
    p.add_argument("--selfplay-frac", type=float, default=0.0, help="Fraction of kitchens using 3-chef all-policy self-play.")
    p.add_argument("--partner-noise", type=float, default=0.0, help="Initial random action perturbation probability on partner chefs.")
    p.add_argument("--partner-noise-end", type=float, default=None, help="Final partner noise (linearly ramped over updates).")
    p.add_argument("--random-spawn-prob", type=float, default=0.0, help="Probability of randomizing all 3 chef spawn cells on reset.")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--no-bc", action="store_true", help="Train L1 PPO from scratch without BC init or BC regularization.")
    p.add_argument("--scratch-init", action="store_true", help="Start from random weights (no BC checkpoint) while still allowing light BC aux loss.")
    p.add_argument("--bc-start", type=float, default=0.05)
    p.add_argument("--bc-end", type=float, default=0.0)
    p.add_argument("--bc-decay-updates", type=int, default=30)
    p.add_argument("--bc-policy", type=Path, default=None)
    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--critic-warmup", type=int, default=2)
    p.add_argument("--no-history", action="store_true", help="Disable history encoder completely (neither used nor trained).")
    p.add_argument("--partner-type", choices=("shared", "fast"), default="shared", help="Rule partner controller: shared (19 meals) or fast (21 meals).")
    args = p.parse_args()

    root = RUNS / args.campaign
    root.mkdir(parents=True, exist_ok=True)
    bc_dir = root / "bc"
    ppo_dir = root / "ppo"
    ppo_dir.mkdir(parents=True, exist_ok=True)

    # 1. BC Pretraining on custom GpuSharedPartners (unless --no-bc or --scratch-init)
    bc_ckpt = None
    if not args.no_bc and not args.scratch_init:
        bc_ckpt = args.bc_policy if args.bc_policy is not None else (bc_dir / "bc_best.pt")
        if not bc_ckpt.exists():
            print("=== Stage 1: BC Pretraining ===", flush=True)
            bc_cfg = SharedBCSettings(seed=args.seed, epochs=args.bc_epochs)
            run_pretraining(bc_cfg, bc_dir, device=args.device)
        else:
            print(f"Reusing existing BC checkpoint: {bc_ckpt}", flush=True)
    else:
        mode_tag = "no BC" if args.no_bc else f"light BC aux ({args.bc_start}->{args.bc_end} over {args.bc_decay_updates} upd)"
        print(f"=== From-scratch L1 PPO ({mode_tag}, shaping_scale={args.shaping_scale}) ===", flush=True)

    # 2. L1 PPO Training
    bc_start = 0.0 if args.no_bc else args.bc_start
    bc_end = 0.0 if args.no_bc else args.bc_end
    bc_decay = 0 if args.no_bc else args.bc_decay_updates
    cfg = L1Settings(
        seed=args.seed, num_envs=args.num_envs, rollout_steps=args.rollout_steps,
        updates=args.updates, epochs=args.epochs, minibatch_size=args.minibatch_size,
        lr=args.lr, gamma=args.gamma, entropy_coef=args.entropy_coef,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        eval_every=args.eval_every,
        shaping_scale=args.shaping_scale,
        selfplay_frac=args.selfplay_frac,
        partner_noise=args.partner_noise,
        partner_noise_end=args.partner_noise_end,
        random_spawn_prob=args.random_spawn_prob,
        bc_start=bc_start,
        bc_end=bc_end,
        bc_decay_updates=bc_decay,
        critic_warmup_updates=0 if (args.no_bc or args.scratch_init) else args.critic_warmup,
        use_history=not args.no_history,
        partner_type=args.partner_type,
    )
    (ppo_dir / "experiment.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # Open Trackio run in the default Trackio directory (matches port 7865 dashboard)
    run = trackio.init(project=PROJECT, name=f"{args.campaign}-l1-seed{args.seed}",
                       group="shared-l1-scratch" if (args.no_bc or args.scratch_init) else "shared-l1",
                       config=asdict(cfg), resume="allow")

    # Log BC curve into Trackio as well
    bc_metrics_path = bc_dir / "bc_metrics.json"
    if (bc_ckpt is not None) and bc_metrics_path.exists():
        for rec in json.loads(bc_metrics_path.read_text(encoding="utf-8")):
            run.log({"bc/val_nll": rec["validation"]["nll"],
                     "bc/val_accuracy": rec["validation"]["action_accuracy"]},
                    step=rec["epoch"])

    trainer = SharedL1Trainer(cfg, device=args.device)
    if bc_ckpt is not None:
        trainer.model.load_state_dict(torch.load(bc_ckpt, map_location=args.device, weights_only=False)["model"])
    if bc_start > 0:
        # Attach fresh demonstrations for annealed BC regularization
        demo_bank, demo_data, _ = collect_custom_expert(63, 5, cfg.horizon, 40_000_000, device=args.device, partner_type=cfg.partner_type)
        trainer.demonstrations = (demo_bank, demo_data)
    if bc_ckpt is not None:
        trainer.warmup_critic()

    # Initial eval (step 0)
    ev0 = evaluate_shared(trainer.model, mode="policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
    ev0_all = evaluate_shared(trainer.model, mode="all_policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
    (ppo_dir / "validation-bc.json").write_text(json.dumps(dict(single=ev0, all_policy=ev0_all), indent=2), encoding="utf-8")
    print(f"BC init eval | single={ev0['team_score']:.1f} (roles={ev0['per_role_score']}) j_succ={ev0['joint_attempt_success_rate']:.3f} | all_policy={ev0_all['team_score']:.1f}", flush=True)
    run.log({
        "eval/score_per_episode": ev0["team_score"],
        "eval/meals_per_100_steps": ev0["meals_per_100_steps"],
        "eval/joint_attempt_success_rate": ev0["joint_attempt_success_rate"],
        "eval/joint_wait_steps": ev0["joint_wait_steps"],
        "eval/three_contributor_meals": ev0["three_contributor_meals"],
        "eval/role_server": ev0["per_role_score"]["server"],
        "eval/role_main": ev0["per_role_score"]["main"],
        "eval/role_side": ev0["per_role_score"]["side"],
        "eval/all_policy_score": ev0_all["team_score"],
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
                "opt/bc_loss": m["bc_loss"],
                "opt/clip_fraction": m["clip_fraction"],
                "opt/approx_kl": m["approx_kl"],
                "opt/grad_norm": m["grad_norm"],
                "opt/entropy": m["entropy"],
                "opt/lr": m["lr"],
                "opt/bc_coef": m["bc_coef"],
                "opt/partner_noise": m["partner_noise"],
                "train/mean_step_reward": m["mean_step_reward"],
                "train/mean_return": m["mean_return"],
                "train/mean_value": m["mean_value"],
            }
            if "mean_score" in m:
                log_dict["train/score_per_episode"] = m["mean_score"]
                log_dict["train/meals_per_100_steps"] = m["meals_per_100_steps"]
            if "selfplay_score" in m:
                log_dict["train/script_score"] = m["script_score"]
                log_dict["train/selfplay_score"] = m["selfplay_score"]
            ev_str = ""
            if u % cfg.eval_every == 0 or u == cfg.updates:
                ev = evaluate_shared(trainer.model, mode="policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
                ev_all = evaluate_shared(trainer.model, mode="all_policy", num_envs=45, episodes=5, seed=10_000_000, device=args.device, partner_type=cfg.partner_type)
                has_noise = cfg.partner_noise > 0 or (cfg.partner_noise_end is not None and cfg.partner_noise_end > 0)
                noisy5 = evaluate_noisy_all_policy(trainer.model, noise=0.05, num_envs=45, episodes=3, seed=10_000_000, device=args.device) if has_noise else None
                rspawn = evaluate_random_spawn_all_policy(trainer.model, num_envs=45, episodes=3, seed=10_000_000, device=args.device) if cfg.random_spawn_prob > 0 else None
                (vdir / f"{u:03d}.json").write_text(json.dumps(dict(single=ev, all_policy=ev_all, noisy5_all=noisy5, random_spawn_all=rspawn), indent=2), encoding="utf-8")
                atomic_save(dict(model=trainer.model.state_dict(), config=asdict(cfg), update=u), ppo_dir / f"policy_{u:06d}.pt")
                atomic_save(trainer.state_dict(), ppo_dir / "latest.pt")
                log_dict.update({
                    "eval/score_per_episode": ev["team_score"],
                    "eval/meals_per_100_steps": ev["meals_per_100_steps"],
                    "eval/joint_attempt_success_rate": ev["joint_attempt_success_rate"],
                    "eval/joint_wait_steps": ev["joint_wait_steps"],
                    "eval/three_contributor_meals": ev["three_contributor_meals"],
                    "eval/role_server": ev["per_role_score"]["server"],
                    "eval/role_main": ev["per_role_score"]["main"],
                    "eval/role_side": ev["per_role_score"]["side"],
                    "eval/all_policy_score": ev_all["team_score"],
                })
                if noisy5 is not None:
                    log_dict["eval/noisy5_all_policy_score"] = noisy5
                if rspawn is not None:
                    log_dict["eval/random_spawn_all_policy_score"] = rspawn
                n_str = f" n5={noisy5:.1f}" if noisy5 is not None else ""
                rs_str = f" rsp={rspawn:.1f}" if rspawn is not None else ""
                ev_str = f" | eval={ev['team_score']:.1f} (S={ev['per_role_score']['server']:.0f}/M={ev['per_role_score']['main']:.0f}/D={ev['per_role_score']['side']:.0f}) all={ev_all['team_score']:.1f}{n_str}{rs_str}"
            run.log(log_dict, step=m["env_steps"])
            (ppo_dir / "status.json").write_text(json.dumps(dict(
                update=u, total_updates=cfg.updates, status="complete" if u == cfg.updates else "running",
                env_steps=m["env_steps"], train_score=m.get("mean_score"),
            ), indent=2), encoding="utf-8")
            sc_str = f"{m['mean_score']:.1f}" if "mean_score" in m else "None"
            print(f"L1 {u:2d}/{cfg.updates} score={sc_str} kl={m['approx_kl']:.5f} gn={m['grad_norm']:.2f} ent={m['entropy']:.3f} os={m['optimizer_steps']} sps={m['steps_per_second']:,.0f}{ev_str}", flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

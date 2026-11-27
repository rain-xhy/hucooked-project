"""Twenty-million-parameter E3T on SharedKitchen. Model selection uses development games only."""

import argparse
import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .shared_train import (
    ENGINE,
    INPUT_SIZE,
    Model,
    Pool,
    Training,
    load,
    load_weights,
    log_record,
    rewind_progress,
    save_json,
)
from .shared_work import SharedWorkEnv

ARCH = "shared-work-residual20m-v1"
MODES = ("sample", "cool", "greedy")


@dataclass(frozen=True)
class LargeConfig:
    history: int = 5
    seed: int = 42
    learning_rate: float = 0.00012
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip: float = 0.12
    grad_clip: float = 0.7
    target_kl: float = 0.025
    epochs: int = 3
    minibatch: int = 512
    reward_scale: float = 0.05
    shaping_steps: int = 2_000_000
    explore_steps: int = 2_000_000
    epsilon_start: float = 0.16
    epsilon_end: float = 0.025
    temperature_start: float = 1.25
    temperature_end: float = 0.9
    entropy_start: float = 0.025
    entropy_end: float = 0.006

    @classmethod
    def from_checkpoint(cls, settings):
        # The archived configuration also stored two inactive warm-up options.
        values = dict(settings)
        values.pop("guided_warmup_steps", None)
        values.pop("guided_lr_factor", None)
        return cls(**values)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.ff = nn.Sequential(nn.Linear(1024, 2240), nn.SiLU(), nn.Linear(2240, 1024))

    def forward(self, x):
        return x + 0.5 * self.ff(self.norm(x))


class ActionPolicy(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.input = nn.Linear(INPUT_SIZE + 2 * actions, 1024)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(4)), nn.LayerNorm(1024))
        self.actor = nn.Linear(1024, actions)
        self.critic = nn.Sequential(nn.Linear(1024, 128), nn.SiLU(), nn.Linear(128, 1))
        with torch.no_grad():
            self.actor.weight.mul_(0.1)
            self.actor.bias.zero_()

    def forward(self, obs, condition, mask):
        x = self.blocks(self.input(torch.cat((obs, condition.flatten(-2)), -1)))
        return self.actor(x).masked_fill(~mask, -1.0e9), self.critic(x).squeeze(-1)


class LargePredictor(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.input = nn.Sequential(nn.Linear(INPUT_SIZE + 2 * actions + 1, 256), nn.SiLU())
        self.gru = nn.GRU(256, 256, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(INPUT_SIZE + 256, 652),
            nn.LayerNorm(652),
            nn.SiLU(),
            nn.Linear(652, 2 * actions),
        )

    def forward(self, obs, history):
        encoded = self.input(history) * history[..., -1:]
        _, memory = self.gru(encoded)
        return self.head(torch.cat((obs, memory[-1]), -1)).unflatten(-1, (2, self.actions))


@contextmanager
def math_mode(device, fast):
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = fast
            torch.backends.cudnn.allow_tf32 = fast
        yield
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class LargeModel(Model):
    env_class = SharedWorkEnv

    def __init__(self, config, device):
        self.config, self.device = config, torch.device(device)
        self.actions = SharedWorkEnv.action_count
        self.policy = ActionPolicy(self.actions).to(device)
        self.predictor = LargePredictor(self.actions).to(device)

    def evaluate(self, *args, **kwargs):
        with math_mode(self.device, fast=False):
            return super().evaluate(*args, **kwargs)


class LargeTraining(Training, LargeModel):
    def __init__(self, config, device, envs, workers):
        LargeModel.__init__(self, config, device)
        self.opts = self.optimizers(config.learning_rate)
        self.pool = Pool(envs, workers, self.env_class)
        self.ids = np.arange(envs)
        self.ids_t = self.tensor(self.ids)
        self.steps = self.episode = self.update_num = 0
        self.best = -float("inf")
        self.run_metadata = {"architecture": ARCH}
        self.selection = None
        self.reset()

    def optimizers(self, lr):
        return [
            torch.optim.Adam(
                net.parameters(),
                lr=lr,
                eps=1e-5,
                fused=True if self.device.type == "cuda" else None,
            )
            for net in (self.policy, self.predictor)
        ]

    def roles(self):
        super().roles()
        # Stable four-cohort partner behaviours and mild ego-temperature variation per episode.
        c = self.tensor((self.ids + self.episode) % 4)
        self.ego_scale = self.tensor([0.85, 0.95, 1.05, 1.15])[c]
        self.partner_scale = self.tensor([0.4, 0.8, 1.2, 1.6])[c]
        self.partner_temp = self.tensor([0.7, 0.9, 1.1, 1.35])[c]

    def schedule(self, start, end):
        ratio = min(self.steps / max(1, self.config.explore_steps), 1.0)
        return start + (end - start) * ratio

    def behaviour(self, logits, masks):
        c = self.config
        temp = self.schedule(c.temperature_start, c.temperature_end)
        ego_temp = temp * self.ego_scale
        temperatures = (temp * self.partner_temp)[:, None].expand(-1, 3).clone()
        temperatures[self.ids_t, self.ego_t] = ego_temp
        z = logits / temperatures[..., None]
        policy_probs = z.softmax(-1)
        eps = (self.schedule(c.epsilon_start, c.epsilon_end) * self.partner_scale).clamp(0, 1)
        uniform = masks / masks.sum(-1, keepdim=True)
        mixed = torch.lerp(policy_probs, uniform, eps[:, None, None])
        mixed[self.ids_t, self.ego_t] = policy_probs[self.ids_t, self.ego_t]
        return z, mixed, {"temperature": ego_temp}

    def optimize(self, b):
        cfg = self.config
        delta = {}
        lr = cfg.learning_rate
        for opt in self.opts:
            for group in opt.param_groups:
                group["lr"] = lr
        first = self.update_num == 0
        if first:
            with torch.no_grad():
                reference = [
                    next(net.parameters()).detach().clone() for net in (self.policy, self.predictor)
                ]
        adv = (b["adv"] - b["adv"].mean()) / b["adv"].std(unbiased=False).clamp_min(1.0e-8)
        ent_coef = self.schedule(cfg.entropy_start, cfg.entropy_end)
        all_stats = []
        for epoch in range(cfg.epochs):
            epoch_kl = []
            for ix in torch.randperm(len(adv), device=self.device).split(cfg.minibatch):
                logits, value = self.policy(b["obs"][ix], b["cond"][ix], b["masks"][ix])
                lp = (logits / b["temperature"][ix, None]).log_softmax(-1)
                logp = lp.gather(1, b["act"][ix, None]).squeeze(1)
                logratio = logp - b["logp"][ix]
                if first and not all_stats:
                    # Tolerate small BLAS batch-size/TF32 accumulation differences.
                    torch.testing.assert_close(logp, b["logp"][ix], rtol=0.002, atol=0.002)
                ratio = logratio.exp()
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                actor_loss = -torch.minimum(ratio * adv[ix], clipped * adv[ix]).mean()
                entropy = -(lp.exp() * lp).sum(-1).mean()
                loss = actor_loss + 0.5 * F.smooth_l1_loss(value, b["ret"][ix]) - ent_coef * entropy
                pred = self.predictor(b["obs"][ix], b["hist"][ix])
                ce = F.cross_entropy(pred.flatten(0, 1), b["targets"][ix].flatten())
                for opt in self.opts:
                    opt.zero_grad(set_to_none=True)
                (loss + ce).backward()
                if first and not all_stats:
                    for key, net in (("policy", self.policy), ("predictor", self.predictor)):
                        params = list(net.parameters())
                        if any(p.grad is None for p in params):
                            raise AssertionError(f"Disconnected parameters in {key}")
                        nonzero = (
                            torch.stack([p.grad.detach().abs().amax() for p in params]).gt(0).sum()
                        )
                        delta[f"{key}_grad_tensors"] = int(nonzero.item())
                        delta[f"{key}_parameter_tensors"] = len(params)
                norms = [
                    nn.utils.clip_grad_norm_(
                        net.parameters(), cfg.grad_clip, error_if_nonfinite=True
                    )
                    for net in (self.policy, self.predictor)
                ]
                for opt in self.opts:
                    opt.step()
                kl = ((ratio - 1) - logratio).detach().mean()
                epoch_kl.append(kl)
                all_stats.append(
                    torch.stack([loss.detach(), ce.detach(), *norms, entropy.detach(), kl])
                )
            if torch.stack(epoch_kl).mean().item() > cfg.target_kl:
                break
        stats = torch.stack(all_stats)
        if not torch.isfinite(stats).all():
            raise FloatingPointError("Nonfinite PPO statistics")
        if first:
            for old, key, net in zip(
                reference, ("policy", "predictor"), (self.policy, self.predictor)
            ):
                value = (next(net.parameters()).detach() - old).abs().amax().item()
                if value <= 0:
                    raise AssertionError(f"{key} failed to step")
                delta[key + "_first_update_max_delta"] = value
        self.update_num += 1
        return {
            **dict(
                zip(
                    (
                        "ppo_loss",
                        "predictor_ce",
                        "ppo_grad_norm",
                        "predictor_grad_norm",
                        "entropy",
                        "approx_kl",
                    ),
                    stats.mean(0).tolist(),
                )
            ),
            "entropy_coef": ent_coef,
            "ppo_epochs": epoch + 1,
            "ppo_learning_rate": lr,
            "partner_epsilon": self.schedule(cfg.epsilon_start, cfg.epsilon_end),
            "ego_temperature": b["temperature"].mean().item(),
            **delta,
        }

    def weights(self, path):
        temp = path.with_suffix(".tmp")
        torch.save(
            {
                "architecture": ARCH,
                "engine": ENGINE,
                "config": asdict(self.config),
                "actions": self.actions,
                "policy": self.policy.state_dict(),
                "predictor": self.predictor.state_dict(),
                "steps": self.steps,
            },
            temp,
        )
        os.replace(temp, path)


def link_checkpoint(src, dst):
    if dst.is_file() and src.samefile(dst):
        return
    tmp = dst.with_suffix(".link-tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def publish_checkpoint(tr, source, folder, milestone):
    """Complete publications from the committed checkpoint. Safe to repeat after an interruption."""
    selection = tr.selection
    if selection is not None and selection["selected_step"] == tr.steps:
        best_weights = folder / "best.pt"
        best_resume = folder / "best-resume.pt"
        selection_file = folder / "selection.json"
        if not (
            best_weights.is_file()
            and best_resume.is_file()
            and source.samefile(best_resume)
            and selection_file.is_file()
            and json.loads(selection_file.read_text()) == selection
        ):
            tr.weights(best_weights)
            link_checkpoint(source, best_resume)
            save_json(selection_file, selection)
    if milestone:
        target = folder / f"milestone-{tr.steps:09d}.pt"
        if not target.is_file():
            if selection is not None and selection["selected_step"] == tr.steps:
                link_checkpoint(folder / "best.pt", target)
            else:
                tr.weights(target)


def resume_outputs(tr, source, folder, milestone):
    """Restore earlier best artifacts when continuing a run in another directory."""
    recorded = source.parent / "selection.json"
    previous = json.loads(recorded.read_text()) if recorded.is_file() else None
    # Historical A100 checkpoints stored the score in the checkpoint and the mode in a sidecar.
    if (
        tr.selection is None
        and previous is not None
        and previous["selected_step"] <= tr.steps
        and previous["team_score"] == tr.best
    ):
        tr.selection = previous
    if tr.selection is not None and tr.selection["selected_step"] < tr.steps:
        if previous != tr.selection:
            raise ValueError("The earlier best checkpoint needs its matching selection.json")
        for name in ("best.pt", "best-resume.pt"):
            if not (source.parent / name).is_file():
                raise FileNotFoundError(
                    f"Missing earlier selection artifact: {source.parent / name}"
                )
        if source.parent.resolve() != folder.resolve():
            for name in ("best.pt", "best-resume.pt"):
                link_checkpoint(source.parent / name, folder / name)
            save_json(folder / "selection.json", tr.selection)
    publish_checkpoint(tr, source, folder, milestone)


def train(args):
    c = LargeConfig(**{k: getattr(args, k) for k in LargeConfig.__dataclass_fields__})
    if (
        min(args.steps, args.envs, c.epochs, c.minibatch, args.rollout, args.eval_every) <= 0
        or args.steps % args.envs
    ):
        raise ValueError("Invalid step/batch count; steps must be divisible by envs")
    if min(args.workers, args.eval_workers, args.eval_episodes, args.milestone_every) < 0:
        raise ValueError("Worker, validation episode and milestone counts must be nonnegative")
    modes = args.eval_modes.split(",")
    if any(m not in MODES for m in modes):
        raise ValueError("Unsupported development mode")
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / "latest.pt"
    if latest.is_file() and (not args.resume or not args.resume.samefile(latest)):
        raise FileExistsError("Continue this output with --resume latest.pt, or use a new output")
    if not args.resume:
        if any(
            (folder / name).is_file() for name in ("best.pt", "best-resume.pt", "selection.json")
        ):
            raise FileExistsError("The output already contains a selected checkpoint; use --resume")
        # An interrupted first update has no committed state yet; restart with the same seed.
        rewind_progress(folder, -1)
    torch.manual_seed(c.seed)
    tr = LargeTraining(c, args.device, args.envs, args.workers)
    try:
        if args.resume:
            d = load(args.resume)
            tr.restore(d)
            milestone = d.get("publish_milestone", False)
            del d
            resume_outputs(tr, args.resume, folder, milestone)
            rewind_progress(folder, tr.steps)
            link_checkpoint(args.resume, latest)
        counts = {
            name + "_params": sum(p.numel() for p in net.parameters())
            for name, net in (("policy", tr.policy), ("predictor", tr.predictor))
        }
        log_record(
            folder,
            "initialization",
            tr.steps,
            **counts,
            trainable_params=sum(counts.values()),
            architecture=ARCH,
            initialization="resume" if args.resume else "random",
            engine=ENGINE,
            device=str(tr.device),
            envs=args.envs,
            actions=tr.actions,
            config=asdict(c),
        )

        def validate(stage):
            candidates = []
            for mode in modes:
                ev = tr.evaluate(
                    args.eval_episodes,
                    args.eval_seed,
                    mode,
                    args.eval_workers,
                    parallel=args.eval_episodes,
                )
                ev.update(checkpoint_steps=tr.steps, stage=stage)
                save_json(folder / f"validation-{tr.steps:09d}-{mode}.json", ev)
                candidates.append(ev)
            winner = max(candidates, key=lambda ev: ev["summary"]["team_score"])
            s = winner["summary"]
            log_record(
                folder,
                "validation",
                tr.steps,
                **s,
                mode=winner["mode"],
                stage=stage,
                **{ev["mode"] + "_team_score": ev["summary"]["team_score"] for ev in candidates},
            )
            return {
                "selected_step": tr.steps,
                "mode": winner["mode"],
                "stage": stage,
                "team_score": s["team_score"],
                "validation_seed": args.eval_seed,
                "validation_episodes": args.eval_episodes,
            }

        if not args.resume:
            tr.weights(folder / "zero.pt")
            if args.eval_episodes:
                validate("initial")
        bucket = tr.steps // args.eval_every
        mile = tr.steps // args.milestone_every if args.milestone_every else 0
        with math_mode(tr.device, fast=True):
            while tr.steps < args.steps:
                n = min(args.rollout, (args.steps - tr.steps) // args.envs)
                t0 = time.monotonic()
                batch, stats = tr.collect(n)
                stats.update(tr.optimize(batch))
                elapsed = time.monotonic() - t0
                log_record(
                    folder,
                    "train",
                    tr.steps,
                    **stats,
                    update=tr.update_num,
                    steps_per_second=n * len(tr.ids) / elapsed,
                )
                now = tr.steps // args.eval_every
                deadline = (
                    args.deadline_epoch
                    and time.time() + max(30, 2 * elapsed) >= args.deadline_epoch
                )
                stop_requested = args.stop_file and args.stop_file.is_file()
                if (
                    now > bucket
                    or tr.update_num == 1
                    or tr.steps == args.steps
                    or deadline
                    or stop_requested
                ):
                    bucket = now
                    selected = validate("ppo") if args.eval_episodes else None
                    improved = selected and selected["team_score"] > tr.best
                    if improved:
                        tr.best = selected["team_score"]
                        tr.selection = selected
                    milestone = bool(
                        args.milestone_every and tr.steps // args.milestone_every > mile
                    )
                    tr.checkpoint(latest, milestone=milestone)
                    publish_checkpoint(tr, latest, folder, milestone)
                    if milestone:
                        mile = tr.steps // args.milestone_every
                    if deadline or stop_requested:
                        log_record(
                            folder,
                            "deadline" if deadline else "stopped",
                            tr.steps,
                            deadline_epoch=args.deadline_epoch,
                        )
                        break
    finally:
        tr.pool.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser(
        "train",
        help="Run one training stage from random initialization or a full checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    t.add_argument("--out", type=Path, required=True, help="Training output directory")
    t.add_argument("--resume", type=Path, help="Restore a full training checkpoint (latest.pt)")
    t.add_argument(
        "--steps", type=int, default=1966080, help="Cumulative target joint environment steps"
    )
    t.add_argument("--envs", type=int, default=64, help="Parallel three-chef environments")
    t.add_argument("--rollout", type=int, default=64, help="Joint ticks per environment per update")
    t.add_argument(
        "--workers", type=int, default=0, help="Environment worker processes (0 runs locally)"
    )
    t.add_argument(
        "--eval-workers", type=int, default=0, help="Development evaluation worker processes"
    )
    t.add_argument("--eval-every", type=int, default=196608)
    t.add_argument("--eval-episodes", type=int, default=32)
    t.add_argument("--eval-seed", type=int, default=918401)
    t.add_argument(
        "--eval-modes",
        default="sample,cool,greedy",
        help="Decoding modes to compare on the development seeds, in tie-breaking order",
    )
    t.add_argument("--milestone-every", type=int, default=1572864)
    t.add_argument(
        "--deadline-epoch",
        type=float,
        default=0.0,
        help="Unix deadline for stopping and saving (0 disables)",
    )
    t.add_argument(
        "--stop-file", type=Path, help="Stop and save after an update when this file exists"
    )
    for k in LargeConfig.__dataclass_fields__:
        default = getattr(LargeConfig(), k)
        t.add_argument("--" + k.replace("_", "-"), type=type(default), default=default)
    e = sub.add_parser(
        "evaluate",
        help="Evaluate one policy on normal starts and write official per-episode metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    e.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Policy weights (best.pt) or a full checkpoint",
    )
    e.add_argument(
        "--out", required=True, type=Path, help="Destination for the complete JSON evaluation"
    )
    e.add_argument("--seed", type=int, default=928401)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument(
        "--mode",
        choices=MODES,
        default="sample",
        help="Decoding mode; use the mode selected on development games",
    )
    e.add_argument("--workers", type=int, default=0)
    e.add_argument("--parallel", type=int, default=100)
    e.add_argument("--log-dir", type=Path)
    for s in (e, t):
        s.add_argument("--device", default="cpu")
        s.add_argument("--torch-threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    if a.command == "train":
        train(a)
        return
    d = load(a.checkpoint)
    model = LargeModel(LargeConfig.from_checkpoint(d["config"]), a.device)
    load_weights(model, d)
    ev = model.evaluate(a.episodes, a.seed, a.mode, a.workers, a.parallel)
    ev["checkpoint_steps"] = d["steps"]
    save_json(a.out, ev)
    if a.log_dir:
        a.log_dir.mkdir(parents=True, exist_ok=True)
        log_record(a.log_dir, "holdout", d["steps"], mode=a.mode, **ev["summary"])
    else:
        print(json.dumps(ev["summary"]), flush=True)


if __name__ == "__main__":
    main()

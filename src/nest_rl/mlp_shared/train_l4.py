"""L3-to-L4 sparse PPO with global random spawns and perturbed self-play partners."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
import trackio

from .env import FLOOR, SharedKitchenConfig, TorchSharedKitchen
from .pretrain import atomic_save
from .train_l1 import L1Settings, PROJECT, RUNS, SharedL1Trainer


@dataclass(frozen=True)
class L4Settings(L1Settings):
    num_envs: int = 1008
    rollout_steps: int = 300
    updates: int = 200
    lr: float = 5e-5
    entropy_coef: float = 0.002
    bc_start: float = 0.0
    bc_end: float = 0.0
    bc_decay_updates: int = 0
    critic_warmup_updates: int = 0
    selfplay_frac: float = 1.0
    partner_noise: float = 0.05
    partner_noise_end: float = 0.05
    random_spawn_prob: float = 1.0
    use_history: bool = False
    partner_type: str = "fast"
    eval_every: int = 5

    def __post_init__(self):
        if self.shaping_scale != 0 or self.bc_start != 0 or self.bc_end != 0:
            raise ValueError("L4 requires sparse rewards and zero BC auxiliary loss")
        if self.random_spawn_prob != 1 or self.selfplay_frac != 1 or self.use_history:
            raise ValueError("L4 uses global random spawns and no-history self-play")
        if self.num_envs < 3 or self.num_envs % 3 or not 0 <= self.partner_noise <= 1:
            raise ValueError("Expected role-balanced environments and a valid perturbation probability")


def initialize_trainer(cfg, checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    trainer = SharedL1Trainer(cfg, device=device)
    trainer.model.load_state_dict(payload["model"], strict=True)
    return trainer


@torch.inference_mode()
def evaluate_l4(model, *, noise=0.05, random_spawn=True, num_envs=45, episodes=3,
                horizon=300, seed=20_000_000, device="cuda"):
    was_training = model.training
    model.eval()
    devices = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    totals = {}
    try:
        with torch.random.fork_rng(devices=devices):
            env = TorchSharedKitchen(SharedKitchenConfig(horizon=horizon), num_envs=num_envs,
                                     device=device, use_cuda_graph=True)
            floor = torch.tensor(sorted(FLOOR), device=device)
            generator = torch.Generator(device=device).manual_seed(seed)
            focal = env.lanes % 3
            for episode in range(episodes):
                obs, _ = env.reset(seed=seed + episode * 1000)
                if random_spawn:
                    order = torch.rand((num_envs, len(floor)), device=device, generator=generator).argsort(dim=1)
                    env.state["pos"].copy_(floor[order[:, :3]])
                    env._write_observations()
                    obs = env._obs_out.clone()
                for _ in range(horizon):
                    joint = torch.stack([model.actor(obs[:, chef]).argmax(-1) for chef in range(3)], dim=1)
                    mask = torch.rand((num_envs, 3), device=device, generator=generator) < noise
                    mask[env.lanes, focal] = False
                    replacements = torch.randint(0, 6, (num_envs, 3), device=device, generator=generator)
                    joint = torch.where(mask, replacements, joint)
                    obs, _, _, _, _ = env.step(joint)
                for key, value in env.episode_metrics().items():
                    totals[key] = totals.get(key, 0.0) + float(value)
    finally:
        model.train(was_training)
    result = {key: value / episodes for key, value in totals.items()}
    result.update(episodes=num_envs * episodes, partner_noise=noise, random_spawn_prob=float(random_spawn))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--init-policy", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=1008)
    parser.add_argument("--minibatch-size", type=int, default=3024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--partner-noise", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = L4Settings(seed=args.seed, updates=args.updates, num_envs=args.num_envs,
                     minibatch_size=args.minibatch_size, lr=args.lr, partner_noise=args.partner_noise,
                     partner_noise_end=args.partner_noise, eval_every=args.eval_every)
    destination = RUNS / args.campaign / "l4"
    destination.mkdir(parents=True, exist_ok=False)
    validation_dir = destination / "validation"
    validation_dir.mkdir()
    config = dict(asdict(cfg), stage="l4", init_policy=str(args.init_policy.resolve()),
                  init_policy_sha256=hashlib.sha256(args.init_policy.read_bytes()).hexdigest(),
                  reward="sparse_delivery_only", project=PROJECT)
    (destination / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trainer = initialize_trainer(cfg, args.init_policy, args.device)
    run = trackio.init(project=PROJECT, name=f"{args.campaign}-l4-seed{args.seed}",
                       group="shared-l4", config=config, resume="never")
    best_score = float("-inf")
    journal = destination / "train.jsonl"

    def save_status(status, **extra):
        payload = dict(status=status, update=trainer.update, total_updates=cfg.updates,
                       env_steps=trainer.update * cfg.num_envs * cfg.rollout_steps, **extra)
        (destination / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        save_status("initial_evaluation")
        initial = evaluate_l4(trainer.model, noise=cfg.partner_noise, device=args.device)
        (destination / "validation-init.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        best_score = initial["team_score"]
        initial_checkpoint = dict(model=trainer.model.state_dict(), config=config, update=0,
                                  selection_score=best_score)
        atomic_save(initial_checkpoint, destination / "policy_000000.pt")
        atomic_save(initial_checkpoint, destination / "best.pt")
        run.log({f"eval/l4_{key}": value for key, value in initial.items()}, step=0)
        print(f"L4 initialized from {args.init_policy} | random-spawn + noise score={initial['team_score']:.2f}", flush=True)
        save_status("running", initial_score=initial["team_score"])
        for _ in range(cfg.updates):
            metrics = trainer.step()
            update = trainer.update
            if abs(metrics["mean_step_reward"] * cfg.horizon - metrics["mean_score"]) > 0.01:
                raise RuntimeError("Rollout rewards differ from sparse delivery scores")
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(update=update, **metrics), allow_nan=False) + "\n")
            logs = {f"train/{key}": value for key, value in metrics.items()}
            if update == 1 or update % cfg.eval_every == 0 or update == cfg.updates:
                evaluation = evaluate_l4(trainer.model, noise=cfg.partner_noise, device=args.device)
                clean = evaluate_l4(trainer.model, noise=0, random_spawn=False, device=args.device)
                (validation_dir / f"{update:03d}.json").write_text(
                    json.dumps(dict(l4=evaluation, clean=clean), indent=2), encoding="utf-8")
                logs.update({f"eval/l4_{key}": value for key, value in evaluation.items()})
                logs["eval/clean_all_policy_score"] = clean["team_score"]
                checkpoint = dict(model=trainer.model.state_dict(), config=config, update=update,
                                  selection_score=evaluation["team_score"])
                atomic_save(checkpoint, destination / f"policy_{update:06d}.pt")
                atomic_save(trainer.state_dict(), destination / "latest.pt")
                if evaluation["team_score"] > best_score:
                    best_score = evaluation["team_score"]
                    atomic_save(checkpoint, destination / "best.pt")
            run.log(logs, step=update)
            save_status("running", train_score=metrics["mean_score"], best_l4_score=best_score)
            print(f"L4 {update}/{cfg.updates} score={metrics['mean_score']:.2f} "
                  f"sparse_reward={metrics['mean_step_reward']:.5f} "
                  f"best_eval={best_score:.2f} sps={metrics['steps_per_second']:,.0f}", flush=True)
        save_status("completed", best_l4_score=best_score)
    except BaseException as error:
        save_status("failed", error=str(error))
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    main()

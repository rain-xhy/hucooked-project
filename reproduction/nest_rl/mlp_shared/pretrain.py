"""Behavior cloning pretraining on custom GpuSharedPartners demonstrations (ready to run; do not auto-start)."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .env import SharedKitchenConfig, TorchSharedKitchen
from .evaluation import evaluate_shared
from .fast_partners import GpuFastPartners
from .gpu_partners import GpuSharedPartners
from .history import OnlineHistory, RolloutHistory
from .policy import V3Policy, partner_action_ids


def atomic_save(data, path):
    path = Path(path)
    tmp = path.with_suffix(f".tmp.{time.time_ns()}")
    torch.save(data, tmp)
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    # Fallback on persistent Windows Defender lock: overwrite directly and clean temp.
    try:
        torch.save(data, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class SharedBCSettings:
    seed: int = 1
    num_envs: int = 96
    episodes: int = 10
    horizon: int = 300
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 1e-3
    validation_envs: int = 18
    validation_episodes: int = 5
    validation_seed: int = 30_000_000
    patience: int = 4
    use_history: bool = True
    partner_type: str = "shared"
    label_smoothing: float = 0.0


@torch.no_grad()
def collect_custom_expert(num_envs, episodes, horizon, seed, device="cuda", partner_type="shared"):
    """Collect balanced 3-role demonstrations from GpuSharedPartners or GpuFastPartners."""
    cfg = SharedKitchenConfig(horizon=horizon)
    env = TorchSharedKitchen(cfg, num_envs=num_envs, device=device, use_cuda_graph=True)
    roles = torch.arange(num_envs, device=device) % 3
    partner_cls = GpuFastPartners if partner_type == "fast" else GpuSharedPartners
    expert = partner_cls(env, roles=roles, use_cuda_graph=True)
    live = OnlineHistory(num_envs, device)
    bank = RolloutHistory(live, episodes * horizon)
    actions = []
    scores = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep * 1000)
        expert.reset()
        for _ in range(horizon):
            learner = expert.learner_ids
            cur = obs[env.lanes, learner]
            joint = expert.actions()
            choice = joint[env.lanes, learner].clone()
            obs, reward, _, _, _ = env.step(joint)
            bank.record(cur, partner_action_ids(joint)[env.lanes, learner])
            actions.append(choice)
            scores += float(reward.sum())
        bank.finish_episode()
    bank.seal()
    data = dict(action=torch.cat(actions))
    meta = dict(seed=seed, episodes=num_envs * episodes, samples=len(data["action"]),
                mean_score=scores / (num_envs * episodes),
                controller=expert.version,
                action_counts=torch.bincount(data["action"], minlength=6).tolist())
    return bank, data, meta


def _bc_epoch(model, bank, data, batch_size, optimizer=None, label_smoothing=0.0):
    training = optimizer is not None
    model.train(training)
    count = len(data["action"])
    order = (torch.randperm(count, device=bank.obs.device) if training
             else torch.arange(count, device=bank.obs.device))
    total_nll = 0.0
    total_ent = 0.0
    correct = 0
    with torch.set_grad_enabled(training):
        for index in order.split(batch_size):
            logp, _, _, _ = bank.replay(model, index)
            targets = data["action"][index]
            nll = F.nll_loss(logp, targets)
            if training and label_smoothing > 0.0:
                loss = (1.0 - label_smoothing) * nll + label_smoothing * (-logp.mean())
            else:
                loss = nll
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                total_nll += float(nll) * len(index)
                ent = -(logp.exp() * logp).sum(-1).mean()
                total_ent += float(ent) * len(index)
                correct += int((logp.argmax(-1) == targets).sum())
    return dict(nll=total_nll / count, action_accuracy=correct / count, mean_entropy=total_ent / count)


def run_pretraining(cfg: SharedBCSettings, output, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model = V3Policy(3, use_history=cfg.use_history).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-8)
    started = time.perf_counter()
    train_bank, train_data, train_meta = collect_custom_expert(
        cfg.num_envs, cfg.episodes, cfg.horizon, cfg.seed, device, partner_type=cfg.partner_type)
    val_bank, val_data, val_meta = collect_custom_expert(
        cfg.validation_envs, cfg.validation_episodes, cfg.horizon, cfg.validation_seed, device, partner_type=cfg.partner_type)
    n_params = sum(p.numel() for p in model.parameters())
    (output / "bc_config.json").write_text(
        json.dumps(dict(config=asdict(cfg), num_params=n_params, training=train_meta, validation=val_meta), indent=2),
        encoding="utf-8",
    )
    best, stale = float("inf"), 0
    records = []
    for ep in range(cfg.epochs + 1):
        tr = _bc_epoch(model, train_bank, train_data, cfg.batch_size, optimizer, cfg.label_smoothing) if ep else None
        va = _bc_epoch(model, val_bank, val_data, cfg.batch_size)
        rec = dict(epoch=ep, training=tr, validation=va, elapsed_seconds=time.perf_counter() - started)
        records.append(rec)
        if va["nll"] < best:
            best, stale = va["nll"], 0
            atomic_save(dict(model=model.state_dict(), config=asdict(cfg), use_history=cfg.use_history,
                             epoch=ep, validation=va, source=f"shared-gpu-bc-{cfg.partner_type}"),
                        output / "bc_best.pt")
        else:
            stale += 1
        (output / "bc_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(rec), flush=True)
        if stale >= cfg.patience:
            break
    eval_single = evaluate_shared(model, mode="policy", device=device, partner_type=cfg.partner_type)
    eval_all = evaluate_shared(model, mode="all_policy", device=device, partner_type=cfg.partner_type)
    (output / "bc_eval.json").write_text(json.dumps(dict(single_policy=eval_single, all_policy=eval_all), indent=2), encoding="utf-8")
    return output / "bc_best.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--partner-type", choices=("shared", "fast"), default="shared")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    args = p.parse_args()
    cfg = SharedBCSettings(
        seed=args.seed, num_envs=args.num_envs, episodes=args.episodes, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, use_history=not args.no_history,
        partner_type=args.partner_type, label_smoothing=args.label_smoothing,
    )
    run_pretraining(cfg, args.output, device=args.device)


if __name__ == "__main__":
    main()

"""Causal, indexed-history imitation of the parallel kitchen controller.

The expert controls all three chefs. Each lane contributes the designated role's
observation/action, with the other two executed actions as history. No controller
plans or style labels enter the policy observation. Validation kitchens use a
separate seed stream. Critic weights are deliberately left for PPO to fit.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from overcooked_ai_py.cooperative.partition import PartitionConfig
from .controllers import make_partners
from .features import partner_action_ids
from .history import OnlineHistory, RolloutHistory
from .kitchen import V3PolicyKitchen
from .policy import V3Policy
from .train import atomic_save


@dataclass(frozen=True)
class BCSettings:
    seed: int = 1
    num_envs: int = 63
    episodes: int = 10
    horizon: int = 400
    epochs: int = 10
    batch_size: int = 900
    lr: float = .001
    validation_envs: int = 9
    validation_episodes: int = 5
    validation_seed: int = 30_000_000
    aux_coefficient: float = .1
    patience: int = 4
    controller_version: str = "parallel-v2"


def validate_seed_streams(cfg):
    # Partition engine.reset uses SeedSequence(seed + lane).spawn(2).
    # Seedless later resets continue those same per-lane spawn/order streams.
    # Thus disjoint base-seed intervals are needed, not just unequal first seeds.
    if min(cfg.seed, cfg.validation_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    if max(cfg.seed, cfg.validation_seed) < min(
            cfg.seed + cfg.num_envs, cfg.validation_seed + cfg.validation_envs):
        raise ValueError("Training and validation lane seed ranges must not overlap")


@torch.no_grad()
def collect_expert(num_envs, episodes, horizon, seed, device, controller_version="parallel-v2"):
    if num_envs < 9 or num_envs % 9 or min(episodes, horizon) < 1:
        raise ValueError("Positive episode lengths and a lane count divisible by nine required")
    env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=horizon),
                          num_envs=num_envs, device=device)
    obs, _ = env.reset(seed=seed)
    group = torch.arange(num_envs, device=device) % 9
    roles, styles = group // 3, group % 3
    expert = make_partners(env, roles, styles, controller_version)
    expert.reset()
    live = OnlineHistory(num_envs, device)
    bank = RolloutHistory(live, episodes * horizon)
    actions = []
    scores = 0.
    for episode in range(episodes):
        for _ in range(horizon):
            learner = expert.learner_ids
            current = obs[env.lanes, learner]
            joint = expert.actions().clone()
            choice = joint[env.lanes, learner].clone()
            obs, reward, _, _, _ = env.step(joint)
            # record stores the history table BEFORE adding this executed step.
            bank.record(current, partner_action_ids(joint)[env.lanes, learner])
            actions.append(choice)
            scores += reward.sum().item()
        bank.finish_episode()
        print(json.dumps(dict(stage="collect_expert", seed=seed, episode=episode + 1,
                              episodes=episodes, samples=(episode + 1) * horizon * num_envs)), flush=True)
        if episode + 1 < episodes:
            obs, _ = env.reset()
            expert.reset()
    bank.seal()
    data = dict(action=torch.cat(actions), labels=styles.repeat(episodes * horizon))
    metadata = dict(seed=seed, episodes=num_envs * episodes, samples=len(data["action"]),
                    mean_score=scores / (num_envs * episodes),
                    action_counts=torch.bincount(data["action"], minlength=6).tolist())
    return bank, data, metadata


def epoch(model, bank, data, batch_size, optimizer=None, aux_coefficient=.1):
    training = optimizer is not None
    model.train(training)
    count = len(data["action"])
    order = (torch.randperm(count, device=bank.obs.device) if training else
             torch.arange(count, device=bank.obs.device))
    totals = torch.zeros(5, device=bank.obs.device)
    confusion = torch.zeros(6, 6, device=bank.obs.device, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for index in order.split(batch_size):
            logp, _, aux, has = bank.replay(model, index)
            targets, labels = data["action"][index], data["labels"][index]
            nll = F.nll_loss(logp, targets)
            aux_sum = (F.cross_entropy(aux, labels, reduction="none") * has).sum()
            aux_loss = aux_sum / has.sum().clamp_min(1)
            loss = nll + aux_coefficient * aux_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 15., error_if_nonfinite=True)
                optimizer.step()
            predicted = logp.argmax(-1)
            with torch.no_grad():
                totals += torch.stack((nll * len(index), (predicted == targets).sum(),
                                       aux_sum, ((aux.argmax(-1) == labels) & has).sum(), has.sum()))
                confusion += torch.bincount(targets * 6 + predicted, minlength=36).reshape(6, 6)
    nll, correct, aux_sum, aux_correct, valid = totals.tolist()
    rows = confusion.sum(1)
    per_action = confusion.diag().float() / rows.clamp_min(1)
    return dict(nll=nll / count, action_accuracy=correct / count,
                balanced_action_accuracy=per_action[rows > 0].mean().item(),
                per_action_accuracy=[float(a) if n else None for a, n in zip(per_action.tolist(), rows.tolist())],
                confusion_matrix=confusion.cpu().tolist(),
                aux_loss=aux_sum / max(1, valid), aux_accuracy=aux_correct / max(1, valid))


def run_pretraining(cfg, output, device="cuda", callback=None):
    if min(cfg.epochs, cfg.batch_size, cfg.patience) < 1 or cfg.lr <= 0:
        raise ValueError("Positive optimizer settings required")
    validate_seed_streams(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model = V3Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-8)
    started = time.perf_counter()
    train_bank, train_data, train_meta = collect_expert(
        cfg.num_envs, cfg.episodes, cfg.horizon, cfg.seed, device, controller_version=cfg.controller_version)
    val_bank, val_data, val_meta = collect_expert(
        cfg.validation_envs, cfg.validation_episodes, cfg.horizon, cfg.validation_seed, device,
        controller_version=cfg.controller_version)
    manifest = dict(config=asdict(cfg), training=train_meta, validation=val_meta,
                    controller_version=cfg.controller_version, recipe_rewards=[20, 10, 50],
                    history_episodes=5, history_steps=40,
                    sampling="Uniform shuffled samples; ordinary unweighted action NLL",
                    critic_pretrained=False)
    (output / "bc_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    best, stale = float("inf"), 0
    records = []
    for iteration in range(cfg.epochs + 1):
        train_metrics = (epoch(model, train_bank, train_data, cfg.batch_size, optimizer,
                               cfg.aux_coefficient) if iteration else None)
        validation = epoch(model, val_bank, val_data, cfg.batch_size)
        record = dict(epoch=iteration, training=train_metrics, validation=validation,
                      elapsed_seconds=time.perf_counter() - started)
        records.append(record)
        # Select using held-out action NLL. Classification cannot mask poor action imitation.
        if validation["nll"] < best:
            best, stale = validation["nll"], 0
            atomic_save(dict(model=model.state_dict(), config=asdict(cfg), epoch=iteration,
                             validation=validation, source=f"{cfg.controller_version}-behavior-cloning",
                             critic_pretrained=False), output / "bc_best.pt")
        else:
            stale += 1
        (output / "bc_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if callback:
            callback(record)
        if stale >= cfg.patience:
            break
    (output / "bc_finished.json").write_text(json.dumps(dict(
        config=asdict(cfg), epochs_completed=records[-1]["epoch"], best_validation_nll=best,
        checkpoint="bc_best.pt"), indent=2), encoding="utf-8")
    return output / "bc_best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    for name, field in BCSettings.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(parser.parse_args())
    output, device = args.pop("output"), args.pop("device")
    torch.set_num_threads(4)
    run_pretraining(BCSettings(**args), output, device)


if __name__ == "__main__":
    main()

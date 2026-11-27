"""Left, cook, courier, then joint PPO with rehearsal and explicit promotion gates."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .bc_experiment import compact_metrics, load_policy
from .evaluation import evaluate, ROLE_NAMES
from .experiment import read, save
from .history import RolloutHistory
from .pretrain import collect_expert
from .tracking import RUNS, COOPERATION_PROJECT, open_run, recorded_steps
from .train import Settings, Trainer, atomic_save

STAGES = ("left", "right_cook", "right_courier", "joint")
COUNTS = ((45, 9, 9), (9, 45, 9), (9, 9, 45), (21, 21, 21))
DEMO_SEED = 40_000_000
VALIDATION_SEED = 10_000_000


def kitchen_counts(num_envs):
    """Largest-remainder role allocation, retaining at least three lanes/role."""
    if type(num_envs) is not int or num_envs < 9:
        raise ValueError("num_envs must be an integer >=9")
    output = []
    for weights in COUNTS:
        row = [max(3, num_envs * weight // 63) for weight in weights]
        while sum(row) > num_envs:
            index = max((i for i in range(3) if row[i] > 3),
                        key=lambda i: row[i] * 63 - num_envs * weights[i])
            row[index] -= 1
        while sum(row) < num_envs:
            index = max(range(3), key=lambda i: num_envs * weights[i] - row[i] * 63)
            row[index] += 1
        output.append(tuple(row))
    return tuple(output)


def evaluation_due(update, samples, interval, stage_updates):
    return update == stage_updates or update * samples // interval > (update - 1) * samples // interval


def promotion_boundary(update, rollout_steps, horizon):
    return (update * rollout_steps) % (5 * horizon) == 0


def promotion(result, wait, references, stage):
    """Every learned role retains >=90% of its reference; focus must help."""
    reasons = []
    learned = ROLE_NAMES[:min(stage + 1, 3)]
    for role in learned:
        score = result["per_role"][role]["mean_score"]
        if score < .9 * references[role]:
            reasons.append(f"{role}: score {score:.2f} below retention floor {.9 * references[role]:.2f}")
    focused = ROLE_NAMES if stage == 3 else (ROLE_NAMES[stage],)
    for role in focused:
        if result["per_role"][role]["mean_score"] <= wait["per_role"][role]["mean_score"]:
            reasons.append(f"{role}: no positive contribution over waiting")
    # Retain actual cross-wall ability in the left role throughout the curriculum.
    for recipe in ("tomato_egg", "potato_tomato_beef"):
        if result["per_role"]["left"]["per_recipe"][recipe]["delivered"] <= 0:
            reasons.append(f"left: no {recipe} deliveries")
    return not reasons, reasons


def demonstrations(root, horizon, device, controller_version="parallel-v2"):
    path = root / "demonstrations.pt"
    if not path.exists():
        bank, data, metadata = collect_expert(63, 5, horizon, DEMO_SEED, device, controller_version=controller_version)
        atomic_save(dict(obs=bank.obs.cpu(), ids=bank.ids.cpu(), tables=bank.tables.cpu(),
                         prefix=bank.prefix, data={k: v.cpu() for k, v in data.items()},
                         metadata=metadata, horizon=horizon), path)
        return bank, data
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["horizon"] != horizon or saved["metadata"]["seed"] != DEMO_SEED:
        raise ValueError("Demonstration configuration changed")
    bank = RolloutHistory.__new__(RolloutHistory)
    for key in ("obs", "ids", "tables", "prefix"):
        setattr(bank, key, saved[key])
    return bank, saved["data"]


def isolated_evaluation(model, device, **args):
    nstate, pstate = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
            return evaluate(model, device=device, **args)
    finally:
        np.random.set_state(nstate)
        random.setstate(pstate)


def run_curriculum(root, initial_policy, *, device="cuda", seed=1, stage_updates=10,
                   horizon=400, rollout_steps=800, epochs=4, minibatch_size=900,
                   stop_after=None, focus_lr=.0003, joint_lr=.0001, num_envs=128,
                   eval_env_steps=252_000):
    root, initial_policy = Path(root), Path(initial_policy)
    role_counts = kitchen_counts(num_envs)
    if type(eval_env_steps) is not int or eval_env_steps < 1:
        raise ValueError("eval_env_steps must be a positive integer")
    if any(not math.isfinite(lr) or lr <= 0 for lr in (focus_lr, joint_lr)):
        raise ValueError("Learning rates must be finite and positive")
    if any(type(value) is not int or value < 1 for value in (stage_updates, rollout_steps, horizon)):
        raise ValueError("Stage updates, rollout steps and horizon must be positive integers")
    if not promotion_boundary(stage_updates, rollout_steps, horizon):
        raise ValueError("Stage budget must end at a complete five-episode history group")
    root.mkdir(parents=True, exist_ok=True)
    config = dict(version="role-curriculum-v1", seed=seed, stage_updates=stage_updates,
                  num_envs=num_envs,
                  eval_env_steps=eval_env_steps,
                  horizon=horizon, rollout_steps=rollout_steps, epochs=epochs,
                  minibatch_size=minibatch_size, initial_policy=str(initial_policy.resolve()),
                  initial_sha256=hashlib.sha256(initial_policy.read_bytes()).hexdigest(),
                  role_counts=[list(row) for row in role_counts], learning_rates=[focus_lr] * 3 + [joint_lr],
                  bc_coefficient=.1, target_kl=.02, retention=.9, consecutive_passes=2,
                  controller_version="parallel-v2", recipe_rewards=[20., 10., 50.],
                  shaping=True, demo_seed=DEMO_SEED, validation_seed=VALIDATION_SEED)
    manifest = root / "experiment.json"
    if manifest.exists():
        previous_config = read(manifest)
        previous_config.setdefault("num_envs", 63)
        previous_config.setdefault("eval_env_steps", 252_000)
        if previous_config != config:
            raise ValueError("Curriculum configuration changed; use a new campaign")
    save(manifest, config)
    evargs = dict(seed=VALIDATION_SEED, groups=1, horizon=horizon,
                  controller_version="parallel-v2", recipe_rewards=(20., 10., 50.))
    baselines = {}
    for mode in ("wait", "script", "bc"):
        path = root / f"validation-{mode}.json"
        if not path.exists():
            print(f"Evaluating {mode} baseline", flush=True)
            model = load_policy(initial_policy, device) if mode == "bc" else None
            save(path, isolated_evaluation(model, device, mode="policy" if mode == "bc" else mode, **evargs))
        baselines[mode] = read(path)
    bank, data = demonstrations(root, horizon, device)
    latest = root / "latest.pt"
    saved = torch.load(latest, map_location=device, weights_only=False) if latest.exists() else None
    state = saved["curriculum"] if saved else dict(stage=0, global_update=0, passes=0,
        references={r: baselines["bc"]["per_role"][r]["mean_score"] for r in ROLE_NAMES},
        status="running", reasons=[])
    if state["status"] in ("complete", "needs_attention"):
        print(f"Curriculum {state['status']}: {state['reasons']}", flush=True)
        return root / "summary.json"
    if stop_after is not None and state["global_update"] >= stop_after:
        return root / "summary.json"
    journal = root / "train.jsonl"
    if journal.exists():
        retained = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break
            if entry["global_update"] <= state["global_update"]:
                retained.append(line)
        journal.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    name = f"{root.name}-ppo-seed{seed}"
    published = recorded_steps(name, project=COOPERATION_PROJECT)
    run = open_run(name, "role-curriculum", config, True, project=COOPERATION_PROJECT)
    samples = num_envs * rollout_steps

    def publish(update, stage, metrics=None, result=None):
        step = update * samples
        if step not in published:
            values = compact_metrics(metrics, result, baselines["wait"], baselines["script"])
            values["curriculum/stage"] = stage + 1
            run.log(values, step=step)
            published.add(step)

    def persist(trainer):
        atomic_save(dict(**trainer.state_dict(), curriculum=state), latest)
        save(root / "summary.json", dict(**state, stage_name=STAGES[state["stage"]],
             latest_checkpoint=str(latest.resolve())))

    def status(trainer):
        save(root / "status.json", dict(**state, stage_name=STAGES[state["stage"]],
             stage_update=trainer.update, stage_budget=stage_updates,
             role_counts=list(trainer.cfg.role_counts)))

    trainer = None
    try:
        publish(0, 0, result=baselines["bc"])
        while state["stage"] < 4:
            stage = state["stage"]
            cfg = Settings(config["learning_rates"][stage], seed + stage * 1000, stage_updates,
                num_envs=num_envs, horizon=horizon, rollout_steps=rollout_steps, epochs=epochs,
                minibatch_size=minibatch_size, role_counts=role_counts[stage],
                shaping=True, bc_coefficient=.1, target_kl=.02)
            previous = trainer
            trainer = Trainer(cfg, device)
            trainer.attach_demonstrations(bank, data)
            if saved is not None:
                trainer.load(latest)
                saved = None
                # Repair missing Trackio publication after checkpoint persistence.
                ep = root / "validation" / f"{state['global_update']:03d}.json"
                if ep.exists():
                    publish(state["global_update"], stage, result=read(ep))
            elif previous is not None:
                trainer.model.load_state_dict(previous.model.state_dict())
                trainer.optimizer.load_state_dict(previous.optimizer.state_dict())
            else:
                trainer.model.load_state_dict(torch.load(initial_policy, map_location=device, weights_only=False)["model"])
            print(f"Stage {stage + 1}/4 {STAGES[stage]}: counts={role_counts[stage]}, local update={trainer.update}", flush=True)
            status(trainer)
            promoted = state["passes"] >= 2 and promotion_boundary(trainer.update, rollout_steps, horizon)
            result = read(root / "validation" / f"{state['global_update']:03d}.json") if promoted else None
            while trainer.update < stage_updates and not promoted:
                metrics = trainer.step()
                state["global_update"] += 1
                update = state["global_update"]
                with (root / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(global_update=update, stage=stage,
                        stage_update=trainer.update, **metrics), allow_nan=False) + "\n")
                result = None
                if evaluation_due(trainer.update, samples, eval_env_steps, stage_updates):
                    result = isolated_evaluation(trainer.model, device, **evargs)
                    save(root / "validation" / f"{update:03d}.json", result)
                    atomic_save(dict(model=trainer.model.state_dict(), config=asdict(cfg),
                        global_update=update, stage=stage), root / f"policy_{update:06d}.pt")
                    passed, state["reasons"] = promotion(result, baselines["wait"], state["references"], stage)
                    state["passes"] = state["passes"] + 1 if passed else 0
                    promoted = state["passes"] >= 2 and promotion_boundary(trainer.update, rollout_steps, horizon)
                    if not promoted and trainer.update == stage_updates:
                        state["status"] = "needs_attention"
                        state["reasons"].append(
                            f"Insufficient consecutive validation passes ({state['passes']}/2) at stage budget")
                    if promoted and stage == 3:
                        state["status"] = "complete"
                    if promoted:
                        atomic_save(dict(model=trainer.model.state_dict(), config=asdict(cfg),
                            global_update=update, stage=stage, validation=result),
                            root / f"accepted_{STAGES[stage]}.pt")
                    persist(trainer)
                publish(update, stage, metrics, result)
                status(trainer)
                print(f"Curriculum {STAGES[stage]} {trainer.update}/{stage_updates} global={update} "
                      f"score={metrics.get('mean_score')} passes={state['passes']} reasons={state['reasons']}", flush=True)
                if state["status"] != "running":
                    return root / "summary.json"
                if promoted:
                    break
                if stop_after is not None and update >= stop_after:
                    persist(trainer)
                    return root / "summary.json"
            if promoted:
                # Every promotion is at a complete 5-episode-group boundary.
                for role in ROLE_NAMES:
                    state["references"][role] = max(state["references"][role], result["per_role"][role]["mean_score"])
                state["stage"] += 1
                state["passes"] = 0
                # The next loop resets kitchens/history but keeps network and Adam.
                # Persist the next-stage initial state before taking any samples.
                next_cfg = Settings(config["learning_rates"][stage + 1], seed + (stage + 1) * 1000,
                    stage_updates, num_envs=num_envs, horizon=horizon, rollout_steps=rollout_steps,
                    epochs=epochs, minibatch_size=minibatch_size, role_counts=role_counts[stage + 1],
                    shaping=True, bc_coefficient=.1, target_kl=.02)
                next_trainer = Trainer(next_cfg, device)
                next_trainer.model.load_state_dict(trainer.model.state_dict())
                next_trainer.optimizer.load_state_dict(trainer.optimizer.state_dict())
                persist(next_trainer)
                status(next_trainer)
                saved = True
                if stop_after is not None and state["global_update"] >= stop_after:
                    return root / "summary.json"
    finally:
        run.finish()


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="counter-curriculum-128-lr0003-seed1")
    parser.add_argument("--bc-policy", type=Path, default=RUNS / "counter-bc-shaped-seed1/bc/bc_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stage-updates", type=int, default=10)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=900)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--eval-env-steps", type=int, default=252_000)
    parser.add_argument("--focus-lr", type=float, default=.0003)
    parser.add_argument("--joint-lr", type=float, default=.0001)
    args = vars(parser.parse_args())
    campaign = args.pop("campaign")
    if not campaign or not campaign.replace("-", "").replace("_", "").isalnum():
        parser.error("Use a simple campaign name")
    run_curriculum(RUNS / campaign, args.pop("bc_policy"), **args)


if __name__ == "__main__":
    main()

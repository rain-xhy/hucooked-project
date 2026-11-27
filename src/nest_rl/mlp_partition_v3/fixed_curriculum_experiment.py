"""Fixed ten-update role stages: evaluation reports results but never gates training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .bc_experiment import compact_metrics, load_policy
from .curriculum_experiment import demonstrations, isolated_evaluation, kitchen_counts, promotion_boundary, STAGES
from .experiment import read, save
from .policy import V3Policy
from .tracking import RUNS, COOPERATION_PROJECT, open_run, recorded_steps
from .train import Settings, Trainer, atomic_save

STAGE_UPDATES = 10
VALIDATION_SEED = 10_000_000


def _parse_stage_updates(value):
    if value is None:
        return [STAGE_UPDATES] * 4
    if isinstance(value, int):
        return [value] * 4
    if isinstance(value, str):
        parts = [int(p.strip()) for p in value.split(",") if p.strip()]
        if len(parts) == 1:
            return parts * 4
        if len(parts) == 4:
            return parts
        raise ValueError("stage_updates must be one integer or four comma-separated integers")
    parts = [int(x) for x in value]
    if len(parts) != 4:
        raise ValueError("stage_updates must have four values")
    return parts


def _stage_for_update(global_update, budgets):
    if global_update <= 0:
        return 0
    seen = 0
    for idx, budget in enumerate(budgets):
        seen += budget
        if global_update <= seen:
            return idx
    return len(budgets) - 1


def run_fixed_curriculum(root, initial_policy=None, *, device="cuda", seed=1, num_envs=504,
                         horizon=400, rollout_steps=800, epochs=4, minibatch_size=900,
                         focus_lr=.0003, joint_lr=.0001, entropy_coef=.03,
                         warmup_fraction=.02, warmup_start_factor=.1,
                         max_grad_norm=15.0, stagger_history=False,
                         controller_version="parallel-v2", stage_updates=None,
                         stop_after=None, from_scratch=False):
    root = Path(root)
    if not from_scratch and initial_policy is None:
        raise ValueError("Supply an initial policy or choose from_scratch")
    initial_policy = None if from_scratch else Path(initial_policy)
    counts = kitchen_counts(num_envs)
    budgets = _parse_stage_updates(stage_updates)
    total_budget = sum(budgets)
    if min(horizon, rollout_steps, *budgets) < 1:
        raise ValueError("Positive horizon, rollout_steps, and stage_updates required")
    for budget in budgets:
        if stagger_history:
            if (budget * rollout_steps) % horizon != 0:
                raise ValueError("Stage updates must end at an episode boundary")
        elif not promotion_boundary(budget, rollout_steps, horizon):
            raise ValueError("Stage updates must end at a complete five-episode history group")
    root.mkdir(parents=True, exist_ok=True)
    stage_updates_cfg = budgets[0] if len(set(budgets)) == 1 else list(budgets)
    config = dict(version="fixed-role-curriculum-v1", seed=seed, num_envs=num_envs,
                  stage_updates=stage_updates_cfg, stages=list(STAGES), role_counts=[list(c) for c in counts],
                  horizon=horizon, rollout_steps=rollout_steps, epochs=epochs, minibatch_size=minibatch_size,
                  learning_rates=[focus_lr, focus_lr, focus_lr, joint_lr], bc_coefficient=0. if from_scratch else .1, target_kl=.02,
                  controller_version=controller_version, recipe_rewards=[20., 10., 50.], shaping=True,
                  eval_every_updates=2, validation_seed=VALIDATION_SEED, demo_seed=40_000_000,
                  initial_policy=str(initial_policy.resolve()) if initial_policy is not None else None,
                  initial_sha256=hashlib.sha256(initial_policy.read_bytes()).hexdigest() if initial_policy is not None else None,
                  advancement="Exactly ten updates per stage; scores do not control progression")
    if entropy_coef != .03:
        config["entropy_coef"] = entropy_coef
    if warmup_fraction != .02:
        config["warmup_fraction"] = warmup_fraction
    if warmup_start_factor != .1:
        config["warmup_start_factor"] = warmup_start_factor
    if max_grad_norm != 15.0:
        config["max_grad_norm"] = max_grad_norm
    if stagger_history:
        config["stagger_history"] = True
    if from_scratch:
        config.update(from_scratch=True, initialization="random", initialization_seed=seed,
                      demo_seed=None)
    manifest = root / "experiment.json"
    if manifest.exists() and read(manifest) != config:
        raise ValueError("Fixed curriculum configuration changed; use a new campaign")
    save(manifest, config)
    latest = root / "latest.pt"
    saved = torch.load(latest, map_location=device, weights_only=False) if latest.exists() else None
    state = dict(saved["curriculum"]) if saved else dict(stage=0, global_update=0, status="running")
    if state["status"] == "complete" or (stop_after is not None and state["global_update"] >= stop_after):
        if saved is not None:
            # A process can exit after atomic latest.pt replacement but before
            # the human-readable status files are written. The checkpoint wins.
            save(root / "summary.json", dict(**state, stage_name=STAGES[state["stage"]],
                 stage_update=saved["update"], latest_checkpoint=str(latest.resolve())))
            save(root / "status.json", dict(**state, stage_name=STAGES[state["stage"]],
                 stage_update=saved["update"], stage_budget=budgets[state["stage"]],
                 total_budget_updates=total_budget,
                 role_counts=list(saved["config"]["role_counts"])))
        return root / "summary.json"
    evargs = dict(seed=VALIDATION_SEED, groups=1, horizon=horizon,
                  controller_version=controller_version, recipe_rewards=(20., 10., 50.))
    baselines = {}
    initial_name = "initial" if from_scratch else "bc"
    if from_scratch:
        initial_policy = root / "initial_random.pt"
        if not initial_policy.exists():
            # Match Trainer's seeded initialization without consuming the RNG
            # subsequently used for action sampling or optimizer minibatches.
            with torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
                torch.manual_seed(seed)
                initial_model = V3Policy().to(device)
                atomic_save(dict(model=initial_model.state_dict(), source="random-initialization", seed=seed), initial_policy)
    for mode in ("wait", "script", initial_name):
        path = root / f"validation-{mode}.json"
        if not path.exists():
            print(f"Evaluating {mode} baseline", flush=True)
            model = load_policy(initial_policy, device) if mode == initial_name else None
            save(path, isolated_evaluation(model, device, mode="policy" if mode == initial_name else mode, **evargs))
        baselines[mode] = read(path)
    bank, data = (None, None) if from_scratch else (
        demonstrations(root, horizon, device) if controller_version == "parallel-v2"
        else demonstrations(root, horizon, device, controller_version=controller_version))
    journal = root / "train.jsonl"
    if journal.exists():
        retained = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            if record["global_update"] <= state["global_update"]:
                retained.append(line)
        journal.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    name = f"{root.name}-ppo-seed{seed}"
    published = recorded_steps(name, project=COOPERATION_PROJECT)
    run = open_run(name, "fixed-role-curriculum", config, True, project=COOPERATION_PROJECT)
    samples = num_envs * rollout_steps

    def publish(update, stage, metrics=None, result=None):
        step = update * samples
        if step not in published:
            values = compact_metrics(metrics, result, baselines["wait"], baselines["script"])
            values["curriculum/stage"] = stage + 1
            run.log(values, step=step)
            published.add(step)

    def status(trainer):
        save(root / "status.json", dict(**state, stage_name=STAGES[state["stage"]],
             stage_update=trainer.update, stage_budget=budgets[state["stage"]],
             total_budget_updates=total_budget,
             role_counts=list(trainer.cfg.role_counts)))

    def persist(trainer):
        atomic_save(dict(**trainer.state_dict(), curriculum=dict(state)), latest)
        save(root / "summary.json", dict(**state, stage_name=STAGES[state["stage"]],
             stage_update=trainer.update, latest_checkpoint=str(latest.resolve())))
        status(trainer)

    def build(stage):
        cfg = Settings(config["learning_rates"][stage], seed + stage * 1000, budgets[stage],
                       num_envs=num_envs, horizon=horizon, rollout_steps=rollout_steps, epochs=epochs,
                       minibatch_size=minibatch_size, controller_version=controller_version,
                       role_counts=counts[stage], shaping=True,
                       bc_coefficient=config["bc_coefficient"], target_kl=.02,
                       entropy_coef=entropy_coef, warmup_fraction=warmup_fraction,
                       warmup_start_factor=warmup_start_factor, max_grad_norm=max_grad_norm,
                       stagger_history=stagger_history)
        trainer = Trainer(cfg, device)
        if not from_scratch:
            trainer.attach_demonstrations(bank, data)
        return trainer

    try:
        publish(0, 0, result=baselines[initial_name])
        trainer = build(state["stage"])
        if saved is not None:
            trainer.load(latest)
            path = root / "validation" / f"{state['global_update']:03d}.json"
            if path.exists():
                # A boundary checkpoint can already describe the next stage;
                # the evaluation belongs to the stage that produced the update.
                evaluated_stage = _stage_for_update(state["global_update"], budgets)
                publish(state["global_update"], evaluated_stage, result=read(path))
        elif not from_scratch:
            trainer.model.load_state_dict(torch.load(initial_policy, map_location=device, weights_only=False)["model"])
        while True:
            stage = state["stage"]
            stage_budget = budgets[stage]
            print(f"Fixed stage {stage + 1}/4 {STAGES[stage]} counts={counts[stage]} update={trainer.update}/{stage_budget}", flush=True)
            status(trainer)
            while trainer.update < stage_budget:
                metrics = trainer.step()
                state["global_update"] += 1
                update = state["global_update"]
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(global_update=update, stage=stage,
                        stage_update=trainer.update, **metrics), allow_nan=False) + "\n")
                result = None
                if trainer.update % 2 == 0 or trainer.update == stage_budget:
                    result = isolated_evaluation(trainer.model, device, **evargs)
                    save(root / "validation" / f"{update:03d}.json", result)
                    atomic_save(dict(model=trainer.model.state_dict(), config=asdict(trainer.cfg),
                        global_update=update, stage=stage), root / f"policy_{update:06d}.pt")
                    persist(trainer)
                publish(update, stage, metrics, result)
                status(trainer)
                print(f"Fixed {STAGES[stage]} {trainer.update}/{stage_budget} global={update}/{total_budget} score={metrics.get('mean_score')}", flush=True)
                if trainer.update < stage_budget and stop_after is not None and update >= stop_after:
                    persist(trainer)
                    return root / "summary.json"
            # Also runs immediately after restoring a stage-end checkpoint.
            # Scores never enter this transition: exact stage budget is done.
            if trainer.tick != 0 or (not stagger_history and trainer.history.episode != 0):
                raise RuntimeError("Stage transition would discard an unfinished history group")
            atomic_save(dict(model=trainer.model.state_dict(), config=asdict(trainer.cfg),
                global_update=state["global_update"], stage=stage), root / f"finished_{STAGES[stage]}.pt")
            if stage == 3:
                state["status"] = "complete"
                persist(trainer)
                return root / "summary.json"
            next_trainer = build(stage + 1)
            next_trainer.model.load_state_dict(trainer.model.state_dict())
            next_trainer.optimizer.load_state_dict(trainer.optimizer.state_dict())
            # New stage restarts its warmup while preserving learned weights and
            # Adam moments. Kitchens/history reset only at the checked boundary.
            state["stage"] += 1
            trainer = next_trainer
            persist(trainer)
            if stop_after is not None and state["global_update"] >= stop_after:
                return root / "summary.json"
    finally:
        run.finish()


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="counter-fixed-504x800-lr0003-seed1")
    parser.add_argument("--bc-policy", type=Path, default=RUNS / "counter-bc-shaped-seed1/bc/bc_best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=504)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--rollout-steps", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=900)
    parser.add_argument("--focus-lr", type=float, default=.0003)
    parser.add_argument("--joint-lr", type=float, default=.0001)
    parser.add_argument("--entropy-coef", type=float, default=.03)
    parser.add_argument("--warmup-fraction", type=float, default=.02)
    parser.add_argument("--warmup-start-factor", type=float, default=.1)
    parser.add_argument("--max-grad-norm", type=float, default=15.0)
    parser.add_argument("--stagger-history", action="store_true")
    parser.add_argument("--controller-version", default="parallel-v2")
    parser.add_argument("--stage-updates", default=None, help="Single int or 4 comma-separated ints (left,rcook,rcour,joint)")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--from-scratch", action="store_true", help="Random initialization; no BC or demonstration rehearsal")
    args = vars(parser.parse_args())
    campaign = args.pop("campaign")
    if not campaign or not campaign.replace("-", "").replace("_", "").isalnum():
        parser.error("Use a simple campaign name")
    run_fixed_curriculum(RUNS / campaign, args.pop("bc_policy"), **args)


if __name__ == "__main__":
    main()

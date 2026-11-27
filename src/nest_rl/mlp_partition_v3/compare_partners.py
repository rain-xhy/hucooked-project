"""Evaluate frozen L1 policies and baselines under both partner versions."""
import argparse
import hashlib
from pathlib import Path
import torch
from .evaluation import evaluate
from .experiment import TEST_SEED, save, read, metrics
from .policy import V3Policy
from .tracking import RUNS, open_run

RECIPE_REWARDS = (20, 10, 50)
BEST_UPDATES = {1: 60, 2: 50, 3: 65}


def fingerprint(controller, checkpoint=None):
    module = Path(__file__).parent / ("legacy_partners.py" if controller == "serial-v1" else "partners.py")
    values = dict(controller_source_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
    if checkpoint is not None:
        values["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return values


def compare(root, source, device="cuda", replays=True, controllers=None):
    config = dict(test_seed=TEST_SEED, groups=10, episodes=450, horizon=400,
                  recipe_rewards=list(RECIPE_REWARDS), controllers=["serial-v1", "parallel-v2"],
                  checkpoints={str(s): str(source / f"formal-lr0.001-seed{s}" / f"policy_{u:06}.pt")
                               for s, u in BEST_UPDATES.items()},
                  metric_definitions=dict(dual_pots_active_steps="Both pots nonempty after transition",
                      dual_pots_progress_steps="Both pots receive ingredient, heating step, or ready-meal pickup in same transition",
                      idle_with_outstanding_work_steps="Chef STAY steps while any ticket or physical work remains; opportunity proxy, not causal avoidable-idle proof",
                      cross_wall_pickups="Actual pickup on opposite side from recorded depositor; excludes same-side retrieval"))
    manifest = root / "config.json"
    if manifest.exists() and read(manifest) != config:
        raise ValueError("Comparison configuration differs; choose a new output directory")
    save(manifest, config)
    rows = {}
    for controller in config["controllers"]:
        for label in ("script", "wait", *[f"policy-seed{s}-u{u}" for s,u in BEST_UPDATES.items()]):
            name = f"{controller}-{label}"
            if (root / f"{name}.json").exists():
                rows[name] = read(root / f"{name}.json")
    for controller in controllers or config["controllers"]:
        for label, mode, policy_seed in [("script", "script", None), ("wait", "wait", None),
                                         *[(f"policy-seed{s}-u{u}", "policy", s) for s,u in BEST_UPDATES.items()]]:
            name = f"{controller}-{label}"
            path = root / f"{name}.json"
            expected_fingerprint = fingerprint(controller, config["checkpoints"][str(policy_seed)] if policy_seed is not None else None)
            if path.exists():
                rows[name] = read(path)
                if rows[name].get("execution_fingerprint") != expected_fingerprint:
                    raise ValueError(f"Cached result has a different or missing source fingerprint: {path}")
                continue
            model = None
            if policy_seed is not None:
                model = V3Policy().to(device)
                state = torch.load(config["checkpoints"][str(policy_seed)], weights_only=True, map_location=device)
                model.load_state_dict(state["model"])
                model.eval()
            print(f"Evaluating {name}: 450 paired episodes", flush=True)
            result = evaluate(model, seed=TEST_SEED, groups=10, mode=mode, device=device, horizon=400,
                              controller_version=controller, recipe_rewards=RECIPE_REWARDS,
                              replay_dir=root / "replays" / name if replays else None)
            result["execution_fingerprint"] = expected_fingerprint
            result["comparison_result_version"] = 2
            run = open_run(f"{root.name}-{name}", "partner-comparison", dict(config, controller_version=controller,
                           mode=mode, policy_seed=policy_seed), resume=True)
            try:
                run.log(metrics(result), step=0)
            finally:
                run.finish()
            save(path, result)
            rows[name] = result
            print(f"{name}: mean_score={result['mean_score']:.3f}", flush=True)
    summary = dict(config=config, status="complete" if len(rows) == 10 else "partial", results={name: {key: row[key] for key in
                   ("episodes", "total_score", "mean_score", "macro_delivery_rate", "cross_delivery_rate",
                    "deliveries_per_episode", "dual_pots_active_steps", "dual_pots_progress_steps",
                    "idle_with_outstanding_work_steps", "idle_actionable_ingredient_steps", "cross_wall_pickups", "per_recipe",
                    "execution_fingerprint")} for name,row in rows.items()})
    save(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS / "parallel-v2-comparison")
    parser.add_argument("--source", type=Path, default=RUNS / "counter-400-lr001")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--controllers", nargs="+", choices=("serial-v1", "parallel-v2"))
    args = parser.parse_args()
    compare(args.root, args.source, args.device, not args.no_replays, args.controllers)

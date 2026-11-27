"""Train three counter-layout policies at a fixed learning rate and evaluate them."""
import argparse
import json
import time
from numbers import Real

import torch

from .evaluation import evaluate
from .policy import V3Policy
from .tracking import RUNS, open_run, recorded_steps


VALIDATION_SEED = 10_000_000
TEST_SEED = 20_000_000
HORIZON = 400
ROLLOUT = 800
SAMPLES = 63 * ROLLOUT
# This entry point reproduces the completed L1 experiment.
CONTROLLER_VERSION = "serial-v1"
RECIPE_REWARDS = (20., 20., 20.)
RANK_METRICS = ("macro_delivery_rate", "cross_delivery_rate", "deliveries_per_episode")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers can briefly open a status file without delete sharing.
    # Keep atomic replacement, but tolerate a short-lived sharing/access denial.
    for attempt in range(30):
        try:
            tmp.replace(path)
            break
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 29:
                raise
            time.sleep(.1)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(result):
    """Keep common curves easy to find. The JSON also has every subgroup."""
    output = {}
    def collect(prefix, row):
        for name, value in row.items():
            if isinstance(value, Real):
                output[prefix + name] = value
    collect("", result)
    for section in ("per_recipe", "per_role", "per_style"):
        for name, row in result[section].items():
            collect(f"{section}/{name}/", row)
    for i, row in enumerate(result["per_episode"], start=1):
        if row["partner_accuracy"] is not None:
            output[f"episode_{i}/partner_accuracy"] = row["partner_accuracy"]
    return output


def rank(rows):
    return tuple(sum(row[key] for row in rows) / len(rows) for key in RANK_METRICS)


def publish_comparison(root, summary):
    """Place screen scores and independent-test comparisons together in Trackio."""
    name = f"{root.name}-comparison"
    existing = recorded_steps(name)
    points = {}
    for candidate in summary.get("candidates", []):
        if "rank" in candidate:
            values = points.setdefault(0, {})
            for metric, score in zip(RANK_METRICS, candidate["rank"]):
                values[f"screen/lr_{candidate['lr']:g}/{metric}"] = score
    if "selected_lr" in summary:
        points.setdefault(0, {})["learning_rate"] = summary["selected_lr"]
    for seed, result in summary.get("formal", {}).items():
        values = {f"test/{mode}/macro_delivery_rate": score
                  for mode, score in result["macro_delivery_rate"].items()}
        values.update(history_gain=result["history_gain"], gain_over_wait=result["gain_over_wait"],
                      best_update=result["best_update"])
        points[int(seed)] = values
    pending = sorted(points.keys() - existing)
    if pending:
        run = open_run(name, "comparison", dict(step_axis="0=screen, 1/2/3=formal training seed"), True)
        try:
            for step in pending:
                run.log(points[step], step=step)
        finally:
            run.finish()


def evaluate_once(root, name, mode, groups, seed, model=None, step=0):
    path = root / f"{name}.json"
    if path.exists():
        return read(path)
    report = evaluate(model, seed=seed, mode=mode, groups=groups, horizon=HORIZON,
                      controller_version=CONTROLLER_VERSION, recipe_rewards=RECIPE_REWARDS)
    run = open_run(f"{root.name}-{name}", "evaluation",
                   dict(mode=mode, groups=groups, seed=seed, horizon=HORIZON,
                        controller_version=CONTROLLER_VERSION, recipe_rewards=RECIPE_REWARDS), True)
    try:
        run.log(metrics(report), step=step)
    finally:
        run.finish()
    save(path, report)
    return report


def fit(root, phase, lr, seed, updates):
    from .train import train
    name = f"{phase}-lr{lr:g}-seed{seed}"
    folder = root / name
    validations = folder / "validation"
    if (folder / "finished.json").is_file():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(folder.glob("checkpoint_*.pt"))
    config = dict(phase=phase, seed=seed, learning_rate=lr, updates=updates, layout="counter",
                  num_envs=63, horizon=HORIZON, rollout_steps=ROLLOUT, mini_batch=900, epochs=4, warmup_fraction=.02,
                  gamma=.99, gae_lambda=.95, clip=.2, entropy=.03, value_loss=.5,
                  max_grad_norm=15, adam_eps=1e-8, class_loss=.1, history=[5, 40],
                  hidden=256, validation_seed=VALIDATION_SEED,
                  controller_version=CONTROLLER_VERSION, recipe_rewards=RECIPE_REWARDS)
    save(folder / "experiment.json", config)
    run_name = f"{root.name}-{name}"
    published = recorded_steps(run_name)
    run = open_run(run_name, phase, config, True)

    def report(update, model, losses):
        step = SAMPLES * update
        values = dict(losses, env_steps=step, update=update)
        if update % 5 == 0:
            result = evaluate(model, seed=VALIDATION_SEED, horizon=HORIZON,
                              controller_version=CONTROLLER_VERSION, recipe_rewards=RECIPE_REWARDS)
            save(validations / f"{update:03}.json", result)
            values.update({"eval/" + k: v for k, v in metrics(result).items()})
        if step not in published:
            run.log(values, step=step)
        print(f"{name} {update}/{updates} env_steps={step} "
              f"eval={values.get('eval/macro_delivery_rate', '-')}", flush=True)

    try:
        if checkpoints:
            restored = int(checkpoints[-1].stem.split("_")[-1])
            journal = folder / "train.jsonl"
            history = {}
            if journal.exists():
                for line in journal.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    history[row["update"]] = row
            for update in range(1, restored + 1):
                step = update * SAMPLES
                path = validations / f"{update:03}.json"
                if update % 5 == 0 and not path.exists():
                    report(update, load_policy(folder, update), history.get(update, {}))
                elif step not in published and update in history:
                    values = history[update].copy()
                    if update % 5 == 0:
                        values.update({"eval/" + k: v for k, v in metrics(read(path)).items()})
                    run.log(values, step=step)
        train(folder, lr, seed, updates, resume=checkpoints[-1] if checkpoints else None,
              callback=report, horizon=HORIZON, rollout_steps=ROLLOUT,
              controller_version=CONTROLLER_VERSION, recipe_rewards=RECIPE_REWARDS)
        # Recover a finished update whose validation callback was interrupted.
        if not (validations / f"{updates:03}.json").exists():
            report(updates, load_policy(folder, updates), {})
    finally:
        run.finish()
    save(folder / "finished.json", dict(updates=updates))
    return folder


def load_policy(folder, update):
    model = V3Policy().cuda()
    state = torch.load(folder / f"policy_{update:06}.pt", weights_only=True, map_location="cuda")
    model.load_state_dict(state["model"])
    return model.eval()


def campaign(root):
    root.mkdir(parents=True, exist_ok=True)
    lr = .001
    config = dict(lr=lr, horizon=HORIZON, rollout_steps=ROLLOUT, num_envs=63, updates=80, seeds=[1, 2, 3])
    manifest = root / "config.json"
    if (manifest.exists() and read(manifest) != config) or (
            not manifest.exists() and (root / "script-preflight.json").exists()):
        raise ValueError("This campaign already contains a different configuration. Use a new campaign name.")
    save(manifest, config)
    baseline = evaluate_once(root, "script-preflight", "script", 1, VALIDATION_SEED)
    if any(row["delivered"] == 0 for style in baseline["per_style"].values()
           for row in style["per_recipe"].values()):
        raise RuntimeError("Scripted preflight missed a style/recipe combination")
    evaluate_once(root, "wait-diagnostic", "wait", 1, VALIDATION_SEED)
    summary = dict(selected_lr=lr, status="formal_training", formal={}, **config)
    save(root / "summary.json", summary)
    publish_comparison(root, summary)
    for seed in (1, 2, 3):
        folder = fit(root, "formal", lr, seed, 80)
        best = max(range(5, 81, 5), key=lambda u: rank([read(folder / "validation" / f"{u:03}.json")]))
        model = load_policy(folder, best)
        scores = {}
        for mode in ("policy", "empty", "script", "wait"):
            controlled = mode in ("policy", "empty")
            name = f"test-s{seed}-{mode}" if controlled else f"test-{mode}"
            result = evaluate_once(root, name, mode, 10, TEST_SEED,
                                   model=model if controlled else None, step=best * SAMPLES if controlled else 0)
            scores[mode] = result["macro_delivery_rate"]
        summary["formal"][seed] = dict(best_update=best, macro_delivery_rate=scores,
                                       history_gain=scores["policy"]-scores["empty"],
                                       gain_over_wait=scores["policy"]-scores["wait"])
        save(root / "summary.json", summary)
        publish_comparison(root, summary)
    summary["status"] = "complete"
    save(root / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="counter-400-lr001", help="Name for saved files and Trackio runs")
    args = parser.parse_args()
    if not args.campaign or not args.campaign.replace("-", "").replace("_", "").isalnum():
        parser.error("Use letters, numbers, hyphens or underscores for the campaign name")
    campaign(RUNS / args.campaign)

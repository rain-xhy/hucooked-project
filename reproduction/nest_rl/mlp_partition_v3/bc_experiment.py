"""Script imitation followed by shaped PPO, with a compact cooperation dashboard."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .evaluation import evaluate, ROLE_NAMES
from .experiment import read, save
from .policy import V3Policy
from .tracking import COOPERATION_PROJECT, RUNS, open_run, recorded_steps

VALIDATION_SEED = 10_000_000
RECIPES = ("tomato_egg", "potato", "potato_tomato_beef")


def compact_metrics(losses=None, result=None, wait=None, script=None):
    """Only behavioral outcomes go to Trackio; complete diagnostics stay on disk."""
    output = {}
    for source, target in (("mean_score", "train/score_per_episode"),
                           ("mean_shaping_reward", "train/process_reward_per_episode")):
        if losses and source in losses:
            output[target] = losses[source]
    if result is not None:
        output.update({"eval/score_per_episode": result["mean_score"],
                       "eval/deliveries_per_episode": result["deliveries_per_episode"]})
        if wait is not None:
            output["eval/wait_score"] = wait["mean_score"]
            output["eval/model_contribution"] = result["mean_score"] - wait["mean_score"]
            for role in ROLE_NAMES:
                output[f"contribution/{role}"] = (result["per_role"][role]["mean_score"]
                                                    - wait["per_role"][role]["mean_score"])
        if script is not None:
            output["eval/script_score"] = script["mean_score"]
        for recipe in RECIPES:
            row = result["per_recipe"][recipe]
            if row["delivery_rate"] is not None:
                output[f"delivery_rate/{recipe}"] = row["delivery_rate"]
            if row["mean_delivery_steps"] is not None:
                output[f"delivery_steps/{recipe}"] = row["mean_delivery_steps"]
    return output


def load_policy(path, device):
    model = V3Policy().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def repair_validation_log(folder, restored, samples, published, run, wait, script):
    """Repair a callback interrupted after JSON persistence but before logging."""
    path = folder / "validation" / f"{restored:03}.json"
    step = restored * samples
    if path.is_file() and step not in published:
        run.log(compact_metrics(result=read(path), wait=wait, script=script), step=step)
        published.add(step)


def fit(root, initial_policy, *, seed=1, updates=80, device="cuda", stop_after=None,
        num_envs=63, horizon=400, rollout_steps=800, minibatch_size=900, epochs=4):
    from .train import train
    root = Path(root)
    folder = root / "ppo"
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(lr=.001, seed=seed, updates=updates, num_envs=num_envs,
                  horizon=horizon, rollout_steps=rollout_steps, minibatch_size=minibatch_size,
                  epochs=epochs, controller_version="parallel-v2", recipe_rewards=[20., 10., 50.],
                  shaping=True, initial_policy=str(Path(initial_policy).resolve()),
                  validation_seed=VALIDATION_SEED, validation_episodes=45)
    manifest = folder / "experiment.json"
    if manifest.exists() and read(manifest) != config:
        raise ValueError("Existing campaign configuration differs; use a new output directory")
    save(manifest, config)
    evargs = dict(seed=VALIDATION_SEED, groups=1, device=device, horizon=horizon,
                  controller_version="parallel-v2", recipe_rewards=(20., 10., 50.))
    baselines = {}
    for mode in ("wait", "script"):
        path = root / f"validation-{mode}.json"
        if not path.exists():
            save(path, evaluate(None, mode=mode, **evargs))
        baselines[mode] = read(path)
    checkpoints = sorted(folder.glob("checkpoint_*.pt"))
    samples = num_envs * rollout_steps
    name = f"{root.name}-ppo-seed{seed}"
    published = recorded_steps(name, project=COOPERATION_PROJECT)
    run = open_run(name, "bc-then-ppo", config, True, project=COOPERATION_PROJECT)

    def report(update, model, losses):
        step = update * samples
        result = None
        if update % 5 == 0 or update == updates:
            path = folder / "validation" / f"{update:03}.json"
            # Re-evaluate after a replayed update: its policy may differ from a
            # previously interrupted update that had not reached a full save.
            result = evaluate(model, **evargs)
            save(path, result)
        if step not in published:
            run.log(compact_metrics(losses, result, baselines["wait"], baselines["script"]), step=step)
            published.add(step)
        print(f"PPO {update}/{updates}: train score={losses.get('mean_score', '-')}; "
              f"eval score={result['mean_score'] if result else '-'}", flush=True)

    try:
        if 0 not in published:
            model = load_policy(initial_policy, device)
            initial = evaluate(model, **evargs)
            save(folder / "validation" / "000.json", initial)
            run.log(compact_metrics(result=initial, wait=baselines["wait"], script=baselines["script"]), step=0)
        if checkpoints:
            restored = int(checkpoints[-1].stem.split("_")[-1])
            # A full save precedes callback evaluation. Repair that boundary if
            # the prior process stopped between the two operations.
            path = folder / "validation" / f"{restored:03}.json"
            if (restored % 5 == 0 or restored == updates) and not path.exists():
                report(restored, load_policy(folder / f"policy_{restored:06}.pt", device), {})
            repair_validation_log(folder, restored, samples, published, run,
                                  baselines["wait"], baselines["script"])
        final = train(folder, .001, seed, updates, device=device,
                      resume=checkpoints[-1] if checkpoints else None,
                      init_policy=None if checkpoints else initial_policy,
                      callback=report, stop_after=stop_after, num_envs=num_envs,
                      horizon=horizon, rollout_steps=rollout_steps, epochs=epochs,
                      minibatch_size=minibatch_size, controller_version="parallel-v2",
                      recipe_rewards=(20., 10., 50.), shaping=True)
    finally:
        run.finish()
    validations = sorted((folder / "validation").glob("*.json"))
    best = max(validations, key=lambda path: (read(path)["macro_delivery_rate"] or 0,
                                             read(path)["cross_delivery_rate"] or 0,
                                             read(path)["mean_score"]))
    completed = int(final.stem.split("_")[-1])
    summary = dict(status="complete" if completed == updates else "paused", last_update=completed,
                   best_update=int(best.stem), best_validation=read(best),
                   wait_baseline=baselines["wait"], script_baseline=baselines["script"],
                   last_checkpoint=str(final))
    save(root / "summary.json", summary)
    return final


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="counter-bc-shaped-seed1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--bc-policy", type=Path, help="Reuse an already completed BC artifact")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--bc-episodes", type=int, default=10)
    parser.add_argument("--bc-epochs", type=int, default=10)
    args = parser.parse_args()
    if not args.campaign or not args.campaign.replace("-", "").replace("_", "").isalnum():
        parser.error("Use letters, numbers, hyphens and underscores in campaign names")
    root = RUNS / args.campaign
    root.mkdir(parents=True, exist_ok=True)
    if args.bc_policy is None:
        from .pretrain import BCSettings, run_pretraining
        bc_cfg = BCSettings(seed=args.seed, episodes=args.bc_episodes, epochs=args.bc_epochs,
                            validation_seed=30_000_000)
        finished = root / "bc-finished.json"
        if finished.exists():
            previous = read(finished)
            if previous["config"] != asdict(bc_cfg):
                raise ValueError("BC configuration changed; use a new campaign")
            args.bc_policy = Path(previous["policy"])
            if not args.bc_policy.is_file():
                raise FileNotFoundError(args.bc_policy)
        else:
            run = open_run(f"{root.name}-bc", "behavior-cloning", asdict(bc_cfg), True,
                           project=COOPERATION_PROJECT)
            def callback(record):
                values = record["validation"]
                run.log({"bc/validation_nll": values["nll"],
                         "bc/validation_accuracy": values["action_accuracy"]}, step=record["epoch"])
            try:
                args.bc_policy = run_pretraining(bc_cfg, root / "bc", device=args.device, callback=callback)
                save(finished, dict(config=asdict(bc_cfg), policy=str(Path(args.bc_policy).resolve())))
            finally:
                run.finish()
    fit(root, args.bc_policy, seed=args.seed, updates=args.updates, device=args.device,
        stop_after=args.stop_after)


if __name__ == "__main__":
    main()

"""Manifest-driven native evaluation of the selected SharedKitchen policies.

Run from ``reproduction/stq`` with the pace_mixed environment active, for
example::

    python scripts/evaluate_final_suite.py --gpus 1 --episodes 12 --seed 100001

Every experiment runs in its own process so CUDA state is isolated.  The
manifest selects the checkpoint; this script never substitutes ``best.pt``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = Path("experiments.json")
DEFAULT_OUTPUT = Path("results/final_evaluation.json")
DEFAULT_SEED = 100001
DEFAULT_EPISODES = 12
METRICS = (
    "team_score", "meals_completed", "normal_completed", "joint_completed",
    "joint_successes", "joint_attempt_success_rate", "joint_wait_steps",
    "three_contributor_meals", "resource_conflicts", "movement_conflicts",
    "ticks", "meals_per_100_steps",
)
FAMILIES = {"lipo", "joint_lipo", "comedi", "joint_comedi", "hipt", "joint_hipt", "ppo"}


def _root_path(path: str | Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def load_manifest(path: str | Path, root: Path = ROOT) -> tuple[dict, Path]:
    manifest_path = _root_path(path, root).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments = data.get("experiments") if isinstance(data, dict) else None
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Manifest must contain a non-empty top-level 'experiments' list")
    required = {"experiment_id", "family", "result_dir", "checkpoint", "actual_env_steps", "training_state"}
    seen: set[str] = set()
    for index, item in enumerate(experiments):
        if not isinstance(item, dict):
            raise ValueError(f"experiments[{index}] must be an object")
        missing = required - item.keys()
        if missing:
            raise ValueError(f"experiments[{index}] missing fields: {', '.join(sorted(missing))}")
        experiment_id = str(item["experiment_id"])
        if not experiment_id or experiment_id in seen:
            raise ValueError(f"Missing or duplicate experiment_id: {experiment_id!r}")
        seen.add(experiment_id)
        if item["family"] not in FAMILIES:
            raise ValueError(f"Unsupported family for {experiment_id}: {item['family']!r}")
        if not isinstance(item["actual_env_steps"], int) or item["actual_env_steps"] < 0:
            raise ValueError(f"actual_env_steps must be a nonnegative integer for {experiment_id}")
        checkpoint_name = Path(str(item["checkpoint"])).name.lower()
        if checkpoint_name == "best.pt":
            raise ValueError(f"Final-policy manifest cannot select best.pt: {experiment_id}")
        if item["family"] == "ppo" and (not item.get("backbone") or not item.get("model_size")):
            raise ValueError(f"PPO entry needs backbone and model_size: {experiment_id}")
    return data, manifest_path


def select_experiments(experiments: list[dict], only: list[str] | None) -> list[dict]:
    if not only:
        return list(experiments)
    by_id = {str(item["experiment_id"]): item for item in experiments}
    unknown = sorted(set(only) - by_id.keys())
    if unknown:
        raise ValueError("Unknown experiment_id(s): " + ", ".join(unknown))
    if len(set(only)) != len(only):
        raise ValueError("--only contains duplicate experiment ids")
    return [by_id[item] for item in only]


def resolve_checkpoint_path(item: dict, root: Path = ROOT) -> Path:
    checkpoint = Path(str(item["checkpoint"]))
    if checkpoint.is_absolute():
        return checkpoint.resolve()
    direct = root / checkpoint
    if direct.is_file() or len(checkpoint.parts) > 1:
        return direct.resolve()
    return (root / str(item["result_dir"]) / checkpoint).resolve()


def extract_checkpoint_steps(family: str, checkpoint: dict) -> int | None:
    for key in ("actual_env_steps", "total_env_steps", "train_env_steps", "env_steps"):
        value = checkpoint.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    if family == "comedi":
        counters = checkpoint.get("counters", {})
        for key in ("total_train_env_steps", "total_env_steps", "total_rl_env_steps"):
            value = counters.get(key) if isinstance(counters, dict) else None
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def ppo_architecture(item: dict, checkpoint: dict) -> tuple[str, str]:
    payload_config = checkpoint.get("config", {})
    if not isinstance(payload_config, dict):
        payload_config = {}
    checkpoint_backbone = checkpoint.get("backbone")
    checkpoint_size = checkpoint.get("model_size") or payload_config.get("model_size")
    backbone = checkpoint_backbone or item.get("backbone")
    model_size = checkpoint_size or item.get("model_size")
    if not backbone or not model_size:
        raise ValueError("PPO checkpoint/manifest must define backbone and model_size")
    if checkpoint_backbone and item.get("backbone") and checkpoint_backbone != item["backbone"]:
        raise ValueError(f"PPO backbone mismatch: manifest={item['backbone']} payload={checkpoint_backbone}")
    if checkpoint_size and item.get("model_size") and checkpoint_size != item["model_size"]:
        raise ValueError(f"PPO model_size mismatch: manifest={item['model_size']} payload={checkpoint_size}")
    return str(backbone), str(model_size)


def native_metrics(record: dict) -> dict[str, int | float]:
    missing = [key for key in METRICS if key not in record]
    if missing:
        raise RuntimeError("Missing native metric(s): " + ", ".join(missing))
    result: dict[str, int | float] = {}
    for key in METRICS:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Native metric {key} is not a finite number")
        result[key] = int(value) if key == "ticks" else float(value)
    return result


def summarize(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("No native episode metrics were collected")
    means, stds = {}, {}
    for key in METRICS:
        values = [float(row[key]) for row in records]
        mean = sum(values) / len(values)
        means[key] = mean
        stds[key] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return {
        "episodes": len(records),
        "means": means,
        "stds": stds,
        "episode_metrics": records,
        "episode_scores": [row["team_score"] for row in records],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_native_actions(action_fn, *, episodes: int, seed: int, gamma: float = 0.9985) -> list[dict]:
    import numpy as np
    from baseline_common.envs import KitchenBatch

    env = KitchenBatch(num_envs=episodes, seed=seed, stage="standard", workers=0,
                       train=False, gamma=gamma)
    records: list[dict] = []
    try:
        observation = env.reset()
        while len(records) < episodes:
            actions = np.asarray(action_fn(observation))
            if actions.shape != (episodes, 3):
                raise RuntimeError(f"Policy actions have shape {actions.shape}, expected {(episodes, 3)}")
            observation, _, dones, infos = env.step(actions.astype(np.int64, copy=False))
            for done, info in zip(dones, infos):
                if done and len(records) < episodes:
                    metrics = info.get("metrics")
                    if not isinstance(metrics, dict):
                        raise RuntimeError("Native episode ended without info['metrics']")
                    record = native_metrics(metrics)
                    if record["ticks"] != 300:
                        raise RuntimeError(f"Expected a 300-tick native episode, got {record['ticks']}")
                    records.append(record)
    finally:
        env.close()
    return records


def _lipo_evaluation(checkpoint: dict, family: str, device, episodes: int, seed: int):
    import numpy as np
    import torch

    if family == "lipo":
        from lipo.models import LipoMember as Member, ModelConfig as Config
    else:
        from joint_lipo.model import JointLipoMember as Member, JointLipoConfig as Config
    config = Config(**checkpoint["model_config"])
    states = checkpoint.get("member_states")
    if not isinstance(states, list) or len(states) != 8:
        raise RuntimeError(f"LIPO final checkpoint must contain all 8 members, got {len(states or [])}")
    combined, member_results = [], []
    for index, member_state in enumerate(states):
        member = Member(config)
        member.load_state_dict(member_state["model"] if "model" in member_state else member_state, strict=True)
        member.to(device).eval()
        latent_rng = np.random.default_rng(seed + index * 1009)
        latent = torch.as_tensor(latent_rng.integers(config.latent_dim, size=(episodes, 3)),
                                 dtype=torch.long, device=device)
        if family == "lipo":
            roles = torch.arange(3, device=device).repeat(episodes)

            def act(observation, model=member, z=latent, role=roles):
                vector = torch.as_tensor(observation["vector"], dtype=torch.float32, device=device)
                vector = vector.reshape(-1, config.obs_dim)
                return model.actor_logits(vector, role, z.reshape(-1)).argmax(-1).reshape(episodes, 3).cpu().numpy()
        else:
            def act(observation, model=member, z=latent):
                state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device)
                return model.actor(state, z).argmax(-1).cpu().numpy()
        records = _collect_native_actions(act, episodes=episodes, seed=seed)
        combined.extend(records)
        member_results.append({"member_index": index, **summarize(records)})
        del member
    return combined, member_results, {
        "policy": "all 8 final population members; each controls a full 3-chef team",
        "stage": "population_final",
        "latent_rule": "independent z per chef and episode; seeded by seed + member_index * 1009",
    }


def _hipt_evaluation(checkpoint: dict, family: str, device, episodes: int, seed: int):
    if family == "hipt":
        from hipt.model import HiPTPolicy as Model
        from hipt.evaluate import evaluate
    else:
        from joint_hipt.model import JointHiPTPolicy as Model
        from joint_hipt.evaluate import evaluate
    arguments = checkpoint["config"]["arguments"]
    if checkpoint.get("stage") != "hierarchical":
        raise RuntimeError(f"HiPT checkpoint is not at the final hierarchical stage: {checkpoint.get('stage')}")
    model = Model(z_dim=int(arguments["z_dim"]))
    model.load_state_dict(checkpoint["hierarchical"]["model"], strict=True)
    model.to(device).eval()
    outcome = evaluate(model, episodes=episodes, num_envs=episodes, workers=0, seed=seed,
                       device=device, gamma=float(arguments.get("gamma", 0.9985)),
                       p_min=int(arguments.get("p_min", 20)), p_max=int(arguments.get("p_max", 40)))
    records = [native_metrics(row) for row in outcome["records"]]
    return records, None, {
        "policy": "final hierarchical policy",
        "stage": "hierarchical_final",
        "action_rule": "greedy high/low action logits; option durations sampled from the seeded evaluation RNG",
        "z_dim": int(arguments["z_dim"]),
    }


def _comedi_evaluation(checkpoint: dict, family: str, device, episodes: int, seed: int):
    import numpy as np
    import torch
    if family == "comedi":
        from comedi.model import ConvActor as Actor
        state = checkpoint["cbr"]["model"]
        actor = Actor()
    else:
        from joint_comedi.model import JointActor as Actor
        state = checkpoint["policy"]["model"]
        actor = Actor()
    actor_state = {key[6:]: value for key, value in state.items() if key.startswith("actor.")}
    if not actor_state:
        raise RuntimeError("Final CBR checkpoint does not contain actor weights")
    actor.load_state_dict(actor_state, strict=True)
    actor.to(device).eval()

    def act(observation):
        if family == "comedi":
            spatial = torch.as_tensor(observation["spatial"], dtype=torch.float32, device=device)
            vector = torch.as_tensor(observation["vector"], dtype=torch.float32, device=device)
            logits = actor(spatial.reshape(-1, 26, 7, 7), vector.reshape(-1, 866))
            return logits.argmax(-1).reshape(episodes, 3).cpu().numpy()
        spatial = torch.as_tensor(observation["spatial"][:, 0], dtype=torch.float32, device=device)
        state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device)
        return actor(spatial, state).argmax(-1).cpu().numpy()

    records = _collect_native_actions(act, episodes=episodes, seed=seed)
    return records, None, {"policy": "final CBR main actor", "stage": "cbr_final"}


def _ppo_evaluation(checkpoint: dict, item: dict, device, episodes: int, seed: int):
    import torch
    from joint_ppo.models import JointPolicy

    backbone, model_size = ppo_architecture(item, checkpoint)
    policy = JointPolicy(backbone, model_size=model_size)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    actor = policy.actor.to(device).eval()

    def act(observation):
        spatial = torch.as_tensor(observation["spatial"][:, 0], dtype=torch.float32, device=device)
        state = torch.as_tensor(observation["state"], dtype=torch.float32, device=device)
        return actor(spatial, state).argmax(-1).cpu().numpy()

    records = _collect_native_actions(act, episodes=episodes, seed=seed)
    return records, None, {
        "policy": "final PPO actor",
        "stage": "actor_final",
        "backbone": backbone,
        "model_size": model_size,
    }


def evaluate_experiment(item: dict, *, episodes: int, seed: int, device_name: str, physical_gpu: int | None) -> dict:
    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this evaluation process")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")

    path = resolve_checkpoint_path(item)
    if not path.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
    started = time.monotonic()
    digest, size = _sha256(path), path.stat().st_size
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Expected a dictionary checkpoint payload")
    family = str(item["family"])
    payload_steps = extract_checkpoint_steps(family, checkpoint)
    manifest_steps = int(item["actual_env_steps"])
    if payload_steps is not None and payload_steps != manifest_steps:
        raise RuntimeError(f"Checkpoint step mismatch: manifest={manifest_steps}, payload={payload_steps}")

    with torch.inference_mode():
        if family in ("lipo", "joint_lipo"):
            records, members, policy_info = _lipo_evaluation(checkpoint, family, device, episodes, seed)
        elif family in ("hipt", "joint_hipt"):
            records, members, policy_info = _hipt_evaluation(checkpoint, family, device, episodes, seed)
        elif family in ("comedi", "joint_comedi"):
            records, members, policy_info = _comedi_evaluation(checkpoint, family, device, episodes, seed)
        elif family == "ppo":
            records, members, policy_info = _ppo_evaluation(checkpoint, item, device, episodes, seed)
        else:
            raise RuntimeError(f"Unsupported family: {family}")
    expected = 8 * episodes if family in ("lipo", "joint_lipo") else episodes
    if len(records) != expected or any(row["ticks"] != 300 for row in records):
        raise RuntimeError(f"Incomplete evaluation: expected {expected} 300-tick episodes, got {len(records)}")

    result = {
        "experiment_id": item["experiment_id"],
        "family": family,
        "checkpoint": str(path),
        "checkpoint_bytes": size,
        "checkpoint_sha256": digest,
        "actual_env_steps": manifest_steps,
        "payload_env_steps": payload_steps,
        "training_state": item["training_state"],
        "physical_gpu": physical_gpu if device_name == "cuda" else None,
        "device": str(device),
        "evaluation_seed": seed,
        "episodes_per_policy": episodes,
        "policy_info": policy_info,
        **summarize(records),
        "members": members,
        "elapsed_seconds": time.monotonic() - started,
        "evaluation_status": "completed",
    }
    del checkpoint
    return result


def _run_child(item: dict, *, manifest_path: Path, episodes: int, seed: int,
               device: str, gpu: int | None) -> dict:
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--manifest", str(manifest_path), "--episodes", str(episodes),
               "--seed", str(seed), "--device", device,
               "--_job-id", str(item["experiment_id"])]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1", SDL_VIDEODRIVER="dummy",
                      SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu) if device == "cuda" and gpu is not None else ""
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    marker = "EVAL_RESULT "
    lines = [line[len(marker):] for line in process.stdout.splitlines() if line.startswith(marker)]
    if process.returncode != 0 or len(lines) != 1:
        tail = process.stdout[-12000:]
        raise RuntimeError(f"job {item['experiment_id']} failed (exit={process.returncode}):\n{tail}")
    return json.loads(lines[0])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate final SharedKitchen policies from experiments.json.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1],
                        help="Physical GPU ids assigned round-robin (default: 1)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Maximum concurrent isolated jobs (default: 4)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Episodes per policy/member (default: 12)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="+", help="Evaluate only these experiment_id values")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Use CUDA or CPU; CPU is intended for focused validation")
    parser.add_argument("--_job-id", dest="_job_id", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be positive")
    if args.device == "cuda" and not args.gpus:
        parser.error("--gpus must contain at least one physical GPU id")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicate physical GPU ids")
    try:
        manifest, manifest_path = load_manifest(args.manifest)
        all_experiments = manifest["experiments"]
        if args._job_id:
            item = next((entry for entry in all_experiments
                         if str(entry["experiment_id"]) == args._job_id), None)
            if item is None:
                raise ValueError(f"Unknown internal experiment id: {args._job_id}")
            physical_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"]) if args.device == "cuda" else None
            result = evaluate_experiment(item, episodes=args.episodes, seed=args.seed,
                                         device_name=args.device, physical_gpu=physical_gpu)
            print("EVAL_RESULT " + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
            return 0

        selected = select_experiments(all_experiments, args.only)
        if args.device == "cuda" and not args.gpus:
            parser.error("--gpus must contain at least one physical GPU id")
        results_by_id, errors = {}, []
        def run_one(index: int, item: dict):
            gpu = args.gpus[index % len(args.gpus)] if args.device == "cuda" else None
            print(json.dumps({"started": item["experiment_id"], "physical_gpu": gpu,
                              "device": args.device}), flush=True)
            try:
                return _run_child(item, manifest_path=manifest_path, episodes=args.episodes,
                                  seed=args.seed, device=args.device, gpu=gpu), None
            except Exception as error:
                return None, {"experiment_id": item["experiment_id"], "error": str(error)}

        with ThreadPoolExecutor(max_workers=min(len(selected), args.max_parallel)) as pool:
            futures = [pool.submit(run_one, index, item) for index, item in enumerate(selected)]
            for future in as_completed(futures):
                result, error = future.result()
                if result is not None:
                    results_by_id[result["experiment_id"]] = result
                    print("EVAL_RESULT " + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
                if error is not None:
                    errors.append(error)
        ordered_results = [results_by_id[str(item["experiment_id"])]
                           for item in selected if str(item["experiment_id"]) in results_by_id]
        output_path = _root_path(args.output).resolve()
        payload = {
            "schema_version": 1,
            "environment": manifest.get("environment", "SharedKitchen"),
            "manifest": str(manifest_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "seed": args.seed,
                "episodes_per_policy": args.episodes,
                "native_metrics": list(METRICS),
                "evaluation_mode": "train=False; raw SharedKitchen episode metrics",
                "action_selection": "greedy argmax; HiPT option durations use a seeded RNG",
                "max_parallel": args.max_parallel,
                "selection": "manifest checkpoint only; best.pt is rejected",
            },
            "selected_experiments": [str(item["experiment_id"]) for item in selected],
            "completed": len(ordered_results),
            "failed": errors,
            "results": ordered_results,
        }
        _write_json(output_path, payload)
        print(json.dumps({"suite_output": str(output_path), "completed": len(ordered_results),
                          "failed": len(errors)}, ensure_ascii=False), flush=True)
        return 1 if errors else 0
    except Exception as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

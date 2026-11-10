"""CLI: python -m reproduction.haha.train STAGE [options].

STAGE is prepare, bc, worker, manager or evaluate. No work is performed on import.
"""

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from . import ENVIRONMENT_COMMIT, UPSTREAM_COMMIT
from .artifacts import (check_loaded_policy, checkpoint_info, read_json, save_policy,
                        write_json)
from .bc import BCPartner, RandomPartner, prepare, train_bc
from .env import CookingEnv
from .subtasks import LAYOUTS


def arguments():
    parser = argparse.ArgumentParser(description="Original-environment HA²-BCP reproduction")
    parser.add_argument("stage", choices=("prepare", "bc", "worker", "manager", "evaluate"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "cache" / "2019_hh_trials_all.pickle",
                        help="Trusted author human-data pickle; prepare downloads it if absent")
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cpu",
                        help="Device for frozen Workers and BC partners in environment processes")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--task-timeout", type=int, default=40)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=.001)
    parser.add_argument("--timesteps", type=int, help="Additional steps; defaults to 31M Worker / 35M Manager")
    parser.add_argument("--max-hours", type=float,
                        help="Training cap (default BC=3h, Worker=18h, Manager=16h); BC divides it among candidates")
    parser.add_argument("--save-every", type=int, default=250_000)
    parser.add_argument("--resume", type=Path, help="Resume an SB3 checkpoint, including optimizer state")
    parser.add_argument("--resume-bc", action="store_true", help="Resume complete BC epochs and saved trajectory splits")
    parser.add_argument("--worker", type=Path, help="Fixed Worker; prefer RUN/worker/best.zip, otherwise last.zip")
    parser.add_argument("--manager", type=Path, help="Evaluation checkpoint; defaults to RUN/manager/last.zip")
    parser.add_argument("--unknown-penalty", type=float, default=-.1,
                        help="Author-code training penalty, not an evaluation team reward")
    parser.add_argument("--failure-limit", type=int, default=5,
                        help="Author-code Manager training cutoff; 0 disables it; disabled in evaluation")
    parser.add_argument("--no-worker-shaping", action="store_true")
    parser.add_argument("--epochs", type=int, default=300, help="BC epochs per candidate")
    parser.add_argument("--bc-hidden", type=int, default=64)
    parser.add_argument("--bc-lr", type=float, default=.001)
    parser.add_argument("--bc-batch-size", type=int, default=32)
    parser.add_argument("--bc-eval-every", type=int, default=10)
    parser.add_argument("--selection-episodes", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=1_000_000)
    parser.add_argument("--validation-episodes", type=int, default=40, help="Worker skill episodes per layout/seat")
    parser.add_argument("--validation-games", type=int, default=3, help="Manager full games per layout/seat")
    parser.add_argument("--validation-min-samples", type=int, default=3, help="Minimum observations for each core Worker skill")
    parser.add_argument("--validation-seconds", type=float, default=180)
    parser.add_argument("--evaluate-stage", choices=("worker", "manager"), default="manager")
    parser.add_argument("--partners", nargs="+", choices=("bc", "human_proxy", "random"),
                        default=["human_proxy", "random"])
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per layout, partner and player seat")
    parser.add_argument("--deterministic", action="store_true", help="Greedy trained-policy evaluation")
    parser.add_argument("--trajectories", type=int, default=0,
                        help="Export this many native JSONL trajectories per manager evaluation condition")
    args = parser.parse_args()
    if args.max_hours is None:
        args.max_hours = {"bc": 3, "worker": 18, "manager": 16}.get(args.stage, 1)
    positive = ("torch_threads", "horizon", "task_timeout", "n_envs", "hidden", "n_steps",
                "batch_size", "ppo_epochs", "max_hours", "save_every", "epochs", "bc_hidden",
                "bc_batch_size", "bc_eval_every", "selection_episodes", "episodes",
                "validate_every", "validation_episodes", "validation_games",
                "validation_min_samples", "validation_seconds")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.bc_lr <= 0:
        parser.error("learning rates must be positive")
    if args.timesteps is not None and args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.failure_limit < 0 or args.trajectories < 0:
        parser.error("failure-limit and trajectories cannot be negative")
    if len(set(args.layouts)) != len(args.layouts):
        parser.error("--layouts cannot contain duplicates")
    if args.stage in {"worker", "manager"}:
        if args.n_envs < len(args.layouts):
            parser.error("n-envs must cover all selected layouts; reduce --layouts if necessary")
        if args.n_steps * args.n_envs < 2 or args.batch_size < 2:
            parser.error("PPO rollout and batch sizes must be at least two")
        if (args.n_steps * args.n_envs) % args.batch_size:
            parser.error("batch-size must divide n-steps * n-envs")
    if args.resume and args.stage not in {"worker", "manager"}:
        parser.error("--resume currently supports Worker/Manager PPO stages only")
    if args.resume_bc and args.stage != "bc":
        parser.error("--resume-bc is only valid for the BC stage")
    args.run_dir = (args.run_dir or Path(__file__).parent / "runs" / f"bcp_seed{args.seed}").resolve()
    args.dataset = args.dataset.resolve()
    preferred_worker = args.run_dir / "worker" / "best.zip"
    args.worker = (args.worker or (preferred_worker if preferred_worker.is_file()
                                  else args.run_dir / "worker" / "last.zip")).resolve()
    preferred_manager = args.run_dir / "manager" / "best.zip"
    args.manager = (args.manager or (preferred_manager if preferred_manager.is_file()
                                    else args.run_dir / "manager" / "last.zip")).resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    return args


def make_env(args, stage, layout, rank, *, evaluation=False, partner_role="bc", player=None):
    torch.set_num_threads(args.torch_threads)
    partner = (RandomPartner() if partner_role == "random" else BCPartner.load(
        args.run_dir / "bc" / layout / f"{partner_role}.pt", args.inference_device
    ))
    worker = None
    if stage == "manager":
        worker = PPO.load(args.worker, device=args.inference_device)
        check_loaded_policy(worker, checkpoint_info(args.worker))
        worker.policy.set_training_mode(False)
        worker.policy.requires_grad_(False)
    env = CookingEnv(layout, partner, stage=stage, worker=worker, horizon=args.horizon,
                     task_timeout=args.task_timeout, player=player, evaluation=evaluation,
                     worker_shaping=not args.no_worker_shaping,
                     unknown_penalty=args.unknown_penalty, failure_limit=args.failure_limit,
                     deterministic=args.deterministic if evaluation else False)
    env.action_space.seed(args.seed + rank)
    return env


class TrainingProgress(BaseCallback):
    def __init__(self, args, output, config, deadline, session):
        super().__init__()
        self.args, self.output, self.config = args, output, config
        self.deadline, self.session = deadline, session
        self.best_score = None
        best = output / "best.json"
        if best.exists():
            saved = read_json(best)
            if saved["artifact_id"] == config["artifact_id"]:
                self.best_score = saved["config"].get("validation_score")

    def _on_training_start(self):
        self.started = time.monotonic()
        self.start_steps = self.model.num_timesteps
        self.next_save = self.start_steps + self.args.save_every
        self.next_validation = self.start_steps + self.args.validate_every
        self.stream = (self.session / "rollout_events.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["timesteps", "layout", "player", "subtask", "success",
                              "invalidated", "timeout", "steps", "worker_failed"])
        save_policy(self.output / "last.zip", self.model, self.config)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done or info.get("worker_failed"):
                self.writer.writerow([
                    self.model.num_timesteps, info.get("layout"), info.get("player"),
                    info.get("subtask"), info.get("task_success"), info.get("task_invalidated"),
                    info.get("task_timeout"), info.get("task_steps"), info.get("worker_failed"),
                ])
        return time.monotonic() < self.deadline

    def _on_rollout_start(self):
        # Called after the preceding PPO update; saved weights include that update.
        if self.model.num_timesteps >= self.next_save:
            save_policy(self.output / f"step_{self.model.num_timesteps}.zip", self.model, self.config)
            save_policy(self.output / "last.zip", self.model, self.config)
            self.stream.flush()
            self.next_save = self.model.num_timesteps + self.args.save_every
        if self.model.num_timesteps >= self.next_validation:
            self.validation()
            self.next_validation = self.model.num_timesteps + self.args.validate_every

    def validation(self):
        from .evaluate import validate

        if time.monotonic() >= self.deadline:
            return
        # Validation must not perturb subsequent training's sampling RNGs.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        was_training = self.model.policy.training
        try:
            with torch.random.fork_rng():
                self.model.policy.set_training_mode(False)
                result = validate(self.args, self.model, self.args.stage,
                                  min(self.deadline, time.monotonic() + self.args.validation_seconds))
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.model.policy.set_training_mode(was_training)
        result["timesteps"] = self.model.num_timesteps
        write_json(self.session / f"validation_{self.model.num_timesteps}.json", result)
        score = result["score"]
        if score is not None and (self.best_score is None or score > self.best_score):
            self.best_score = score
            selected_config = {**self.config, "validation_score": score,
                               "validation_partner": "bc", "validation": result}
            save_policy(self.output / "best.zip", self.model, selected_config)
        print(f"BC-only validation at {self.model.num_timesteps}: score={score}, status={result['status']}", flush=True)

    def _on_rollout_end(self):
        self.logger.record("time/effective_steps_per_second", (
            self.model.num_timesteps - self.start_steps
        ) / max(time.monotonic() - self.started, .001))
        self.logger.record("time/base_env_steps", sum(self.training_env.env_method("get_base_steps")))

    def close(self):
        if hasattr(self, "stream"):
            self.stream.close()


def settings(args, device):
    values = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    values.update(device_resolved=device, upstream_commit=UPSTREAM_COMMIT,
                  environment_commit=ENVIRONMENT_COMMIT, evaluation_unstuck=False,
                  implementation="Gymnasium/SB3 2.x port; trajectory-level BC data split", schema=2,
                  checkpoint_selection="BC-only held-out seeds; Worker core-skill macro success, Manager sparse team return")
    expected = {"torch": "2.3.1", "numpy": "1.26.4", "pandas": "2.2.2",
                "gym": "0.26.2", "gymnasium": "0.29.1",
                "stable-baselines3": "2.3.2", "sb3-contrib": "2.3.0"}
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use the isolated Python 3.10 environment, not AutoDL's Python 3.12 base")
    versions = {package: importlib.metadata.version(package) for package in expected}
    for package, expected_version in expected.items():
        if versions[package].split("+")[0] != expected_version:
            raise RuntimeError(f"{package}={versions[package]}, expected {expected_version}")
    values["runtime_versions"] = versions
    distribution = importlib.metadata.distribution("overcooked-ai")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    values["environment_install_source"] = direct_url
    archive_name = f"overcooked_ai-{ENVIRONMENT_COMMIT}.tar.gz"
    pinned_archive = ("archive_info" in direct_url and
                      direct_url.get("url", "").endswith("/" + archive_name))
    if (direct_url.get("vcs_info", {}).get("commit_id") != ENVIRONMENT_COMMIT
            and not pinned_archive):
        raise ValueError("Install the pinned author environment from requirements.txt; shared hucooked is not equivalent")
    return values


def check_config(saved, current, keys):
    for key in keys:
        if saved.get(key) != current.get(key):
            raise ValueError(f"Configuration changed: {key}; use a new --run-dir")


def preflight(args, device):
    config = settings(args, device)
    stage = args.evaluate_stage if args.stage == "evaluate" else args.stage
    if stage == "prepare":
        return config
    if stage == "bc":
        for layout in args.layouts:
            path = args.run_dir / "data" / f"{layout}.npz"
            with np.load(path, allow_pickle=False) as data:
                if data["observations"].ndim != 2 or data["observations"].shape[1] != 96:
                    raise ValueError(f"Wrong BC observation shape: {path}")
                if len(data["actions"]) != len(data["groups"]) or len(data["actions"]) != len(data["observations"]):
                    raise ValueError(f"Inconsistent prepared data lengths: {path}")
                source = json.loads(str(data["source"]))
                if source["upstream_commit"] != UPSTREAM_COMMIT or source["layout"] != layout:
                    raise ValueError(f"Wrong prepared data source: {path}")
        return config
    roles = args.partners if args.stage == "evaluate" else ["bc"]
    config["partner_artifacts"] = {}
    for layout in args.layouts:
        split = read_json(args.run_dir / "bc" / layout / "split.json") if roles != ["random"] else None
        if split is not None and not split.get("roles_ready"):
            raise ValueError(f"BC role assignment is incomplete for {layout}")
        if split is not None and max(split["candidate_selfplay_returns"]) <= 0:
            print(f"WARNING: both BC candidates on {layout} have zero selection return", flush=True)
        for role in roles:
            if role == "random":
                continue
            path = args.run_dir / "bc" / layout / f"{role}.pt"
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["layout"] != layout or saved["role"] != role:
                raise ValueError(f"Wrong partner role/layout in {path}")
            config["partner_artifacts"][f"{layout}/{role}"] = saved["artifact_id"]
    if stage == "manager":
        worker = checkpoint_info(args.worker)
        if worker["stage"] != "worker":
            raise ValueError("--worker is not a Worker checkpoint")
        if not set(args.layouts).issubset(worker["config"]["layouts"]):
            raise ValueError("Worker was not trained on all requested layouts")
        check_config(worker["config"], config, ("horizon", "task_timeout"))
        config["worker_checkpoint_id"] = worker["checkpoint_id"]
        if "validation_score" not in worker["config"]:
            print("WARNING: using a Worker without a valid core-skill validation score; inspect Worker evaluation before full Manager training", flush=True)
    if args.stage == "evaluate":
        path = args.worker if stage == "worker" else args.manager
        checkpoint = checkpoint_info(path)
        if checkpoint["stage"] != stage:
            raise ValueError("Evaluation checkpoint stage differs")
        check_config(checkpoint["config"], config, ("horizon", "task_timeout"))
        if not set(args.layouts).issubset(checkpoint["config"]["layouts"]):
            raise ValueError("Evaluation layouts are outside the checkpoint's training layouts")
        if stage == "manager":
            check_config(checkpoint["config"], config, ("worker_checkpoint_id",))
    return config


def train_rl(args, device):
    deadline = time.monotonic() + args.max_hours * 3600
    config = preflight(args, device)  # Validate artifacts before constructing any environments.
    output = args.run_dir / args.stage
    saved = None
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output exists: {output}; use --resume or a new --run-dir")
    if args.resume:
        saved = checkpoint_info(args.resume)
        check_config(saved["config"], config, (
            "schema", "stage", "layouts", "horizon", "task_timeout", "hidden", "seed",
            "n_envs", "n_steps", "batch_size", "ppo_epochs", "no_worker_shaping",
            "unknown_penalty", "failure_limit", "partner_artifacts", "worker_checkpoint_id",
            "validation_episodes", "validation_games", "validation_min_samples", "deterministic",
        ))
        config["artifact_id"] = saved["artifact_id"]
    else:
        config["artifact_id"] = uuid.uuid4().hex
    session = output / f"session_{time.time_ns()}"
    session.mkdir(parents=True, exist_ok=False)
    factories = [partial(make_env, args, args.stage, args.layouts[i % len(args.layouts)], i)
                 for i in range(args.n_envs)]
    factories = [partial(monitored_env, factory, str(session / f"env_{i}"))
                 for i, factory in enumerate(factories)]
    vec = (DummyVecEnv(factories) if args.vec_env == "dummy"
           else SubprocVecEnv(factories, start_method="spawn"))
    vec.seed(args.seed)
    algorithm = PPO if args.stage == "worker" else MaskablePPO
    progress = TrainingProgress(args, output, config, deadline, session)
    model = None
    try:
        if args.resume:
            model = algorithm.load(args.resume, env=vec, device=device)
            check_loaded_policy(model, saved)
            model.learning_rate = args.learning_rate
            model.lr_schedule = lambda _: args.learning_rate
            model.ent_coef = args.entropy_coef
        else:
            model = algorithm(
                "MultiInputPolicy", vec, seed=args.seed, device=device, verbose=1,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.ppo_epochs,
                learning_rate=args.learning_rate, ent_coef=args.entropy_coef,
                vf_coef=.3, gamma=.99, gae_lambda=.95, clip_range=.2,
                policy_kwargs={"net_arch": {"pi": [args.hidden] * 2, "vf": [args.hidden] * 2},
                               "activation_fn": torch.nn.Tanh},
            )
            model.haha_artifact_id = config["artifact_id"]
        model.set_logger(configure(str(session / "log"), ["stdout", "csv"]))
        config["policy_parameters"] = sum(p.numel() for p in model.policy.parameters())
        write_json(session / "config.json", config)
        if not args.resume:
            write_json(output / "config.json", config)
        steps = args.timesteps or (31_000_000 if args.stage == "worker" else 35_000_000)
        interrupted = False
        try:
            if time.monotonic() < deadline:
                model.learn(total_timesteps=steps, callback=progress,
                            reset_num_timesteps=not bool(args.resume))
                progress.validation()
        except KeyboardInterrupt:
            interrupted = True
        save_policy(output / "last.zip", model, config)
        write_json(session / "summary.json", {
            "interrupted": interrupted, "timesteps": model.num_timesteps,
            "time_limit_reached": time.monotonic() >= deadline,
            "base_env_steps_this_process": sum(vec.env_method("get_base_steps")),
            "checkpoint": str(output / "last.zip"), "best_validation_score": progress.best_score,
            "resume_note": "PPO optimizer/weights resume; live kitchens and partially collected rollouts restart.",
        })
    finally:
        progress.close()
        vec.close()


def monitored_env(factory, filename):
    return Monitor(factory(), filename=filename)


def main():
    args = arguments()
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if (device == "cuda" or args.inference_device == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.stage == "prepare":
        preflight(args, device)
        prepare(args.dataset, args.run_dir / "data", args.layouts, args.seed)
    elif args.stage == "bc":
        directory = args.run_dir / "bc"
        config = preflight(args, device)
        config_path = directory / "config.json"
        if directory.exists() and any(directory.iterdir()):
            if not args.resume_bc:
                raise FileExistsError(f"BC outputs exist: {directory}; use --resume-bc or a new --run-dir")
            check_config(read_json(config_path), config, (
                "schema", "layouts", "seed", "bc_hidden", "bc_lr", "bc_batch_size",
                "bc_eval_every", "selection_episodes", "horizon", "epochs",
            ))
        else:
            write_json(config_path, config)
        train_bc(args, device)
    elif args.stage in {"worker", "manager"}:
        train_rl(args, device)
    else:
        from .evaluate import evaluate
        evaluate(args, device)


if __name__ == "__main__":
    main()

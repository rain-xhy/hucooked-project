"""Train fresh NestRL policies using the pinned Overcooked PPO implementation."""

import argparse
from functools import partial, partialmethod
from math import ceil, isfinite
import json
from pathlib import Path
import random
import sys
from time import perf_counter

from overcooked_setup import CONFIG, RECIPES, ROOT, UPSTREAM, activate_upstream
from rollout_env import HISTORY_SIZE, kitchen_factory
from rollout_profile import RolloutProfile


def training_options(options, train_size, eval_size, processes):
    rollout_size = options.rollout_steps * processes
    steps = rollout_size * options.updates
    mini_batches = 18 if options.ppo_mini_batch_size is None else rollout_size // options.ppo_mini_batch_size
    eval_interval = steps if options.eval_every_updates is None else rollout_size * options.eval_every_updates
    save_interval = getattr(options, "checkpoint_every_steps", None) or 5_000_000
    return [
        "--env-name", "Overcooked", "--env_config", CONFIG,
        "--player-id", str(options.player), "--recipe-type", "cross",
        "--seed", str(options.seed), "--exp-name", options.run_name,
        "--algo", "ppo",
        *(["--use-dummy-vec-env"] if options.env_mode == "serial" else []),
        *(["--no-cuda"] if options.device == "cpu" else []),
        "--num-steps", str(options.rollout_steps), "--num-processes", str(processes),
        "--num-env-steps", str(steps), "--save-interval", str(save_interval),
        "--eval-interval", str(eval_interval), "--eval-episodes", str(HISTORY_SIZE),
        "--log-interval", str(options.rollout_steps * processes),
        "--train-pool-size", str(train_size), "--eval-pool-size", str(eval_size),
        "--rule-based-opponents", str(train_size if options.level == "l1" else 0),
        "--use-gae", "--use-proper-time-limit", "--eps", "1e-8",
        "--lr", str(options.learning_rate), "--clip-param", "0.2", "--entropy-coef", "0.03",
        "--value-loss-coef", "0.5", "--num-epochs", "4", "--num-mini-batch", str(mini_batches),
        "--gamma", "0.99", "--gae-lambda", "0.95", "--max-grad-norm", "15",
        "--latent-training", "--e2e-obj", "--deterministic-latent", "--use-latent-critic",
        "--pre-hidden-dims", "128", "128", "--encoder-base", "mlp", "--agg-func", "mean",
        "--post-hidden-dims", "128", "--latent-dim", "128", "--kl-coef", "0",
        "--hidden-dims", "128", "128", "--act-func", "relu",
        "--history-middle-sampling", "--merge-encoder-computation",
        "--history-size", str(HISTORY_SIZE), "--history-full-size", str(HISTORY_SIZE),
        "--history-use-episodes", "--include-current-episode", "--use-meta-episode",
        "--self-obs-mode", "--clear-history-on-full", "--separate-history",
    ]


def main(*, output_root=ROOT, recipes=RECIPES, gpu_vec_class=None, invocation_dir=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=("l1", "l2"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--recipes", nargs="+", choices=recipes, default=list(recipes))
    parser.add_argument("--partners", type=Path, nargs="+")
    parser.add_argument("--eval-partners", type=Path, nargs="+")
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--processes", type=int, help="number of training kitchens")
    parser.add_argument("--workers", type=int, help="subprocess count when kitchens share a worker")
    parser.add_argument("--ppo-mini-batch-size", type=int, help="samples per PPO optimizer step")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--lr-warmup-updates", type=int, default=0,
                        help="initial PPO rollout updates over which to warm up the learning rate")
    warmup.add_argument("--lr-warmup-fraction", type=float,
                        help="fraction of all planned PPO optimizer steps used for linear warmup")
    parser.add_argument("--eval-every-updates", type=int, help="PPO updates between held-out evaluations")
    parser.add_argument("--eval-rounds", type=int, default=1, help="five-episode rounds per Trackio evaluation")
    parser.add_argument("--trackio-project", help="record metrics to a local Trackio project")
    if gpu_vec_class is not None:
        parser.add_argument("--checkpoint-every-steps", type=int)
        parser.add_argument("--resume", type=Path, help="resume from a trusted training state into a new run")
        parser.add_argument("--defer-rollout-stats", action="store_true")
        parser.add_argument("--profile-cuda-stages", action="store_true")
    parser.add_argument("--trackio-sample-every", type=int, default=0,
                        help="vector steps between intermediate training samples logged to Trackio")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--env-mode", choices=("serial", "subprocess", "gpu"), default="serial",
                        help="run training kitchens serially, across subprocesses, or as a CUDA batch")
    parser.add_argument("--profile", action="store_true", help="write rollout timings to profile.json")
    parser.add_argument("--history-encoder", choices=("original", "batched"), default="batched",
                        help="choose the learner's history encoder for CUDA rollout actions")
    parser.add_argument("--history-storage", choices=("cpu", "gpu"),
                        help="learner history device (defaults to GPU for GPU kitchens)")
    parser.add_argument("--observation", choices=("dense", "spatial"), default="dense",
                        help="73-feature observations or 7x8 maps with chef positions and previous actions")
    options = parser.parse_args()
    if getattr(options, "resume", None) is not None:
        options.resume = ((invocation_dir or Path.cwd()) / options.resume).resolve()
    needs_training_patch = (
        getattr(options, "checkpoint_every_steps", None) is not None
        or getattr(options, "defer_rollout_stats", False)
    )

    if gpu_vec_class is not None and (options.env_mode != "gpu" or options.level != "l1"
                                      or options.observation != "dense" or options.history_storage == "cpu"):
        parser.error("The standard kitchen requires GPU L1 training, dense observations and GPU history")
    if options.player != getattr(gpu_vec_class, "learner_player", options.player):
        parser.error(f"{gpu_vec_class.layout_name} trains the middle cook (--player {gpu_vec_class.learner_player})")
    if not options.run_name.isascii() or not options.run_name.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-name must contain only ASCII letters, digits, hyphens and underscores")
    if options.rollout_steps < 40 or options.updates < 1:
        parser.error("use at least 40 rollout steps and one PPO update")
    if not isfinite(options.learning_rate) or options.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= options.lr_warmup_updates <= options.updates:
        parser.error("--lr-warmup-updates must be between zero and --updates")
    if options.lr_warmup_fraction is not None and not 0 < options.lr_warmup_fraction <= 1:
        parser.error("--lr-warmup-fraction must be greater than zero and at most one")
    if options.eval_every_updates is not None and (
        options.eval_every_updates < 1 or options.updates % options.eval_every_updates
    ):
        parser.error("--eval-every-updates must divide --updates so the final policy is evaluated")
    if options.eval_rounds < 1 or (options.eval_rounds != 1 and not options.trackio_project):
        parser.error("--eval-rounds must be positive and requires --trackio-project when greater than one")
    if options.trackio_sample_every < 0 or (options.trackio_sample_every and not options.trackio_project):
        parser.error("--trackio-sample-every must be nonnegative and requires --trackio-project when enabled")
    if options.env_mode == "gpu" and options.device != "cuda":
        parser.error("--env-mode gpu requires --device cuda")
    if options.history_storage is None:
        batched_cuda = options.env_mode == "gpu" and options.history_encoder == "batched"
        options.history_storage = "gpu" if batched_cuda else "cpu"
    if options.history_storage == "gpu" and (options.env_mode != "gpu" or options.history_encoder != "batched"):
        parser.error("GPU history storage requires --env-mode gpu and --history-encoder batched")
    if options.level == "l2":
        if not options.partners or not options.eval_partners:
            parser.error("L2 requires both training and held-out evaluation L1 checkpoints")
        options.partners = [path.resolve() for path in options.partners]
        options.eval_partners = [path.resolve() for path in options.eval_partners]
        if set(options.partners) & set(options.eval_partners):
            parser.error("training and evaluation checkpoint paths must be disjoint")
        for path in options.partners + options.eval_partners:
            if not path.is_file():
                parser.error(f"checkpoint does not exist: {path}")
    elif options.partners or options.eval_partners:
        parser.error("checkpoint pools are used only by L2")

    output = output_root / "runs" / options.run_name
    if output.exists():
        parser.error(f"run directory already exists: {output}")
    if not (UPSTREAM / "train_.py").is_file():
        parser.error("initialize the adaptive-RL submodule first")
    activate_upstream()

    import numpy as np
    import torch

    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch installation and an available GPU")

    from learning.arguments import get_args
    from learning import envs
    import train_

    if needs_training_patch:
        from inspect import signature

        if "training_state" not in signature(train_.train_embedding).parameters:
            parser.error("Apply adaptive-rl-training.patch as documented in mlp_standard/README.md.")

    VecPyTorch = envs.VecPyTorch

    original_step_async = VecPyTorch.step_async

    def step_async(env, actions):
        # Upstream squeezes CPU LongTensors before passing NumPy actions to Gym.
        actions = actions.cpu()
        if actions.dtype == torch.long and actions.ndim == 1:
            actions = actions.unsqueeze(1)
        return original_step_async(env, actions)

    VecPyTorch.step_async = step_async
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    torch.set_num_threads(1)
    train_pool = options.recipes if options.level == "l1" else options.partners
    eval_pool = options.recipes if options.level == "l1" else options.eval_partners
    processes = len(train_pool) if options.processes is None else options.processes
    if processes < len(train_pool):
        parser.error("--processes must be at least the training pool size")
    if options.env_mode == "subprocess" and processes < 2:
        parser.error("--env-mode subprocess requires at least two training environments")
    if options.workers is not None:
        if options.env_mode != "subprocess" or options.workers < 2 or processes % options.workers:
            parser.error("--workers requires subprocess mode and must be a divisor of --processes of at least two")
        if options.workers < processes:
            from grouped_vec_env import GroupedSubprocVecEnv

            envs.SubprocVecEnv = partial(GroupedSubprocVecEnv, workers=options.workers)
    if options.ppo_mini_batch_size is not None:
        if not 2 <= options.ppo_mini_batch_size <= processes * options.rollout_steps:
            parser.error("--ppo-mini-batch-size must be between two and the rollout sample count")
        storage = train_.RolloutStorage
        storage.feed_forward_generator = partialmethod(
            storage.feed_forward_generator, mini_batch_size=options.ppo_mini_batch_size,
        )
    if processes % len(train_pool):
        # The pinned trainer expects equal slots per pool entry. Repeat the recipe
        # cycle to provide one slot per kitchen when its count is not divisible.
        train_pool = [train_pool[rank % len(train_pool)] for rank in range(processes)]
    launch_args = training_options(options, len(train_pool), len(eval_pool), processes)
    sys.argv = ["train_.py", *launch_args]
    args = get_args()
    rollout_size = processes * options.rollout_steps
    checkpoint_interval = getattr(options, "checkpoint_every_steps", None)
    resume = getattr(options, "resume", None)
    if checkpoint_interval is not None and (checkpoint_interval < 1 or checkpoint_interval % rollout_size):
        parser.error("--checkpoint-every-steps must be a positive multiple of one rollout")
    if resume is not None and checkpoint_interval is None:
        parser.error("--resume requires --checkpoint-every-steps")
    batch_size = options.ppo_mini_batch_size or rollout_size // args.num_mini_batch
    optimizer_steps_per_update = args.num_epochs * (rollout_size // batch_size)
    warmup_steps = options.lr_warmup_updates * optimizer_steps_per_update
    if options.lr_warmup_fraction is not None:
        warmup_steps = ceil(options.lr_warmup_fraction * options.updates * optimizer_steps_per_update)
    options.lr_warmup_optimizer_steps = warmup_steps
    training_state = None
    completed_updates = 0
    if checkpoint_interval is not None:
        from mlp_training_state import TrainingState

        training_state = TrainingState(options, output, rollout_size)
        completed_updates = training_state.completed_updates
    output.mkdir(parents=True)
    launch = {"options": vars(options), "upstream_args": launch_args}
    if gpu_vec_class is not None:
        launch["environment"] = {"layout": gpu_vec_class.layout_name, "observation_size": gpu_vec_class.observation_size,
                                 "horizon": gpu_vec_class.horizon, "device": "cuda"}
    (output / "launch.json").write_text(
        json.dumps(launch, default=str, indent=2),
        encoding="utf-8",
    )
    args.log_dir = args.save_dir = str(output)
    args.num_agents = 2
    args.policy_id_max = torch.tensor([len(train_pool)])
    args.policy_id_all = torch.arange(len(train_pool)).unsqueeze(0)

    device_name = torch.cuda.get_device_name() if args.cuda else "CPU"
    print(f"Training device: {device_name}", flush=True)
    training_factory = None
    evaluation_factory = None
    spatial_obs = options.observation == "spatial"
    if options.env_mode == "gpu":
        from torch_kitchen_vec import TorchKitchenVecEnv, evaluate_cuda_rules, install_cuda_kitchen_history

        training_factory = partial(gpu_vec_class or TorchKitchenVecEnv, player=options.player, profile=options.profile,
                                   spatial_obs=spatial_obs)
        install_cuda_kitchen_history(train_.RolloutStorage)
        if options.history_storage == "gpu":
            from cuda_history import install_cuda_history
            from learning.model import Encoder

            install_cuda_history(train_.RolloutStorage, Encoder)
            if options.level == "l1":
                evaluation_factory = training_factory
                train_.evaluate = evaluate_cuda_rules
        if getattr(options, "defer_rollout_stats", False):
            training_factory = partial(training_factory, defer_rollout_stats=True)
    profile = RolloutProfile(options.profile, options.level == "l2", training_factory, evaluation_factory,
                             cuda_stages=getattr(options, "profile_cuda_stages", False))
    if warmup_steps:
        from ppo_warmup import install_lr_warmup

        install_lr_warmup(train_.algo.PPO_, warmup_steps,
                          completed_updates * optimizer_steps_per_update, options.learning_rate)
    tracker = None
    if options.trackio_project:
        from trackio_training import TrackioTraining

        tracker = TrackioTraining(options, output, train_pool, eval_pool, completed_updates * rollout_size)
        tracker.install(train_, args, defer_rollout_stats=getattr(options, "defer_rollout_stats", False))
        if options.profile:
            tracker.log_rollout_samples = profile.timed(tracker.log_rollout_samples, "rollout_logging")
    envs.make_env = kitchen_factory(envs.make_env, options.profile, spatial_obs=spatial_obs)
    if options.history_encoder == "batched" and args.cuda:
        from rollout_history import install_batched_history

        install_batched_history(train_.LatentPolicy)
    profile.install(train_, VecPyTorch)
    if spatial_obs:
        from spatial_observation import make_spatial_policy

        train_.LatentPolicy = make_spatial_policy
    start = perf_counter()
    try:
        if training_state is None:
            train_.train_embedding(args, train_pool, eval_pool)
        else:
            train_.train_embedding(args, train_pool, eval_pool, training_state=training_state)
        if args.cuda:
            torch.cuda.synchronize()
    finally:
        try:
            profile.close_remaining()
        finally:
            if tracker is not None:
                tracker.finish()
    elapsed = perf_counter() - start
    steps_run = args.num_env_steps - completed_updates * rollout_size
    profile.report(output, elapsed, steps_run)
    print(f"Checkpoint: {output / 'ppo' / 'latest.pt'} ({elapsed:.1f}s)")
    print(f"Training throughput: {steps_run / elapsed:.1f} env steps/s")


if __name__ == "__main__":
    main()

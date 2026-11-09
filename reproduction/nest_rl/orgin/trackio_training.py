"""Record local NestRL PPO training and repeated held-out evaluation."""

import json
from math import isfinite
from numbers import Real
import os
import random
from time import perf_counter


class TrackioTraining:
    def __init__(self, options, output, train_pool, eval_pool):
        self.project = options.trackio_project
        self.rounds = options.eval_rounds
        self.sample_every = options.trackio_sample_every
        self.output = output
        self.config = json.loads(json.dumps({
            **vars(options), "train_pool": train_pool, "eval_pool": eval_pool,
        }, default=str))
        self.run = None
        self.step = None
        self.metrics = {}
        self.pending_sample = {}

    def init(self, *, config, name, **kwargs):
        os.environ["TRACKIO_DIR"] = str(self.output.parent / "trackio")
        os.environ["TRACKIO_STORAGE_MODE"] = "sqlite"
        for key in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID", "TRACKIO_BUCKET_ID"):
            os.environ.pop(key, None)

        import trackio

        self.started = perf_counter()
        self.samples_per_update = config.num_steps * config.num_processes
        self.run = trackio.init(
            project=self.project, name=name, config={
                **self.config, "samples_per_update": self.samples_per_update,
                "env_step_budget": config.num_env_steps,
            },
            embed=False, auto_log_gpu=False, auto_log_cpu=False,
        )
        return self

    def log(self, metrics, *, step):
        if self.step is not None and step != self.step:
            self.flush()
        self.step = step
        self.metrics.update({
            key: float(value) for key, value in metrics.items()
            if isinstance(value, Real) and isfinite(value)
        })
        raw_grad_norm = metrics.get("train/raw_grad_norm")
        if raw_grad_norm is not None:
            grad_norm = float(raw_grad_norm)
            self.metrics["train/raw_grad_norm_nonfinite"] = int(not isfinite(grad_norm))
            if isfinite(grad_norm):
                self.metrics["train/raw_grad_norm"] = grad_norm
        self.logged_at = perf_counter()
        if "env_steps" in metrics:
            self.flush()

    def flush(self):
        if "env_steps" in self.metrics:
            env_steps = int(self.metrics["env_steps"])
            self.run.log({
                **self.pending_sample, **self.metrics, "env_steps": env_steps,
                "wall_seconds": self.logged_at - self.started,
            }, step=env_steps)
            self.pending_sample.clear()
        self.metrics.clear()

    def finish(self):
        if self.run is not None:
            self.flush()
            self.run.finish()
            self.run = None

    def install(self, trainer, args):
        import numpy as np
        import torch
        import evaluation_

        args.wandb_user_name = "local-trackio"
        trainer.wandb = evaluation_.wandb = self
        evaluate = trainer.evaluate

        def repeat(args, pool, episodes, envs, history, policy, log_steps, **kwargs):
            np_state, random_state = np.random.get_state(), random.getstate()
            results = []
            devices = [torch.cuda.current_device()] if args.cuda else []
            try:
                with torch.random.fork_rng(devices=devices):
                    for round_index in range(self.rounds):
                        seed = 100_000 + round_index
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.random.default_generator.manual_seed(seed)
                        if args.cuda:
                            torch.cuda.manual_seed(seed)
                        result, _ = evaluate(args, pool, episodes, envs, history, policy, None, **kwargs)
                        results.append(result)
            finally:
                np.random.set_state(np_state)
                random.setstate(random_state)
            count = len(pool) * episodes * self.rounds
            success_rate = float(np.mean([r["success_rate"] for r in results]))
            self.log({
                "eval/reward": np.mean([r["reward"] for r in results]),
                "eval/success_rate": success_rate,
                "eval/success_count": round(count * success_rate),
                "eval/episode_count": count,
            }, step=log_steps)

        trainer.evaluate = repeat
        if self.sample_every:
            self.install_sampling_log(trainer, args.num_steps)

    def install_sampling_log(self, trainer, rollout_steps):
        insert = trainer.RolloutStorage.insert
        vector_steps = env_steps = episodes = successes = 0
        total_reward = 0.0

        def record(storage, obs, next_obs, recurrent_hidden_states, actions, action_log_probs,
                   value_preds, rewards, masks, bad_masks, infos, *args, **kwargs):
            nonlocal vector_steps, env_steps, episodes, successes, total_reward
            result = insert(storage, obs, next_obs, recurrent_hidden_states, actions, action_log_probs,
                            value_preds, rewards, masks, bad_masks, infos, *args, **kwargs)
            vector_steps += 1
            env_steps += len(infos)
            for info in infos:
                if "episode" in info:
                    episodes += 1
                    total_reward += float(info["episode"]["r"])
                    successes += info["termination_info"].endswith("completed")
            end_of_rollout = vector_steps % rollout_steps == 0
            if not end_of_rollout and vector_steps % self.sample_every != 0:
                return result
            metrics = {"sample/episodes": episodes}
            if episodes:
                metrics.update({
                    "sample/reward": total_reward / episodes,
                    "sample/success_rate": successes / episodes,
                })
            if end_of_rollout:
                self.pending_sample = metrics
            else:
                self.run.log({
                    **metrics, "env_steps": env_steps,
                    "wall_seconds": perf_counter() - self.started,
                }, step=env_steps)
            episodes = successes = 0
            total_reward = 0.0
            return result

        trainer.RolloutStorage.insert = record

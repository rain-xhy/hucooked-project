"""Measure training wall time separately from per-kitchen worker time."""

from collections import defaultdict
from contextlib import suppress
from functools import wraps
import json
from pathlib import Path
from time import perf_counter


class RolloutProfile:
    def __init__(self, enabled, adaptive, training_factory=None, evaluation_factory=None, cuda_stages=False):
        self.enabled = enabled
        self.adaptive = adaptive
        self.wall = defaultdict(float)
        self.kitchens = {}
        self.backends = {}
        self.active = {}
        self.evaluating = False
        self.measuring = False
        self.training_factory = training_factory
        self.evaluation_factory = evaluation_factory
        self.cuda_stages = cuda_stages
        self.cuda_spans = defaultdict(list)

    def timed(self, method, bucket, *, cuda=False):
        if cuda and self.cuda_stages:
            from torch.cuda import Event
        else:
            Event = None

        @wraps(method)
        def measure(*args, **kwargs):
            if self.evaluating or self.measuring:
                return method(*args, **kwargs)
            self.measuring = True
            start = perf_counter()
            if Event is not None:
                gpu_start, gpu_end = Event(enable_timing=True), Event(enable_timing=True)
                gpu_start.record()
            try:
                return method(*args, **kwargs)
            finally:
                if Event is not None:
                    gpu_end.record()
                    self.cuda_spans[bucket].append((gpu_start, gpu_end))
                self.wall[bucket] += perf_counter() - start
                self.measuring = False

        return measure

    def install(self, trainer, vec_class):
        make_vec = trainer.make_vec_envs
        close = vec_class.close

        def collect_vector_stats(vec):
            role = self.active.pop(vec, None)
            if role is not None and (self.enabled or self.adaptive):
                self.kitchens[role] = vec.env_method("rollout_stats")

        def make_vector(args, env_name, seed, num_processes, log_dir, device, *extra, **kwargs):
            role = "eval" if self.active else "train"
            factory = self.evaluation_factory if role == "eval" else self.training_factory
            if role == "eval" and factory is None:
                # Keep evaluation identical across benchmarks and separate its monitor files.
                kwargs["always_use_dummy"] = True
                log_dir = Path(log_dir) / "eval"
                log_dir.mkdir(exist_ok=True)
                log_dir = str(log_dir)
            start = perf_counter()
            if factory is not None:
                vec = factory(num_processes, device, seed)
                backend = type(vec).__name__
                finish = vec.close

                def close_training_vector():
                    try:
                        collect_vector_stats(vec)
                    finally:
                        finish()

                vec.close = close_training_vector
                if self.enabled:
                    vec.step = self.timed(vec.step, "vector_wait", cuda=True)
                    if getattr(vec, "defer_rollout_stats", False):
                        vec.finish_rollout = self.timed(vec.finish_rollout, "rollout_statistics")
            else:
                vec = make_vec(args, env_name, seed, num_processes, log_dir, device, *extra, **kwargs)
                backend = type(vec.venv).__name__
            self.wall["vector_startup"] += perf_counter() - start
            self.active[vec] = role
            self.backends[role] = backend
            return vec

        def close_vector(vec):
            try:
                collect_vector_stats(vec)
            finally:
                close(vec)

        trainer.make_vec_envs = make_vector
        vec_class.close = close_vector
        if self.enabled:
            evaluate = trainer.evaluate

            def measure_evaluation(*args, **kwargs):
                self.evaluating = True
                start = perf_counter()
                try:
                    return evaluate(*args, **kwargs)
                finally:
                    self.evaluating = False
                    self.wall["evaluation"] += perf_counter() - start

            trainer.evaluate = measure_evaluation
            trainer.LatentPolicy.act = self.timed(trainer.LatentPolicy.act, "policy", cuda=True)
            trainer.algo.PPO_.update = self.timed(trainer.algo.PPO_.update, "ppo", cuda=True)
            trainer.RolloutStorage.insert = self.timed(trainer.RolloutStorage.insert, "rollout_insert", cuda=True)
            trainer.RolloutStorage.compute_returns = self.timed(trainer.RolloutStorage.compute_returns, "returns", cuda=True)
            vec_class.step_async = self.timed(vec_class.step_async, "vector_send")
            vec_class.step_wait = self.timed(vec_class.step_wait, "vector_wait")

    def close_remaining(self):
        for vec in list(self.active):
            # On a failed rollout, close pipes without making another worker RPC.
            self.active.pop(vec)
            with suppress(BrokenPipeError, ConnectionResetError, EOFError):
                vec.close()

    def report(self, output, elapsed, steps):
        print("Vector backends:", self.backends)
        if self.adaptive:
            for role, kitchens in self.kitchens.items():
                for worker, stats in enumerate(kitchens):
                    print(f"{role}[{worker}] {stats['path']}: {stats['episodes']} finished episodes, "
                          f"{stats['episodes_with_context']} started with prior context, "
                          f"{stats['rounds']} round resets")
        if self.enabled:
            data = {
                "env_steps": steps,
                "elapsed_seconds": elapsed,
                "backends": self.backends,
                "wall_seconds": dict(self.wall),
                "kitchens": self.kitchens,
            }
            if self.cuda_stages:
                data["cuda_stream_seconds"] = {
                    name: sum(start.elapsed_time(end) for start, end in spans) / 1000
                    for name, spans in self.cuda_spans.items()
                }
            (output / "profile.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

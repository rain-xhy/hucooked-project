"""Reproducible V3 wall-time benchmark and synchronized diagnostic breakdown.

Normal runs synchronize only around collect/optimize; diagnostic wrappers insert
extra synchronization and are not throughput measurements. No training artifacts
are modified. BC demonstrations are read from the existing curriculum cache.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from .curriculum_experiment import demonstrations
from .history import RolloutHistory
from .train import Settings, Trainer
from . import shaping

RUNS = Path(__file__).resolve().parent / "runs"


class GPUSampler:
    """Whole-device nvidia-smi samples, not process-exclusive utilization."""
    def __init__(self, device, enabled):
        self.rows = []
        self.process = self.thread = None
        if enabled and str(device).startswith("cuda"):
            index = torch.device(device).index or 0
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen([
                "nvidia-smi", "-i", str(index),
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=flags)
            def read():
                for line in self.process.stdout:
                    try:
                        gpu, memory, power = map(float, line.strip().split(","))
                        self.rows.append(dict(time=time.perf_counter(), gpu_percent=gpu,
                                              memory_used_mib=memory, power_watts=power))
                    except ValueError:
                        pass
            self.thread = threading.Thread(target=read, daemon=True)
            self.thread.start()

    def finish(self, start, end):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.thread.join(timeout=5)
            self.process.stdout.close()
        samples = [row for row in self.rows if start <= row["time"] <= end]
        if not samples:
            return None
        values = [row["gpu_percent"] for row in samples]
        return dict(samples=len(samples), mean_gpu_percent=sum(values) / len(values),
                    min_gpu_percent=min(values), max_gpu_percent=max(values),
                    max_device_memory_used_mib=max(row["memory_used_mib"] for row in samples),
                    mean_power_watts=sum(row["power_watts"] for row in samples) / len(samples),
                    note="Whole GPU includes desktop/other processes; 200 ms sampling, not a process-exclusive counter.")


class Timings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.data = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
        self.stack = []
        self.originals = []

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)
        self.originals.append((owner, attribute, original))
        def measured(*args, **kwargs):
            self.sync()
            start = time.perf_counter()
            self.stack.append(0.)
            try:
                return original(*args, **kwargs)
            finally:
                self.sync()
                elapsed = time.perf_counter() - start
                children = self.stack.pop()
                row = self.data[label]
                row["calls"] += 1
                row["inclusive_seconds"] += elapsed
                row["exclusive_seconds"] += elapsed - children
                if self.stack:
                    self.stack[-1] += elapsed
        setattr(owner, attribute, measured)

    def restore(self):
        for owner, attribute, original in reversed(self.originals):
            setattr(owner, attribute, original)


def timed(device, fn):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def make_trainer(lanes, steps, device, policy, demo):
    if lanes == 128:
        counts = (92, 18, 18)
    elif lanes > 0 and lanes % 63 == 0:
        counts = tuple(lanes // 63 * n for n in (45, 9, 9))
    else:
        raise ValueError("Benchmark lanes must be 128 or a positive multiple of 63")
    stage_updates = {128: 10, 252: {400: 10, 800: 5, 1000: 4}.get(steps, 5),
                     504: 10}.get(lanes, 20)
    cfg = Settings(.0003, 1, stage_updates, num_envs=lanes, horizon=400,
                   rollout_steps=steps, role_counts=counts,
                   shaping=True, bc_coefficient=.1, target_kl=.02)
    trainer = Trainer(cfg, device)
    trainer.model.load_state_dict(policy)
    trainer.attach_demonstrations(*demo)
    return trainer


def normal(lanes, steps, device, policy, demo, sample_gpu=False):
    trainer = make_trainer(lanes, steps, device, policy, demo)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    sampler = GPUSampler(device, sample_gpu)
    start = time.perf_counter()
    try:
        (bank, data, rollout_metrics), collect = timed(device, trainer.collect)
        optimize_metrics, optimize = timed(device, lambda: trainer.optimize(bank, data))
    finally:
        sampled = sampler.finish(start, time.perf_counter())
    memory = (dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                   device_total_bytes=torch.cuda.get_device_properties(device).total_memory)
              if cuda else None)
    return dict(config=asdict(trainer.cfg), collect_seconds=collect, optimize_seconds=optimize,
                total_seconds=collect + optimize,
                environment_steps_per_second=lanes * steps / (collect + optimize),
                optimizer_steps=optimize_metrics["optimizer_steps"],
                mean_score=rollout_metrics.get("mean_score"),
                completed_episodes=rollout_metrics.get("episodes"),
                score_episodes=rollout_metrics.get("score_episodes"),
                ending_episode_tick=trainer.tick,
                kl_early_stop=optimize_metrics["kl_early_stop"], memory=memory,
                gpu_sampling=sampled)


def diagnostic(steps, device, policy, demo):
    trainer = make_trainer(63, steps, device, policy, demo)
    meter = Timings(device)
    for owner, name, label in (
        (trainer.history, "forward", "policy_forward_with_history"),
        (trainer.history, "append", "history_append"),
        (trainer.history, "refresh", "history_refresh"),
        (trainer.partners, "actions", "scripted_partners"),
        (trainer.env, "step", "environment_step"),
        (trainer.env, "observations", "observation_encoding"),
        (trainer.env, "reset", "environment_reset"),
        (shaping, "potential", "progress_potential"),
        (shaping, "transition_reward", "shaping_transition"),
        (RolloutHistory, "record", "history_index_record"),
    ):
        meter.wrap(owner, name, label)
    try:
        (_, _, _), elapsed = timed(device, trainer.collect)
    finally:
        meter.restore()
    exclusive = sum(row["exclusive_seconds"] for row in meter.data.values())
    return dict(collect_seconds=elapsed, sections=dict(meter.data),
                uninstrumented_seconds=elapsed - exclusive,
                warning="Extra synchronization distorts timings; compare throughput using normal runs only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--lanes", type=int, nargs="+", default=[63])
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=200)
    parser.add_argument("--scale-steps", type=int, default=0)
    parser.add_argument("--demo-root", type=Path, default=RUNS / "counter-curriculum-seed1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    demo = demonstrations(args.demo_root, 400, args.device)
    policy = torch.load(RUNS / "counter-bc-shaped-seed1/bc/bc_best.pt",
                        map_location=args.device, weights_only=False)["model"]
    # One short untimed pass warms allocations and CUDA libraries.
    normal(63, 20, args.device, policy, demo)
    source_dir = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("train.py", "features.py", "shaping.py", "partners.py", "history.py")}
    report = dict(device=str(args.device), gpu=torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
                  cpu_threads=torch.get_num_threads(), source_sha256=source_hashes, normal=[], scaling=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for lanes in args.lanes:
        for repeat in range(args.repeats):
            row = normal(lanes, args.steps, args.device, policy, demo, args.sample_gpu)
            report["normal"].append(row)
            print(json.dumps(dict(repeat=repeat, **row)), flush=True)
            save()
    if args.diagnostic_steps:
        report["diagnostic"] = diagnostic(args.diagnostic_steps, args.device, policy, demo)
        save()
        print(json.dumps(report["diagnostic"]), flush=True)
    if args.scale_steps:
        for lanes in (63, 126, 252):
            row = normal(lanes, args.scale_steps, args.device, policy, demo)
            report["scaling"].append(row)
            save()
            print(json.dumps(dict(scaling_lanes=lanes, **row)), flush=True)


if __name__ == "__main__":
    main()

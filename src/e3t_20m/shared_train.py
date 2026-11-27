"""Shared rollouts, checkpointing and normal-start evaluation for the three-chef E3T run."""

import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict

import numpy as np
import torch

from .shared_adapter import ARCHIVED_ENGINE_COMMIT, ENGINE_COMMIT
from .shared_work import SharedWorkEnv

ENGINE = ENGINE_COMMIT
PARTNERS = np.array([[1, 2], [0, 2], [0, 1]])
METRICS = (
    "team_score",
    "meals_completed",
    "joint_attempt_success_rate",
    "joint_wait_steps",
    "joint_attempt_steps",
    "joint_successes",
    "joint_ready_steps",
    "normal_completed",
    "joint_completed",
    "three_contributor_meals",
    "ticks",
)
INPUT_SIZE = 446


def gae(rewards, values, dones, bootstrap, gamma, gae_lambda, cutoff_values=None):
    if cutoff_values is not None:
        # Time-limit bootstraps enter the return, while dones still cut the trace.
        rewards = rewards + gamma * cutoff_values
    advantages = torch.empty_like(rewards)
    next_value = bootstrap
    running = torch.zeros_like(bootstrap)
    for index in reversed(range(len(rewards))):
        carry = (~dones[index]).to(rewards.dtype)
        delta = rewards[index] + gamma * carry * next_value - values[index]
        running = delta + gamma * gae_lambda * carry * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def features(obs):
    """Drop only the constant 7x7x10 layout; append the identified cook's own 70 features."""
    if obs.shape[-2:] != (3, 866):
        raise ValueError(f"Expected the native three-chef observation; got {obs.shape}")
    chefs = obs[..., 0, 491:701].reshape(*obs.shape[:-2], 3, 70)
    return np.concatenate((obs[..., :1], obs[..., 491:], chefs), axis=-1)


def next_history(old, obs, others, actions):
    h = np.empty_like(old)
    h[..., :-1, :] = old[..., 1:, :]
    h[..., -1, :INPUT_SIZE] = obs
    h[..., -1, INPUT_SIZE:-1] = np.eye(actions, dtype=np.float32)[others].reshape(
        *obs.shape[:-1], -1
    )
    h[..., -1, -1] = 1.0
    return h


def _operate(envs, cmd, arg):
    if cmd == "reset":
        return [env.reset(int(seed)) for env, seed in zip(envs, arg)]
    if cmd == "restore":
        results = []
        for env, (seed, trace) in zip(envs, arg):
            o, m = env.reset(int(seed))
            for actions in trace:
                o, m, *_ = env.step(actions)
            results.append((o, m))
        return results
    if cmd != "step":
        raise ValueError(f"Unknown environment operation: {cmd}")
    results = []
    for env, actions in zip(envs, arg):
        o, m, raw, shaped, term, trunc, info = env.step(actions)
        metrics = {k: info["metrics"][k] for k in METRICS} if term or trunc else None
        results.append((o, m, raw, shaped, term, trunc, metrics))
    return results


def _worker(pipe, count, env_class):
    try:
        envs = [env_class() for _ in range(count)]
        while True:
            cmd, arg = pipe.recv()
            if cmd == "close":
                break
            pipe.send((True, _operate(envs, cmd, arg)))
    except EOFError:
        pass
    except Exception:  # noqa: BLE001 - propagate any child error and traceback to the parent
        pipe.send((False, traceback.format_exc()))
    finally:
        pipe.close()


class Pool:
    def __init__(self, count, workers, env_class=SharedWorkEnv):
        self.count = count
        self.pipes, self.procs = [], []
        self.groups = np.array_split(np.arange(count), min(max(1, workers), count))
        self.local = []
        if len(self.groups) == 1:
            self.local = [env_class() for _ in range(count)]
        else:
            # Forkserver avoids inheriting an initialized PyTorch GPU context.
            ctx = mp.get_context(
                "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            )
            for group in self.groups:
                pipe, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(child, len(group), env_class), daemon=True)
                proc.start()
                child.close()
                self.pipes.append(pipe)
                self.procs.append(proc)

    def call(self, cmd, arg):
        if self.local:
            return _operate(self.local, cmd, arg)
        for pipe, group in zip(self.pipes, self.groups):
            pipe.send((cmd, arg[group[0] : group[-1] + 1]))
        rows = []
        for pipe in self.pipes:
            ok, result = pipe.recv()
            if not ok:
                raise RuntimeError(result)
            rows.extend(result)
        return rows

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            pipe.close()
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()


def unpack(pairs):
    obs, masks = zip(*pairs)
    stacked = np.stack(obs)
    if stacked.shape[-1] == 869:
        # Work adapter appends each chef's observed initial position category.
        # All three active-player indicators are constant ones in this task.
        encoded = features(stacked[..., :866])
        encoded[..., [1, 71, 141]] = stacked[..., 866:]
    else:
        encoded = features(stacked)
    return encoded, np.stack(masks)


def summarize(rows):
    pooled_attempts = sum(r["joint_attempt_steps"] for r in rows)
    successes = sum(r["joint_successes"] for r in rows)
    stats = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    stats.update(
        episodes=len(rows),
        team_score_total=sum(r["team_score"] for r in rows),
        meals_completed_total=sum(r["meals_completed"] for r in rows),
        joint_attempt_success_rate_pooled=successes / max(1, pooled_attempts),
        joint_attempt_steps_total=pooled_attempts,
        joint_successes_total=successes,
        episodes_with_joint_meal=sum(r["joint_completed"] > 0 for r in rows),
        episodes_without_attempts=sum(r["joint_attempt_steps"] == 0 for r in rows),
    )
    return stats


class Model:
    env_class = SharedWorkEnv

    def tensor(self, data):
        return torch.as_tensor(data, device=self.device)

    def history(self, *batch):
        return np.zeros(
            (*batch, self.config.history, INPUT_SIZE + 2 * self.actions + 1), dtype=np.float32
        )

    @torch.no_grad()
    def evaluate(self, episodes, seed, mode, workers=0, parallel=16):
        if episodes < 1 or parallel < 1 or workers < 0:
            raise ValueError(
                "Use positive episode and parallel counts, and a nonnegative worker count"
            )
        results = []
        for start in range(0, episodes, parallel):
            size = min(episodes - start, parallel)
            seeds = list(range(seed + start, seed + start + size))
            rngs = [np.random.default_rng(s) for s in seeds]
            pool = Pool(size, workers, self.env_class)
            try:
                obs, masks = unpack(pool.call("reset", seeds))
                hist = self.history(size, 3)
                total = np.zeros(size)
                while True:
                    ob = self.tensor(obs.reshape(-1, INPUT_SIZE))
                    h = self.tensor(hist.reshape(-1, *hist.shape[-2:]))
                    predictions = self.predictor(ob, h).softmax(-1)
                    logits, _ = self.policy(
                        ob, predictions, self.tensor(masks.reshape(-1, self.actions))
                    )
                    if mode == "greedy":
                        actions = logits.argmax(-1).cpu().numpy().reshape(size, 3)
                    else:
                        temperature = 0.65 if mode == "cool" else 1.0
                        p = (
                            (logits / temperature)
                            .softmax(-1)
                            .cpu()
                            .numpy()
                            .reshape(size, 3, self.actions)
                        )
                        cum = np.cumsum(p, axis=-1, dtype=np.float64)
                        cum /= cum[..., -1:]
                        cum[..., -1] = 1.0
                        u = np.stack([rng.random(3) for rng in rngs])
                        actions = (u[..., None] >= cum).sum(-1)
                    rows = pool.call("step", actions)
                    total += [r[2] for r in rows]
                    hist = next_history(hist, obs, actions[:, PARTNERS], self.actions)
                    obs, masks = unpack([r[:2] for r in rows])
                    ended = [r[4] or r[5] for r in rows]
                    if any(ended):
                        if not all(ended):
                            raise RuntimeError(
                                "Continuous default runs must end at the common horizon"
                            )
                        for i, row in enumerate(rows):
                            metrics = row[6]
                            if metrics["ticks"] != 300 or metrics["team_score"] != total[i]:
                                raise AssertionError("Official terminal score / horizon mismatch")
                            results.append({"seed": seeds[i], **metrics})
                        break
            finally:
                pool.close()
        return {
            "seed": seed,
            "mode": mode,
            "engine": ENGINE,
            "summary": summarize(results),
            "per_episode": results,
        }


class Training(Model):
    def seeds(self):
        return (1 << 30) + self.config.seed + self.ids + self.episode * len(self.ids)

    def reset(self):
        self.obs, self.masks = unpack(self.pool.call("reset", self.seeds()))
        self.hist = self.history(len(self.ids))
        self.trace = []
        self.roles()

    def roles(self):
        self.ego = (self.ids + self.episode) % 3
        self.ego_t = self.tensor(self.ego)

    @torch.no_grad()
    def values(self, obs, masks, hist):
        o = self.tensor(obs[self.ids, self.ego])
        cond = self.predictor(o, self.tensor(hist)).softmax(-1)
        return self.policy(o, cond, self.tensor(masks[self.ids, self.ego]))[1]

    @torch.no_grad()
    def collect(self, steps):
        buf, terminal_rows = {}, []
        raw_sum, shaped_sum = 0.0, 0.0
        for _ in range(steps):
            o = self.tensor(self.obs)
            masks = self.tensor(self.masks)
            h = self.tensor(self.hist)
            oe, me = o[self.ids_t, self.ego_t], masks[self.ids_t, self.ego_t]
            cond = self.predictor(oe, h).softmax(-1)
            q = torch.randn(len(self.ids), 3, 2, self.actions, device=self.device).softmax(-1)
            q[self.ids_t, self.ego_t] = cond
            logits, values = self.policy(o, q, masks)
            logits, prob, behaviour_data = self.behaviour(logits, masks)
            acts = torch.multinomial(prob.flatten(0, 1), 1).reshape(-1, 3)
            ego_act = acts[self.ids_t, self.ego_t]
            old_logp = (
                logits[self.ids_t, self.ego_t]
                .log_softmax(-1)
                .gather(1, ego_act[:, None])
                .squeeze(1)
            )
            joint = acts.cpu().numpy()
            others = joint[self.ids[:, None], PARTNERS[self.ego]]
            rows = self.pool.call("step", joint)
            obs, mask = unpack([r[:2] for r in rows])
            raw = np.asarray([r[2] for r in rows], dtype=np.float32)
            shaping = np.asarray([r[3] for r in rows], dtype=np.float32)
            term = np.asarray([r[4] for r in rows])
            trunc = np.asarray([r[5] for r in rows])
            done = term | trunc
            raw_sum += float(raw.sum())
            shaped_sum += float(shaping.sum())
            weight = (
                max(0.0, 1 - self.steps / max(self.config.shaping_steps, 1))
                if self.config.shaping_steps
                else 0.0
            )
            hist = next_history(self.hist, self.obs[self.ids, self.ego], others, self.actions)
            cutoff = torch.zeros(len(self.ids), device=self.device)
            if np.any(trunc):
                cutoff = self.values(obs, mask, hist) * self.tensor(trunc & ~term)
            record = dict(
                obs=oe,
                masks=me,
                hist=h,
                cond=cond,
                act=ego_act,
                logp=old_logp,
                values=values[self.ids_t, self.ego_t],
                targets=self.tensor(others),
                reward=self.tensor((raw + weight * shaping) * self.config.reward_scale),
                done=self.tensor(done),
                cutoff=cutoff,
                **behaviour_data,
            )
            for key, value in record.items():
                buf.setdefault(key, []).append(value)
            self.steps += len(self.ids)
            if done.any():
                if not done.all():
                    raise RuntimeError("Expected equal-length continuous episodes")
                terminal_rows.extend(r[6] for r in rows)
                self.episode += 1
                self.reset()
            else:
                self.obs, self.masks, self.hist = obs, mask, hist
                self.trace.append(joint.copy())
        b = {k: torch.stack(v) for k, v in buf.items()}
        boot = self.values(self.obs, self.masks, self.hist)
        b["adv"], b["ret"] = gae(
            b["reward"],
            b["values"],
            b["done"],
            boot,
            self.config.gamma,
            self.config.gae_lambda,
            b["cutoff"],
        )
        count = steps * len(self.ids)
        stats = {
            "raw_reward_per_step": raw_sum / count,
            "shaping_reward_per_step": shaped_sum / count,
            "shaping_weight": weight,
            "episodes_finished": len(terminal_rows),
        }
        if terminal_rows:
            stats.update(summarize(terminal_rows))
        return {k: v.flatten(0, 1) for k, v in b.items()}, stats

    def checkpoint(self, path, *, milestone=False):
        state = {
            "engine": ENGINE,
            "config": asdict(self.config),
            "actions": self.actions,
            "policy": self.policy.state_dict(),
            "predictor": self.predictor.state_dict(),
            "opts": [opt.state_dict() for opt in self.opts],
            "steps": self.steps,
            "update": self.update_num,
            "episode": self.episode,
            "envs": len(self.ids),
            "trace": torch.as_tensor(np.asarray(self.trace, dtype=np.int64)),
            "history": torch.from_numpy(self.hist),
            "best": self.best,
            "selection": self.selection,
            "publish_milestone": milestone,
            "run_metadata": self.run_metadata,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        os.replace(temp, path)

    def restore(self, d):
        if (
            self.config != type(self.config).from_checkpoint(d["config"])
            or len(self.ids) != d["envs"]
        ):
            raise ValueError(
                "Resume needs the original model configuration and vector environment count"
            )
        metadata = d.get("run_metadata", {})
        if (
            metadata.get("guided", False)
            or metadata.get("protocol") is not None
            or metadata.get("architecture") != self.run_metadata["architecture"]
        ):
            raise ValueError("Resume requires a pure E3T checkpoint with the same architecture")
        load_weights(self, d)
        for opt, st in zip(self.opts, d["opts"]):
            opt.load_state_dict(st)
        self.steps, self.episode, self.update_num = d["steps"], d["episode"], d["update"]
        self.best = d["best"]
        self.selection = d.get("selection")
        traces = d["trace"].cpu().numpy()
        self.trace = list(traces)
        per_env = traces.swapaxes(0, 1).tolist() if len(traces) else [[] for _ in self.ids]
        self.obs, self.masks = unpack(self.pool.call("restore", list(zip(self.seeds(), per_env))))
        self.hist = d["history"].cpu().numpy()
        self.roles()
        torch.set_rng_state(d["rng"].cpu())
        if d["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(d["cuda_rng"].cpu(), self.device)


def load_weights(model, d):
    if d["engine"] not in (ENGINE, ARCHIVED_ENGINE_COMMIT) or d["actions"] != model.actions:
        raise ValueError("Engine or action-space checkpoint mismatch")
    model.policy.load_state_dict(d["policy"])
    model.predictor.load_state_dict(d["predictor"])


def log_record(folder, kind, steps, **data):
    record = json.dumps(
        {"kind": kind, "env_steps": steps, "wall_time": time.time(), **data}, allow_nan=False
    )
    with (folder / "progress.jsonl").open("a") as f:
        f.write(record + "\n")
    print(record, flush=True)


def rewind_progress(folder, steps):
    """Keep the committed prefix, dropping uncheckpointed updates and a torn last line."""
    path = folder / "progress.jsonl"
    if path.is_file():
        temp = path.with_suffix(".recovering")
        with path.open("rb") as source, temp.open("wb") as kept:
            for line in source:
                if steps < 0 or not line.endswith(b"\n"):
                    break
                if json.loads(line)["env_steps"] > steps:
                    break
                kept.write(line)
        os.replace(temp, path)
    for result in folder.glob("validation-*-*.json"):
        step = result.name.split("-", 2)[1]
        if step.isdigit() and int(step) > steps:
            result.unlink()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)

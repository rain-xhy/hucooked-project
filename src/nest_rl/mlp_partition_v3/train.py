"""Rollout collection, indexed-history PPO and restartable training."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES
from .features import partner_action_ids
from .history import OnlineHistory, RolloutHistory
from .kitchen import V3PolicyKitchen
from .controllers import make_partners
from .policy import V3Policy
from . import shaping as progress_shaping


@dataclass(frozen=True)
class Settings:
    lr: float
    seed: int
    updates: int
    num_envs: int = 63
    rollout_steps: int = 800
    horizon: int = 400
    epochs: int = 4
    minibatch_size: int = 900
    controller_version: str = "parallel-v2"
    recipe_rewards: tuple = (20., 10., 50.)
    shaping: bool = False
    shaping_version: str = progress_shaping.VERSION
    shaping_coefficient: float = progress_shaping.COEFFICIENT
    role_counts: tuple = None
    bc_coefficient: float = 0.
    bc_batch_size: int = 180
    target_kl: float = None
    entropy_coef: float = 0.03
    warmup_fraction: float = 0.02
    warmup_start_factor: float = 0.1
    max_grad_norm: float = 15.0
    stagger_history: bool = False


def checkpoint_settings(config):
    config = dict(config)
    config.setdefault("controller_version", "serial-v1")
    config["recipe_rewards"] = tuple(config.get("recipe_rewards", (20., 20., 20.)))
    if config.get("role_counts") is not None:
        config["role_counts"] = tuple(config["role_counts"])
    return Settings(**config)


def scheduled_lr(base, optimizer_step, total_steps, warmup_fraction=.02, start_factor=.1):
    if warmup_fraction <= 0:
        return base
    warmup = max(1, math.ceil(warmup_fraction * total_steps))
    if warmup == 1:
        return base
    return base * (start_factor + (1. - start_factor) * min(1., optimizer_step / (warmup - 1)))


class Trainer:
    def __init__(self, cfg, device):
        if min(cfg.num_envs, cfg.horizon, cfg.rollout_steps, cfg.updates,
                                   cfg.minibatch_size, cfg.epochs) < 1:
            raise ValueError("Use positive sizes")
        self.cfg = cfg
        if cfg.role_counts is not None and (len(cfg.role_counts) != 3 or
                any(type(n) is not int or n < 3 for n in cfg.role_counts) or
                sum(cfg.role_counts) != cfg.num_envs):
            raise ValueError("role_counts must contain three integers >= 3 summing to num_envs")
        if not math.isfinite(cfg.bc_coefficient) or cfg.bc_coefficient < 0 or cfg.bc_batch_size < 1:
            raise ValueError("Invalid demonstration settings")
        if cfg.target_kl is not None and (not math.isfinite(cfg.target_kl) or cfg.target_kl <= 0):
            raise ValueError("target_kl must be positive")
        if not math.isfinite(cfg.entropy_coef) or cfg.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        if not math.isfinite(cfg.warmup_fraction) or not (0. <= cfg.warmup_fraction <= 1.):
            raise ValueError("warmup_fraction must be in [0, 1]")
        if not math.isfinite(cfg.warmup_start_factor) or not (0. < cfg.warmup_start_factor <= 1.):
            raise ValueError("warmup_start_factor must be in (0, 1]")
        if not math.isfinite(cfg.max_grad_norm) or cfg.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if cfg.shaping_version != progress_shaping.VERSION or cfg.shaping_coefficient != progress_shaping.COEFFICIENT:
            raise ValueError("Unsupported shaping configuration")
        self.device = torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=cfg.horizon,
                                                  recipe_rewards=cfg.recipe_rewards),
                                   num_envs=cfg.num_envs, device=device)
        self.obs, _ = self.env.reset(seed=cfg.seed)
        group = torch.arange(cfg.num_envs, device=device) % 9
        self.roles, self.styles = group // 3, group % 3
        if cfg.role_counts is not None:
            self.roles = torch.repeat_interleave(torch.arange(3, device=device),
                                                torch.tensor(cfg.role_counts, device=device))
            self.styles = torch.arange(cfg.num_envs, device=device) % 3
        self.partners = make_partners(self.env, self.roles, self.styles, cfg.controller_version,
                                     learner_mask=torch.ones(cfg.num_envs, device=device, dtype=torch.bool))
        self.partners.reset()
        self.history = OnlineHistory(cfg.num_envs, device, stagger=cfg.stagger_history)
        self.model = V3Policy(3).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        self.tick = self.update = 0
        self.optimizer_step_count = 0
        self.demonstrations = None
        # Reward accounting spans rollout boundaries, including partial episodes.
        self.episode_totals = torch.zeros(2, dtype=torch.float64, device=device)
        self.episode_totals_valid = True

    def attach_demonstrations(self, bank, data):
        """Attach immutable, reproducibly rebuilt expert data before updates.

        Attach before load() when rebuilding on resume: load restores sampling RNG.
        Demonstration tensors are not copied into every full checkpoint.
        """
        device = next(self.model.parameters()).device
        if bank.obs.device != device or data["action"].device != device:
            raise ValueError("Demonstrations and trainer must use the same device")
        if len(data["action"]) < 1:
            raise ValueError("Empty demonstration dataset")
        self.demonstrations = bank, data

    def state_dict(self):
        return dict(config=asdict(self.cfg), model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(), update=self.update, tick=self.tick,
                    optimizer_step_count=self.optimizer_step_count,
                    episode_totals=self.episode_totals,
                    episode_totals_valid=self.episode_totals_valid,
                    env=self.env.state, draws=self.env.draws,
                    lane_rng=[[r.bit_generator.state for r in pair] for pair in self.env.rng],
                    partners=self.partners.state_dict(), history=self.history.state_dict(),
                    roles=self.roles, styles=self.styles,
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        restored = checkpoint_settings(state["config"])
        current = asdict(self.cfg)
        for key in ("controller_version", "recipe_rewards", "entropy_coef",
                    "warmup_fraction", "warmup_start_factor", "max_grad_norm", "stagger_history"):
            if key not in state["config"]:
                current[key] = getattr(restored, key)
        if current != asdict(restored):
            raise ValueError("Checkpoint configuration differs from requested training schedule")
        if self.cfg != restored:
            self.__init__(restored, self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update, self.tick = state["update"], state["tick"]
        if "episode_totals" in state:
            self.episode_totals.copy_(state["episode_totals"])
            self.episode_totals_valid = state["episode_totals_valid"]
        else:
            self.episode_totals.zero_()
            # An old mid-episode save has no earlier reward accounting. Exclude
            # its first finished episode from score means instead of undercounting.
            self.episode_totals_valid = self.tick == 0
        self.optimizer_step_count = state.get("optimizer_step_count", self.update * self.cfg.epochs *
            math.ceil(self.cfg.num_envs * self.cfg.rollout_steps / self.cfg.minibatch_size))
        self.env.state = state["env"]
        self.env.draws.copy_(state["draws"])
        for generators, saved in zip(self.env.rng, state["lane_rng"]):
            for generator, source in zip(generators, saved):
                generator.bit_generator.state = source
        self.roles.copy_(state["roles"])
        self.styles.copy_(state["styles"])
        self.partners.load_state_dict(state["partners"])
        self.history.load_state_dict(state["history"])
        self.obs = self.env.observations()
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])

    @torch.no_grad()
    def collect(self):
        cfg = self.cfg
        bank = RolloutHistory(self.history, cfg.rollout_steps)
        actions, logprobs, values, rewards, labels, terminals = [], [], [], [], [], []
        lanes = self.env.lanes
        totals = self.obs.new_zeros(3)
        shaping_total = self.obs.new_zeros(())
        recipe_deliveries = torch.zeros(3, dtype=torch.long, device=self.device)
        episodes = 0
        scored_episodes = 0
        completed_totals = torch.zeros_like(self.episode_totals)
        # Only local to this rollout: restore/external state edits cannot leave
        # a stale potential. Within an episode, next Phi is the next step's Phi.
        before = None
        self.history.refresh(self.model)
        for _ in range(cfg.rollout_steps):
            learner = self.partners.learner_ids
            current = self.obs[lanes, learner]
            logp, value, _ = self.history.forward(self.model, current)
            choice = logp.exp().multinomial(1)
            joint = self.partners.actions().clone()
            joint[lanes, learner] = choice[:, 0]
            if cfg.shaping and before is None:
                before = progress_shaping.potential(self.env.state)
            self.obs, reward, _, _, info = self.env.step(joint)
            peers = partner_action_ids(joint)[lanes, learner]
            bank.record(current, peers)
            self.history.append(current, peers, self.model)
            actions.append(choice[:, 0])
            logprobs.append(logp.gather(1, choice)[:, 0])
            values.append(value[:, 0])
            labels.append(self.styles.clone())
            self.tick += 1
            done = self.tick == cfg.horizon
            if cfg.shaping:
                after = torch.zeros_like(before) if done else progress_shaping.potential(self.env.state)
                shaped = progress_shaping.COEFFICIENT * (progress_shaping.GAMMA * after - before)
                before = after
            else:
                shaped = torch.zeros_like(reward)
            rewards.append(reward.float() + shaped.float())
            shaping_total += shaped.sum()
            self.episode_totals += torch.stack((reward.sum(), shaped.sum()))
            terminals.append(done)
            totals += torch.stack((reward.sum(), info["delivered"].sum(), info["expired"].sum()))
            recipe_deliveries.scatter_add_(0, info["recipe"].flatten(), info["delivered"].long().flatten())
            if done:
                if self.episode_totals_valid:
                    completed_totals += self.episode_totals
                    scored_episodes += cfg.num_envs
                self.episode_totals.zero_()
                self.episode_totals_valid = True
                self.history.finish_episode()
                bank.finish_episode()
                self.obs, _ = self.env.reset()
                self.partners.reset()
                self.tick = 0
                before = None
                episodes += cfg.num_envs
        bank.seal()
        current = self.obs[lanes, self.partners.learner_ids]
        next_value = self.history.forward(self.model, current)[1][:, 0]
        values = torch.stack(values)
        advantages = torch.zeros_like(values)
        carry = torch.zeros_like(next_value)
        for t in reversed(range(cfg.rollout_steps)):
            discount = 0.0 if terminals[t] else 0.99
            delta = rewards[t] + discount * next_value - values[t]
            carry = delta + discount * 0.95 * carry
            advantages[t] = carry
            next_value = values[t]
        returns = (advantages + values).flatten()
        adv = advantages.flatten()
        data = dict(action=torch.cat(actions), logp=torch.cat(logprobs), labels=torch.cat(labels),
                    returns=returns, old_values=values.flatten(), adv=(adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8))
        reward, deliveries, expired = totals.tolist()
        metrics = dict(episodes=episodes, reward=reward, deliveries=deliveries, expirations=expired,
                       total_score=reward, reward_per_lane=reward / cfg.num_envs,
                       shaping_reward=shaping_total.item())
        metrics["score_episodes"] = scored_episodes
        if scored_episodes:
            completed_score, completed_shaping = completed_totals.tolist()
            metrics["mean_score"] = completed_score / scored_episodes
            metrics["mean_shaping_reward"] = completed_shaping / scored_episodes
        for recipe, count, points in zip(RECIPE_NAMES, recipe_deliveries.tolist(), cfg.recipe_rewards):
            metrics[f"per_recipe/{recipe}/score"] = count * points
        return bank, data, metrics

    def optimize(self, bank, data):
        cfg = self.cfg
        if cfg.bc_coefficient and self.demonstrations is None:
            raise ValueError("Attach expert demonstrations before BC-regularized updates")
        batch = len(data["action"])
        metrics = []
        bc_values = []
        stopped = False
        steps_per_update = cfg.epochs * math.ceil(batch / cfg.minibatch_size)
        total_steps = cfg.updates * steps_per_update
        for _ in range(cfg.epochs):
            for index in torch.randperm(batch, device=self.device).split(cfg.minibatch_size):
                lr = scheduled_lr(cfg.lr, self.optimizer_step_count, total_steps,
                                  cfg.warmup_fraction, cfg.warmup_start_factor)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                logp, value, aux, has = bank.replay(self.model, index)
                selected = logp.gather(1, data["action"][index, None])[:, 0]
                logratio = selected - data["logp"][index]
                ratio = logratio.exp()
                # PPO clipping alone is not a bound on the change in policy.
                # Check before another optimizer step, stop all remaining epochs.
                with torch.no_grad():
                    current_kl = (ratio - 1 - logratio).mean()
                    if not torch.isfinite(current_kl):
                        raise FloatingPointError("Non-finite PPO likelihood ratio / KL")
                if cfg.target_kl is not None and metrics and current_kl > 1.5 * cfg.target_kl:
                    stopped = True
                    break
                clipped = ratio.clamp(0.8, 1.2)
                advantage = data["adv"][index]
                actor = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                old_value = data["old_values"][index]
                clipped_value = old_value + (value[:, 0] - old_value).clamp(-.2, .2)
                target = data["returns"][index]
                critic = .5 * torch.maximum((value[:, 0] - target).square(),
                                            (clipped_value - target).square()).mean()
                entropy = -(logp.exp() * logp).sum(-1).mean()
                labels = data["labels"][index]
                valid = has.sum().clamp_min(1)
                ce = (F.cross_entropy(aux, labels, reduction="none") * has).sum() / valid
                accuracy = ((aux.argmax(-1) == labels) & has).sum() / valid
                loss = actor + .5 * critic - cfg.entropy_coef * entropy + .1 * ce
                bc_loss = loss.new_zeros(())
                if cfg.bc_coefficient:
                    expert_bank, expert_data = self.demonstrations
                    expert_index = torch.randint(len(expert_data["action"]),
                                                 (cfg.bc_batch_size,), device=self.device)
                    expert_logp, *_ = expert_bank.replay(self.model, expert_index)
                    bc_loss = F.nll_loss(expert_logp, expert_data["action"][expert_index])
                    loss = loss + cfg.bc_coefficient * bc_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                except RuntimeError as error:
                    if "non-finite" in str(error):
                        raise FloatingPointError(str(error)) from error
                    raise
                self.optimizer.step()
                self.optimizer_step_count += 1
                with torch.no_grad():
                    kl = (ratio - 1 - logratio).mean()
                    clip_fraction = (ratio != clipped).float().mean()
                    metrics.append(torch.stack((loss, actor, critic, entropy, ce, accuracy, kl,
                                                clip_fraction, norm)).detach())
                    bc_values.append(bc_loss.detach())
            if stopped:
                break
        names = ("loss", "policy_loss", "value_loss", "entropy", "aux_loss", "aux_accuracy",
                 "approx_kl", "clip_fraction", "grad_norm")
        result = dict(zip(names, torch.stack(metrics).mean(0).tolist()))
        if cfg.bc_coefficient or cfg.target_kl is not None:
            result.update(bc_loss=torch.stack(bc_values).mean().item(),
                          optimizer_steps=len(metrics), kl_early_stop=int(stopped))
        return result

    def step(self):
        self.model.train()
        warmup = max(1, math.ceil(.02 * self.cfg.updates))
        lr = self.cfg.lr * min(1., (self.update + 1) / warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        start = time.perf_counter()
        bank, data, metrics = self.collect()
        metrics.update(self.optimize(bank, data))
        self.update += 1
        count = self.cfg.num_envs * self.cfg.rollout_steps
        metrics.update(env_steps=self.update * count, lr=self.optimizer.param_groups[0]["lr"], steps_per_second=count / (time.perf_counter() - start))
        return metrics


def atomic_save(data, path):
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def train(run_dir, lr, seed, updates, *, device="cuda", resume=None, callback=None,
          stop_after=None, init_policy=None, **sizes):
    """Return the last full checkpoint. stop_after interrupts a fixed total schedule."""
    cfg = Settings(lr, seed, updates, **sizes)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, device)
    if resume is not None and init_policy is not None:
        raise ValueError("Choose resume or init_policy, not both")
    if init_policy is not None:
        artifact = torch.load(init_policy, map_location=device, weights_only=False)
        trainer.model.load_state_dict(artifact["model"])
    if resume is not None:
        trainer.load(resume)
    journal = directory / "train.jsonl"
    if journal.exists():
        retained = []
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                break  # A process may have stopped in the middle of a write.
            if entry["update"] <= trainer.update:
                retained.append(line)
        tmp = journal.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
        tmp.replace(journal)
    end = min(updates, stop_after) if stop_after is not None else updates
    final = directory / f"checkpoint_{trainer.update:06d}.pt"
    for update in range(trainer.update + 1, end + 1):
        metrics = trainer.step()
        if update % 5 == 0 or update == end:
            atomic_save(dict(model=trainer.model.state_dict(), update=update, config=asdict(trainer.cfg)),
                        directory / f"policy_{update:06d}.pt")
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(update=update, **metrics), allow_nan=False) + "\n")
        if update % 20 == 0 or update == end:
            final = directory / f"checkpoint_{update:06d}.pt"
            atomic_save(trainer.state_dict(), final)
        if callback:
            # Isolate evaluation draws from the next rollout and minibatch shuffle.
            np_state, py_state = np.random.get_state(), random.getstate()
            cuda_devices = [trainer.device] if trainer.device.type == "cuda" else []
            try:
                with torch.random.fork_rng(devices=cuda_devices):
                    callback(update, trainer.model, metrics)
            finally:
                np.random.set_state(np_state)
                random.setstate(py_state)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--init-policy")
    parser.add_argument("--shaping", action="store_true")
    train(**vars(parser.parse_args()))


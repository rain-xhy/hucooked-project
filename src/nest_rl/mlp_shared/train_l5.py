"""DEPRECATED L5 experiment: capped coordination penalties; not adopted.

This recipe did not retain all L4 metrics. Keep L4 U165 as the selected model.
The implementation is retained only for inspecting the discarded experiment.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import trackio

from .env import FLOOR, SharedKitchenConfig, TorchSharedKitchen
from .fast_partners import GpuFastPartners
from .l5_evaluation import L5Evaluator, assess_retention, selection_objective
from .policy import V3Policy
from .pretrain import atomic_save
from .train_l1 import PROJECT, RUNS, scheduled_lr

EXPERIMENT_STATUS = 'deprecated'


@dataclass(frozen=True)
class L5Settings:
    seed: int = 1
    num_envs: int = 1008
    horizon: int = 300
    updates: int = 100
    epochs: int = 2
    minibatch_size: int = 3024
    lr: float = 1e-5
    gamma: float = .995
    gae_lambda: float = .95
    entropy_coef: float = .001
    target_kl: float = .005
    clip_ratio: float = .1
    max_grad_norm: float = 2.
    random_spawn_prob: float = .8
    unmatched_penalty: float = .01
    wait_penalty: float = .002
    penalty_cap: float = 2.
    eval_every: int = 5

    def __post_init__(self):
        if self.num_envs < 30 or self.num_envs % 3:
            raise ValueError('Need at least 30 environments, balanced across three roles')
        if self.horizon != 300 or self.updates < 1 or self.epochs < 1 or self.eval_every < 1:
            raise ValueError('L5 retains the 300-step task and requires positive run counts')
        if min(self.unmatched_penalty, self.wait_penalty, self.penalty_cap) < 0:
            raise ValueError('Penalty magnitudes must be nonnegative')


def peer_counts(num_envs):
    each = max(3, (round(num_envs * .1) // 3) * 3)
    return each, each, num_envs - 2 * each


class PenaltyAccumulator:
    def __init__(self, n, device, *, unmatched=.01, waiting=.002, cap=2.):
        self.used = torch.zeros(n, device=device)
        self.unmatched, self.waiting, self.cap = unmatched, waiting, cap

    def reset(self):
        self.used.zero_()

    def step(self, unmatched_delta, waiting_delta):
        remaining = (self.cap - self.used).clamp_min(0)
        unmatched = torch.minimum(unmatched_delta.float() * self.unmatched, remaining)
        waiting = torch.minimum(waiting_delta.float() * self.waiting, (remaining - unmatched).clamp_min(0))
        self.used.add_(unmatched + waiting)
        return unmatched, waiting


class L5Trainer:
    def __init__(self, cfg, checkpoint, device='cuda'):
        self.cfg, self.device = cfg, torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get('update') != 165:
            raise ValueError('L5 must initialize from the selected L4 U165 checkpoint')
        self.model = V3Policy(3, use_history=False).to(device)
        self.model.load_state_dict(payload['model'], strict=True)
        self.frozen_l4 = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-8)
        self.env = TorchSharedKitchen(SharedKitchenConfig(horizon=cfg.horizon, shaping_scale=0),
            num_envs=cfg.num_envs, device=device, use_cuda_graph=True)
        self.env.reset(seed=cfg.seed)
        self.roles = self.env.lanes % 3
        self.partners = GpuFastPartners(self.env, roles=self.roles,
            learner_mask=torch.ones(cfg.num_envs, device=device, dtype=torch.bool))
        self.n_fast, self.n_frozen, self.n_current = peer_counts(cfg.num_envs)
        self.floor = torch.tensor(sorted(FLOOR), device=device)
        self.penalties = PenaltyAccumulator(cfg.num_envs, device,
            unmatched=cfg.unmatched_penalty, waiting=cfg.wait_penalty, cap=cfg.penalty_cap)
        self.update = self.optimizer_step_count = 0

    def reset(self):
        cfg, env = self.cfg, self.env
        obs, _ = env.reset()
        mask = torch.rand(cfg.num_envs, device=self.device) < cfg.random_spawn_prob
        order = torch.rand((cfg.num_envs, len(self.floor)), device=self.device).argsort(dim=1)
        positions = self.floor[order[:, :3]]
        env.state['pos'].copy_(torch.where(mask[:, None], positions, env.state['pos']))
        env._write_observations()
        self.partners.reset()
        self.penalties.reset()
        # Repeat each draw across a triplet to balance all focal roles within each noise class.
        category = torch.rand(cfg.num_envs // 3, device=self.device).repeat_interleave(3)
        self.noise = torch.where(category < .2, 0., torch.where(category < .8, .05, .10))
        self.random_spawn_fraction = float(mask.float().mean())
        return env._obs_out.clone()

    @torch.no_grad()
    def collect(self):
        cfg, env = self.cfg, self.env
        obs = self.reset()
        lanes = env.lanes
        learner = self.partners.learner_ids
        peers = torch.tensor([[1, 2], [0, 2], [0, 1]], device=self.device)[self.roles]
        peer_slots = self.partners.chef_ids.gather(1, peers)
        currents, actions, logprobs, values, rewards = [], [], [], [], []
        sparse_sum = torch.zeros(cfg.num_envs, device=self.device)
        unmatched_sum = torch.zeros_like(sparse_sum)
        wait_sum = torch.zeros_like(sparse_sum)
        frozen_start, current_start = self.n_fast, self.n_fast + self.n_frozen
        for _ in range(cfg.horizon):
            current = obs[lanes, learner]
            logp, value, _ = self.model(current)
            choice = logp.exp().multinomial(1)[:, 0]
            joint = self.partners.actions()
            joint[lanes, learner] = choice
            for begin, end, policy in ((frozen_start, current_start, self.frozen_l4),
                                        (current_start, cfg.num_envs, self.model)):
                indices = lanes[begin:end]
                peer_obs = obs[indices[:, None], peer_slots[begin:end]]
                peer_actions = policy.actor(peer_obs.reshape(-1, 376)).argmax(-1).reshape(-1, 2)
                joint[indices[:, None], peer_slots[begin:end]] = peer_actions
            mask = torch.rand((cfg.num_envs, 3), device=self.device) < self.noise[:, None]
            mask[lanes, learner] = False
            replacement = torch.randint(0, 6, joint.shape, device=self.device)
            before_unmatched = env.state['unmatched_attempt_steps'].clone()
            before_wait = (env.state['joint_ready_steps'] - env.state['joint_successes']).clone()
            obs, sparse, _, _, _ = env.step(torch.where(mask, replacement, joint))
            unmatched, waiting = self.penalties.step(
                env.state['unmatched_attempt_steps'] - before_unmatched,
                env.state['joint_ready_steps'] - env.state['joint_successes'] - before_wait)
            # Native env.score and its reward buffer remain untouched.
            reward = sparse - unmatched - waiting
            sparse_sum.add_(sparse)
            unmatched_sum.add_(unmatched)
            wait_sum.add_(waiting)
            currents.append(current)
            actions.append(choice)
            logprobs.append(logp.gather(1, choice[:, None])[:, 0])
            values.append(value[:, 0])
            rewards.append(reward)
        torch.testing.assert_close(sparse_sum, env.state['score'], rtol=0, atol=1e-5)
        if bool((self.penalties.used > cfg.penalty_cap + 1e-5).any()):
            raise RuntimeError('Episode penalty cap exceeded')
        vals, rew = torch.stack(values), torch.stack(rewards)
        advantages = torch.zeros_like(vals)
        carry = torch.zeros(cfg.num_envs, device=self.device)
        next_value = torch.zeros_like(carry)
        for t in reversed(range(cfg.horizon)):
            discount = 0. if t == cfg.horizon - 1 else cfg.gamma
            delta = rew[t] + discount * next_value - vals[t]
            carry = delta + discount * cfg.gae_lambda * carry
            advantages[t] = carry
            next_value = vals[t]
        returns = advantages + vals
        for role in range(3):
            selected = self.roles == role
            group = advantages[:, selected]
            advantages[:, selected] = (group - group.mean()) / (group.std(unbiased=False) + 1e-8)
        data = dict(obs=torch.stack(currents).reshape(-1, 376), actions=torch.stack(actions).flatten(),
            logp=torch.stack(logprobs).flatten(), old_values=vals.flatten(),
            advantages=advantages.flatten(), returns=returns.flatten())
        scores = env.state['score']
        metrics = dict(mean_score=float(scores.mean()), score_per_episode=float(scores.mean()),
            sparse_reward=float(sparse_sum.mean()), unmatched_penalty=float(unmatched_sum.mean()),
            joint_wait_penalty=float(wait_sum.mean()), total_penalty=float(self.penalties.used.mean()),
            max_episode_penalty=float(self.penalties.used.max()), training_return=float(rew.sum(0).mean()),
            mean_step_reward=float(rew.mean()), random_spawn_fraction=self.random_spawn_fraction,
            partner_noise=float(self.noise.mean()), bc_coef=0., bc_loss=0.,
            fast_peer_score=float(scores[:self.n_fast].mean()),
            frozen_l4_peer_score=float(scores[frozen_start:current_start].mean()),
            current_peer_score=float(scores[current_start:].mean()))
        metrics.update({f'role_{role}_score': float(scores[self.roles == i].mean())
                        for i, role in enumerate(('server', 'main', 'side'))})
        metrics.update({f'episode_{key}': value for key, value in env.episode_metrics().items()})
        return data, metrics

    def optimize(self, data):
        cfg = self.cfg
        lr = scheduled_lr(cfg.lr, self.update, cfg.updates, .1, .05)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        accumulated = []
        stopped = False
        for _ in range(cfg.epochs):
            for index in torch.randperm(len(data['actions']), device=self.device).split(cfg.minibatch_size):
                logp, value, _ = self.model(data['obs'][index])
                selected = logp.gather(1, data['actions'][index, None])[:, 0]
                logratio = selected - data['logp'][index]
                ratio = logratio.exp()
                with torch.no_grad():
                    kl = (ratio - 1 - logratio).mean()
                    if not bool(torch.isfinite(kl)):
                        raise FloatingPointError('Non-finite PPO KL')
                    if accumulated and kl > 1.5 * cfg.target_kl:
                        stopped = True
                        break
                clipped = ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                advantage = data['advantages'][index]
                actor = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                old_value = data['old_values'][index]
                clipped_value = old_value + (value[:, 0] - old_value).clamp(-.2, .2)
                target = data['returns'][index]
                critic = .5 * torch.maximum((value[:, 0] - target).square(), (clipped_value - target).square()).mean()
                entropy = -(logp.exp() * logp).sum(-1).mean()
                loss = actor + .5 * critic - cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer_step_count += 1
                accumulated.append(torch.stack((loss, actor, critic, entropy, kl, grad)).detach())
            if stopped:
                break
        names = ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'grad_norm')
        result = dict(zip(names, torch.stack(accumulated).mean(0).tolist()))
        result.update(lr=lr, optimizer_steps=len(accumulated), kl_early_stop=int(stopped))
        return result

    def step(self):
        self.model.train()
        started = time.perf_counter()
        data, metrics = self.collect()
        metrics.update(self.optimize(data))
        self.update += 1
        steps = self.cfg.num_envs * self.cfg.horizon
        metrics.update(env_steps=self.update * steps, steps_per_second=steps / (time.perf_counter() - started))
        return metrics

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    update=self.update, config=asdict(self.cfg), optimizer_steps=self.optimizer_step_count,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def evaluation_logs(suite, prefix='eval'):
    result = {}
    for case, metrics in suite.items():
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                result[f'{prefix}/{case}/{key}'] = value
        for role, value in metrics['per_role_score'].items():
            result[f'{prefix}/{case}/role_{role}_score'] = value
    if 'standard' in suite:
        result.update({'eval/clean_all_policy_score': suite['standard']['team_score'],
            'eval/all_policy_score': suite['standard']['team_score'],
            'eval/team_score': suite['standard']['team_score'],
            'eval/joint_attempt_success_rate': suite['standard']['pooled_sync_success_rate'],
            'eval/joint_wait_steps': suite['standard']['joint_wait_steps']})
    if 'fast_rule' in suite:
        result['eval/vs_rule_score'] = suite['fast_rule']['team_score']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--init-policy', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    print('DEPRECATED: this L5 recipe was not adopted; the selected model remains L4 U165.', flush=True)
    torch.set_num_threads(4)
    cfg = L5Settings(updates=args.updates, seed=args.seed)
    destination = RUNS / args.campaign / 'l5'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'validation').mkdir()
    root = Path(__file__).resolve().parents[3]
    source_sha = hashlib.sha256(args.init_policy.read_bytes()).hexdigest()
    config = dict(asdict(cfg), experiment_status=EXPERIMENT_STATUS, adopted=False,
        init_policy=str(args.init_policy.resolve()), init_policy_sha256=source_sha,
        init_update=165, project=PROJECT, group='shared-l5', device=args.device,
        gpu=torch.cuda.get_device_name(), peer_counts=peer_counts(cfg.num_envs),
        noise_probabilities=[.2, .6, .2], noise_values=[0, .05, .10],
        shaping_scale=0, use_history=False, bc_coef=0,
        selection_seed=33_000_000, holdout_seed=34_000_000,
        selection_episodes_per_gpu_case=135, selection_episodes_per_native_case=96,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        environment_commit=subprocess.check_output(['git', '-C', 'overcooked_env', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        source_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('train_l5.py', 'l5_evaluation.py')})
    write_json(destination / 'experiment.json', config)
    trainer = L5Trainer(cfg, args.init_policy, device=args.device)
    run = trackio.init(project=PROJECT, name=f'{args.campaign}-l5-seed{cfg.seed}', group='shared-l5', config=config, resume='never')
    evaluator = L5Evaluator(args.device)
    baseline = {}
    best_objective = None
    best_update = None
    def status(state, **extra):
        write_json(destination / 'status.json', dict(status=state, update=trainer.update, total_updates=cfg.updates,
            env_steps=trainer.update * cfg.num_envs * cfg.horizon, best_selection_update=best_update, **extra))
    def check_source():
        if hashlib.sha256(args.init_policy.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError('L4 source checkpoint changed unexpectedly')
    try:
        status('baseline_gpu')
        gpu_base, gpu_raw = evaluator.gpu_suite(trainer.frozen_l4)
        write_json(destination / 'baseline_gpu.json', dict(summary=gpu_base, episodes=gpu_raw))
        print('L4 GPU retention baseline ready', flush=True)
        status('baseline_native')
        native_base, native_raw = evaluator.native_suite(trainer.frozen_l4)
        baseline = {**gpu_base, **native_base}
        write_json(destination / 'baseline_native.json', dict(summary=native_base, episodes=native_raw))
        write_json(destination / 'baseline_summary.json', baseline)
        best_objective = selection_objective(baseline)
        run.log(evaluation_logs(baseline, prefix='baseline_l4'), step=0)
        print(f'L4 baseline complete: {len(baseline)} cases, objective={best_objective}', flush=True)
        status('training')
        for _ in range(cfg.updates):
            metrics = trainer.step()
            update = trainer.update
            logs = {f'train/{key}': value for key, value in metrics.items()}
            logs.update({f'opt/{key}': metrics[key] for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'grad_norm', 'lr')})
            with (destination / 'train.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(update=update, **metrics), allow_nan=False) + '\n')
            if update == 1 or update % cfg.eval_every == 0 or update == cfg.updates:
                status('evaluating_gpu')
                gpu_suite, raw = evaluator.gpu_suite(trainer.model)
                guard = assess_retention(gpu_suite, gpu_base)
                suite = dict(gpu_suite)
                full = False
                if guard['passed']:
                    status('evaluating_native')
                    native_suite, native_raw = evaluator.native_suite(trainer.model)
                    suite.update(native_suite)
                    raw.update(native_raw)
                    guard = assess_retention(suite, baseline)
                    full = True
                objective = selection_objective(suite)
                result = dict(summary=suite, episodes=raw, retention=guard, full_evaluation=full, objective=objective)
                write_json(destination / 'validation' / f'{update:03d}.json', result)
                logs.update(evaluation_logs(suite))
                logs.update({'eval/retention_pass': int(full and guard['passed']),
                    'eval/guard_violation_count': len(guard['violations']),
                    'eval/full_native_evaluation': int(full)})
                checkpoint = dict(model=trainer.model.state_dict(), config=config, update=update,
                    retention_pass=bool(full and guard['passed']), objective=objective)
                atomic_save(checkpoint, destination / f'policy_{update:06d}.pt')
                atomic_save(trainer.state_dict(), destination / 'latest.pt')
                if full and guard['passed'] and objective > best_objective:
                    best_objective, best_update = objective, update
                    atomic_save(checkpoint, destination / 'best_selection.pt')
                    print(f'L5 selection candidate U{update}: {objective}', flush=True)
                print(f'L5 U{update} guard={full and guard["passed"]} first_violations={guard["violations"][:3]}', flush=True)
            run.log(logs, step=update)
            status('training', train_score=metrics['mean_score'], last_training_return=metrics['training_return'])
            print(f'L5 {update}/{cfg.updates} sparse={metrics["mean_score"]:.2f} '
                  f'cost={metrics["total_penalty"]:.4f} return={metrics["training_return"]:.2f} '
                  f'sps={metrics["steps_per_second"]:,.0f} best_selection={best_update}', flush=True)
        if best_update is not None:
            status('holdout_validation')
            checkpoint = torch.load(destination / 'best_selection.pt', map_location=args.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model'], strict=True)
            evaluator.seed = 34_000_000
            base_gpu, base_raw = evaluator.gpu_suite(trainer.frozen_l4)
            base_native, base_native_raw = evaluator.native_suite(trainer.frozen_l4)
            trial_gpu, trial_raw = evaluator.gpu_suite(trainer.model)
            trial_native, trial_native_raw = evaluator.native_suite(trainer.model)
            holdout_base, holdout_trial = {**base_gpu, **base_native}, {**trial_gpu, **trial_native}
            holdout_guard = assess_retention(holdout_trial, holdout_base)
            improved = selection_objective(holdout_trial) > selection_objective(holdout_base)
            write_json(destination / 'holdout.json', dict(baseline=holdout_base, candidate=holdout_trial,
                baseline_episodes={**base_raw, **base_native_raw}, candidate_episodes={**trial_raw, **trial_native_raw},
                retention=holdout_guard, improved=improved))
            if holdout_guard['passed'] and improved:
                checkpoint['holdout_retention_pass'] = True
                atomic_save(checkpoint, destination / 'best.pt')
                final_state = 'completed_guarded_l5'
            else:
                final_state = 'completed_no_holdout_qualified_l5'
            run.log({'holdout/retention_pass': int(holdout_guard['passed']), 'holdout/improved': int(improved)}, step=trainer.update + 1)
        else:
            final_state = 'completed_no_qualified_l5'
        check_source()
        status(final_state, accepted_best_exists=(destination / 'best.pt').exists())
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        run.finish()


if __name__ == '__main__':
    main()

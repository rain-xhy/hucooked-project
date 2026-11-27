"""Evaluation and retention checks for the discarded, non-adopted L5 experiment."""
import itertools
import math
from pathlib import Path
import sys

import numpy as np
import torch

from .env import FLOOR, SharedKitchenConfig, TorchSharedKitchen
from .evaluation import role_slot_map
from .fast_partners import GpuFastPartners

ROLES = ('server', 'main', 'side')
GPU_CASES = (
    ('standard', 'standard', 0., False),
    ('jitter1', 'jitter1', 0., False),
    ('random_floor', 'random_floor', 0., False),
    ('noise05', 'standard', .05, False),
    ('noise10', 'standard', .10, False),
    ('random_floor_noise05', 'random_floor', .05, False),
    ('fast_rule', 'standard', 0., True),
    ('fast_rule_noise05', 'standard', .05, True),
    ('fast_rule_noise10', 'standard', .10, True),
)
LOWER_BOUNDS = ('team_score', 'meals_completed', 'normal_completed', 'joint_completed',
                'three_contributor_meals', 'score_min', 'score_p10',
                'joint_attempt_success_rate', 'pooled_sync_success_rate')
UPPER_BOUNDS = ('movement_conflicts', 'resource_conflicts', 'zero_episodes', 'joint_wait_steps')


def assess_retention(candidate, reference, tolerance=1e-6):
    violations = []
    for case, base in reference.items():
        trial = candidate.get(case)
        if trial is None:
            violations.append(f'{case}: missing evaluation')
            continue
        for field in LOWER_BOUNDS + UPPER_BOUNDS:
            value = trial.get(field)
            if value is None or not math.isfinite(float(value)):
                violations.append(f'{case}/{field}: missing or non-finite')
            elif field in LOWER_BOUNDS and value + tolerance < base[field]:
                violations.append(f'{case}/{field}: {value:.6f} < L4 {base[field]:.6f}')
            elif field in UPPER_BOUNDS and value > base[field] + tolerance:
                violations.append(f'{case}/{field}: {value:.6f} > L4 {base[field]:.6f}')
        for role, baseline in base['per_role_score'].items():
            value = trial.get('per_role_score', {}).get(role)
            if value is None or not math.isfinite(float(value)) or value + tolerance < baseline:
                violations.append(f'{case}/role/{role}: {value} < L4 {baseline:.6f}')
    return dict(passed=not violations, violations=violations)


def selection_objective(suite):
    return (sum(m['pooled_sync_success_rate'] for m in suite.values()) / len(suite),
            -sum(m['joint_wait_steps'] for m in suite.values()) / len(suite))


def summarize(records):
    n = len(records)
    scores = np.asarray([r['team_score'] for r in records])
    keys = ('team_score', 'meals_completed', 'normal_completed', 'joint_completed',
            'three_contributor_meals', 'movement_conflicts', 'resource_conflicts',
            'joint_wait_steps', 'joint_attempt_success_rate', 'unmatched_attempt_steps')
    result = {key: sum(r[key] for r in records) / n for key in keys}
    result.update(episodes=n, score_min=float(scores.min()), score_p10=float(np.percentile(scores, 10)),
                  zero_episodes=int((scores == 0).sum()),
                  pooled_sync_success_rate=sum(r['joint_successes'] for r in records) /
                  max(1, sum(r['joint_attempt_steps'] for r in records)),
                  per_role_score={role: float(np.mean([r['team_score'] for r in records if r['role'] == role]))
                                  for role in ROLES})
    return result


class L5Evaluator:
    def __init__(self, device='cuda', seed=33_000_000, num_envs=45, batches=3, native_episodes=96):
        if num_envs % 3 or native_episodes % 3:
            raise ValueError('Evaluations must balance all three roles')
        self.device = torch.device(device)
        self.seed, self.batches, self.native_episodes = seed, batches, native_episodes
        self.env = TorchSharedKitchen(SharedKitchenConfig(), num_envs=num_envs, device=device, use_cuda_graph=True)
        self.roles = self.env.lanes % 3
        self.partners = GpuFastPartners(self.env, roles=self.roles,
            learner_mask=torch.ones(num_envs, device=device, dtype=torch.bool))

    # CUDA graphs persist between evaluations. Avoid inference tensors escaping
    # into the subsequent training graph capture in this shared CUDA process.
    @torch.no_grad()
    def gpu_suite(self, model):
        training = model.training
        model.eval()
        summaries, raw = {}, {}
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                for name, spawn, noise, rule in GPU_CASES:
                    records = []
                    generator = torch.Generator(device=self.device).manual_seed(self.seed + 1_000_000)
                    for batch in range(self.batches):
                        seed = self.seed + batch * 1000
                        obs, _ = self.env.reset(seed=seed)
                        mapping = role_slot_map(self.env)
                        focal = mapping[self.env.lanes, self.roles]
                        if spawn != 'standard':
                            rng = np.random.default_rng(seed + 2_000_000)
                            initial = self.env.state['pos'].cpu().tolist()
                            floor = sorted(FLOOR)
                            positions = []
                            for before in initial:
                                if spawn == 'random_floor':
                                    after = rng.choice(floor, size=3, replace=False).tolist()
                                else:
                                    options = [[p for p in floor if abs(p % 7 - start % 7) + abs(p // 7 - start // 7) <= 1]
                                               for start in before]
                                    combinations = [v for v in itertools.product(*options) if len(set(v)) == 3]
                                    after = combinations[int(rng.integers(len(combinations)))]
                                positions.append(after)
                            self.env.state['pos'].copy_(torch.tensor(positions, device=self.device))
                            self.env._write_observations()
                            obs = self.env._obs_out.clone()
                        self.partners.reset()
                        for _ in range(300):
                            if rule:
                                joint = self.partners.actions()
                                joint[self.env.lanes, focal] = model.actor(obs[self.env.lanes, focal]).argmax(-1)
                            else:
                                joint = model.actor(obs.reshape(-1, 376)).argmax(-1).reshape(-1, 3)
                            mask = torch.rand(joint.shape, device=self.device, generator=generator) < noise
                            mask[self.env.lanes, focal] = False
                            replacement = torch.randint(0, 6, joint.shape, device=self.device, generator=generator)
                            obs, _, _, _, _ = self.env.step(torch.where(mask, replacement, joint))
                        state = {key: self.env.state[key].cpu().tolist() for key in (
                            'score', 'normal_completed', 'joint_completed', 'three_contributor_meals',
                            'movement_conflicts', 'resource_conflicts', 'joint_ready_steps',
                            'joint_successes', 'joint_attempt_steps', 'unmatched_attempt_steps')}
                        for lane in range(self.env.num_envs):
                            r = {key: value[lane] for key, value in state.items()}
                            r['team_score'] = r.pop('score')
                            r.update(seed=seed, lane=lane, role=ROLES[lane % 3],
                                meals_completed=r['normal_completed'] + r['joint_completed'],
                                joint_wait_steps=r['joint_ready_steps'] - r['joint_successes'],
                                joint_attempt_success_rate=r['joint_successes'] / max(1, r['joint_attempt_steps']))
                            records.append(r)
                    summaries[name], raw[name] = summarize(records), records
        finally:
            model.train(training)
        return summaries, raw

    @torch.no_grad()
    def native_suite(self, model):
        # This is the real native controller, not the distinct GPU fast controller.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'overcooked_env/src'))
        from overcooked_ai_py.cooperative.shared.kitchen import SharedKitchen
        from overcooked_ai_py.cooperative.shared.controller import ScriptedController
        from overcooked_ai_py.cooperative.shared.layouts import SPAWNS
        training = model.training
        model.eval()
        summaries, raw = {}, {}
        try:
            for noise in (0., .05, .10):
                name = 'native_rule' + (f'_noise{round(noise * 100):02d}' if noise else '')
                n = self.native_episodes
                envs = [SharedKitchen() for _ in range(n)]
                obs, controllers, focal = [], [], []
                for i, env in enumerate(envs):
                    observation, _ = env.reset(seed=self.seed + 10_000 + i)
                    positions = [c.position for c in env.state.chefs]
                    focal.append(positions.index(SPAWNS[i % 3]))
                    controllers.append(ScriptedController(env))
                    obs.append(np.asarray(observation, dtype=np.float32))
                focal = np.asarray(focal)
                observations = np.stack(obs)
                assert observations.shape == (n, 3, 866)
                generator = np.random.default_rng(self.seed + 3_000_000)
                for _ in range(300):
                    compact = np.concatenate((observations[:, :, :1], observations[:, :, 491:]), axis=-1)
                    current = torch.from_numpy(compact[np.arange(n), focal]).to(self.device)
                    actions = np.asarray([controller.actions(env) for controller, env in zip(controllers, envs)])
                    actions[np.arange(n), focal] = model.actor(current).argmax(-1).cpu().numpy()
                    mask = generator.random((n, 3)) < noise
                    mask[np.arange(n), focal] = False
                    actions = np.where(mask, generator.integers(0, 6, size=(n, 3)), actions)
                    for i, env in enumerate(envs):
                        next_obs, _, _, _, _ = env.step(actions[i])
                        observations[i] = next_obs
                records = [dict(env.episode_metrics(), seed=self.seed + 10_000 + i, role=ROLES[i % 3])
                           for i, env in enumerate(envs)]
                summaries[name], raw[name] = summarize(records), records
        finally:
            model.train(training)
        return summaries, raw

"""Exact planner parity against the original NumPy lane/collision batching."""
import unittest

import numpy as np
import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import INTERACT, OFFSETS, STAY
from .kitchen import V3PolicyKitchen
from .partners import Partners


class NumpyBatchPartners(Partners):
    """Pre-optimization batching, retained as a scalar-representation oracle."""

    def actions(self):
        keys = ('pos', 'facing', 'held', 'ids', 'due', 'recipe', 'board',
                'counter', 'mix', 'locked', 'cuts', 'heat')
        state = self.env.state
        packed = torch.cat([state[k] for k in keys], 1).detach().cpu().numpy()
        arrays = np.split(packed, np.cumsum([state[k].shape[1] for k in keys])[:-1], 1)
        batch = dict(zip(keys, arrays))
        self._learner_cpu = self.learner_ids.cpu().numpy()
        for lane in self.rows:
            self.destinations[lane] = self._lane(lane, {k: v[lane] for k, v in batch.items()})
        pos, facing = batch['pos'], batch['facing']
        blocked = np.zeros_like(pos)
        for lane in self.rows:
            a, b = self.chef_ids[lane, 1:]
            blocked[lane, a], blocked[lane, b] = pos[lane, b], pos[lane, a]
        moves = self.paths[self.destinations, blocked, pos].astype(np.int64)
        moves = np.where(moves >= 6, np.where(facing == moves - 6, INTERACT, moves - 6), moves)
        for lane in self.rows:
            if self.learner_mask[lane]:
                moves[lane, self._learner_cpu[lane]] = STAY
            a, b = self.chef_ids[lane, 1:]
            off = (*OFFSETS, 0, 0)
            if pos[lane, a] + off[moves[lane, a]] == pos[lane, b] + off[moves[lane, b]]:
                moves[lane, b] = STAY
        self.idle_actionable = self.eligible_work & (moves == STAY)
        return torch.as_tensor(moves, device=self.env.device)


class PlannerPerformanceTests(unittest.TestCase):
    def compare(self, device, external):
        n = 9
        env = V3PolicyKitchen(PartitionConfig(layout='counter', horizon=400), num_envs=n, device=device)
        env.reset(seed=811)
        roles, styles = torch.arange(n, device=device) // 3, torch.arange(n, device=device) % 3
        mask = torch.full((n,), external, dtype=torch.bool, device=device)
        reference, current = [cls(env, roles, styles, learner_mask=mask)
                              for cls in (NumpyBatchPartners, Partners)]
        rng = np.random.default_rng(72)
        for tick in range(400):
            expected, actual = reference.actions(), current.actions()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key, value in reference.state_dict().items():
                other = current.state_dict()[key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(other, value, rtol=0, atol=0)
                else:
                    self.assertEqual(other, value, key)
            if external:
                actual[env.lanes, current.learner_ids] = torch.as_tensor(
                    rng.integers(6, size=n), device=device)
            env.step(actual)
            if tick == 203:
                restored = Partners(env, roles, styles, learner_mask=mask)
                restored.load_state_dict(current.state_dict())
                current = restored

    def test_cpu_full_script_exact_400_steps(self):
        self.compare('cpu', False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_external_learner_exact_400_steps(self):
        self.compare('cuda', True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()

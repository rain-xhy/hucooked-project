"""Regression: baseline evaluation must not poison the first training graph."""
from pathlib import Path
import unittest

import torch

from src.nest_rl.mlp_shared.l5_evaluation import L5Evaluator
from src.nest_rl.mlp_shared.train_l5 import L5Settings, L5Trainer

CHECKPOINT = Path(__file__).resolve().parents[1] / 'runs/no-hist-l4-from-l3u025-rspawn100-noise5-sparse-seed1/l4/best.pt'


@unittest.skipUnless(CHECKPOINT.is_file() and torch.cuda.is_available(), 'Requires local L4 U165 weights and CUDA')
class L5CudaOrderTests(unittest.TestCase):
    def test_baseline_then_first_training_update(self):
        torch.set_num_threads(4)
        trainer = L5Trainer(L5Settings(num_envs=30, updates=2, epochs=1, minibatch_size=9000), CHECKPOINT)
        evaluator = L5Evaluator(num_envs=6, batches=1, native_episodes=3)
        evaluator.gpu_suite(trainer.frozen_l4)
        evaluator.native_suite(trainer.frozen_l4)
        metrics = trainer.step()
        self.assertEqual(trainer.update, 1)
        self.assertGreater(trainer.optimizer_step_count, 0)
        self.assertLessEqual(metrics['max_episode_penalty'], 2.00001)


if __name__ == '__main__':
    unittest.main()

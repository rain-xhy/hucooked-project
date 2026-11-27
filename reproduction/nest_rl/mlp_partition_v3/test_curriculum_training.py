"""Role emphasis, retained demonstrations, and bounded PPO updates."""
import tempfile
import unittest
from pathlib import Path

import torch

from .pretrain import collect_expert
from .train import Settings, Trainer, atomic_save, checkpoint_settings


class CurriculumTrainingTests(unittest.TestCase):
    def test_role_counts_and_legacy_defaults(self):
        trainer = Trainer(Settings(.001, 7, 2, role_counts=(45, 9, 9)), "cpu")
        self.assertEqual(trainer.roles.bincount().tolist(), [45, 9, 9])
        for role, count in enumerate((45, 9, 9)):
            self.assertEqual(trainer.styles[trainer.roles == role].bincount().tolist(), [count // 3] * 3)
        with self.assertRaises(ValueError):
            Trainer(Settings(.001, 7, 2, role_counts=(44, 10, 8)), "cpu")
        old = checkpoint_settings(dict(lr=.001, seed=1, updates=2))
        self.assertIsNone(old.role_counts)
        self.assertEqual(old.bc_coefficient, 0.)
        self.assertIsNone(old.target_kl)

    def test_128_roles_and_balanced_styles(self):
        for counts in ((92, 18, 18), (18, 92, 18), (18, 18, 92), (43, 43, 42)):
            with self.subTest(counts=counts):
                trainer = Trainer(Settings(.001, 7, 2, num_envs=128, role_counts=counts), "cpu")
                self.assertEqual(trainer.roles.bincount().tolist(), list(counts))
                self.assertEqual(trainer.styles.bincount().tolist(), [43, 43, 42])
                for role in range(3):
                    frequency = trainer.styles[trainer.roles == role].bincount(minlength=3)
                    self.assertLessEqual((frequency.max() - frequency.min()).item(), 1)
        default = Trainer(Settings(.001, 7, 2), "cpu")
        self.assertEqual(default.roles.bincount().tolist(), [21, 21, 21])
        self.assertEqual(default.styles.bincount().tolist(), [21, 21, 21])
        for counts in ((92, 18, 17), (124, 2, 2), (92., 18, 18)):
            with self.assertRaises(ValueError):
                Trainer(Settings(.001, 7, 2, num_envs=128, role_counts=counts), "cpu")

    def test_kl_stops_before_second_update(self):
        cfg = Settings(.1, 17, 2, num_envs=9, horizon=4, rollout_steps=4,
                       epochs=4, minibatch_size=36, target_kl=1e-12)
        trainer = Trainer(cfg, "cpu")
        bank, data, _ = trainer.collect()
        metrics = trainer.optimize(bank, data)
        self.assertEqual(metrics["optimizer_steps"], 1)
        self.assertEqual(metrics["kl_early_stop"], 1)
        self.assertEqual(trainer.optimizer_step_count, 1)

    def test_partial_rollouts_report_only_complete_episode_scores(self):
        cfg = Settings(.001, 17, 2, num_envs=9, horizon=4, rollout_steps=10,
                       epochs=1, minibatch_size=90)
        trainer = Trainer(cfg, "cpu")
        native_step = trainer.env.step
        def fixed_reward(actions):
            obs, reward, *rest = native_step(actions)
            return obs, torch.ones_like(reward), *rest
        trainer.env.step = fixed_reward
        _, _, first = trainer.collect()
        self.assertEqual(trainer.tick, 2)
        self.assertEqual(first["episodes"], 18)
        self.assertEqual(first["score_episodes"], 18)
        self.assertEqual(first["reward"], 90)
        self.assertEqual(first["mean_score"], 4)
        self.assertEqual(trainer.episode_totals[0].item(), 18)
        _, _, second = trainer.collect()
        self.assertEqual(second["score_episodes"], 27)
        self.assertEqual(second["mean_score"], 4)
        self.assertEqual(trainer.tick, 0)
        self.assertEqual(trainer.episode_totals[0].item(), 0)

    def test_old_partial_checkpoint_excludes_unknown_reward_prefix(self):
        cfg = Settings(.001, 17, 2, num_envs=9, horizon=4, rollout_steps=2,
                       epochs=1, minibatch_size=18)
        trainer = Trainer(cfg, "cpu")
        trainer.collect()
        state = trainer.state_dict()
        del state["episode_totals"]
        del state["episode_totals_valid"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            atomic_save(state, path)
            restored = Trainer(cfg, "cpu")
            restored.load(path)
            self.assertFalse(restored.episode_totals_valid)
            _, _, metrics = restored.collect()
            self.assertEqual(metrics["episodes"], 9)
            self.assertEqual(metrics["score_episodes"], 0)
            self.assertNotIn("mean_score", metrics)
            self.assertTrue(restored.episode_totals_valid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_partial_episode_accounting_exact_resume(self):
        cfg = Settings(.001, 41, 3, num_envs=9, horizon=4, rollout_steps=10,
                       epochs=1, minibatch_size=90, shaping=True)
        first = Trainer(cfg, "cuda")
        first.env.state["held"][:, 0] = 2
        first.obs = first.env.observations()
        first.step()
        self.assertEqual(first.tick, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.pt"
            atomic_save(first.state_dict(), path)
            expected = first.step()
            second = Trainer(cfg, "cuda")
            second.load(path)
            self.assertEqual(second.tick, 2)
            actual = second.step()
            for name in expected.keys() - {"steps_per_second"}:
                self.assertAlmostEqual(expected[name], actual[name], places=6, msg=name)
            self.assertEqual(actual["score_episodes"], 27)
            for a, b in zip(first.model.parameters(), second.model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(first.episode_totals, second.episode_totals, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_demonstrations_and_exact_resume(self):
        demo_bank, demo_data, _ = collect_expert(9, 1, 6, 40000000, "cuda")
        cfg = Settings(.001, 41, 3, num_envs=9, horizon=4, rollout_steps=5,
                       epochs=2, minibatch_size=15, role_counts=(3, 3, 3),
                       bc_coefficient=.1, bc_batch_size=18, target_kl=.02, shaping=True)
        first = Trainer(cfg, "cuda")
        first.attach_demonstrations(demo_bank, demo_data)
        metrics = first.step()
        self.assertGreater(metrics["bc_loss"], 0.)
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_save(first.state_dict(), path)
            expected = first.step()
            second = Trainer(cfg, "cuda")
            second.attach_demonstrations(demo_bank, demo_data)
            second.load(path)
            actual = second.step()
            for name in expected.keys() - {"steps_per_second"}:
                self.assertAlmostEqual(expected[name], actual[name], places=6, msg=name)
            for a, b in zip(first.model.parameters(), second.model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(first.optimizer_step_count, second.optimizer_step_count)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

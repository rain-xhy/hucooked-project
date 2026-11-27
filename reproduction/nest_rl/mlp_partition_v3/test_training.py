"""End-to-end PPO replay and checkpoint regressions on the actual kitchen."""
import tempfile
import unittest
from pathlib import Path

import torch

from .history import OnlineHistory, RolloutHistory
from .policy import HISTORY_STEPS, V3Policy
from .train import Settings, Trainer, atomic_save, scheduled_lr, checkpoint_settings


class TrainingTests(unittest.TestCase):
    def test_legacy_checkpoint_keeps_rewards_and_controller(self):
        cfg = Settings(.001, 17, 2, num_envs=9, horizon=4, rollout_steps=4,
                       epochs=1, minibatch_size=36, controller_version="serial-v1",
                       recipe_rewards=(20., 20., 20.))
        trainer = Trainer(cfg, "cpu")
        trainer.collect()
        state = trainer.state_dict()
        del state["config"]["controller_version"]
        del state["config"]["recipe_rewards"]
        self.assertEqual(checkpoint_settings(state["config"]), cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            atomic_save(state, path)
            new = Trainer(Settings(.001, 17, 2, num_envs=9, horizon=4, rollout_steps=4,
                                   epochs=1, minibatch_size=36), "cpu")
            new.load(path)
            self.assertEqual(new.cfg, cfg)
            self.assertEqual(tuple(new.env.config.recipe_rewards), (20., 20., 20.))
            torch.testing.assert_close(trainer.partners.actions(), new.partners.actions())

    def test_optimizer_step_warmup(self):
        self.assertAlmostEqual(scheduled_lr(.001, 0, 1500), .0001)
        self.assertAlmostEqual(scheduled_lr(.001, 29, 1500), .001)
        self.assertAlmostEqual(scheduled_lr(.001, 100, 1500), .001)
        self.assertLess(scheduled_lr(.001, 10, 1500), .001)

    def test_indexed_forward_at_ring_and_period_boundaries(self):
        torch.manual_seed(33)
        model = V3Policy()
        lanes = 2
        live = OnlineHistory(lanes, "cpu")
        observation = torch.randn(49, lanes, 153)
        joint_ids = torch.randint(36, (49, lanes))
        for _ in range(3):
            live.append(observation[0], joint_ids[0])
            live.finish_episode()
        bank = RolloutHistory(live, 49)
        outputs = []
        for t, (current, ids) in enumerate(zip(observation, joint_ids)):
            mask = torch.arange(HISTORY_STEPS)[None, :] >= torch.tensor(live.lengths)[:, None]
            mask = mask.expand(lanes, -1, -1)
            dense = model(current, live.obs.permute(2, 0, 1, 3), live.ids.permute(2, 0, 1),
                          mask.all(-1), mask)
            online = live.forward(model, current)
            for a, b in zip(online, dense):
                torch.testing.assert_close(a, b)
            outputs.append(online)
            bank.record(current, ids)
            live.append(current, ids, model)
            if t in (HISTORY_STEPS + 3, HISTORY_STEPS + 5):
                live.finish_episode()
                bank.finish_episode()
        bank.seal()
        perm = torch.randperm(lanes * len(outputs))
        replay = bank.replay(model, perm)
        for i in range(3):
            torch.testing.assert_close(replay[i], torch.cat([x[i] for x in outputs])[perm])
        expected = torch.ones(len(perm), dtype=torch.bool)
        expected[(HISTORY_STEPS + 6) * lanes:(HISTORY_STEPS + 7) * lanes] = False
        torch.testing.assert_close(replay[3], expected[perm])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_rollout_probabilities_finite_head_gradients_and_resume(self):
        cfg = Settings(3e-4, 41, 3, num_envs=9, horizon=3, rollout_steps=5,
                       epochs=1, minibatch_size=15)
        first = Trainer(cfg, "cuda")
        bank, data, _ = first.collect()
        indices = torch.randperm(len(data["action"]), device="cuda")
        logp, *_ = bank.replay(first.model, indices)
        replayed = logp.gather(1, data["action"][indices, None])[:, 0]
        torch.testing.assert_close(replayed, data["logp"][indices])
        metrics = first.optimize(bank, data)
        self.assertTrue(all(torch.isfinite(torch.tensor(list(metrics.values())))))
        for module in (first.model.actor, first.model.critic, first.model.partner_head,
                       first.model.episode, first.model.period):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and g.isfinite().all() for g in grads))
            self.assertTrue(any(g.abs().max() > 0 for g in grads))
        first.update = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_save(first.state_dict(), path)
            expected = first.step()
            second = Trainer(cfg, "cuda")
            second.load(path)
            actual = second.step()
            for name in expected.keys() - {"steps_per_second"}:
                self.assertAlmostEqual(expected[name], actual[name], places=6, msg=name)
            for a, b in zip(first.model.parameters(), second.model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for key in first.env.state:
                torch.testing.assert_close(first.env.state[key], second.env.state[key])


if __name__ == "__main__":
    unittest.main()



"""Cached rollout potentials agree with uncached state-transition rewards."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import shaping
from .train import Settings, Trainer, atomic_save


class ShapingCacheTests(unittest.TestCase):
    def test_uncached_rewards_returns_updates_and_call_count(self):
        self.verify_uncached("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_uncached_equivalence(self):
        self.verify_uncached("cuda")

    def verify_uncached(self, device):
        torch.set_num_threads(2)
        potential = shaping.potential
        for horizon in (5, 7):
            cfg = Settings(.0003, 19, 4, num_envs=9, horizon=horizon,
                           rollout_steps=5, epochs=1, minibatch_size=45, shaping=True)
            trainer = Trainer(cfg, device)
            for segment in range(3):
                # Nonzero inventory makes stale initial or reset values visible.
                trainer.env.state["held"][:, 0] = 2
                trainer.obs = trainer.env.observations()
                original_step = trainer.env.step
                rewards, bonuses, done_flags = [], [], []

                def oracle_step(actions):
                    before = potential(trainer.env.state)
                    terminal = trainer.tick + 1 == cfg.horizon
                    output = original_step(actions)
                    after = torch.zeros_like(before) if terminal else potential(trainer.env.state)
                    bonus = shaping.COEFFICIENT * (shaping.GAMMA * after - before)
                    bonuses.append(bonus)
                    rewards.append(output[1].float() + bonus.float())
                    done_flags.append(terminal)
                    return output

                with patch.object(trainer.env, "step", side_effect=oracle_step), \
                     patch.object(shaping, "potential", wraps=potential) as calls:
                    bank, data, metrics = trainer.collect()
                # One evaluation per step, plus a nonterminal endpoint; the
                # previous implementation needed 2*steps - terminal_count.
                self.assertEqual(calls.call_count, cfg.rollout_steps + int(not done_flags[-1]))
                self.assertLess(calls.call_count, 2 * cfg.rollout_steps - sum(done_flags))
                total = bonuses[0].new_zeros(())
                for bonus in bonuses:
                    total += bonus.sum()
                self.assertEqual(metrics["shaping_reward"], total.item())

                current = trainer.obs[trainer.env.lanes, trainer.partners.learner_ids]
                next_value = trainer.history.forward(trainer.model, current)[1][:, 0]
                values = data["old_values"].reshape(cfg.rollout_steps, cfg.num_envs)
                advantages = torch.zeros_like(values)
                carry = torch.zeros_like(next_value)
                for t in reversed(range(cfg.rollout_steps)):
                    discount = 0. if done_flags[t] else .99
                    carry = rewards[t] + discount * next_value - values[t] + discount * .95 * carry
                    advantages[t] = carry
                    next_value = values[t]
                returns = (advantages + values).flatten()
                flat = advantages.flatten()
                adv = (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
                torch.testing.assert_close(data["returns"], returns, rtol=0, atol=0)
                torch.testing.assert_close(data["adv"], adv, rtol=0, atol=0)
                if segment == 1:
                    with tempfile.TemporaryDirectory() as directory:
                        checkpoint = Path(directory) / "state.pt"
                        atomic_save(trainer.state_dict(), checkpoint)
                        trainer.optimize(bank, data)
                        reference = Trainer(cfg, device)
                        reference.load(checkpoint)
                        reference.optimize(bank, dict(data, returns=returns, adv=adv))
                        for actual, expected in zip(trainer.model.parameters(), reference.model.parameters()):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

"""Potential shaping cannot pay for cycling items or excess stock."""
import tempfile
import unittest
from pathlib import Path

import torch

from .kitchen import V3PolicyKitchen
from .shaping import GAMMA, potential, transition_reward
from .train import Settings, Trainer, atomic_save


class ShapingTests(unittest.TestCase):
    def state(self):
        env = V3PolicyKitchen(device="cpu")
        env.state["recipe"][:] = 1
        return env.state

    def test_processing_and_stock_cap(self):
        s = self.state()
        s["held"][0, 0] = 2
        self.assertAlmostEqual(potential(s).item(), .25)
        s["held"][0, 0] = 6
        self.assertAlmostEqual(potential(s).item(), .75)
        s["held"][:] = 0
        s["mix"][0, 0] = 2
        self.assertAlmostEqual(potential(s).item(), 1.)
        s["locked"][0, 0] = 1
        s["heat"][0, 0] = 4
        self.assertAlmostEqual(potential(s).item(), 3.)
        s["held"][:] = 6
        s["counter"][:] = 6
        self.assertAlmostEqual(potential(s).item(), 4.5)
        s["held"][:] = 0
        s["counter"][:] = 0
        s["mix"][:] = 0
        s["locked"][:] = -1
        s["heat"][:] = 0
        s["held"][0, 0] = 8
        s["pos"][0, 0] = 14
        self.assertAlmostEqual(potential(s).item(), 3.5)

    def test_transfer_cycle_and_terminal_telescoping(self):
        s = self.state()
        s["held"][0, 0] = 6
        start = potential(s)
        s["held"][0, 0] = 0
        s["counter"][0, 0] = 6
        first = transition_reward(start, s)
        middle = potential(s)
        s["held"][0, 0] = 6
        s["counter"][0, 0] = 0
        second = transition_reward(middle, s)
        self.assertLessEqual((first + GAMMA * second).item(), 0)
        terminal = transition_reward(potential(s), s, True)
        torch.testing.assert_close(first + GAMMA * second + GAMMA ** 2 * terminal, -start)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_update_exact_resume(self):
        cfg = Settings(.001, 41, 3, num_envs=9, horizon=7, rollout_steps=5,
                       epochs=1, minibatch_size=45, shaping=True)
        first = Trainer(cfg, "cuda")
        first.env.state["held"][:, 0] = 2
        first.obs = first.env.observations()
        metrics = first.step()
        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
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


if __name__ == "__main__":
    unittest.main()

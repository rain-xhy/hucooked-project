"""Sequence BC must carry recurrent gradients over complete trajectories."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from hipt.behavior_cloning import _low_logits_for_sequence, warm_start_hierarchical
from hipt.model import HiPTPolicy, VECTOR_DIM


class SequenceBehaviorCloningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_final_step_loss_backpropagates_through_prior_hidden_states(self):
        torch.manual_seed(19)
        model = HiPTPolicy(z_dim=4)
        spatial = torch.randn(3, 2, 26, 7, 7, requires_grad=True)
        vector = torch.randn(3, 2, VECTOR_DIM, requires_grad=True)

        logits = _low_logits_for_sequence(model, spatial, vector)
        logits[-1].square().mean().backward()

        self.assertEqual(tuple(logits.shape), (3, 2, 4, 6))
        self.assertGreater(float(spatial.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(vector.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(model.lstm.weight_ih_l0.grad.abs().sum()), 0.0)

    def test_sequence_bc_rejects_data_without_episode_metadata(self):
        rng = np.random.default_rng(21)
        with tempfile.TemporaryDirectory(prefix="hipt-sequence-bc-") as temporary:
            dataset = Path(temporary) / "missing_boundaries.npz"
            np.savez_compressed(
                dataset,
                spatial=rng.normal(size=(6, 26, 7, 7)).astype(np.float32),
                vector=rng.normal(size=(6, VECTOR_DIM)).astype(np.float32),
                actions=np.zeros(6, dtype=np.int64),
                validation=np.asarray([False] * 3 + [True] * 3),
                env_steps=np.asarray(1),
            )
            with self.assertRaisesRegex(ValueError, "episode_lengths"):
                warm_start_hierarchical(
                    HiPTPolicy(z_dim=4), dataset_path=dataset,
                    device=torch.device("cpu"), seed=0, epochs=1,
                )


if __name__ == "__main__":
    unittest.main()

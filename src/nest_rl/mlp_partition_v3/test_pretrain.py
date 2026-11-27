"""Expert demonstration causality and BC optimization regressions."""
import tempfile
import unittest

import torch

from .policy import V3Policy
from .pretrain import BCSettings, collect_expert, epoch, run_pretraining, validate_seed_streams


class PretrainTests(unittest.TestCase):
    def test_disjoint_lane_seed_streams(self):
        for first, second in ((1, 1), (1, 2), (1, 63), (10, 2)):
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_seed_streams(BCSettings(seed=first, validation_seed=second))
        # Half-open ranges: training 1..63, validation 64..72 are disjoint.
        validate_seed_streams(BCSettings(seed=1, validation_seed=64))
        validate_seed_streams(BCSettings(seed=10, validation_seed=1))
        validate_seed_streams(BCSettings())

    def test_causal_history_and_five_episode_boundary(self):
        bank, data, meta = collect_expert(9, 6, 3, 321, "cpu")
        self.assertEqual(meta["samples"], 162)
        self.assertTrue((bank.tables[0] == -1).all())
        self.assertTrue((bank.tables[15] == -1).all())
        for tick in range(18):
            self.assertTrue((bank.tables[tick] < tick).all())
        self.assertEqual(bank.tables[12].ge(0).sum().item(), 12)
        self.assertEqual(data["labels"].bincount().tolist(), [54, 54, 54])
        # Replaying identical independent simulations gives identical expert labels.
        other, labels, _ = collect_expert(9, 6, 3, 321, "cpu")
        torch.testing.assert_close(data["action"], labels["action"])
        torch.testing.assert_close(bank.obs, other.obs)

    def test_optimization_and_critic_unchanged(self):
        torch.manual_seed(11)
        bank, data, _ = collect_expert(9, 1, 12, 333, "cpu")
        model = V3Policy()
        critic = {key: value.clone() for key, value in model.critic.state_dict().items()}
        initial = epoch(model, bank, data, 108)["nll"]
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(8):
            epoch(model, bank, data, 108, optimizer)
        final = epoch(model, bank, data, 108)
        self.assertLess(final["nll"], initial)
        self.assertEqual(sum(map(sum, final["confusion_matrix"])), 108)
        for key, value in model.critic.state_dict().items():
            torch.testing.assert_close(value, critic[key], rtol=0, atol=0)

    def test_best_artifact_and_callback(self):
        with tempfile.TemporaryDirectory() as output:
            records = []
            cfg = BCSettings(num_envs=9, episodes=1, horizon=4, epochs=1,
                             batch_size=36, validation_episodes=1)
            path = run_pretraining(cfg, output, "cpu", callback=records.append)
            artifact = torch.load(path, weights_only=False)
            V3Policy().load_state_dict(artifact["model"])
            self.assertFalse(artifact["critic_pretrained"])
            self.assertEqual(len(records), 2)
            self.assertEqual(artifact["validation"]["nll"], min(r["validation"]["nll"] for r in records))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()

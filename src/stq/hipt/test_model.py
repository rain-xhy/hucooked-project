"""Focused HiPT model and influence-reward tests; runnable with unittest."""

import unittest

import torch

from hipt.model import (
    HIDDEN_SIZE,
    HEAD_SIZE,
    FlatActorCritic,
    HiPTPolicy,
    SPATIAL_FEATURE_CHANNELS,
    VECTOR_DIM,
    influence_kl,
)


class InfluenceRewardTests(unittest.TestCase):
    def test_identical_subpolicies_have_zero_kl(self):
        low = torch.zeros(3, 4, 6)
        high = torch.randn(3, 4)
        reward = influence_kl(low, high)
        self.assertEqual(tuple(reward.shape), (3, 4))
        self.assertTrue(torch.allclose(reward, torch.zeros_like(reward), atol=1e-7))

    def test_kl_is_directed_current_policy_to_high_weighted_mixture(self):
        # With two z values, the second subpolicy exactly matches the marginal;
        # the first has a nonzero forward KL, while reversing the distributions
        # would not yield the same result.
        low = torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]])
        high = torch.tensor([[-3.0, 3.0]])
        all_rewards = influence_kl(low, high)
        self.assertGreater(float(all_rewards[0, 0]), float(all_rewards[0, 1]))
        selected = influence_kl(low, high, torch.tensor([0]))
        self.assertTrue(torch.allclose(selected, all_rewards[:, 0]))
        self.assertGreater(float(selected[0]), 0.0)
        low_prob = low.softmax(-1)
        marginal = (high.softmax(-1).unsqueeze(-1) * low_prob).sum(-2)
        reverse = (marginal * (marginal.log() - low_prob[:, 0].log())).sum(-1)
        self.assertFalse(torch.allclose(selected, reverse))


class RecurrentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _observation(self, batch=2, requires_grad=False):
        spatial = torch.randn(batch, 26, 7, 7, requires_grad=requires_grad)
        vector = torch.randn(batch, VECTOR_DIM, requires_grad=requires_grad)
        return spatial, vector

    def test_episode_done_clears_each_recurrent_stream(self):
        torch.manual_seed(4)
        model = HiPTPolicy(z_dim=4)
        spatial, vector = self._observation(batch=2)
        state = (torch.randn(1, 2, HIDDEN_SIZE), torch.randn(1, 2, HIDDEN_SIZE))
        done = torch.tensor([1.0, 0.0])
        out, value, low, low_value, next_state = model.step(spatial, vector, state, done)
        zero_state = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))
        reference = model.step(spatial[:1], vector[:1], zero_state, torch.zeros(1))
        self.assertTrue(torch.allclose(out[:1], reference[0], atol=1e-6))
        self.assertTrue(torch.allclose(value[:1], reference[1], atol=1e-6))
        self.assertTrue(torch.allclose(low[:1], reference[2], atol=1e-6))
        self.assertTrue(torch.allclose(low_value[:1], reference[3], atol=1e-6))
        self.assertEqual(tuple(next_state[0].shape), (1, 2, HIDDEN_SIZE))

    def test_unroll_preserves_bptt_and_updates_both_policy_levels(self):
        torch.manual_seed(9)
        model = HiPTPolicy(z_dim=4)
        time, batch = 3, 2
        spatial = torch.randn(time, batch, 26, 7, 7, requires_grad=True)
        vector = torch.randn(time, batch, VECTOR_DIM, requires_grad=True)
        done = torch.zeros(time, batch)
        initial = model.initial_state(batch)
        high_logits, high_value, low_logits, low_values, _, _ = model.unroll(
            spatial, vector, initial, done
        )
        high_loss = high_logits[-1].square().mean() + high_value[-1].square().mean()
        low_loss = low_logits[-1].square().mean() + low_values[-1].square().mean()
        (high_loss + low_loss).backward()
        self.assertIsNotNone(spatial.grad)
        self.assertGreater(float(spatial.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(model.high_actor.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.low_actor.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.lstm.weight_ih_l0.grad.abs().sum()), 0.0)

    def test_sharedkitchen_encoder_keeps_observation_shape_and_uses_compact_layers(self):
        model = HiPTPolicy(z_dim=4)
        convolutions = [layer for layer in model.spatial if isinstance(layer, torch.nn.Conv2d)]
        self.assertEqual([layer.kernel_size for layer in convolutions], [(5, 5), (3, 3), (3, 3)])
        self.assertEqual(
            [(layer.in_channels, layer.out_channels) for layer in convolutions],
            [(26, SPATIAL_FEATURE_CHANNELS),
             (SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS),
             (SPATIAL_FEATURE_CHANNELS, SPATIAL_FEATURE_CHANNELS)],
        )
        self.assertEqual(model.encoder[0].in_features, 16 * 7 * 7 + VECTOR_DIM)
        self.assertEqual(model.lstm.hidden_size, HIDDEN_SIZE)
        self.assertEqual(model.high_feature[0].out_features, HEAD_SIZE)
        partner = FlatActorCritic()
        self.assertEqual(partner.trunk[0].in_features, 16 * 7 * 7 + VECTOR_DIM)
        self.assertEqual(
            [layer.out_features for layer in partner.trunk if isinstance(layer, torch.nn.Linear)],
            [HIDDEN_SIZE, HIDDEN_SIZE],
        )


if __name__ == "__main__":
    unittest.main()

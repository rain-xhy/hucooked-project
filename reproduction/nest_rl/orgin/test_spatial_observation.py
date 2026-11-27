import unittest

import torch

from spatial_observation import OBSERVATION_SIZE, SpatialFeatures, pack_kitchen, unpack_observation
from torch_kitchen import HEIGHT, WIDTH, TorchKitchen


class SpatialObservationTests(unittest.TestCase):
    def test_map_and_player_order_at_reset_and_after_step(self):
        kitchen = TorchKitchen(1, "cpu")
        kitchen.reset(layout=([[22, 40]], [[7, 14, 34, 41, 21, 48, 24]]))
        obs = pack_kitchen(kitchen)
        self.assertEqual(obs.shape, (1, 2, OBSERVATION_SIZE))
        grid, players = unpack_observation(obs)
        self.assertEqual(grid.shape, (1, 2, 33, 8, 7))
        self.assertEqual(grid[0, 0, 2, 4, 0], 1)  # Left cut board.
        self.assertEqual(grid[0, 0, 4, 1, 0], 1)  # Tomato.
        self.assertEqual(grid[0, 0, 16, 3, 3], 1)  # Plate.
        self.assertEqual(grid[0, 0, 23, 3, 1], 1)  # Self position.
        self.assertEqual(grid[0, 1, 23, 5, 5], 1)
        self.assertEqual(players[..., 4:].count_nonzero(), 0)

        actions = torch.tensor([[1, 4]])
        kitchen.step(actions)
        kitchen.chopped[0, 0] = True
        kitchen.plated[0, 0] = True
        kitchen.object_pos[0, 0] = 24
        grid, players = unpack_observation(pack_kitchen(kitchen, actions))
        self.assertEqual(grid[0, 0, 10, 3, 3], 1)  # Chopped tomato.
        self.assertEqual(grid[0, 0, 17, 3, 3], 1)  # Tomato in plate.
        torch.testing.assert_close(
            players[0, 0, :4], torch.tensor([1 / WIDTH, 3 / HEIGHT, 5 / WIDTH, 4 / HEIGHT]),
        )
        self.assertEqual(players[0, 0, 4:10].argmax(), 1)
        self.assertEqual(players[0, 1, 4:10].argmax(), 4)

    def test_cnn_accepts_history_batches_and_preserves_critic_latent(self):
        kitchen = TorchKitchen(1, "cpu")
        kitchen.reset()
        packed = pack_kitchen(kitchen)
        encoder = SpatialFeatures()
        convolutions = [layer for layer in encoder.map_net if isinstance(layer, torch.nn.Conv2d)]
        self.assertEqual([(layer.in_channels, layer.out_channels) for layer in convolutions],
                         [(33, 16), (16, 16)])
        self.assertTrue(all(layer.kernel_size == (3, 3) for layer in convolutions))
        history = packed.unsqueeze(0).expand(3, -1, -1, -1)
        out = encoder(history)
        self.assertEqual(out.shape, (3, 1, 2, 144))
        out.sum().backward()
        self.assertGreater(encoder.map_net[0].weight.grad.abs().sum().item(), 0)
        latents = torch.ones(1, 2, 128)
        critic_features = encoder(torch.cat((packed, latents), dim=-1))
        torch.testing.assert_close(critic_features[..., :144], out[0])
        torch.testing.assert_close(critic_features[..., 144:], latents)


if __name__ == "__main__":
    unittest.main()

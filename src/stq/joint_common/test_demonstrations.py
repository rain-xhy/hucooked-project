from pathlib import Path
import tempfile
import unittest

import numpy as np

from joint_common.demonstrations import load_joint_dataset


class DatasetTests(unittest.TestCase):
    def test_grouping_preserves_team_observation_actions_and_step_budget(self):
        rng = np.random.default_rng(4)
        states = rng.normal(size=(7, 863)).astype(np.float32)
        vectors = np.concatenate((np.repeat(states[:, None], 3, axis=1),
                                  np.broadcast_to(np.eye(3), (7, 3, 3))), axis=-1)
        spatial = rng.normal(size=(7, 3, 26, 7, 7)).astype(np.float32)
        actions = rng.integers(0, 6, (7, 3))
        flags = np.repeat(np.arange(7) >= 4, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data.npz'
            np.savez(path, vector=vectors.reshape(-1, 866),
                     spatial=spatial.reshape(-1, 26, 7, 7), actions=actions.reshape(-1),
                     validation=flags, episode_lengths=np.array([4, 3]), env_steps=np.array(4))
            data = load_joint_dataset(path)
        np.testing.assert_array_equal(data['state'], states)
        np.testing.assert_array_equal(data['spatial'], spatial[:, 0])
        np.testing.assert_array_equal(data['actions'], actions)
        np.testing.assert_array_equal(data['validation'], np.arange(7) >= 4)
        self.assertEqual(data['env_steps'], 4)
        self.assertEqual(data['actions'].shape, (7, 3))


if __name__ == '__main__':
    unittest.main()

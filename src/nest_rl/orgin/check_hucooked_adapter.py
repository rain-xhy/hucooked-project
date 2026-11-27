"""Check the bridge against native hucooked states and actions."""

import unittest

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, SoupState

from hucooked_adapter import HucookedObservationAdapter, hucooked_action, nest_action_index


class HucookedAdapterTest(unittest.TestCase):
    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.adapter = HucookedObservationAdapter(self.mdp)
        self.state = self.mdp.get_standard_start_state()

    def test_action_order_in_both_directions(self):
        native_actions = ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), "interact")
        self.assertEqual(tuple(hucooked_action(i) for i in range(6)), native_actions)
        self.assertEqual([nest_action_index(a) for a in native_actions], list(range(6)))
        with self.assertRaises(ValueError):
            hucooked_action(-1)

    def test_initial_observation_positions_and_feature_order(self):
        obs = self.adapter.observe(self.state, 1)
        self.assertEqual(obs.shape, (73,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(
            obs[:24],
            [3/5, 1/4, -2/5, 1/4, -1/5, -1/4, 0, 0, 0, 2/4,
             -2/5, 2/4, 0, 0, 0, 0, 0, 0, 1/5, 0, 0, 0, 0, 0],
        )
        np.testing.assert_array_equal(obs[24:32], [0, 0, 0, 1] * 2)
        np.testing.assert_array_equal(obs[38:42], [1, 0, 0, 1])
        np.testing.assert_array_equal(np.flatnonzero(obs[55:67]), [0])

    def test_soup_at_pot_and_in_hands(self):
        self.state.players[1].update_pos_and_or((2, 1), (0, -1))
        self.state.add_object(SoupState.get_soup((2, 0), num_onions=3, finished=True))
        obs = self.adapter.observe(self.state, 1)
        np.testing.assert_array_equal(np.flatnonzero(obs[32:38]), [3])
        np.testing.assert_array_equal(np.flatnonzero(obs[55:67]), [1, 5])
        np.testing.assert_array_equal(np.flatnonzero(obs[67:73]), [2])

        self.state.players[1].set_object(ObjectState("dish", (2, 1)))
        next_state, _ = self.mdp.get_state_transition(self.state, ((0, 0), "interact"))
        obs = self.adapter.observe(next_state, 1)
        np.testing.assert_array_equal(np.flatnonzero(obs[42:49]), [2, 3])
        np.testing.assert_array_equal(np.flatnonzero(obs[49:55]), [2])
        np.testing.assert_array_equal(np.flatnonzero(obs[55:67]), [1])
        np.testing.assert_array_equal(obs[10:12], [0, 0])


if __name__ == "__main__":
    unittest.main()

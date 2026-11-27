"""Checks for option timing, terminal clearing, and two-scale bootstrapping."""

import unittest

import numpy as np

from hipt.common import OptionClock, PendingOption, high_level_gae, sample_durations


class OptionClockTests(unittest.TestCase):
    def test_z_is_held_for_exact_inclusive_duration_and_then_resampled(self):
        clock = OptionClock((1, 3))
        mask = np.array([[True, False, False]])
        z = np.array([[2, 0, 0]])
        duration = np.array([[3, 0, 0]])
        clock.start(mask, z, duration)
        self.assertFalse(clock.due[0, 0])
        self.assertFalse(clock.advance(np.array([False]))[0, 0])
        self.assertEqual(clock.z[0, 0], 2)
        self.assertFalse(clock.advance(np.array([False]))[0, 0])
        self.assertTrue(clock.advance(np.array([False]))[0, 0])
        self.assertEqual(clock.z[0, 0], 2)
        clock.start(mask, np.array([[1, 0, 0]]), np.array([[2, 0, 0]]))
        self.assertEqual(clock.z[0, 0], 1)

    def test_terminal_clears_z_timer_for_every_role(self):
        clock = OptionClock((2, 3))
        mask = np.ones((2, 3), dtype=bool)
        clock.start(mask, np.full((2, 3), 3), np.full((2, 3), 25))
        clock.advance(np.array([True, False]))
        self.assertTrue(np.all(clock.remaining[0] == 0))
        self.assertTrue(np.all(clock.z[0] == 0))
        self.assertTrue(np.all(clock.remaining[1] == 24))
        self.assertTrue(np.all(clock.z[1] == 3))

    def test_duration_sampling_includes_both_bounds(self):
        draws = sample_durations(np.random.default_rng(7), (10000,), 20, 40)
        self.assertEqual(int(draws.min()), 20)
        self.assertEqual(int(draws.max()), 40)


class HighAdvantageTests(unittest.TestCase):
    def _record(self, reward, value=0.0, terminal=False, bootstrap=0.0):
        return {"reward": reward, "old_value": value, "terminal": terminal,
                "bootstrap_value": bootstrap}

    def test_terminal_macro_reward_flows_to_earlier_option(self):
        records = [self._record(0.0), self._record(10.0, terminal=True)]
        high_level_gae([records], gamma=1.0, gae_lambda=1.0)
        self.assertEqual([record["advantage"] for record in records], [10.0, 10.0])

    def test_terminal_options_separate_adjacent_episodes(self):
        first, second = self._record(2.0, terminal=True), self._record(7.0, terminal=True)
        high_level_gae([[first, second]], gamma=1.0, gae_lambda=1.0)
        self.assertEqual(first["advantage"], 2.0)
        self.assertEqual(second["advantage"], 7.0)

    def test_pending_macro_averages_only_observed_steps(self):
        history = []
        pending = PendingOption(history, 0, 1, -0.5, 2.0)
        pending.add_step(4.0)
        pending.add_step(8.0)
        record = pending.finish(bootstrap_value=3.0, terminal=False)
        self.assertEqual(record["duration"], 2)
        self.assertEqual(record["reward"], 6.0)
        self.assertEqual(record["bootstrap_value"], 3.0)
        self.assertFalse(record["terminal"])


if __name__ == "__main__":
    unittest.main()

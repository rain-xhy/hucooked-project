import unittest

import numpy as np

from baseline_common.envs import KitchenBatch as OriginalBatch
from joint_common.envs import (
    JointKitchenSlot, KitchenBatch, REWARD_PROFILES, TaskMilestones, navigation_potential,
    potential,
)
from overcooked_ai_py.cooperative.shared.controller import ScriptedController


class RewardTests(unittest.TestCase):
    def test_legacy_profile_is_identical_to_existing_training(self):
        rng = np.random.default_rng(8)
        with OriginalBatch(2, seed=4) as original, KitchenBatch(2, seed=4) as joint:
            a, b = original.reset(), joint.reset()
            for _ in range(12):
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                actions = rng.integers(0, 6, (2, 3))
                a, x, done_a, info_a = original.step(actions)
                b, y, done_b, info_b = joint.step(actions)
                np.testing.assert_array_equal(x, y)
                np.testing.assert_array_equal(done_a, done_b)
                self.assertEqual(info_a, info_b)

    def test_dense_profile_does_not_change_native_task_or_metrics(self):
        gamma = .9985
        slot = JointKitchenSlot(4, 0, 'standard', True, gamma, 'ppo_task')
        slot.reset()
        controller = ScriptedController(slot.core)
        weights = REWARD_PROFILES['ppo_task']
        initial = weights['native_scale'] * (
            weights['inventory_weight'] * potential(slot.core)
            + weights['navigation_weight'] * navigation_potential(slot.core))
        discounted_potential = raw_total = train_total = 0.
        for step in range(300):
            _, reward, done, info = slot.step(controller.actions(slot.core))
            components = info['task_reward_components']
            self.assertAlmostEqual(reward, sum(components.values()), places=7)
            raw_total += info['team_reward']
            train_total += reward
            discounted_potential += gamma ** step * (
                components['inventory'] + components['navigation'])
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(info['metrics']['team_score'], raw_total)
        self.assertGreater(info['metrics']['joint_completed'], 0)
        self.assertAlmostEqual(info['episode_summary']['training_reward'], train_total)
        self.assertAlmostEqual(info['episode_summary']['training_native_reward'], raw_total * .05)
        self.assertAlmostEqual(discounted_potential, -initial, places=5)

    def test_preparation_rewards_cannot_be_farmed_by_repeating_or_discarding(self):
        milestones = TaskMilestones()
        self.assertEqual(milestones.award('cut', (1,), .75, quota=2), .75)
        self.assertEqual(milestones.award('cut', (1,), .75, quota=2), 0.)
        self.assertEqual(milestones.award('cut', (2,), .75, quota=2), .75)
        self.assertEqual(milestones.award('cut', (3,), .75, quota=2), 0.)
        self.assertEqual(milestones.award('cut', (3,), .75, quota=3), .75)

    def test_dense_parallel_workers_match_serial_transitions(self):
        rng = np.random.default_rng(19)
        with KitchenBatch(2, seed=5, reward_profile='ppo_task') as serial, \
                KitchenBatch(2, seed=5, workers=2, reward_profile='ppo_task') as parallel:
            a, b = serial.reset(), parallel.reset()
            for _ in range(8):
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                actions = rng.integers(0, 6, (2, 3))
                a, x, da, ia = serial.step(actions)
                b, y, db, ib = parallel.step(actions)
                np.testing.assert_array_equal(x, y)
                np.testing.assert_array_equal(da, db)
                self.assertEqual(ia, ib)

    def test_dense_profile_evaluation_uses_unscaled_native_reward(self):
        slot = JointKitchenSlot(4, 0, 'standard', False, .9985, 'ppo_task')
        slot.reset()
        controller = ScriptedController(slot.core)
        for _ in range(300):
            _, reward, done, info = slot.step(controller.actions(slot.core))
            self.assertEqual(reward, info['team_reward'])
            self.assertEqual(info['training_shaping'], 0.)
            if done:
                break
        self.assertEqual(info['episode_summary']['training_reward'], info['metrics']['team_score'])


if __name__ == '__main__':
    unittest.main()

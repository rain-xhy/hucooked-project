import copy
import importlib
import unittest

import torch


class L5Tests(unittest.TestCase):
    def component(self, name):
        module = 'src.nest_rl.mlp_shared.' + name
        self.assertIsNotNone(importlib.util.find_spec(module), 'Missing L5 implementation: ' + module)
        return importlib.import_module(module)

    def metrics(self):
        return {'standard': dict(team_score=420., meals_completed=21., normal_completed=11.,
            joint_completed=10., three_contributor_meals=21., movement_conflicts=4.,
            resource_conflicts=0., score_min=400., score_p10=400., zero_episodes=0,
            joint_attempt_success_rate=.26, pooled_sync_success_rate=.26, joint_wait_steps=43.,
            per_role_score=dict(server=420., main=420., side=420.))}

    def test_penalties_are_separate_capped_and_reset_per_episode(self):
        module = self.component('train_l5')
        costs = module.PenaltyAccumulator(2, 'cpu', unmatched=.01, waiting=.002, cap=2.)
        u, w = costs.step(torch.tensor([1., 0.]), torch.tensor([1., 1.]))
        torch.testing.assert_close(u, torch.tensor([.01, 0.]))
        torch.testing.assert_close(w, torch.tensor([.002, .002]))
        costs.step(torch.tensor([1000., 1000.]), torch.tensor([1000., 1000.]))
        torch.testing.assert_close(costs.used, torch.tensor([2., 2.]))
        u, w = costs.step(torch.ones(2), torch.ones(2))
        torch.testing.assert_close(u + w, torch.zeros(2))
        costs.reset()
        torch.testing.assert_close(costs.used, torch.zeros(2))
        u, w = costs.step(torch.zeros(2), torch.zeros(2))
        torch.testing.assert_close(u + w, torch.zeros(2))

    def test_score_regression_cannot_be_offset_by_sync_gain(self):
        module = self.component('l5_evaluation')
        base = self.metrics()
        trial = copy.deepcopy(base)
        trial['standard'].update(team_score=400., pooled_sync_success_rate=.95)
        result = module.assess_retention(trial, base)
        self.assertFalse(result['passed'])
        self.assertTrue(any('team_score' in x for x in result['violations']))

    def test_role_conflict_wait_and_missing_cases_are_guarded(self):
        module = self.component('l5_evaluation')
        base = self.metrics()
        for field, value in [('movement_conflicts', 5.), ('joint_wait_steps', 44.),
                             ('pooled_sync_success_rate', .25), ('score_min', 380.)]:
            trial = copy.deepcopy(base)
            trial['standard'][field] = value
            self.assertFalse(module.assess_retention(trial, base)['passed'], field)
        trial = copy.deepcopy(base)
        trial['standard']['per_role_score']['main'] = 400.
        self.assertFalse(module.assess_retention(trial, base)['passed'])
        self.assertFalse(module.assess_retention({}, base)['passed'])

    def test_improvement_requires_real_change_after_retention(self):
        module = self.component('l5_evaluation')
        base = self.metrics()
        self.assertTrue(module.assess_retention(base, base)['passed'])
        self.assertEqual(module.selection_objective(base), module.selection_objective(copy.deepcopy(base)))
        trial = copy.deepcopy(base)
        trial['standard'].update(pooled_sync_success_rate=.30, joint_attempt_success_rate=.30, joint_wait_steps=42.)
        self.assertTrue(module.assess_retention(trial, base)['passed'])
        self.assertGreater(module.selection_objective(trial), module.selection_objective(base))

    def test_peer_cohorts_keep_three_roles_balanced(self):
        module = self.component('train_l5')
        for n in (30, 1008):
            fast, frozen, current = module.peer_counts(n)
            self.assertEqual(fast + frozen + current, n)
            self.assertTrue(all(x >= 3 and x % 3 == 0 for x in (fast, frozen, current)))


if __name__ == '__main__':
    unittest.main()

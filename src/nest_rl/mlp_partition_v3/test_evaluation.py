import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from .evaluation import _reset_paired, _summary, evaluate
from .kitchen import V3PolicyKitchen
from .policy import V3Policy


class EvaluationTest(unittest.TestCase):
    def test_macro_rates_use_resolved_orders_and_equal_recipe_weights(self):
        orders = np.array([[[1, 3, 2, 11], [4, 0, 1, 28], [2, 8, 0, 12]]])
        counts = np.array([[9, 2, 3, 4, 2, 5, 9, 1]])
        result = _summary(orders, counts, np.zeros((1, 3, 3), dtype=int))
        self.assertAlmostEqual(result["cross_delivery_rate"], (.25 + .2) / 2)
        self.assertAlmostEqual(result["macro_delivery_rate"], (.25 + 1 + .2) / 3)
        self.assertEqual(result["cross_deliveries"], 3)
        self.assertEqual(result["mean_counter_wait"], 4.5)
        self.assertEqual(result["per_recipe"]["tomato_egg"]["unresolved"], 2)
        self.assertEqual(result["per_recipe"]["tomato_egg"]["mean_delivery_steps"], 11)
        self.assertIsNone(result["partner_accuracy"])

    def test_each_pair_shares_spawns_and_draws(self):
        env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=8), num_envs=18, device="cpu")
        _reset_paired(env, 555)
        for v in (*env.state.values(), env.draws):
            for group in v.split(9):
                self.assertTrue(torch.equal(group, group[:1].expand_as(group)))
        self.assertFalse(torch.equal(env.draws[0], env.draws[9]))

    def test_five_episode_modes_are_json_safe(self):
        model = V3Policy()
        config = PartitionConfig(layout="counter", horizon=3)
        with patch(__package__ + ".evaluation.PartitionConfig", return_value=config):
            for mode in ("policy", "empty", "script", "wait"):
                result = evaluate(model, seed=123, mode=mode, device="cpu")
                json.dumps(result, allow_nan=False)
                self.assertTrue(model.training)
                self.assertEqual(result["episodes"], 45)
                self.assertGreaterEqual(result["idle_actionable_ingredient_steps"], 0)
                self.assertEqual(sum(r["unresolved"] for r in result["per_recipe"].values()), 135)
                self.assertEqual([np.sum(r["partner_confusion"]) for r in result["per_episode"]], [9] * 5)
                self.assertEqual(result["per_role_style"]["left"]["fifo"]["episodes"], 5)

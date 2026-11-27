import unittest
import torch
from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import cell, EAST, WEST, INTERACT, STAY
from .kitchen import V3PolicyKitchen
from .process_metrics import ProcessMetrics, EXTRA_NAMES


class ProcessMetricsTest(unittest.TestCase):
    def test_only_opposite_side_pickup_counts_as_cross_wall(self):
        for item, kind in ((1, "raw"), (5, "cut"), (7, "meal")):
            for receiver, expected in ((0, 0), (1, 1)):
                env = V3PolicyKitchen(PartitionConfig(layout="counter"), num_envs=1, device="cpu")
                s = env.state
                s["pos"][0] = torch.tensor([cell(5, 3), cell(7, 3), cell(11, 5)])
                s["facing"][0] = torch.tensor([EAST, WEST, WEST])
                s["held"][0, 0] = item
                meter = ProcessMetrics(env)
                a = torch.tensor([[INTERACT, STAY, STAY]])
                meter.before(a)
                env.step(a)
                meter.after()
                a[0] = STAY
                a[0, receiver] = INTERACT
                meter.before(a)
                env.step(a)
                values = dict(zip(EXTRA_NAMES, meter.after()[0].tolist()))
                self.assertEqual(values["cross_wall_pickups"], expected)
                self.assertEqual(values[f"cross_wall_{kind}_pickups"], expected)
                self.assertEqual(values[f"counter_{kind}_pickups"], 1)
                self.assertEqual(values[f"counter_wait_{kind}_steps"], 1)

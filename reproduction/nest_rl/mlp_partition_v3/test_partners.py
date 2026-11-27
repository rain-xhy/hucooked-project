import unittest

import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import (
    CUT_TOMATO, EGG, FIRST_MEAL, INTERACT, STAY,
)
from .kitchen import V3PolicyKitchen
from .legacy_partners import LegacyPartners as Partners


class PartnersTest(unittest.TestCase):
    def make(self, n):
        env = V3PolicyKitchen(PartitionConfig(layout="counter"), num_envs=n, device="cpu")
        env.reset(seed=37)
        labels = torch.arange(n) % 3
        return env, Partners(env, labels, labels)

    def test_style_selection_and_expiry(self):
        env, partners = self.make(3)
        s = env.state
        s["recipe"][:] = torch.tensor([2, 0, 1])
        s["due"][:] = torch.tensor([70, 20, 20])
        partners.actions()
        self.assertEqual(partners.ticket_ids.tolist(), [0, 1, 2])
        s["due"][:] = torch.tensor([10, 50, 15])
        partners.actions()
        self.assertEqual(partners.ticket_ids.tolist(), [0, 1, 2])
        s["ids"] += 3
        s["recipe"][:] = torch.tensor([0, 0, 2])
        s["due"][:] = 50
        partners.actions()
        self.assertEqual(partners.ticket_ids.tolist(), [3, 3, 3])

    def test_each_style_delivers_every_recipe_with_missed_interactions(self):
        env, partners = self.make(9)
        recipes = torch.arange(9) // 3
        env.state["recipe"][:] = recipes[:, None]
        env.state["due"][:] = env.deadlines[recipes, None]
        delivered = torch.zeros(9, dtype=torch.bool)
        skipped = torch.zeros_like(delivered)
        for step in range(100):
            actions = partners.actions()
            self.assertEqual(actions.shape, (9, 3))
            if step % 7 == 1:
                skipped |= (actions == INTERACT).any(1)
                actions[actions == INTERACT] = STAY
            _, _, _, _, info = env.step(actions)
            delivered |= (info["delivered"] & (info["order_ids"] < 3)).any(1)
            if delivered.all() and skipped.all():
                break
        self.assertTrue(delivered.all(), delivered)
        self.assertTrue(skipped.all(), skipped)

    def test_premature_pickup_and_checkpoint(self):
        env, partners = self.make(3)
        s = env.state
        s["recipe"][:] = 0
        # The lower right chef picked up the tomato before the upper chef
        # reached the counter. A second lane already picked up the meal.
        lower = partners.chef_ids[:, 2]
        s["held"][0, lower[0]] = CUT_TOMATO
        s["held"][1, lower[1]] = FIRST_MEAL
        s["mix"][0, 1] = 4  # Egg was already deposited.
        s["held"][2, partners.chef_ids[2, 1]] = EGG
        saved = partners.state_dict()
        actions = partners.actions()
        partners.reset()
        partners.load_state_dict(saved)
        torch.testing.assert_close(partners.actions(), actions)
        delivered = torch.zeros(3, dtype=torch.bool)
        for _ in range(65):
            _, _, _, _, info = env.step(partners.actions())
            delivered |= (info["delivered"] & (info["recipe"] == 0)).any(1)
            if delivered.all():
                break
        self.assertTrue(delivered.all(), delivered)

    def test_reset_rebinds_roles(self):
        env, partners = self.make(6)
        env.reset(seed=105)
        partners.reset()
        pos = env.state["pos"]
        for lane, role in enumerate(partners.roles):
            ids = sorted(range(3), key=lambda i: (-1 if pos[lane, i] % 13 < 6 else pos[lane, i]))
            self.assertEqual(partners.learner_ids[lane].item(), ids[role])


if __name__ == "__main__":
    unittest.main()

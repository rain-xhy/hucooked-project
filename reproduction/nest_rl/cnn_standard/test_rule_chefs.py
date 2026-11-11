from pathlib import Path
import unittest

from overcooked_ai_py.agents.agent import AgentPair, StayAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.static import LAYOUTS_DIR

from evaluate_rule_chefs import run_episode
from rule_chefs import RuleChef


def play(layout, *agents):
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    env._mp = MotionPlanner(mdp)  # Keep the test planner in memory.
    pair = AgentPair(*agents)
    pair.set_mdp(mdp)
    score = 0
    while not env.is_done():
        actions, infos = zip(*pair.joint_action(env.state))
        _, reward, _, _ = env.step(actions, infos)
        score += reward
    return score, env.game_stats


class RuleChefTests(unittest.TestCase):
    def test_full_chef_can_cook_and_serve(self):
        score, stats = play("cramped_room", RuleChef("full"), StayAgent())
        self.assertGreater(score, 0)
        self.assertGreaterEqual(len(stats["potting_onion"][0]), 3)
        self.assertTrue(stats["soup_delivery"][0])

    def test_prep_and_serve_split_the_work_at_a_shared_pot(self):
        for layout in ("cramped_room", "asymmetric_advantages", "coordination_ring",
                       "counter_circuit_o_1order"):
            with self.subTest(layout=layout):
                score, stats = play(layout, RuleChef("prep"), RuleChef("serve"))
                self.assertGreater(score, 0)
                self.assertGreaterEqual(len(stats["potting_onion"][0]), 3)
                self.assertFalse(stats["potting_onion"][1])
                self.assertTrue(stats["soup_delivery"][1])

    def test_divided_kitchen_passes_ingredients_and_bowls(self):
        score, stats = play("forced_coordination", RuleChef("serve"), RuleChef("prep"))
        self.assertGreater(score, 0)
        self.assertTrue(stats["onion_drop"][1])
        self.assertTrue(stats["dish_drop"][1])
        self.assertGreaterEqual(len(stats["potting_onion"][0]), 3)
        self.assertTrue(stats["soup_delivery"][0])

    def test_prep_and_serve_can_make_the_mixed_bonus_order(self):
        score, stats = play("cramped_room_tomato", RuleChef("prep"), RuleChef("serve"))
        self.assertGreater(score, 0)
        self.assertTrue(stats["potting_tomato"][0])
        self.assertTrue(stats["potting_onion"][0])
        self.assertTrue(stats["soup_delivery"][1])

    def test_default_team_completes_every_native_layout(self):
        layouts = sorted(Path(LAYOUTS_DIR).glob("*.layout"))
        self.assertTrue(layouts)
        for path in layouts:
            with self.subTest(layout=path.stem):
                result = run_episode(path.stem, horizon=400)
                self.assertIsNotNone(result.first_delivery)
                self.assertGreater(result.score, 0)


if __name__ == "__main__":
    unittest.main()

"""Shared setup for the pinned author checkout."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "adaptive-RL"
CONFIG = "environment/overcooked/config/fc_small_test.yaml"
RECIPES = {
    "lettuce-onion": ["Lettuce", "Onion"],
    "tomato-carrot": ["Tomato", "Carrot"],
    "potato-broccoli": ["Potato", "Broccoli"],
}


def activate_upstream():
    overcooked = UPSTREAM / "environment" / "overcooked"
    os.chdir(UPSTREAM)
    sys.path[:0] = [str(UPSTREAM), str(overcooked), str(overcooked / "gym_cooking" / "rebar")]
    os.environ.update(WANDB_MODE="disabled", MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def rule_partner(recipe, mode="full"):
    import yaml
    from environment.overcooked.policy import RuleBasedPolicy

    class RecipePartner(RuleBasedPolicy):
        def get_navigation_action(self, world, agent_loc, target_loc):
            try:
                return super().get_navigation_action(world, agent_loc, target_loc)
            except NotImplementedError:
                # A chef with no legal step toward the target waits for its partner.
                return 0

    layout = yaml.safe_load((UPSTREAM / CONFIG).read_text(encoding="utf-8"))["mode"]
    return RecipePartner(mode, 0, 0, 0, None, layout, ingredient_support_set=RECIPES[recipe])

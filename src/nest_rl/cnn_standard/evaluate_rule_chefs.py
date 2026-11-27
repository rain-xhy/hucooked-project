"""Run the rule chefs on one or all of the native Overcooked layouts."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.static import LAYOUTS_DIR

from rule_chefs import ROLES, make_rule_team


@dataclass(frozen=True)
class EpisodeResult:
    roles: tuple[str, ...]
    score: float
    first_delivery: int | None
    deliveries_by_player: tuple[int, ...]


def run_episode(layout, roles=None, horizon=400):
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env._mp = MotionPlanner(mdp)  # Avoid writing planner caches into the original project.
    for key in ("cumulative_sparse_rewards_by_agent", "cumulative_shaped_rewards_by_agent"):
        # Native layouts can return fractional rewards or an infinite tutorial bonus.
        env.game_stats[key] = env.game_stats[key].astype(float)
    team = make_rule_team(mdp, roles)
    score = 0.0
    first_delivery = None
    while not env.is_done():
        actions, infos = zip(*team.joint_action(env.state))
        _, reward, _, _ = env.step(actions, infos)
        score += reward
        if reward > 0 and first_delivery is None:
            first_delivery = env.state.timestep

    deliveries = tuple(len(steps) for steps in env.game_stats["soup_delivery"])
    return EpisodeResult(tuple(chef.role for chef in team.agents), score, first_delivery, deliveries)


def main():
    layouts = sorted(path.stem for path in Path(LAYOUTS_DIR).glob("*.layout"))
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--layout", choices=layouts)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--roles", nargs="+", choices=ROLES, help="One role per player in layout order")
    parser.add_argument("--horizon", type=int, default=400)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.all and args.roles:
        parser.error("--roles requires a single --layout")

    selected = layouts if args.all else [args.layout]
    completed = 0
    for layout in selected:
        try:
            result = run_episode(layout, args.roles, args.horizon)
        except ValueError as error:
            parser.error(str(error))
        completed += result.first_delivery is not None
        print(f"{layout:39} roles={','.join(result.roles):24} "
              f"score={result.score:7g} first={result.first_delivery!s:4} "
              f"deliveries={result.deliveries_by_player}", flush=True)
    print(f"Completed {completed}/{len(selected)} layouts within {args.horizon} steps.")
    return int(completed != len(selected))


if __name__ == "__main__":
    raise SystemExit(main())

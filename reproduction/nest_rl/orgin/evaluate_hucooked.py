"""Run one native hucooked soup episode with a locally trained NestRL L1."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from overcooked_setup import ROOT, activate_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted local NestRL L1 checkpoint")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

    sys.path.insert(0, str(ROOT.parents[2] / "hucooked" / "src"))
    activate_upstream()

    import numpy as np
    import torch
    from learning.storage_ import PeriodicHistoryStorage
    from overcooked_ai_py.agents.agent import GreedyHumanModel
    from overcooked_ai_py.mdp.actions import Action
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS

    from hucooked_adapter import HucookedObservationAdapter, hucooked_action, nest_action_index

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()

    layout = "cramped_room"
    horizon = 400
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    planner = MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS)
    env._mp = planner.motion_planner  # Share the in-memory planner with the rule chef.
    partner = GreedyHumanModel(planner)
    partner.set_agent_index(0)
    adapter = HucookedObservationAdapter(mdp)

    # Match L1 training: self observations paired with the partner's actions,
    # including history from the current episode.
    history = PeriodicHistoryStorage(
        num_processes=1, num_policies=1, history_storage_size=5, clear_period=5,
        refresh_interval=1, sample_size=None, has_rew_done=False, max_samples_per_period=None,
        step_mode=False, use_episodes=True, has_meta_time_step=False, include_current_episode=True,
        obs_shape=(73,), act_shape=(), max_episode_length=horizon, merge_encoder_computation=True,
        last_episode_only=False, pop_oldest_episode=False,
    )
    history.to("cpu")
    indices = torch.zeros(1, dtype=torch.long)
    masks = torch.zeros(1, 1)
    action_counts = Counter()
    l1_interactions = Counter()
    l1_moves = 0
    with torch.no_grad():
        done = False
        while not done:
            state = env.state
            obs = torch.from_numpy(adapter.observe(state, 1))
            context = (history, (indices,) + history.get_all_current_indices())
            _, action, _, _ = policy.act(obs.unsqueeze(0), None, masks, indices, context)
            index = action.item()
            native = hucooked_action(index)
            action_counts[index] += 1
            chef = state.players[1]
            if native == Action.INTERACT:
                front = Action.move_in_direction(chef.position, chef.orientation)
                l1_interactions[mdp.get_terrain_type_at_pos(front)] += 1

            partner_action, _ = partner.action(state)
            next_state, _, done, info = env.step((partner_action, native))
            l1_moves += next_state.players[1].position != chef.position
            history.add(0, obs, torch.tensor(nest_action_index(partner_action)))
            masks.fill_(1)

    episode = info["episode"]
    stats = episode["ep_game_stats"]
    events = ("onion_pickup", "potting_onion", "dish_pickup", "soup_pickup", "soup_delivery")
    print(json.dumps({
        "python": sys.executable,
        "checkpoint": str(checkpoint),
        "layout": layout,
        "seed": args.seed,
        "steps": episode["ep_length"],
        "sparse_reward": float(episode["ep_sparse_r"]),
        "shaped_reward_by_player": episode["ep_shaped_r_by_agent"].tolist(),
        "events_by_player": {
            name: [len(timesteps) for timesteps in stats[name]] for name in events
        },
        "player_order": ["GreedyHumanModel", "L1"],
        "l1_actions_in_nest_order": [action_counts[i] for i in range(6)],
        "l1_position_changes": l1_moves,
        "l1_interactions_by_front_tile": l1_interactions,
    }, indent=2))


if __name__ == "__main__":
    main()

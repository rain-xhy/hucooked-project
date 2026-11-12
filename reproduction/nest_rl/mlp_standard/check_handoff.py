"""Replay a complete three-chef handoff episode against the native MDP."""

import argparse

from standard_setup import activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    activate()

    import torch
    from overcooked_ai_py.mdp.actions import Action
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from handoff_kitchen import HORIZON, OBSERVATION_SIZE, HandoffKitchen, HandoffRuleTeam
    from hucooked_adapter import hucooked_action, nest_action_index
    from rule_chefs import RuleChef

    torch.set_num_threads(1)
    mdp = OvercookedGridworld.from_layout_name("three_player_handoff")
    state = mdp.get_standard_start_state()
    kitchen = HandoffKitchen(2, args.device)
    team = HandoffRuleTeam(kitchen)
    route_planner = RuleChef()
    route_planner.set_mdp(mdp)
    counter, pot, clear_of_pot = (5, 3), (8, 0), (6, 3)
    assert mdp.terrain_mtx[3][5] == "X" and not mdp.get_sync_handoff_locations()

    deliveries = 0
    obs = kitchen.reset()
    for step in range(HORIZON):
        _, cook, _ = state.players
        occupied = {state.players[0].position, state.players[2].position}
        positions, targets = route_planner._routes(cook.position, cook.orientation, occupied)
        soup = state.objects.get(pot)
        if soup is not None and not soup.is_idle:
            cook_action = positions[clear_of_pot][1]
        elif cook.has_object() or (soup is not None and len(soup.ingredients) == 3):
            cook_action = targets[pot][1]
        else:
            cook_action = targets[counter][1] if counter in state.objects else Action.STAY

        peers = team.act()
        assert peers.device.type == args.device and peers.shape == (2, 2)
        cook_actions = torch.tensor([nest_action_index(cook_action), 0], device=args.device)
        joint = torch.stack((peers[:, 0], cook_actions, peers[:, 1]), dim=1)
        state, info = mdp.get_state_transition(state, tuple(map(hucooked_action, joint[0].tolist())))
        obs, reward, done, timeout, success = kitchen.step(joint)
        actual, expected = kitchen.env.to_native_state(0).to_dict(), state.to_dict()
        for snapshot in (actual, expected):
            snapshot["objects"].sort(key=lambda obj: obj["position"])
        assert actual == expected, (step, actual, expected)
        sparse = sum(info["sparse_reward_by_agent"])
        assert reward[0].item() == sparse + sum(info["shaped_reward_by_agent"])
        assert obs.shape == (2, 3, OBSERVATION_SIZE) and obs.dtype == torch.float32 and obs.isfinite().all()
        assert done.tolist() == timeout.tolist() == [step + 1 == HORIZON] * 2
        deliveries += sparse > 0

    assert deliveries > 0 and success.tolist() == [0, -1]
    kitchen.reset(mask=torch.tensor([True, False], device=args.device))
    assert kitchen.ticks.tolist() == [0, HORIZON]
    print(f"Matched {HORIZON} native transitions and {deliveries} positive deliveries on {args.device}.")


if __name__ == "__main__":
    main()

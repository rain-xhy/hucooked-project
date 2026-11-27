"""Check the Torch soup kitchen and L0 against native hucooked transitions."""

import argparse
import random

from standard_setup import activate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    activate()

    import torch
    from hucooked_adapter import hucooked_action, nest_action_index
    from rule_chefs import make_rule_team
    from soup_kitchen import HORIZON, OBSERVATION_SIZE, SoupKitchen, SoupPrepPartner, native_mdp

    torch.set_num_threads(1)
    mdp = native_mdp()
    kitchen = SoupKitchen(1, args.device)
    partner = SoupPrepPartner(kitchen, player=0)
    randomizer = random.Random(1)
    total = deliveries = 0

    def item(obj):
        if obj is None:
            return 0
        if obj.name == "soup":
            return 2 + len(obj.ingredients)
        return {"onion": 1, "dish": 2}[obj.name]

    def flat(position):
        x, y = position
        return x + y * mdp.width

    for episode in range(5):
        state = mdp.get_standard_start_state()
        kitchen.reset()
        team = make_rule_team(mdp, ("prep", "serve"))
        for step in range(HORIZON):
            decisions = team.joint_action(state)
            reference = [nest_action_index(action) for action, _ in decisions]
            actual = partner.act().item()
            assert actual == reference[0], (episode, step, actual, decisions[0], state.to_dict())
            joint = reference if episode == 0 else [randomizer.randrange(6) for _ in range(2)]
            state, info = mdp.get_state_transition(state, tuple(map(hucooked_action, joint)))
            obs, reward, done, timeout, _ = kitchen.step(torch.tensor([joint], device=args.device))
            assert kitchen.agent_pos.tolist() == [[flat(p.position) for p in state.players]]
            assert kitchen.orientation.tolist() == [[nest_action_index(p.orientation) for p in state.players]]
            assert kitchen.held.tolist() == [[item(p.held_object) for p in state.players]]
            pot = state.objects.get(mdp.get_pot_locations()[0])
            assert kitchen.onions.item() == (len(pot.ingredients) if pot else 0)
            assert kitchen.cook_tick.item() == (pot._cooking_tick if pot else -1)
            for position, value in enumerate(kitchen.counter_items[0].tolist()):
                xy = (position % mdp.width, position // mdp.width)
                if xy in mdp.get_counter_locations():
                    assert value == item(state.objects.get(xy))
            sparse = sum(info["sparse_reward_by_agent"])
            shaped = sum(info["shaped_reward_by_agent"])
            assert kitchen.last_sparse.item() == sparse
            assert reward.item() == sparse + shaped, (episode, step, joint, shaped, reward.item())
            assert bool(done.item()) == bool(timeout.item()) == (step + 1 == HORIZON)
            assert obs.shape == (1, 2, OBSERVATION_SIZE) and obs.isfinite().all()
            total += 1
            deliveries += sparse > 0
    assert deliveries > 0
    print(f"Matched {total} native transitions, {deliveries} positive deliveries, and {total} L0 actions on {args.device}.")


if __name__ == "__main__":
    main()

"""Order-capped progress potential, inspired by NestRL recipe-goal differences.

Training reward = native score + gamma * Phi(next) - Phi(current).
True episode terminals use Phi(next)=0; rollout boundaries do not. Consequently
the discounted shaping return telescopes and repeated transfers cannot farm it.
It is a team learning signal, not evidence of the learner's own contribution.
"""
import torch

VERSION = "order-potential-v1"
GAMMA = .99
COEFFICIENT = 1.


def potential(state):
    """Value useful inventory, processing, and delivery-side progress in points.

    Per required ingredient: raw vegetables .25; prepared vegetables, egg,
    and beef .75; in a pot/meal 1.
Per dish: cooking up to 1, ready 2, held on delivery side another .5.
Counts are capped by the three live orders, prioritizing advanced inventory.
No reward for merely putting down / picking up the same item.
"""
    device = state["held"].device
    masks = torch.tensor([5, 2, 11], device=device)
    order_masks = masks[state["recipe"]]
    order_count = torch.stack([(state["recipe"] == r).sum(1) for r in range(3)], 1)
    loose = torch.cat((state["held"], state["board"], state["counter"]), 1)
    meal_count = torch.stack([(loose == r + 7).sum(1) for r in range(3)], 1)
    meal_count = torch.minimum(meal_count, order_count)
    compatible = ((state["mix"][:, :, None] & order_masks[:, None, :]) == state["mix"][:, :, None])
    compatible &= ((state["locked"][:, :, None] < 0) |
                   (state["locked"][:, :, None] == state["recipe"][:, None, :]))
    useful_pots = compatible.any(2)
    result = torch.zeros(len(loose), device=device)
    for raw, cut, bit in ((1, 5, 1), (2, 6, 2), (3, 3, 4), (4, 4, 8)):
        needed = ((order_masks & bit) != 0).sum(1)
        advanced = (((state["mix"] & bit) != 0) & useful_pots).sum(1)
        advanced += (meal_count * ((masks & bit) != 0)).sum(1)
        used = torch.minimum(advanced, needed)
        result += used.float()
        remaining = needed - used
        prepared = (loose == cut).sum(1)
        used = torch.minimum(prepared, remaining)
        result += .75 * used
        if raw != cut:
            result += .25 * torch.minimum((loose == raw).sum(1), remaining - used)
    for recipe in range(3):
        locked = state["locked"] == recipe
        ready = (locked & (state["heat"] >= 4)).sum(1) + meal_count[:, recipe]
        used = torch.minimum(ready, order_count[:, recipe])
        result += 2. * used
        # At most two pots. Highest heat first if duplicate work exceeds demand.
        heat = torch.where(locked & (state["heat"] < 4), state["heat"].float() / 4, 0.)
        heat = heat.sort(1, descending=True).values
        slots = torch.arange(2, device=device)[None, :] < (order_count[:, recipe] - used)[:, None]
        result += (heat * slots).sum(1)
        side = 1 if recipe == 0 else 0
        on_side = ((state["held"] == recipe + 7) & ((state["pos"] % 13 > 6) == bool(side))).sum(1)
        result += .5 * torch.minimum(on_side, order_count[:, recipe])
    return result


def transition_reward(before, state, terminal=False):
    after = torch.zeros_like(before) if terminal else potential(state)
    return COEFFICIENT * (GAMMA * after - before)

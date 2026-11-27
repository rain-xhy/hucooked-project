"""Paired, balanced evaluation on the counter training map."""
import numpy as np
import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES, STAY

from .features import partner_action_ids
from .kitchen import V3PolicyKitchen

ROLE_NAMES = ("left", "right_cook", "right_courier")
STYLE_NAMES = ("fifo", "earliest_deadline", "fewest_ingredients")
PROCESS_NAMES = ("counter_wait_steps", "counter_pickups", "counter_deposits",
                 "cut_steps", "completed_cuts", "pot_deposits", "pot_heat_steps", "completed_meals")


def _reset_paired(env, seed):
    env.reset(seed=seed)
    # BatchKitchen seeds lane i with seed+i. Each contiguous group of nine
    # receives its first lane's full initial state and future ticket sequence.
    for values in (*env.state.values(), env.draws):
        values.copy_(values[::9].repeat_interleave(9, dim=0))
    return env.observations()


def _ratio(a, b):
    return float(a / b) if b else None


def _summary(orders, processes, confusion, scores=None, recipe_rewards=(20, 10, 50), extra=None, actionable_available=False):
    # Leading axes are episodes and/or lanes; each observation is one episode.
    episodes = orders.size // (3 * 4)
    counts = orders.reshape(-1, 3, 4).sum(axis=0)
    process = dict(zip(PROCESS_NAMES, map(int, processes.reshape(-1, len(PROCESS_NAMES)).sum(axis=0))))
    process["mean_counter_wait"] = _ratio(process["counter_wait_steps"], process["counter_pickups"])
    matrix = confusion.reshape(-1, 3, 3).sum(axis=0)
    process["partner_confusion"] = matrix.tolist()
    process["partner_accuracy"] = _ratio(matrix.trace(), matrix.sum())
    per_recipe = {}
    for name, (delivered, expired, unfinished, age) in zip(RECIPE_NAMES, counts):
        resolved = delivered + expired
        issued = resolved + unfinished
        per_recipe[name] = dict(issued=int(issued), delivered=int(delivered), expired=int(expired),
                                unresolved=int(unfinished), delivery_rate=_ratio(delivered, resolved),
                                timeout_rate=_ratio(expired, resolved), mean_delivery_steps=_ratio(age, delivered))
    for i, name in enumerate(RECIPE_NAMES):
        per_recipe[name].update(total_score=float(counts[i, 0] * recipe_rewards[i]),
                                mean_score=float(counts[i, 0] * recipe_rewards[i] / episodes))
    total_score = float(np.sum(scores)) if scores is not None else sum(r["total_score"] for r in per_recipe.values())
    if extra is not None:
        from .process_metrics import EXTRA_NAMES
        process.update(zip(EXTRA_NAMES, map(int, extra.reshape(-1, len(EXTRA_NAMES)).sum(0))))
        for kind in ("raw", "cut", "meal"):
            process[f"mean_counter_wait_{kind}"] = _ratio(process[f"counter_wait_{kind}_steps"], process[f"counter_{kind}_pickups"])
        if not actionable_available:
            process["idle_actionable_ingredient_steps"] = None
    cross = counts[[0, 2]].sum(axis=0)
    rates = [row["delivery_rate"] for row in per_recipe.values()]
    def macro(selected):
        return float(np.mean(selected)) if all(x is not None for x in selected) else None
    return dict(episodes=episodes, total_score=total_score, mean_score=total_score / episodes, deliveries=int(counts[:, 0].sum()),
                cross_deliveries=int(cross[0]), cross_delivery_rate=macro([rates[0], rates[2]]),
                macro_delivery_rate=macro(rates),
                deliveries_per_episode=float(counts[:, 0].sum() / episodes),
                per_recipe=per_recipe, **process)


@torch.inference_mode()
def evaluate(model, *, seed, groups=1, mode="policy", device="cuda", horizon=400,
             controller_version="parallel-v2", recipe_rewards=(20, 10, 50), replay_dir=None):
    """Greedy evaluation. Rows of each confusion matrix are actual partner styles.

    Each group contributes all nine learner role/style pairs on the same five
    consecutive episode seeds. Seed of group g, episode e is seed + 9*g + e.
    Classification is scored once at episode close with the latest 40 steps
    of that episode and available prior episodes. Empty mode always masks history.
    """
    from .history import OnlineHistory
    from .controllers import make_partners
    from .process_metrics import ProcessMetrics, EXTRA_NAMES

    if type(groups) is not int or groups <= 0:
        raise ValueError("groups must be a positive integer")
    if mode not in ("policy", "empty", "script", "wait"):
        raise ValueError("Unknown evaluation mode")
    if mode in ("policy", "empty") and model is None:
        raise ValueError("Policy evaluation requires a model")
    size = groups * 9
    env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=horizon, recipe_rewards=recipe_rewards), num_envs=size, device=device)
    recipe_rewards = env.config.recipe_rewards
    lanes = env.lanes
    roles, styles = (lanes % 9) // 3, lanes % 3
    partners = make_partners(env, roles, styles, controller_version,
                             learner_mask=torch.full((size,), mode != "script", device=device, dtype=torch.bool))
    history = OnlineHistory(size, device) if model is not None else None
    orders = torch.zeros(5, size, 3, 4, device=device, dtype=torch.long)
    processes = orders.new_zeros(5, size, len(PROCESS_NAMES))
    confusion = orders.new_zeros(5, size, 3, 3)
    scores = torch.zeros(5, size, device=device, dtype=torch.float64)
    extra = orders.new_zeros(5, size, len(EXTRA_NAMES))
    replay_paths = []
    training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for ep in range(5):
            obs = _reset_paired(env, seed + ep)
            partners.reset()
            meter = ProcessMetrics(env)
            recorder = None
            if replay_dir is not None and ep == 0:
                from .replay_export import EvaluationReplays
                recorder = EvaluationReplays(env, dict(controller_version=controller_version, mode=mode,
                    seed=int(seed), recipe_rewards=list(recipe_rewards)), ROLE_NAMES, STYLE_NAMES)
            learner = partners.learner_ids
            for _ in range(env.config.horizon):
                current = obs[lanes, learner]
                actions = partners.actions()
                if mode in ("policy", "empty"):
                    logp, _, _ = history.forward(model, current)
                    actions[lanes, learner] = logp.argmax(-1)
                elif mode == "wait":
                    actions[lanes, learner] = STAY
                if model is not None and mode != "empty":
                    history.append(current, partner_action_ids(actions)[lanes, learner], model)

                state = env.state
                cuts, mix, heat, counter = (state[k].clone() for k in ("cuts", "mix", "heat", "counter"))
                meter.before(actions)
                obs, reward, _, _, info = env.step(actions)
                scores[ep] += reward
                extra[ep] += meter.after()
                if controller_version in ("parallel-v2", "parallel-gpu-v3"):
                    extra[ep, :, -1] += torch.as_tensor(partners.idle_actionable, device=device).sum(-1)
                if recorder is not None:
                    recorder.append(actions, reward, info)
                made_cut = state["cuts"] > cuts
                heated = state["heat"] > heat
                increments = (info["wait_steps"], info["pickups"], ((counter == 0) & (state["counter"] != 0)).sum(-1),
                              made_cut.sum(-1), (made_cut & (state["cuts"] == 2)).sum(-1),
                              (state["mix"] > mix).sum(-1), heated.sum(-1), (heated & (state["heat"] == 4)).sum(-1))
                processes[ep] += torch.stack(increments, -1)
                for col, value in ((0, info["delivered"]), (1, info["expired"]), (3, info["age"] * info["delivered"])):
                    orders[ep, :, :, col].scatter_add_(1, info["recipe"], value.long())
            pending = env.state["recipe"]
            orders[ep, :, :, 2].scatter_add_(1, pending, torch.ones_like(pending))
            if model is not None:
                _, _, logits = history.forward(model, obs[lanes, learner])
                confusion[ep, lanes, styles, logits.argmax(-1)] = 1
                history.finish_episode()
            if recorder is not None:
                replay_paths.extend(recorder.write(replay_dir))
    finally:
        if model is not None:
            model.train(training)

    orders, processes, confusion, scores, extra = (t.cpu().numpy() for t in (orders, processes, confusion, scores, extra))
    def summarize(eps=slice(None), lane=slice(None)):
        return _summary(orders[eps, lane], processes[eps, lane], confusion[eps, lane], scores[eps, lane], recipe_rewards, extra[eps, lane], controller_version in ("parallel-v2", "parallel-gpu-v3"))
    result = summarize()
    role_ids, style_ids = roles.cpu().numpy(), styles.cpu().numpy()
    result.update(mode=mode, seed=int(seed), groups=groups, horizon=env.config.horizon,
                  controller_version=controller_version, recipe_rewards=list(recipe_rewards), replay_paths=replay_paths,
                  metric_definitions={"idle_actionable_ingredient_steps": "Scripted chef STAY steps with an executable unclaimed ingredient task after reservation, resource, side, expiry and route checks; excludes externally controlled learner; null for serial-v1",
                                      "idle_with_outstanding_work_steps": "All chef STAY steps while any ticket or physical work remains; includes blocked or currently unavailable work"},
                  partner_confusion_labels=list(STYLE_NAMES),
                  episode_seeds=[[seed + g * 9 + e for e in range(5)] for g in range(groups)],
                  per_role={n: summarize(lane=role_ids == i) for i, n in enumerate(ROLE_NAMES)},
                  per_style={n: summarize(lane=style_ids == i) for i, n in enumerate(STYLE_NAMES)},
                  per_role_style={r: {s: summarize(lane=(role_ids == i) & (style_ids == j))
                                      for j, s in enumerate(STYLE_NAMES)} for i, r in enumerate(ROLE_NAMES)},
                  per_episode=[summarize(i) for i in range(5)])
    return result

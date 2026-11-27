"""Evaluation harness for TorchSharedKitchen reporting all origin/hy episode_metrics."""
import torch

from .env import SPAWNS, STAY, SharedKitchenConfig, TorchSharedKitchen
from .fast_partners import GpuFastPartners
from .gpu_partners import GpuSharedPartners
from .history import OnlineHistory
from .policy import partner_action_ids

ROLE_NAMES = ("server", "main", "side")


def role_slot_map(env):
    """Map logical roles (0=server, 1=main, 2=side) to chef slot indices [N, 3] from spawn positions."""
    pos = env.state["pos"]
    return torch.stack([(pos == sp).long().argmax(dim=1) for sp in SPAWNS], dim=1)


@torch.inference_mode()
def evaluate_shared(model=None, *, mode="script", num_envs=45, episodes=5, seed=10_000_000,
                    device="cuda", config=None, use_cuda_graph=True, partner_type="shared"):
    """Evaluate script / wait / single-policy / all-policy across `num_envs * episodes` episodes."""
    if mode not in ("script", "wait", "policy", "all_policy"):
        raise ValueError("mode must be script, wait, policy, or all_policy")
    if mode in ("policy", "all_policy") and model is None:
        raise ValueError("Policy modes require a model")
    cfg = config or SharedKitchenConfig()
    env = TorchSharedKitchen(cfg, num_envs=num_envs, device=device, use_cuda_graph=use_cuda_graph)
    roles = torch.arange(num_envs, device=device) % 3
    partner_cls = GpuFastPartners if partner_type == "fast" else GpuSharedPartners
    partners = partner_cls(
        env, roles=roles,
        learner_mask=torch.full((num_envs,), mode in ("wait", "policy"), dtype=torch.bool, device=device),
        use_cuda_graph=use_cuda_graph,
    )
    histories = ([OnlineHistory(num_envs, device) for _ in range(3)] if mode == "all_policy"
                 else OnlineHistory(num_envs, device) if mode == "policy" else None)
    training = model.training if model is not None else False
    if model is not None:
        model.eval()

    acc = {}
    per_role_scores = {r: [] for r in ROLE_NAMES}
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep * 1000)
            partners.reset()
            r2s = role_slot_map(env)
            for _ in range(cfg.horizon):
                if mode == "all_policy":
                    joint = torch.empty((num_envs, 3), dtype=torch.long, device=device)
                    cur_by_role = []
                    for r in range(3):
                        sl = r2s[:, r]
                        cur_r = obs[env.lanes, sl]
                        cur_by_role.append((sl, cur_r))
                        logp, _, _ = histories[r].forward(model, cur_r)
                        joint[env.lanes, sl] = logp.argmax(-1)
                    pids = partner_action_ids(joint)
                    for r in range(3):
                        sl, cur_r = cur_by_role[r]
                        histories[r].append(cur_r, pids[env.lanes, sl], model)
                else:
                    joint = partners.actions()
                    if mode == "policy":
                        learner = partners.learner_ids
                        cur = obs[env.lanes, learner]
                        logp, _, _ = histories.forward(model, cur)
                        joint[env.lanes, learner] = logp.argmax(-1)
                        histories.append(cur, partner_action_ids(joint)[env.lanes, learner], model)
                    elif mode == "wait":
                        joint[env.lanes, partners.learner_ids] = STAY
                obs, _, _, _, _ = env.step(joint)
            m = env.episode_metrics()
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v
            scores = env.state["score"]
            for idx, rname in enumerate(ROLE_NAMES):
                mask = roles == idx
                if bool(mask.any()):
                    per_role_scores[rname].append(float(scores[mask].mean()))
            if mode == "all_policy":
                for c in range(3):
                    histories[c].finish_episode()
            elif mode == "policy":
                histories.finish_episode()
    finally:
        if model is not None:
            model.train(training)

    result = {k: v / episodes for k, v in acc.items()}
    result["mode"] = mode
    result["episodes"] = num_envs * episodes
    result["per_role_score"] = {r: sum(vs) / len(vs) for r, vs in per_role_scores.items() if vs}
    return result

"""Render the executed v3 policy graph with Torchview and Graphviz."""
import os
from pathlib import Path
import shutil

import torch
from torchview import draw_graph

from overcooked_ai_py.cooperative.partition import PartitionConfig
from .policy import HIDDEN_DIM, HISTORY_EPISODES, HISTORY_STEPS
from . import OBS_DIM, V3Policy, V3PolicyKitchen, partner_action_ids


def main():
    here = Path(__file__).resolve().parent
    if not shutil.which("dot"):
        portable = next((here.parent / "orgin" / ".conda" / "graphviz").glob("*/bin/dot.exe"), None)
        if portable:
            os.environ["PATH"] = str(portable.parent) + os.pathsep + os.environ["PATH"]
    torch.manual_seed(0)
    model = V3Policy().eval()
    env = V3PolicyKitchen(PartitionConfig(horizon=HISTORY_STEPS), num_envs=1, device="cpu")
    obs, _ = env.reset(seed=7)
    episodes, steps = HISTORY_EPISODES, HISTORY_STEPS
    hist = torch.zeros(1, episodes, steps, OBS_DIM)
    ids = torch.full((1, episodes, steps), -1, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    # Four complete 40-step sample episodes; the fifth has just begun.
    for ep in range(episodes - 1):
        for t in range(steps):
            actions = torch.randint(6, (1, 3))
            hist[:, ep, t] = obs[:, 0]
            ids[:, ep, t] = partner_action_ids(actions)[:, 0]
            mask[:, ep, t] = False
            obs, *_ = env.step(actions)
        obs, _ = env.reset()
    inputs = (obs[:, 0], hist, ids, mask.all(-1), mask)
    logp, value, partner_logits = model(*inputs)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(1))
    assert value.shape == (1, 1) and partner_logits.shape == (1, model.partner_head.out_features)
    graph = draw_graph(model, input_data=inputs, device="cpu", depth=1,
                       graph_name="V3 policy", expand_nested=False).visual_graph
    # Tensor shapes uniquely identify the five input roles in this example.
    shapes = {
        str((1, OBS_DIM)): "Current self-centric observation",
        str((1, episodes, steps, OBS_DIM)): "Historical observations",
        str((1, episodes, steps)): None,
        str((1, episodes)): "Episode padding mask",
    }
    labels = {
        "HistoryFeatures": "History features<BR/>Observation + 36 joint-action indicators",
        "EpisodeEncoder": f"Episode encoder<BR/>MLP {OBS_DIM + 36} → {HIDDEN_DIM} → {HIDDEN_DIM}<BR/>Masked mean over steps",
        "PeriodEncoder": f"Period encoder<BR/>Masked mean over valid episodes<BR/>MLP {HIDDEN_DIM} → {HIDDEN_DIM} → {HIDDEN_DIM}",
        "Actor": f"Actor<BR/>Concat → MLP {OBS_DIM + HIDDEN_DIM} → {HIDDEN_DIM} → {HIDDEN_DIM}<BR/>6 actions + log-softmax",
        "Critic": f"Critic<BR/>Concat → MLP {OBS_DIM + HIDDEN_DIM} → {HIDDEN_DIM} → {HIDDEN_DIM}<BR/>Value head → 1",
        "Linear": f"Auxiliary partner classifier<BR/>Linear {HIDDEN_DIM} → {model.partner_head.out_features}",
    }
    import re
    nodes = {int(match[1]): line for line in graph.body if (match := re.match(r"\s*(\d+) \[label=", line))}
    edges = [(int(m[1]), int(m[2])) for line in graph.body if (m := re.match(r"\s*(\d+) -> (\d+)", line))]
    for i, line in enumerate(graph.body):
        if "input-tensor<BR/>depth:0" in line:
            number = int(line.split("[", 1)[0])
            shape = next(k for k in shapes if k in line)
            label = shapes[shape]
            if label is None:
                is_action = any(a == number and ">HistoryFeatures<BR/>" in nodes[b] for a, b in edges)
                label = "Joint partner action IDs" if is_action else "Step padding mask"
            line = line.replace("input-tensor<BR/>depth:0", label)
        elif "output-tensor<BR/>depth:0" in line:
            line = line.replace("output-tensor<BR/>depth:0",
                                ("Action log-probabilities" if "(1, 6)" in line else
                                 "State value" if "(1, 1)" in line else "Partner-class logits"))
        else:
            for key, label in labels.items():
                line = line.replace(f">{key}<BR/>depth:1", ">" + label)
        graph.body[i] = line
    graph.node_attr["fontname"] = "Arial"
    graph.attr(label="V3 partition kitchen  |  5 episodes × 40 steps  |  256-wide actor–critic", labelloc="t",
               fontname="Arial", fontsize="22", bgcolor="white", dpi="145", pad="0.2")
    stem = here / "model_graphs" / "v3_policy_5x40_256"
    stem.parent.mkdir(exist_ok=True)
    stem.with_suffix(".dot").write_text(graph.source, encoding="utf8")
    for ext in ("svg", "png"):
        stem.with_suffix("." + ext).write_bytes(graph.pipe(format=ext))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()

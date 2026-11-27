"""Trace the author-published Overcooked PACE policy with Torchview."""

import argparse
import os
import re
import shutil

import torch
from torch import nn
from torchview import draw_graph

from overcooked_setup import ROOT, UPSTREAM, activate_upstream


class HistoryBatch:
    """Expose prepared history tensors through the author's history interface."""

    merge_encoder_computation = False

    def __init__(self, observations, actions, episode_mask, step_mask):
        self.tensors = observations, actions, episode_mask, step_mask

    def get_by_idx(self, *_indices):
        return self.tensors


class HistoryFeatures(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, observations, actions):
        return self.encoder.convert_inputs(observations, actions)


class EpisodeEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, history, step_mask):
        return self.base.episode_forward(history, step_mask)


class PeriodEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, episodes, episode_mask):
        return self.base.period_forward(episodes, episode_mask).relu()


class ActorHead(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation, latent):
        distribution, _, _ = self.actor.get_action_dist(observation, None, None, latent)
        return distribution.logits


class CriticHead(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, observation, latent):
        value, _ = self.critic(torch.cat([observation, latent], dim=-1), None, None)
        return value


class PolicyGraph(nn.Module):
    """Trace the author's tensor computation, with each aggregation as one node."""

    def __init__(self, policy):
        super().__init__()
        self.history_features = HistoryFeatures(policy.encoder)
        self.episode_encoder = EpisodeEncoder(policy.encoder.base)
        self.period_encoder = PeriodEncoder(policy.encoder.base)
        self.actor_head = ActorHead(policy.actor)
        self.critic_head = CriticHead(policy.critic)
        self.partner_head = policy.aux_pol_cls_head

    def forward(self, observation, history_observations, history_actions, episode_mask, step_mask):
        history = self.history_features(history_observations, history_actions)
        episodes = self.episode_encoder(history, step_mask)
        latent = self.period_encoder(episodes, episode_mask)
        return self.actor_head(observation, latent), self.critic_head(observation, latent), self.partner_head(latent)


def label_graph(graph, partner_count):
    """Name the traced tensors and restore the edge obscured by in-place scatter."""
    input_names = iter(("Current observation", "History observations", "Partner action IDs",
                        "Episode mask", "Step mask"))
    output_names = iter(("Action log-probabilities", "State value", "Partner-class logits"))
    layer_names = {
        "HistoryFeatures": "History features<BR/>One-hot action + observation",
        "EpisodeEncoder": "Episode encoder<BR/>MLP 79 → 128 → 128<BR/>Masked mean across steps",
        "PeriodEncoder": "Period encoder<BR/>Masked mean across episodes<BR/>MLP 128 → 128 → 128",
        "ActorHead": "Actor<BR/>Concat → MLP 201 → 128 → 128<BR/>Categorical head → 6",
        "CriticHead": "Critic<BR/>Concat → MLP 201 → 128 → 128<BR/>Value head → 1",
        "Linear": f"Auxiliary partner classifier<BR/>Linear 128 → {partner_count}",
    }
    node_ids = {}
    for index, statement in enumerate(graph.body):
        if "input-tensor<BR/>depth:0" in statement:
            statement = statement.replace("input-tensor<BR/>depth:0", next(input_names))
        elif "output-tensor<BR/>depth:0" in statement:
            statement = statement.replace("output-tensor<BR/>depth:0", next(output_names))
        else:
            for name, label in layer_names.items():
                if f">{name}<BR/>depth:1" in statement:
                    node_ids[name] = re.match(r"\s*(\d+) \[label=", statement).group(1)
                    statement = statement.replace(f">{name}<BR/>depth:1", f">{label}")
                    break
        graph.body[index] = statement

    # episode_forward creates a fresh tensor and assigns episode features in place.
    # Torchview 0.2.7 misses that dependency, so link these two traced module nodes.
    graph.edge(node_ids["EpisodeEncoder"], node_ids["PeriodEncoder"])
    graph.node_attr["fontname"] = "Arial"
    graph.graph_attr.update(fontname="Arial", bgcolor="white", dpi="160")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="ppo_pace_seed1",
        choices=("ppo_pace_seed1", *(f"ppo_pace_two_ing_{i}_seed1" for i in (1, 2, 3))),
    )
    name = parser.parse_args().checkpoint
    checkpoint = UPSTREAM / "logs" / "Overcooked" / name / "ppo" / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Initialize the adaptive-RL submodule to obtain {checkpoint}")

    # Prefer the portable Graphviz installation if present.
    if not shutil.which("dot"):
        for dot in (ROOT / ".conda" / "graphviz").glob("*/bin/dot.exe"):
            os.environ["PATH"] = str(dot.parent) + os.pathsep + os.environ["PATH"]
            break
    if not shutil.which("dot"):
        raise RuntimeError("Install Graphviz and add its bin directory to PATH")

    activate_upstream()
    torch.set_num_threads(1)
    # This file is the fixed author checkpoint shipped with the submodule.
    policy = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    model = PolicyGraph(policy).eval()

    action_count = policy.actor.action_dim
    obs_dim = policy.encoder.base.pre_agg_mlp.mlp[0].in_features - action_count
    batch_size, episodes, steps = 1, 5, 40
    history_shape = batch_size, episodes, steps
    generator = torch.Generator().manual_seed(1)
    step_mask = torch.zeros(history_shape, dtype=torch.bool)
    step_mask[:, -1, :] = True
    inputs = (
        torch.randn(batch_size, obs_dim, generator=generator),
        torch.randn(*history_shape, obs_dim, generator=generator),
        torch.randint(action_count, history_shape, generator=generator),
        step_mask.all(dim=-1),
        step_mask,
    )
    with torch.no_grad():
        logits, value, partner_logits = model(*inputs)
        indices = torch.zeros(batch_size, dtype=torch.long)
        history = HistoryBatch(*inputs[1:]), (indices,) * 4
        masks = torch.ones(batch_size, 1)
        distribution, latent, _ = policy.get_action_dist(inputs[0], None, masks, indices, history)
        expected_value, _ = policy.get_value(inputs[0], None, masks, indices, latent)
        torch.testing.assert_close(logits, distribution.logits)
        torch.testing.assert_close(value, expected_value)
        torch.testing.assert_close(partner_logits, policy.aux_pol_cls_head(latent))
    assert logits.shape == (batch_size, action_count)
    assert value.shape == (batch_size, 1)
    assert partner_logits.shape == (batch_size, policy.aux_pol_cls_head.out_features)

    graph = draw_graph(
        model, input_data=inputs, device="cpu", depth=1,
        graph_name="Author Overcooked PACE policy", expand_nested=False,
    ).visual_graph
    label_graph(graph, policy.aux_pol_cls_head.out_features)
    graph.attr(label=f"Author Overcooked PACE policy  |  {name}", labelloc="t", fontsize="22")
    graph.attr("graph", pad="0.2", ranksep="0.32", nodesep="0.22")
    output = ROOT / "model_graphs"
    output.mkdir(exist_ok=True)
    stem = output / ("author_pace_policy" if name == "ppo_pace_seed1" else name)
    stem.with_suffix(".dot").write_text(graph.source, encoding="utf-8")
    for extension in ("svg", "png"):
        stem.with_suffix(f".{extension}").write_bytes(graph.pipe(format=extension))
    print(f"{stem}.svg  |  logits {tuple(logits.shape)}, value {tuple(value.shape)}, "
          f"partner logits {tuple(partner_logits.shape)}")


if __name__ == "__main__":
    main()

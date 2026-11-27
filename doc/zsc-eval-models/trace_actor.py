"""Trace the official ZSC-Eval R_Actor with its published HSP configs and weights."""

import argparse
import importlib.machinery
import pickle
import re
import sys
import types
from pathlib import Path

import torch
from torchview import draw_graph


class PolicyConfigReader(pickle.Unpickler):
    """Read only the types used in the published policy configuration files."""

    allowed = {
        ("argparse", "Namespace"),
        ("gym.spaces.box", "Box"),
        ("gym.spaces.discrete", "Discrete"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        if (module, name) == ("pathlib", "PosixPath"):
            return Path
        if (module, name) in self.allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Unexpected config type {module}.{name}")


def simplify_graph(graph, action_count):
    """Label traced modules and contract the input dtype/device casts."""
    inputs = iter(("Observation", "Previous GRU state", "Episode mask", "Available actions"))
    outputs = iter(("Next GRU state", "Selected action", "Action log probability"))
    modules = {
        "CNNBase": "Grid encoder<BR/>3× Conv2d + ReLU → Flatten<BR/>MLP 800 → 64 → 64 → 64",
        "MLPBase": "Vector encoder<BR/>LayerNorm → MLP 58 → 64 → 64 → 64",
        "RNNLayer": "Recurrent state<BR/>GRU 64 → 64 + LayerNorm",
        "MLPLayer": "Post-GRU MLP<BR/>64 → 64 → 64",
        "ACTLayer": f"Categorical action head<BR/>Linear 64 → {action_count} → mask + distribution",
    }
    casts = set()
    edges = []
    nodes = []
    for statement in graph.body:
        edge = re.fullmatch(r"\s*(\d+) -> (\d+)\s*", statement)
        if edge:
            edges.append(edge.groups())
            continue
        if "to<BR/>depth:1" in statement:
            casts.add(re.match(r"\s*(\d+) \[", statement).group(1))
            continue
        if "input-tensor<BR/>depth:0" in statement:
            statement = statement.replace("input-tensor<BR/>depth:0", next(inputs))
        elif "output-tensor<BR/>depth:0" in statement:
            statement = statement.replace("output-tensor<BR/>depth:0", next(outputs))
        else:
            for module, label in modules.items():
                statement = statement.replace(f">{module}<BR/>depth:1", f">{label}")
        nodes.append(statement)

    upstream = {target: source for source, target in edges if target in casts}
    graph.body[:] = nodes
    for source, target in edges:
        if target not in casts:
            graph.edge(upstream.get(source, source), target)

    graph.node_attr.update(fontname="Arial", fontsize="11")
    graph.graph_attr.update(ranksep="0.4", nodesep="0.23", pad="0.2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="Output path without extension")
    parser.add_argument("--title", required=True)
    options = parser.parse_args()

    package_dir = options.repository.resolve() / "zsceval"
    if not (package_dir / "config.py").is_file():
        parser.error("repository must point to the official ZSC-Eval checkout")

    # Import the original model modules without initializing the game environments.
    package = types.ModuleType("zsceval")
    package.__spec__ = importlib.machinery.ModuleSpec("zsceval", loader=None, is_package=True)
    package.__path__ = [str(package_dir)]
    sys.modules["zsceval"] = package
    from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    torch.set_num_threads(1)
    with options.config.open("rb") as handle:
        args, observation_space, _, action_space = PolicyConfigReader(handle).load()

    actor = R_Actor(args, observation_space, action_space).eval()
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(state, strict=True)

    batch_size = 1
    inputs = (
        torch.zeros(batch_size, *observation_space.shape),
        torch.zeros(batch_size, args.recurrent_N, args.hidden_size),
        torch.ones(batch_size, 1),
        torch.ones(batch_size, action_space.n),
    )
    with torch.no_grad():
        action, log_prob, hidden = actor(*inputs, deterministic=True)
        probs = actor.get_probs(*inputs)
        if isinstance(probs, tuple):
            probs = probs[0]
        torch.testing.assert_close(action, probs.argmax(dim=-1, keepdim=True))
        torch.testing.assert_close(log_prob, probs.gather(1, action).log())
        assert hidden.shape == inputs[1].shape

    parameter_count = sum(parameter.numel() for parameter in actor.parameters())
    graph = draw_graph(
        actor,
        input_data=inputs,
        device="cpu",
        depth=1,
        hide_inner_tensors=True,
        hide_module_functions=True,
        expand_nested=False,
        deterministic=True,
    ).visual_graph
    simplify_graph(graph, action_space.n)
    graph.attr(
        label=f"{options.title}\nOfficial R_Actor · {parameter_count:,} registered parameters",
        labelloc="t",
        fontname="Arial",
        fontsize="20",
        dpi="144",
    )
    options.output.with_suffix(".dot").write_text(graph.source, encoding="utf-8")
    for extension in ("svg", "png"):
        options.output.with_suffix(f".{extension}").write_bytes(graph.pipe(format=extension))
    print(f"{options.title}: {parameter_count:,} parameters, {action_space.n} actions, hidden={tuple(hidden.shape)}")


if __name__ == "__main__":
    main()

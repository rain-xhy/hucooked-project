"""Use the current native environment or the historical CUDA environment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "orgin"


def activate(*, legacy_gpu=False):
    project = ROOT.parents[2]
    environment = project / (".environment/gpu-dev" if legacy_gpu else "overcooked_env")
    if legacy_gpu and not (environment / "src/overcooked_ai_py/mdp/torch_env.py").is_file():
        raise RuntimeError("Set up .environment/gpu-dev as described in src/nest_rl/mlp_standard/README.md")
    sys.path[:0] = [str(ROOT), str(ORIGINAL), str(ROOT.parent / "cnn_standard"),
                    str(environment / "src")]
    from overcooked_setup import activate_upstream

    activate_upstream()

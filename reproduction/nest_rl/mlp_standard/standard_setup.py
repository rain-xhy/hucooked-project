"""Use the NestRL project runtime with the native hucooked checkout."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "orgin"


def activate():
    sys.path[:0] = [str(ROOT), str(ORIGINAL), str(ROOT.parent / "cnn_standard"),
                    str(ROOT.parents[2] / "hucooked" / "src")]
    sys.path.append(str(ORIGINAL / "runs" / "hucooked-py310-probe-deps"))
    from overcooked_setup import activate_upstream

    activate_upstream()

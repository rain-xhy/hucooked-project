"""Render the README's archived benchmark means (requires matplotlib).

Run from any directory: python scripts/plot_benchmark_summary.py
The CSV retains evaluation provenance; these are not matched-protocol trials.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    ("team_score", "团队得分 ↑", 490, [0, 100, 200, 300, 400]),
    ("sync_success_percent", "同步成功率 (%) ↑", 115, [0, 25, 50, 75, 100]),
    ("joint_wait_steps", "同步等待步数 ↓", 76, [0, 20, 40, 60]),
)


def build_figure():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei") if f in fonts), None)
    if font is None:
        raise RuntimeError("Install a CJK font: Microsoft YaHei or Noto Sans CJK SC.")
    plt.rcParams.update({
        "font.family": font, "font.size": 12, "axes.unicode_minus": False,
        "svg.fonttype": "path", "svg.hashsalt": "hucooked-benchmark",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#24383f", "axes.labelcolor": "#24383f",
        "xtick.color": "#62747a", "ytick.color": "#24383f",
    })
    with (ROOT / "docs/benchmark-summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8 or len({row["checkpoint"] for row in rows}) != 8:
        raise ValueError("Expected eight distinct checkpoints.")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.31, top=0.86, wspace=0.24)
    for ax, (key, title, limit, ticks) in zip(axes, METRICS):
        ordered = sorted(rows, key=lambda row: float(row[key]), reverse=True)
        values = [float(row[key]) for row in ordered]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"Invalid metric: {key}")
        ax.bar(range(len(ordered)), values, width=0.64, color="#26858b", zorder=3)
        ax.set_xticks(range(len(ordered)), [row["checkpoint"] for row in ordered],
                      rotation=50, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, limit)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=11)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=21)
        ax.grid(axis="y", color="#e6ecee", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for index, value in enumerate(values):
            label = f"{value:.0f}" if key == "team_score" else f"{value:.2f}"
            ax.text(index, value + limit * 0.018, label, ha="center", va="bottom", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "svg", "both"), default="both")
    args = parser.parse_args()
    fig = build_figure()
    output = ROOT / "docs/images/benchmark-summary"
    formats = ("png", "svg") if args.format == "both" else (args.format,)
    for extension in formats:
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(output.with_suffix("." + extension), dpi=300, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()

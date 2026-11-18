"""Plot the measured continuation success at each tested action boundary."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    blocks = report["blocks"]
    scenes = sorted({row["seed"] for row in blocks})
    if not scenes:
        raise ValueError("No analyzed replay blocks to plot")
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    methods = [("SFT", "sft_successes", "#196B87", "o"), ("PPO", "ppo_successes", "#C36426", "s")]
    for ax, scene in zip(axes, scenes):
        rows = sorted((r for r in blocks if r["seed"] == scene), key=lambda r: r["step"])
        steps = [row["step"] for row in rows]
        for name, key, color, marker in methods:
            rates = [r["pooled"][key] / r["pooled"]["pairs"] for r in rows]
            ax.plot(steps, rates, color=color, marker=marker, linewidth=2,
                    markerfacecolor="white", markeredgewidth=1.6, markersize=6, label=name)
        for r in rows:
            if r["stable_gap"]:
                ax.axvspan(r["step"]-3, r["step"]+3, color="#47754F", alpha=0.13, linewidth=0)
        ax.set_title(f"Scene {scene}", pad=10)
        ax.set_xlabel("Continuation start step")
        ax.set_xlim(steps[0]-4, steps[-1]+4)
        ax.set_xticks(steps)
        ax.set_ylim(-0.04, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Success within the remaining horizon")
    legend = [Line2D([], [], color=c, marker=m, markerfacecolor="white", label=n)
              for n, _, c, m in methods]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    has_gap = any(r["stable_gap"] for r in blocks)
    description = ("Shading marks steps meeting the stable-difference criterion." if has_gap
                   else "No step met the stable-difference criterion.")
    fig.text(0.5, 0.035, description, ha="center", fontsize=9)
    fig.subplots_adjust(top=0.79, bottom=0.23, left=0.11, right=0.99, wspace=0.12)
    fig.savefig(args.output, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.2)


if __name__ == "__main__":
    main()

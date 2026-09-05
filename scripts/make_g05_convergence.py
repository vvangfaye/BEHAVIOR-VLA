#!/usr/bin/env python
"""G05 (GalaxeaVLA) 300k convergence figure, plus a GR00T-300k reference panel.

Loss sources (real logs):
  G05  : /dev/shm/wangfaye/logs/g05_b1k_300k_20260820_112207.log
         tqdm bar description "Step N, Loss: X" (discrete AR token cross-entropy,
         8 ranks x microbatches interleaved -> bin by step for the trend).
  GR00T: /dev/shm/wangfaye/logs/gr00t_b1k_long300k_20260812_023839.log
         HF Trainer "{'loss': X}" every 10 steps (diffusion action head).

  /dev/shm/wangfaye/conda_envs/b1k-groot/bin/python scripts/make_g05_convergence.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/experiments/viz")
OUT.mkdir(parents=True, exist_ok=True)
G05_LOG = Path("/dev/shm/wangfaye/logs/g05_b1k_300k_20260820_112207.log")
GROOT_LOG = Path("/dev/shm/wangfaye/logs/gr00t_b1k_long300k_20260812_023839.log")
G05C, GROOTC = "#C44E52", "#1F77B4"


def g05_curve(bin_size=250):
    txt = G05_LOG.read_text(errors="ignore")
    pts = {}
    for m in re.finditer(r"Step (\d+), Loss: ([0-9.]+)", txt):
        pts.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    if not pts:
        return np.array([]), np.array([]), np.array([]), np.array([])
    steps = np.array(sorted(pts))
    raw = np.array([np.mean(pts[s]) for s in steps])  # avg over ranks/microbatches per step
    # bin the per-step means for a smooth trend + spread band
    bins = {}
    for s, v in zip(steps, raw):
        bins.setdefault(int(s) // bin_size, []).append(v)
    bx = np.array([k * bin_size + bin_size // 2 for k in sorted(bins)])
    bmean = np.array([np.mean(bins[k]) for k in sorted(bins)])
    bstd = np.array([np.std(bins[k]) for k in sorted(bins)])
    return steps, raw, bx, bmean, bstd


def groot_curve():
    txt = GROOT_LOG.read_text(errors="ignore")
    losses = [float(x) for x in re.findall(r"\{'loss': ([0-9.]+)", txt)]
    xs = 10 * (np.arange(len(losses)) + 1)  # logged every 10 steps from step 0
    return xs, np.array(losses)


def main():
    gs, graw, gbx, gbmean, gbstd = g05_curve()
    kx, ky = groot_curve()
    last_step = int(gs[-1]) if len(gs) else 0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # Panel 1: G05 AR cross-entropy convergence (the model still training)
    a1.scatter(gs, graw, s=2, color=G05C, alpha=0.12, label="per-step mean (noisy)")
    a1.plot(gbx, gbmean, color=G05C, lw=2.0, label="250-step binned mean")
    a1.fill_between(gbx, gbmean - gbstd, gbmean + gbstd, color=G05C, alpha=0.15, label="±1 std")
    a1.axhline(np.log(4096), color="gray", ls=":", lw=1, label="uniform-token CE = ln(4096)")
    if len(gbmean):
        a1.annotate(f"{gbmean[0]:.2f}", (gbx[0], gbmean[0]), fontsize=9)
        a1.annotate(f"{gbmean[-1]:.3f}", (gbx[-1], gbmean[-1]), fontsize=9, fontweight="bold")
    a1.set_xlabel("training step"); a1.set_ylabel("AR action-token cross-entropy loss")
    a1.set_title(f"G05 (GalaxeaVLA) discrete-AR  |  {last_step:,}/300,000 steps ({last_step/3000:.1f}%)")
    a1.legend(fontsize=8, loc="upper right"); a1.grid(alpha=0.3)

    # Panel 2: GR00T-300k reference (completed, diffusion loss, log-y)
    a2.plot(kx, ky, color=GROOTC, lw=0.8, alpha=0.9)
    a2.set_yscale("log")
    if len(ky):
        a2.annotate(f"{ky[0]:.3f}", (kx[0], ky[0]), fontsize=9)
        a2.annotate(f"{ky[-1]:.4f}", (kx[-1], ky[-1]), fontsize=9, fontweight="bold")
    a2.set_xlabel("training step"); a2.set_ylabel("diffusion loss (log)")
    a2.set_title(f"GR00T N1.7 diffusion (reference, done)  final={ky[-1]:.4f} @ {int(kx[-1]):,}")
    a2.grid(alpha=0.3, which="both")

    fig.suptitle("BEHAVIOR-1K 300k-step training convergence "
                 "(different loss types/scales; compare shape, not absolute value)", fontsize=12)
    fig.tight_layout()
    p = OUT / "04_g05_convergence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p, "| G05 last step", last_step, "binned final", round(float(gbmean[-1]), 3) if len(gbmean) else None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot pi0.5 sub-task loss and gradient convergence."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIZ = Path(__file__).resolve().parent.parent / "experiments/viz"
VIZ.mkdir(parents=True, exist_ok=True)
OUT_PI05 = VIZ / "06_pi05_subtask_convergence.png"

PI05_LOG = Path(
    "/mnt/workspace/wangfaye/logs/pi05_b1k_subtask_mix_s32_20260825_185859.log"
)
PI05_RE = re.compile(r"Step (\d+): grad_norm=([0-9.]+), loss=([0-9.]+), param_norm=([0-9.]+)")
PI05_TARGET = 210_000

PI05C, ACCC, GNC = "#1F77B4", "#55A868", "#8172B2"
BIN = 2000


def binned(steps, vals, bin_size=BIN):
    """Median + IQR band per bin -- robust to the heavy-tailed per-step spread."""
    keys = steps // bin_size
    uniq = np.unique(keys)
    x = np.array([k * bin_size + bin_size / 2 for k in uniq])
    med = np.array([np.median(vals[keys == k]) for k in uniq])
    lo = np.array([np.percentile(vals[keys == k], 25) for k in uniq])
    hi = np.array([np.percentile(vals[keys == k], 75) for k in uniq])
    return x, med, lo, hi


def tail_slope(x, med, frac):
    """OLS slope per 10k steps, fit on binned medians (spike-robust)."""
    cut = x[0] + (1 - frac) * (x[-1] - x[0])
    m = x >= cut
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(x[m], med[m], 1)[0] * 10_000)


def load_pi05():
    steps, gn, loss = [], [], []
    with PI05_LOG.open() as f:
        for line in f:
            m = PI05_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                gn.append(float(m.group(2)))
                loss.append(float(m.group(3)))
    return np.array(steps), np.array(gn), np.array(loss)


def callouts(ax, marks):
    for xv, ytxt, txt, style in marks:
        ax.axvline(xv, **style)
        ax.annotate(
            txt,
            xy=(xv, ytxt),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-5, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=8,
            color=style["color"],
        )


ps, pgn, ploss = load_pi05()

px, pmed, plo, phi = binned(ps, ploss)
pgx, pgmed, pglo, pghi = binned(ps, pgn)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

best_i = int(np.argmin(pmed))
epoch = PI05_TARGET // 2

ax = axes[0]
ax.plot(ps, ploss, color=PI05C, lw=0.4, alpha=0.15)
ax.fill_between(px, plo, phi, color=PI05C, alpha=0.25, lw=0, label="IQR")
ax.plot(px, pmed, color=PI05C, lw=2.0, label="binned median")
ax.plot(px[best_i], pmed[best_i], "v", color=PI05C, ms=9, label=f"best {pmed[best_i]:.4f}")
ax.set_yscale("log")
ax.set_ylim(min(plo) * 0.95, max(phi) * 1.05)
ax.set_xlim(0, PI05_TARGET * 1.02)
ax.set_xlabel("step")
ax.set_ylabel("flow-matching loss")
ax.set_title(
    f"pi0.5 sub-task  FM {pmed[0]:.3f} -> {pmed[-1]:.4f}  "
    f"(best {pmed[best_i]:.4f} @ {int(px[best_i]):,})",
    fontsize=10,
)
ax.grid(alpha=0.3, which="both")
callouts(
    ax,
    (
        (ps[-1], 0.55, f"now {ps[-1]:,}", dict(color="k", ls=":", lw=1.3)),
        (epoch, 0.02, f"1 epoch = {epoch:,}", dict(color=ACCC, ls="-.", lw=1.3)),
        (PI05_TARGET, 0.02, f"target {PI05_TARGET:,}", dict(color="k", ls="--", lw=1.0, alpha=0.5)),
    ),
)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
ax.plot(ps, pgn, color=GNC, lw=0.4, alpha=0.15)
ax.fill_between(pgx, pglo, pghi, color=GNC, alpha=0.25, lw=0, label="IQR")
ax.plot(pgx, pgmed, color=GNC, lw=2.0, label="binned median")
ax.set_xlim(0, PI05_TARGET * 1.02)
ax.set_ylim(min(pglo) * 0.9, max(pghi) * 1.1)
ax.set_xlabel("step")
ax.set_ylabel("grad_norm")
tail = pgn[ps >= ps[-1] - 0.1 * (ps[-1] - ps[0])]
ax.set_title(
    f"pi0.5 grad_norm  last10%: mean {tail.mean():.3f}, max {tail.max():.3f}", fontsize=10
)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle(
    f"pi0.5 sub-task+mix stride32  |  step {ps[-1]:,} / {PI05_TARGET:,}  |  "
    f"binned-median slope/10k: last50% {tail_slope(px, pmed, 0.5):+.5f}, "
    f"last10% {tail_slope(px, pmed, 0.1):+.5f}",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT_PI05, dpi=150)
plt.close(fig)

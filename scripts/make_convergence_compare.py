#!/usr/bin/env python3
"""Plot G05 and pi0.5 sub-task convergence as three separate figures."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VIZ = Path("/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/experiments/viz")
OUT_G05 = VIZ / "05_g05_convergence.png"
OUT_PI05 = VIZ / "06_pi05_subtask_convergence.png"
OUT_REL = VIZ / "07_relative_decay.png"

G05_LOG = Path(
    "/mnt/workspace/wangfaye/cache/g05_finetune/b1k/"
    "2026-08-20_11-22-39/logs/model_debug_rank0.log"
)
G05_RE = re.compile(
    r"\[LOSS\] step=(\d+) loss=([0-9.]+) fm_loss=[0-9.]+ ce_loss=[0-9.]+ acc=([0-9.]+)"
)

PI05_LOG = Path(
    "/mnt/workspace/wangfaye/logs/pi05_b1k_subtask_mix_s32_20260825_185859.log"
)
PI05_RE = re.compile(r"Step (\d+): grad_norm=([0-9.]+), loss=([0-9.]+), param_norm=([0-9.]+)")
PI05_TARGET = 210_000

G05C, PI05C, ACCC, GNC = "#C44E52", "#1F77B4", "#55A868", "#8172B2"
BIN = 2000
REBASE_AT = 2000


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


def rebase(x, med, at=REBASE_AT):
    i = int(np.argmin(np.abs(x - at)))
    return med / med[i]


def steps_to(x, rel, thr):
    hit = np.nonzero(rel <= thr)[0]
    return int(x[hit[0]]) if hit.size else None


def load_g05():
    steps, loss, acc = [], [], []
    with G05_LOG.open() as f:
        for line in f:
            m = G05_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                loss.append(float(m.group(2)))
                acc.append(float(m.group(3)))
    return np.array(steps), np.array(loss), np.array(acc)


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


gs, gloss, gacc = load_g05()
ps, pgn, ploss = load_pi05()

gx, gmed, glo, ghi = binned(gs, gloss)
gax, gamed, galo, gahi = binned(gs, gacc)
px, pmed, plo, phi = binned(ps, ploss)
pgx, pgmed, pglo, pghi = binned(ps, pgn)

# ---------------------------------------------------------------- G05 figure
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
ax.fill_between(gx, glo, ghi, color=G05C, alpha=0.25, lw=0, label="IQR")
ax.plot(gx, gmed, color=G05C, lw=2.0, label="binned median")
ax.set_yscale("log")
ax.set_ylim(min(glo) * 0.95, max(ghi) * 1.05)
ax.set_xlabel("step")
ax.set_ylabel("action-token CE loss")
ax.set_title(
    f"G0.5  CE {gmed[0]:.3f} -> {gmed[-1]:.4f}  (min {gmed.min():.4f})", fontsize=10
)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axes[1]
ax.fill_between(gax, galo, gahi, color=ACCC, alpha=0.25, lw=0, label="IQR")
ax.plot(gax, gamed, color=ACCC, lw=2.0, label="binned median")
ax.set_ylim(0, 1)
ax.set_xlabel("step")
ax.set_ylabel("action-token accuracy")
ax.set_title(
    f"G0.5  acc {gamed[0]:.3f} -> {gamed[-1]:.3f}  (max {gamed.max():.3f})", fontsize=10
)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle(
    f"G0.5 finetune  |  {len(gs):,} logged steps, run stopped at {gs[-1]:,}  |  "
    f"binned-median slope/10k: last50% {tail_slope(gx, gmed, 0.5):+.5f}, "
    f"last10% {tail_slope(gx, gmed, 0.1):+.5f}",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT_G05, dpi=150)
plt.close(fig)

# ------------------------------------------------------------- pi0.5 figure
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

# ----------------------------------------------------- relative decay figure
grel = rebase(gx, gmed)
prel = rebase(px, pmed)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(gx, grel, color=G05C, lw=2.0, label=f"G0.5 CE  (final {grel[-1]:.3f})")
ax.plot(px, prel, color=PI05C, lw=2.0, label=f"pi0.5 FM  (final {prel[-1]:.3f})")

thresholds = (0.80, 0.70, 0.65, 0.60)
for thr in thresholds:
    ax.axhline(thr, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.annotate(
        f"{thr:.2f}x",
        xy=(0.995, thr),
        xycoords=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="gray",
    )

rows = [f"{'thr':>5} {'G0.5':>9} {'pi0.5':>9}"]
for thr in thresholds:
    g = steps_to(gx, grel, thr)
    p = steps_to(px, prel, thr)
    rows.append(
        f"{thr:>5.2f} {(f'{g:,}' if g else 'never'):>9} {(f'{p:,}' if p else 'never'):>9}"
    )
ax.text(
    0.98,
    0.97,
    "steps to reach\n" + "\n".join(rows),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    family="monospace",
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
)

ax.set_xlabel("step")
ax.set_ylabel(f"loss / own loss @ step {REBASE_AT:,}")
ax.set_title(
    "Relative decay (each rebased to its own step-2k value)\n"
    "absolute CE and FM losses are not comparable -- only decay rates are",
    fontsize=10,
)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(OUT_REL, dpi=150)
plt.close(fig)

print(f"G05  : {len(gs):,} pts, steps {gs[0]}..{gs[-1]}")
print(f"pi05 : {len(ps):,} pts, steps {ps[0]}..{ps[-1]} of {PI05_TARGET:,}")
for p in (OUT_G05, OUT_PI05, OUT_REL):
    print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")

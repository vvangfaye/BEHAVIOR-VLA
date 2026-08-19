#!/usr/bin/env python3
"""Plot the π0.5 long300k training loss curve.

Reads the trainer log at /dev/shm/wangfaye/logs/pi05_long300k.log, extracts
`Step N: grad_norm=..., loss=..., param_norm=...` records, and writes a
two-panel figure showing raw loss and a 500-step moving average.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_RE = re.compile(
    r"Step (?P<step>\d+): grad_norm=(?P<gn>\S+), loss=(?P<loss>\S+), param_norm=(?P<pn>\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        default="/dev/shm/wangfaye/logs/pi05_long300k.log",
        help="Path to the trainer stdout log.",
    )
    parser.add_argument(
        "--out",
        default="/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/experiments/pi05_long300k/loss_curve.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=500,
        help="Moving-average window in steps (0 disables).",
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, losses, grads = [], [], []
    with path.open() as fh:
        for line in fh:
            m = LOG_RE.search(line)
            if not m:
                continue
            steps.append(int(m["step"]))
            losses.append(float(m["loss"]))
            grads.append(float(m["gn"]))
    if not steps:
        raise SystemExit(f"No step records found in {path}")
    arr = np.array
    return arr(steps, dtype=np.int64), arr(losses, dtype=np.float64), arr(grads, dtype=np.float64)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window >= len(values):
        return values.copy()
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps, losses, grads = load_records(log_path)
    window = args.smooth if args.smooth > 0 else 0
    smoothed = moving_average(losses, window) if window else losses
    smoothed_steps = steps[window - 1 :] if window and window > 1 else steps

    # Anchor the y-axis at the 99th percentile so a handful of single-batch
    # spikes don't compress the rest of the curve into a flat line. The full
    # max is reported in the title so spikes are still discoverable.
    loss_ceil = float(np.quantile(losses, 0.99)) * 1.10
    grad_ceil = float(np.quantile(grads, 0.99)) * 1.10
    loss_max = float(losses.max())
    grad_max = float(grads.max())

    fig, (ax_loss, ax_grad) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_loss.scatter(steps, losses, s=2, alpha=0.25, color="#7a7a7a", label="100-step log")
    ax_loss.plot(
        smoothed_steps,
        smoothed,
        color="#c0392b",
        linewidth=2.0,
        label=f"{window}-step moving avg" if window else "loss",
    )
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_ylim(0, loss_ceil)
    ax_loss.grid(True, linestyle=":", alpha=0.4)
    ax_loss.legend(loc="upper right")
    ax_loss.set_title(
        f"π0.5 long300k — resume from step 80,099 → step {steps[-1]:,}\n"
        f"{len(steps):,} 100-step records, last loss {losses[-1]:.4f}  "
        f"(y-axis clipped at 99th pct; max {loss_max:.3f}, grad max {grad_max:.3f})"
    )

    ax_grad.scatter(steps, grads, s=2, alpha=0.25, color="#7a7a7a")
    ax_grad.plot(
        smoothed_steps,
        moving_average(grads, window) if window else grads,
        color="#2c6fb5",
        linewidth=1.5,
    )
    ax_grad.set_ylabel("Grad norm")
    ax_grad.set_xlabel("Train step")
    ax_grad.set_ylim(0, grad_ceil)
    ax_grad.grid(True, linestyle=":", alpha=0.4)

    # Mark the resume boundary so the curve clearly shows the resume handoff.
    ax_loss.axvline(80_000, color="#27ae60", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_loss.text(
        80_000,
        loss_ceil * 0.95,
        "resume from 80,000",
        color="#27ae60",
        fontsize=9,
        ha="left",
        va="top",
    )
    ax_grad.axvline(80_000, color="#27ae60", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path} ({len(steps)} points, window={window})")


if __name__ == "__main__":
    main()
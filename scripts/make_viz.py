#!/usr/bin/env python
"""Project visualizations: Gantt timeline, training loss curves, open-loop eval
comparison. Every number comes from run logs / eval JSON, not from memory.

  /dev/shm/wangfaye/conda_envs/b1k-groot/bin/python scripts/make_viz.py

Log cadence (verified): pi0.5 logs every 100 steps ("Step N: ... loss=X"),
GR00T logs every 10 steps ("{'loss': X}"). GR00T pre-20k comes from the first
run (groot_full3.log); 20k-50k from the resume after OOM@25k (groot_resume.log,
resumed from the last complete checkpoint-20000).
"""
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path("/dev/shm/wangfaye/logs")
REPO = Path("/mnt/workspace/wangfaye/code/BEHAVIOR-Challenge")
OUT = REPO / "experiments" / "viz"
OUT.mkdir(parents=True, exist_ok=True)
C = {"env": "#4C72B0", "data": "#55A868", "train": "#C44E52", "eval": "#8172B3", "crash": "#D62728"}
BLUE = "#1F77B4"


def pi05_curve():
    """pi0.5: merge first run (0-29k) + resume (25k-50k); resume wins overlap."""
    pts = {}
    for name in ("pi05_full7.log", "pi05_resume.log"):
        f = LOGS / name
        if not f.exists():
            continue
        for line in open(f, errors="ignore"):
            m = re.search(r"Step (\d+): grad_norm=[0-9.]+, loss=([0-9.]+)", line)
            if m:
                pts[int(m.group(1))] = float(m.group(2))
    xs = sorted(pts)
    return np.array(xs), np.array([pts[x] for x in xs])


def groot_curve():
    """GR00T logs every 10 steps. full3 supplies 0-20000, resume supplies
    20000-50000 (checkpoint-20000 was the last complete one before OOM@25k)."""
    pts = {}
    f3 = LOGS / "groot_full3.log"
    if f3.exists():
        losses = re.findall(r"\{'loss': ([0-9.]+)", open(f3, errors="ignore").read())
        for i, v in enumerate(losses):
            step = 10 * (i + 1)
            if step <= 20000:
                pts[step] = float(v)
    fr = LOGS / "groot_resume.log"
    if fr.exists():
        losses = re.findall(r"\{'loss': ([0-9.]+)", open(fr, errors="ignore").read())
        for i, v in enumerate(losses):
            pts[20000 + 10 * (i + 1)] = float(v)
    xs = sorted(pts)
    return np.array(xs), np.array([pts[x] for x in xs])


def gantt():
    fmt = "%Y-%m-%d %H:%M"
    rows = [
        ("Build conda envs (pi05 3.11 / groot 3.10)", "2026-08-03 08:50", "2026-08-03 09:20", "env"),
        ("Install deps (jax/torch, flash-attn 395MB)", "2026-08-03 08:55", "2026-08-03 09:30", "env"),
        ("conda-pack relocatable tarballs", "2026-08-03 09:24", "2026-08-03 09:33", "env"),
        ("Stage dataset OSS->local (~1TB, 3xRGB)", "2026-08-03 10:00", "2026-08-03 10:40", "data"),
        ("GR00T relative-stats (parallel, 3 keys)", "2026-08-04 20:44", "2026-08-04 22:44", "data"),
        ("pi0.5 smoke debug (5 fixes)", "2026-08-03 10:40", "2026-08-03 16:20", "train"),
        ("pi0.5 full train 50k (crash@29k->resume@25k)", "2026-08-03 16:26", "2026-08-04 12:14", "train"),
        ("GR00T full train 50k (OOM@25k->resume@20k)", "2026-08-04 22:55", "2026-08-05 14:08", "train"),
        ("Eval env (Isaac Sim: HW-unsupported)", "2026-08-05 08:00", "2026-08-05 12:30", "eval"),
        ("Open-loop eval (pi0.5 200 samples)", "2026-08-05 16:00", "2026-08-05 16:45", "eval"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (label, s, e, cat) in enumerate(rows):
        t0 = mdates.date2num(datetime.strptime(s, fmt))
        t1 = mdates.date2num(datetime.strptime(e, fmt))
        ax.barh(i, t1 - t0, left=t0, height=0.6, color=C[cat], alpha=0.9)
        ax.text(t0, i + 0.34, label, va="bottom", ha="left", fontsize=8.5)
    for when in ("2026-08-04 03:04", "2026-08-05 03:02"):
        ax.axvline(mdates.date2num(datetime.strptime(when, fmt)), color=C["crash"], ls="--", lw=1, alpha=0.7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(i + 1) for i in range(len(rows))])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("env", "data", "train", "eval")]
    ax.legend(handles, ["Environment", "Data", "Training", "Evaluation"], loc="lower right", fontsize=9)
    ax.set_title("BEHAVIOR-1K: pi0.5 & GR00T reproduction timeline (Aug 3-5, 2026)", fontsize=12)
    ax.set_xlabel("wall-clock time  (red dashed = crash; both recovered via checkpoint resume)")
    fig.tight_layout()
    p = OUT / "01_gantt.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def training():
    px, py = pi05_curve()
    gx, gy = groot_curve()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    if len(px):
        a1.plot(px, py, color=C["train"], lw=1.2)
        a1.axvline(25000, color=C["crash"], ls="--", lw=1, alpha=0.7, label="crash@29k -> resume@25k")
        a1.set_title(f"pi0.5 (flow-matching)  final={py[-1]:.4f} @ {int(px[-1])} steps")
    a1.set_xlabel("training step"); a1.set_ylabel("loss (log)"); a1.set_yscale("log")
    a1.legend(fontsize=9); a1.grid(alpha=0.3, which="both")
    if len(gx):
        a2.plot(gx, gy, color=BLUE, lw=1.0)
        a2.axvline(20000, color=C["crash"], ls="--", lw=1, alpha=0.7, label="OOM@25k -> resume@20k")
        a2.set_title(f"GR00T N1.7 (diffusion)  final={gy[-1]:.4f} @ {int(gx[-1])} steps")
    a2.set_xlabel("training step"); a2.set_ylabel("loss (log)"); a2.set_yscale("log")
    a2.legend(fontsize=9); a2.grid(alpha=0.3, which="both")
    fig.suptitle("Full-parameter training on 8x H20 | 100 tasks / 20000 demos", fontsize=12)
    fig.tight_layout()
    p = OUT / "02_training_loss.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def _eval_pi05_only(p):
    po = p["overall"]
    groups = list(po["model"]["groups"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: overall L1 -- learned policy vs two trivial baselines.
    methods = ["pi0.5\nmodel", "hold-last\nbaseline", "zero\nbaseline"]
    vals = [po["model"]["l1"], po["baseline_hold"]["l1"], po["baseline_zero"]["l1"]]
    colors = [C["train"], "#888888", "#CCCCCC"]
    bars = a1.bar(methods, vals, color=colors)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    a1.set_ylabel("action L1 (raw joint space)")
    a1.set_title("Overall L1: learned policy vs trivial baselines")
    a1.grid(alpha=0.3, axis="y")

    # Panel 2: per action-group L1, model vs hold-last (shows grippers).
    w = 0.38
    xg = np.arange(len(groups))
    a2.bar(xg - w / 2, [po["model"]["groups"][k] for k in groups], w, label="pi0.5 model", color=C["train"])
    a2.bar(xg + w / 2, [po["baseline_hold"]["groups"][k] for k in groups], w, label="hold-last", color="#888888")
    a2.set_xticks(xg); a2.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    a2.set_ylabel("action L1"); a2.set_title("Per action-group L1 (grippers dominate baseline error)")
    a2.legend(); a2.grid(alpha=0.3, axis="y")

    # Panel 3: distribution of per-task L1 across the 100 tasks.
    pt = np.array(list(p["per_task_l1"].values()))
    a3.hist(pt, bins=20, color=C["train"], alpha=0.85, edgecolor="white")
    med = float(np.median(pt))
    a3.axvline(med, color="black", ls="--", lw=1.2, label=f"median={med:.4f}")
    a3.axvline(po["model"]["l1"], color=BLUE, ls=":", lw=1.4, label=f"mean={po['model']['l1']:.4f}")
    a3.set_xlabel("per-task action L1"); a3.set_ylabel("number of tasks (of 100)")
    a3.set_title("Per-task L1 distribution (100 tasks)")
    a3.legend(fontsize=9); a3.grid(alpha=0.3, axis="y")

    fig.suptitle("pi0.5 open-loop action-prediction error (200 samples, 16-step horizon) "
                 "-- fit quality, NOT benchmark success rate  |  GR00T open-loop eval incomplete",
                 fontsize=11)
    return fig


def _eval_two_model(p, g):
    po, go = p["overall"], g["overall"]
    groups = list(po["model"]["groups"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: overall L1 -- both models vs shared trivial baselines
    # (hold/zero are identical across models: same samples & ground truth).
    methods = ["pi0.5\nmodel", "GR00T\nmodel", "hold-last\nbaseline", "zero\nbaseline"]
    vals = [po["model"]["l1"], go["model"]["l1"], po["baseline_hold"]["l1"], po["baseline_zero"]["l1"]]
    colors = [C["train"], BLUE, "#888888", "#CCCCCC"]
    bars = a1.bar(methods, vals, color=colors)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    a1.set_ylabel("action L1 (raw joint space)")
    a1.set_title("Overall L1: both policies vs trivial baselines")
    a1.grid(alpha=0.3, axis="y")

    # Panel 2: per action-group L1, pi0.5 vs GR00T
    w = 0.38
    xg = np.arange(len(groups))
    a2.bar(xg - w / 2, [po["model"]["groups"][k] for k in groups], w, label="pi0.5", color=C["train"])
    a2.bar(xg + w / 2, [go["model"]["groups"][k] for k in groups], w, label="GR00T", color=BLUE)
    a2.set_xticks(xg); a2.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    a2.set_ylabel("action L1"); a2.set_title("Per action-group L1 (pi0.5 vs GR00T)")
    a2.legend(); a2.grid(alpha=0.3, axis="y")

    # Panel 3: per-task head-to-head scatter (x=pi0.5, y=GR00T), y=x diagonal.
    pt, gt = p["per_task_l1"], g["per_task_l1"]
    common = sorted(set(pt) & set(gt))
    xs = np.array([pt[t] for t in common]); ys = np.array([gt[t] for t in common])
    gwin = int((ys < xs).sum()); pwin = len(common) - gwin
    a3.scatter(xs, ys, s=18, color="#444444", alpha=0.7)
    lim = float(max(xs.max(), ys.max())) * 1.05
    a3.plot([0, lim], [0, lim], color=C["crash"], ls="--", lw=1, label="y = x")
    a3.set_xlim(0, lim); a3.set_ylim(0, lim)
    a3.set_xlabel("pi0.5 per-task L1"); a3.set_ylabel("GR00T per-task L1")
    a3.set_title(f"Per-task head-to-head ({len(common)} tasks)\n"
                 f"GR00T lower: {gwin}   pi0.5 lower: {pwin}")
    a3.legend(fontsize=9); a3.grid(alpha=0.3)

    fig.suptitle("Open-loop action-prediction error (200 samples, 16-step horizon, identical frames) "
                 "-- fit quality, NOT benchmark success rate", fontsize=11)
    return fig


def evaluation():
    pj = REPO / "eval_results/openloop_pi05.json"
    gj = REPO / "eval_results/openloop_groot.json"
    if not pj.exists():
        print("SKIP evaluation figure: missing", pj)
        return
    p = json.load(open(pj))
    if gj.exists():
        fig = _eval_two_model(p, json.load(open(gj)))
        mode = "two-model comparison"
    else:
        fig = _eval_pi05_only(p)
        mode = "pi0.5 only (GR00T json absent)"
    fig.tight_layout()
    outp = OUT / "03_evaluation.png"
    fig.savefig(outp, dpi=130)
    plt.close(fig)
    print("wrote", outp, "|", mode)


if __name__ == "__main__":
    gantt()
    training()
    evaluation()

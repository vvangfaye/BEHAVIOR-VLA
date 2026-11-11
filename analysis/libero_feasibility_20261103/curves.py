#!/usr/bin/env python3
"""Learning curves of local RLinf runs from their tensorboard event files.

Usage (RLinf venv):  python curves.py results/ppo_baseline results/ppo_tokenmean [--keys k1,k2] [--json out.json]
Prints one row per iteration with the training-episode success and the main PPO diagnostics.
"""
import argparse
import glob
import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_KEYS = [
    "env/success_once",
    "env/episode_len",
    "traj/len_success_mean",
    "traj/len_fail_mean",
    "traj/fail_weight_share_tokenmean",
    "actor/clip_fraction",
    "actor/approx_kl",
    "actor/grad_norm",
    "critic/value_loss",
    "time/step",
    "time/generate_rollouts",
    "time/actor_training",
    "prefix_cache/hits",
    "prefix_cache/verify_max_abs",
]


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    series = {}
    for f in files:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for ev in acc.Scalars(tag):
                series.setdefault(tag, {})[ev.step] = ev.value
    return series


def resolve(series, key):
    """Match a short key (e.g. env/success_once) against prefixed tags (e.g. train/env/success_once)."""
    if key in series:
        return key
    cands = [t for t in series if t.endswith("/" + key) or t.endswith(key)]
    cands.sort(key=len)
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--list", action="store_true", help="list all scalar tags of the first run")
    args = ap.parse_args()
    out = {}
    for run in args.runs:
        series = load(run)
        if args.list:
            for t in sorted(series):
                print(t)
            return
        keys = [k for k in args.keys.split(",") if k]
        tags = {k: resolve(series, k) for k in keys}
        steps = sorted({s for t in tags.values() if t for s in series[t]})
        print(f"\n== {run}  ({len(steps)} logged steps)")
        shown = [k for k in keys if tags[k]]
        print("step  " + "  ".join(f"{k.split('/')[-1][:14]:>14s}" for k in shown))
        rows = []
        for s in steps:
            row = {k: series[tags[k]].get(s) for k in shown}
            rows.append({"step": s, **row})
            print(f"{s:4d}  " + "  ".join(f"{row[k]:14.4f}" if row[k] is not None else f"{'-':>14s}" for k in shown))
        out[run] = rows
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

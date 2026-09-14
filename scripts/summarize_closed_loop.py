#!/usr/bin/env python3
"""Aggregate closed-loop eval results written by eval_b1k.sh / eval_sweep.sh.

Usage:
    python scripts/summarize_closed_loop.py eval_results/pi05_subtask/pi05
    python scripts/summarize_closed_loop.py eval_results/pi05_baseline/pi05 eval_results/pi05_subtask/pi05
    python scripts/summarize_closed_loop.py --paired A B   # only (task, instance) in both
"""
import argparse
import glob
import json
import os
import statistics


def load(root):
    out = {}
    for path in glob.glob(os.path.join(root, "*", "json", "*.json")):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        key = (d["task"], d["instance_id"], d.get("rollout_id", 0))
        out[key] = d
    return out


def summarize(name, rollouts):
    if not rollouts:
        print(f"{name}: no results")
        return
    q = [r["q_score"]["final"] for r in rollouts]
    nonzero = [x for x in q if x > 0]
    tasks = {r["task"] for r in rollouts}
    scoring_tasks = {r["task"] for r in rollouts if r["q_score"]["final"] > 0}
    base = [r["agent_distance"]["base"] for r in rollouts]
    arms = [r["agent_distance"]["left"] + r["agent_distance"]["right"] for r in rollouts]
    print(f"\n=== {name} ===")
    print(f"rollouts        {len(rollouts)}   tasks {len(tasks)}")
    print(f"success         {sum(bool(r['success']) for r in rollouts)}")
    print(f"q_score mean    {statistics.fmean(q):.4f}   max {max(q):.4f}")
    print(f"q_score > 0     {len(nonzero)} rollouts, {len(scoring_tasks)} tasks")
    if nonzero:
        print(f"  mean|>0       {statistics.fmean(nonzero):.4f}")
    print(f"base dist mean  {statistics.fmean(base):.4f} m   max {max(base):.4f}")
    print(f"arm  dist mean  {statistics.fmean(arms):.4f} m")
    top = sorted(rollouts, key=lambda r: -r["q_score"]["final"])[:5]
    if top and top[0]["q_score"]["final"] > 0:
        print("top rollouts:")
        for r in top:
            if r["q_score"]["final"] <= 0:
                break
            print(f"  {r['q_score']['final']:.4f}  {r['task']} inst={r['instance_id']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--paired", action="store_true",
                    help="restrict every run to the (task, instance, rollout) keys common to all")
    args = ap.parse_args()

    runs = [(r, load(r)) for r in args.roots]
    if args.paired and len(runs) > 1:
        common = set(runs[0][1])
        for _, d in runs[1:]:
            common &= set(d)
        print(f"paired on {len(common)} common rollouts")
        runs = [(r, {k: v for k, v in d.items() if k in common}) for r, d in runs]

    for root, d in runs:
        summarize(root, list(d.values()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print '<n_iters> <success per iteration, %>' from the tensorboard events under a run dir (RLinf venv).
Usage: tb_last.py cluster_runs/<tag> [key=env/success_once]"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run, key = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "env/success_once")
vals = {}
for f in sorted(glob.glob(os.path.join(run, "**", "events.out.tfevents.*"), recursive=True)):
    try:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for ev in acc.Scalars(key) if key in acc.Tags().get("scalars", []) else []:
            vals[ev.step] = ev.value
    except Exception:
        pass
if vals:
    print(len(vals), " ".join(f"{100 * vals[s]:.0f}" for s in sorted(vals)))
else:
    print("0")

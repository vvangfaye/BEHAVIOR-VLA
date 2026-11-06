#!/usr/bin/env python3
"""Offline replay of manager v2 (2026-11-06) on the per-task training counts of the 11/5 v1 pilot (seed 3, 18 iterations):
real gpt-6.1-sol calls with the v2 prompt; the counts are fixed (they came from v1's sampling), so this checks the v2
decision behaviour (tasks flagged per iteration, weight spread, lr), not training. Usage: agent_manager_v2_replay.py"""
import importlib.util
import json
from pathlib import Path

AM = "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/rlinf/b1k_tools/agent_manager.py"
spec = importlib.util.spec_from_file_location("am_v2", AM)
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)
OUT = Path(__file__).resolve().parent / "agent_manager_v2_replay_20261106"
rows = [json.loads(line) for line in open("/mnt/data1/faye/agent_mgr_logs/ppo_balanced_seed3_agent/am/decisions_crashed_20261106.jsonl")]
m = am.Manager("agent", OUT / "am", 20, stats_dir=None, episodes_per_iter=128)
for r in rows:
    w, mult, info = m.decide({int(t): tuple(v) for t, v in r["counts"].items()})
    ag = info.get("agent", {})
    print(f"it {info['iteration']:2d} flagged {info.get('flagged')} p>=0.5 {sum(p >= 0.5 for p in ag.get('prob_regressed', []))} "
          f"weights {[round(x, 2) for x in w]} lr x{mult:.2f} scale {info['lr_scale']:.3f} err {ag.get('error')}", flush=True)

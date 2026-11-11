#!/usr/bin/env python3
"""Paired comparison of RoboCasa365 atomic18 evals (2026-11-05): episodes keyed by (task, env, ep) from the
'[robocasa eval] task=..., env=..., ep=..., seed=..., success=...' lines; McNemar exact test on the shared keys,
overall and split into the RL training tasks vs the other atomic tasks.
Usage: robocasa_paired.py BASELINE_console.log OTHER_console.log [OTHER2 ...]"""
import re
import sys
from math import comb

TRAIN = {"TurnOnElectricKettle", "SlideDishwasherRack", "OpenStandMixerHead", "TurnOnSinkFaucet", "PickPlaceCounterToStove", "CoffeeSetupMug"}
PAT = re.compile(r"\[robocasa eval\] task=(\w+), env=(\d+), ep=(\d+), seed=\d+, success=(True|False)")


def load(p):
    d = {}
    for line in open(p, errors="replace"):
        m = PAT.search(line)
        if m:
            d[(m.group(1), int(m.group(2)), int(m.group(3)))] = m.group(4) == "True"  # seeds repeat across epochs
    return d


def mcnemar(a, b, keys):
    x = sum(1 for k in keys if a[k] and not b[k]); y = sum(1 for k in keys if b[k] and not a[k]); n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return sum(a[k] for k in keys), sum(b[k] for k in keys), len(keys), p


base = load(sys.argv[1])
print(f"baseline {sys.argv[1]}: {sum(base.values())}/{len(base)}")
for other in sys.argv[2:]:
    o = load(other)
    keys = sorted(set(base) & set(o))
    for name, sel in (("all", keys), ("train tasks", [k for k in keys if k[0] in TRAIN]), ("other tasks", [k for k in keys if k[0] not in TRAIN])):
        a, b, n, p = mcnemar(o, base, sel)
        print(f"  {other.split('/')[-2]:28s} {name:12s}: {a}/{n} vs baseline {b}/{n} ({100*(a-b)/max(n,1):+.1f}pp), McNemar p={p:.4f}")
    per = {}
    for k in keys:
        v = per.setdefault(k[0], [0, 0, 0]); v[0] += o[k]; v[1] += base[k]; v[2] += 1
    print("   per task (RL/base/n):", {t: tuple(v) for t, v in sorted(per.items())})

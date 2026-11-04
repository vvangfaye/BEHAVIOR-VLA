#!/usr/bin/env python3
"""Paired comparison of LIBERO evaluations over the same (task_id, trial_id) initial states.

Usage: python3 paired.py results/full500/console.log results/eval_<arm>/console.log [more logs ...]
Each log is parsed for "[libero eval] task_id=T, trial_id=K, success=B" lines (one per episode).
The first log is the reference; every other log is compared to it with an exact McNemar test.
"""
import math
import re
import sys

PAT = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")


def parse(path):
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in out:
                    raise SystemExit(f"{path}: duplicate episode {key}")
                out[key] = m.group(3) == "True"
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b (ref wins) and c (other wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def summary(name, res):
    n = len(res)
    s = sum(res.values())
    tasks = sorted({t for t, _ in res})
    per = " ".join(f"{t}:{sum(v for (tt, _), v in res.items() if tt == t)}" for t in tasks)
    print(f"{name}: {s}/{n} = {100 * s / max(n, 1):.1f}%   per-task successes [{per}]")


def main():
    logs = sys.argv[1:]
    if len(logs) < 1:
        raise SystemExit(__doc__)
    ref = parse(logs[0])
    summary(logs[0], ref)
    for path in logs[1:]:
        other = parse(path)
        summary(path, other)
        common = sorted(set(ref) & set(other))
        b = sum(1 for k in common if ref[k] and not other[k])
        c = sum(1 for k in common if other[k] and not ref[k])
        diff = (sum(other[k] for k in common) - sum(ref[k] for k in common)) / max(len(common), 1)
        print(f"  paired on {len(common)} episodes: delta {100 * diff:+.1f}pp, ref-only {b}, other-only {c}, "
              f"McNemar exact p = {mcnemar_exact(b, c):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired comparison of CALVIN evaluations over the same eval chains (trial_id).

Usage: python3 calvin_paired.py results/<ref>/console.log results/<other>/console.log [...]
Parses "[calvin eval] trial_id=K, completed=C" lines (local logging added to calvin_gym_env.py on 2026-11-04).
Reports avg_len, Len-1..5 success rates and, against the first log, the paired mean difference of completed
subtasks with a sign-flip permutation p-value and a bootstrap 95% CI.
"""
import random
import re
import sys

PAT = re.compile(r"\[calvin eval\] trial_id=(\d+), completed=(\d+)")


def parse(path):
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                k, c = int(m.group(1)), int(m.group(2))
                if k in out:
                    raise SystemExit(f"{path}: duplicate chain {k}")
                out[k] = c
    return out


def summary(name, res):
    n = max(len(res), 1)
    vals = list(res.values())
    lens = " ".join(f"L{j}={sum(v >= j for v in vals) / n:.3f}" for j in range(1, 6))
    print(f"{name}: {len(res)} chains, avg_len {sum(vals) / n:.3f}  [{lens}]")


def main():
    logs = sys.argv[1:]
    if not logs:
        raise SystemExit(__doc__)
    ref = parse(logs[0])
    summary(logs[0], ref)
    rng = random.Random(0)
    for path in logs[1:]:
        other = parse(path)
        summary(path, other)
        common = sorted(set(ref) & set(other))
        d = [other[k] - ref[k] for k in common]
        if not d:
            continue
        obs = sum(d) / len(d)
        perm = 10000
        hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) / len(d) >= abs(obs) - 1e-12 for _ in range(perm))
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
        print(f"  paired on {len(common)} chains: delta avg_len {obs:+.3f} (95% CI {boots[49]:+.3f} to {boots[1949]:+.3f}), "
              f"sign-flip p = {(hits + 1) / (perm + 1):.4f}")


if __name__ == "__main__":
    main()

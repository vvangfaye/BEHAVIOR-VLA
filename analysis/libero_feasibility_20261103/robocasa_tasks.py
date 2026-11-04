#!/usr/bin/env python3
"""Per-task success of RoboCasa365 evals from the '[robocasa eval] task=..., env=..., ep=..., success=...' lines
(local logging added to robocasa365_env.py on 2026-11-04). Usage: robocasa_tasks.py console.log [other.log ...]
With two or more logs, also prints the paired difference on episodes present in both (same task/env/ep)."""
import re
import sys

PAT = re.compile(r"\[robocasa eval\] task=(\S+), env=(\d+), ep=(\d+), seed=\S+, success=(True|False), len=(\d+)")


def parse(path):
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                out[(m.group(1), int(m.group(2)), int(m.group(3)))] = (m.group(4) == "True", int(m.group(5)))
    return out


runs = [(p, parse(p)) for p in sys.argv[1:]]
tasks = sorted({k[0] for _, r in runs for k in r})
print(f"{'task':28s}" + "".join(f"{p.split('/')[-2][:22]:>24s}" for p, _ in runs))
for t in tasks:
    row = f"{t:28s}"
    for _, r in runs:
        v = [s for (tt, _, _), (s, _) in r.items() if tt == t]
        row += f"{(f'{sum(v)}/{len(v)} = {100 * sum(v) / len(v):.0f}%' if v else '-'):>24s}"
    print(row)
for p, r in runs:
    n = len(r); s = sum(v[0] for v in r.values())
    print(f"overall {p}: {s}/{n} = {100 * s / max(n, 1):.1f}%")
if len(runs) > 1:
    ref = runs[0][1]
    for p, r in runs[1:]:
        common = sorted(set(ref) & set(r))
        d = [int(r[k][0]) - int(ref[k][0]) for k in common]
        print(f"paired vs first on {len(common)} episodes: delta {100 * sum(d) / max(len(d), 1):+.1f}pp "
              f"(ref-only {sum(x < 0 for x in d)}, other-only {sum(x > 0 for x in d)})")

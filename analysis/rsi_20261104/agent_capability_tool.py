#!/usr/bin/env python3
"""Follow-up to agent_capability_tests.py (2026-11-05 17:0x): test A with statistics provided ("tool-augmented" agent).
Same 21 cases (same seed); each row now also carries drop, 95% CI of the difference and the one-sided z, plus the
number of tasks compared. Usage: agent_capability_tool.py"""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / "agent_capability_20261105_tool"
sys.argv = ["x", str(OUT)]
import agent_capability_tests as T  # noqa: E402

PROMPT = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy on 10 tasks.
Below are per-task success counts from two batches of episodes: a reference batch R (an earlier checkpoint) and a new batch N (the current checkpoint), about 13 episodes per task per batch.
A statistics tool already computed, per task, the drop in success rate (R minus N), its 95% confidence interval, and the one-sided two-proportion z statistic for a drop. 10 tasks are compared at once.
Decide for which tasks the policy's TRUE success rate has really dropped from R to N. Balance missed regressions against false alarms: a missed regression lets a task degrade, a false alarm wastes training. Only list tasks you believe truly regressed; an empty list is fine.

task | R | N | drop | 95% CI of drop | z
{rows}

Answer with JSON only."""


def row(t, R, N):
    (n1, s1), (n2, s2) = R.get(t, (0, 0)), N.get(t, (0, 0))
    p1, p2 = s1 / max(n1, 1), s2 / max(n2, 1)
    d = p1 - p2
    se_u = math.sqrt(max(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1), 1e-12))
    p = (s1 + s2) / max(n1 + n2, 1)
    se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
    return f"{t} | {s1}/{n1} | {s2}/{n2} | {d:+.2f} | [{d - 1.96 * se_u:+.2f}, {d + 1.96 * se_u:+.2f}] | {d / se:+.2f}"


cases = T.build_A()
jobs = [(c["name"], PROMPT.format(rows="\n".join(row(t, c["R"], c["N"]) for t in range(10))), T.SCHEMA_A) for c in cases]
import concurrent.futures as cf  # noqa: E402
with cf.ThreadPoolExecutor(8) as ex:
    res = dict(zip([j[0] for j in jobs], ex.map(lambda j: T.ask(*j), jobs)))
fp = tn = tp = fn = same_any = same_n = 0
for c in cases:
    flags = set(res.get(c["name"], {}).get("regressed_tasks", []))
    for t, truth in c["truth"].items():
        if truth is None:
            continue
        if truth:
            tp += t in flags; fn += t not in flags
        else:
            fp += t in flags; tn += t not in flags
    if all(v is False for v in c["truth"].values()):
        same_n += 1; same_any += bool(flags)
rep = {"agent + stats tool": {"false_alarm_per_task": round(fp / max(fp + tn, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                              "tp": tp, "fn": fn, "fp": fp, "tn": tn, "no-change cases with any alarm": f"{same_any}/{same_n}"},
       "errors": [k for k, v in res.items() if "error" in v]}
(OUT / "report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))

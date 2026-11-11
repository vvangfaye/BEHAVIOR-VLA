#!/usr/bin/env python3
"""Fixes for the agent's conservative regression calls (2026-11-05 17:1x), on the same 21 cases of test A.
P: ask for a probability per task (with the statistics), threshold outside -> AUC + recall at fixed false-alarm rates.
F: few-shot calibration: 4 labelled examples (drawn with another seed = disjoint subsamples) + binary decision.
Usage: agent_capability_fix.py"""
import concurrent.futures as cf
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv = ["x", str(HERE / "agent_capability_20261105_fix")]
import agent_capability_tests as T  # noqa: E402


def row(t, R, N):
    (n1, s1), (n2, s2) = R.get(t, (0, 0)), N.get(t, (0, 0))
    p1, p2 = s1 / max(n1, 1), s2 / max(n2, 1)
    d = p1 - p2
    se_u = math.sqrt(max(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1), 1e-12))
    p = (s1 + s2) / max(n1 + n2, 1)
    se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
    return f"{t} | {s1}/{n1} | {s2}/{n2} | {d:+.2f} | [{d - 1.96 * se_u:+.2f}, {d + 1.96 * se_u:+.2f}] | {d / se:+.2f}"


def z_of(t, R, N):
    (n1, s1), (n2, s2) = R.get(t, (0, 0)), N.get(t, (0, 0))
    p = (s1 + s2) / max(n1 + n2, 1)
    se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
    return (s1 / max(n1, 1) - s2 / max(n2, 1)) / se


HEAD = "task | R | N | drop | 95% CI of drop | z"
SCHEMA_P = {"type": "object", "additionalProperties": False, "required": ["prob_regressed", "reason"],
            "properties": {"prob_regressed": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                              "minItems": 10, "maxItems": 10},
                           "reason": {"type": "string", "maxLength": 300}}}
PROMPT_P = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy on 10 tasks.
Per-task success counts of a reference batch R (earlier checkpoint) and a new batch N (current checkpoint), about 13 episodes per task per batch, with statistics already computed (drop = R minus N, 95% CI of the drop, one-sided z for a drop).
For EACH task, give your probability (0 to 1) that the policy's TRUE success rate on that task has really dropped from R to N. Give calibrated probabilities, not decisions.

{head}
{rows}

Answer with JSON only."""
PROMPT_F = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy on 10 tasks.
You compare per-task success counts of a reference batch R and a new batch N (about 13 episodes per task per batch) and decide which tasks have TRULY regressed. Calibrate on these four solved examples; their answers were checked against 50 evaluation episodes per task:
{examples}
Now the new case:
{head}
{rows}
List the tasks that truly regressed (an empty list is fine). Answer with JSON only."""


def table(c):
    return "\n".join(row(t, c["R"], c["N"]) for t in range(10))


def main():
    cases = T.build_A(seed=0)
    ex_cases = T.build_A(seed=1)
    pick = [c for c in ex_cases if c["name"] in ("A_same_SFT_1", "A_diff_SFT_to_PPO_s2_1", "A_same_PPO_s2_0", "A_diff_SFT_to_PPO_s3_1")]
    exs = []
    for i, c in enumerate(pick):
        ans = sorted(t for t, v in c["truth"].items() if v)
        exs.append(f"Example {i + 1}:\n{HEAD}\n{table(c)}\nAnswer: truly regressed tasks = {ans}")
    exs = "\n\n".join(exs)
    jobs = []
    for c in cases:
        jobs.append(("P_" + c["name"], PROMPT_P.format(head=HEAD, rows=table(c)), SCHEMA_P))
        jobs.append(("F_" + c["name"], PROMPT_F.format(examples=exs, head=HEAD, rows=table(c)), T.SCHEMA_A))
    with cf.ThreadPoolExecutor(8) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: T.ask(*j), jobs)))
    # scored task instances
    items = []
    for c in cases:
        pr = res.get("P_" + c["name"], {}).get("prob_regressed")
        fl = set(res.get("F_" + c["name"], {}).get("regressed_tasks", []))
        for t, truth in c["truth"].items():
            if truth is None:
                continue
            items.append({"truth": truth, "p": (pr[t] if pr else None), "z": z_of(t, c["R"], c["N"]), "few": t in fl})

    def auc(key):
        pos = [i[key] for i in items if i["truth"] and i[key] is not None]
        neg = [i[key] for i in items if not i["truth"] and i[key] is not None]
        return round(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(len(pos) * len(neg), 1), 3)

    def recall_at(key, max_fa):
        """Best recall over thresholds whose false-alarm rate <= max_fa."""
        best = 0.0
        for thr in sorted({i[key] for i in items if i[key] is not None}):
            fa = sum(1 for i in items if not i["truth"] and i[key] >= thr) / max(sum(1 for i in items if not i["truth"]), 1)
            rc = sum(1 for i in items if i["truth"] and i[key] >= thr) / max(sum(1 for i in items if i["truth"]), 1)
            if fa <= max_fa:
                best = max(best, rc)
        return round(best, 3)

    tp = sum(1 for i in items if i["truth"] and i["few"]); fn = sum(1 for i in items if i["truth"] and not i["few"])
    fp = sum(1 for i in items if not i["truth"] and i["few"]); tn = sum(1 for i in items if not i["truth"] and not i["few"])
    rep = {"P agent probabilities": {"AUC": auc("p"), "recall@FA<=2%": recall_at("p", 0.02), "recall@FA<=5%": recall_at("p", 0.05),
                                     "recall@FA<=10%": recall_at("p", 0.10)},
           "z statistic (rule)": {"AUC": auc("z"), "recall@FA<=2%": recall_at("z", 0.02), "recall@FA<=5%": recall_at("z", 0.05),
                                  "recall@FA<=10%": recall_at("z", 0.10)},
           "F agent few-shot (binary)": {"false_alarm": round(fp / max(fp + tn, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                                         "tp": tp, "fn": fn, "fp": fp, "tn": tn},
           "errors": [k for k, v in res.items() if "error" in v]}
    out = HERE / "agent_capability_20261105_fix" / "report.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

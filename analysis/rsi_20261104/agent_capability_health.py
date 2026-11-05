#!/usr/bin/env python3
"""Capability 6 (2026-11-05 17:2x): read a training curve -> improving / flat / collapsing.
Curves = per-iteration training success (%) of finished runs; truth from the final paired evals (see queue_log)."""
import concurrent.futures as cf, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.argv = ["x", str(HERE / "agent_capability_20261105_health")]
import agent_capability_tests as T  # noqa: E402
AR3 = [40, 40, 37, 41, 49, 40, 52, 48, 47, 51, 46, 44, 34, 34, 43, 45, 44, 34, 43, 39]
RUNS = {  # name: (benchmark, curve %, truth)
    "PPO_s1": ("LIBERO-Long", [32, 28, 31, 34, 34, 34, 44, 30, 34, 32, 38, 31, 24, 34, 28, 34, 43, 34, 40, 37], "improving"),
    "CALVIN_PPO": ("CALVIN", [20, 20, 20, 28, 30, 23, 22, 30, 29, 27, 30, 23, 27, 27, 28, 33, 34, 31, 25, 31], "improving"),
    "SUB_s2": ("LIBERO-Long", [33, 35, 23, 35, 39, 35, 26, 32, 38, 35, 39, 38, 37, 33, 37, 36, 47, 34, 42, 37], "improving"),
    "PPO_s2": ("LIBERO-Long", [35, 37, 26, 38, 35, 37, 36, 36, 39, 46, 37, 32, 41, 42, 48, 50, 43, 32, 33, 38], "flat"),
    "PPO_s3": ("LIBERO-Long", [35, 34, 33, 38, 44, 34, 28, 36, 37, 43, 40, 31, 35, 35, 41, 40, 40, 30, 34, 44], "flat"),
    "RoboCasa_PPO": ("RoboCasa365", [43, 47, 48, 52, 38, 45, 42, 43, 46, 47], "flat"),
    "AR_s2": ("LIBERO-Long", [34, 37, 33, 29, 38, 29, 41, 40, 40, 29, 37, 37, 32, 40, 42, 34, 38, 29, 41, 34], "flat"),
    "AR_s3": ("LIBERO-Long", [round(100 * x / 128) for x in AR3], "flat"),
    "RSI_c0": ("LIBERO-Long (3 tasks)", [34, 34, 44, 45, 34, 38, 46, 31], "flat"),
    "RSI_c1": ("LIBERO-Long (3 tasks)", [45, 33, 35, 41, 29, 35, 37, 40], "flat"),
    "RSI_c2": ("LIBERO-Long (3 tasks)", [38, 33, 33, 20, 8, 9, 6, 30], "collapsing"),
    "ESPO_v1": ("LIBERO-Long", [38, 12, 3], "collapsing"),
    "ESPO_v2": ("LIBERO-Long", [45, 33, 38, 39, 23, 27], "collapsing"),
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["label", "confidence", "reason"],
          "properties": {"label": {"type": "string", "enum": ["improving", "flat", "collapsing"]},
                         "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                         "reason": {"type": "string", "maxLength": 300}}}
PROMPT = """You monitor reinforcement-learning fine-tuning (PPO) of a robot manipulation policy on {bench}.
Below is the training success rate (%) of each iteration so far (about 128 stochastic training episodes per iteration, so each point is noisy; evaluation uses a separate deterministic protocol).
{curve}
Is this training run improving the policy, flat (no real progress), or collapsing (the policy is getting worse)? Answer with JSON only."""


def rule(c):
    k = 5 if len(c) >= 10 else max(2, len(c) // 3)
    a, b = sum(c[:k]) / k, sum(c[-k:]) / k
    return "collapsing" if b < a - 8 else ("improving" if b > a + 5 else "flat")


jobs = [(f"H_{n}_{r}", PROMPT.format(bench=b, curve=", ".join(f"it{i + 1}: {v}" for i, v in enumerate(c))), SCHEMA)
        for n, (b, c, _) in RUNS.items() for r in range(2)]
with cf.ThreadPoolExecutor(8) as ex:
    res = dict(zip([j[0] for j in jobs], ex.map(lambda j: T.ask(*j), jobs)))
rows, acc_a, acc_r, n = [], 0, 0, 0
for name, (b, c, truth) in RUNS.items():
    labs = [res.get(f"H_{name}_{r}", {}).get("label") for r in range(2)]
    rl = rule(c)
    rows.append({"run": name, "truth": truth, "agent": labs, "rule": rl})
    acc_a += sum(l == truth for l in labs); acc_r += 2 * (rl == truth); n += 2
rep = {"agent_accuracy": f"{acc_a}/{n}", "rule_accuracy": f"{acc_r // 2}/{n // 2}", "rows": rows,
       "errors": [k for k, v in res.items() if "error" in v]}
(HERE / "agent_capability_20261105_health" / "report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))

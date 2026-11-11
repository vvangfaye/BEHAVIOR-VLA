#!/usr/bin/env python3
"""Offline capability tests for an LLM "RL training manager" (2026-11-05 17:0x; user: results within 30 min, call Sol).

Same model and restrictions as sol_ver.py (gpt-6.1-sol via the local Codex CLI, tools/web/MCP disabled, JSON schema,
shared daily ledger). Text only: the agent sees per-task success counts and public task instructions, never frames.

A  noise vs real regression: per-task counts of two batches (~13 episodes per task, like one training rollout).
   Cases are subsamples of our 500-episode paired evals: same policy twice (truth: no regression) or SFT vs a PPO
   policy with known per-task drops (truth from the full 50-episode rates), plus the five real re-measurements of the
   same anchor policy from the AR v2.3 seed-3 run (training stats). Compared with fixed rules.
B  learnability: predict per-task change after PPO (20 iters x 128 episodes) from SFT per-task rates;
   truth = mean change over PPO seeds 1-3.
C  learning-rate choice in the RSI round-1 situation (truth: the default 5e-6 beat 1.25e-6 on the held-out tasks).
Usage: agent_capability_tests.py <out_dir>
"""
import concurrent.futures as cf
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sol_ver  # noqa: E402  (ledger, restrictions, model settings)

RES = HERE.parent / "libero_feasibility_20261103" / "results"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "agent_capability_20261105")
OUT.mkdir(parents=True, exist_ok=True)
PAT = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")
LOGS = {"SFT": "eval_sft_c10", "PPO_s1": "eval_balanced_s20_c10", "PPO_s2": "eval_seed2_5090_s20_c10",
        "PPO_s3": "eval_balanced_seed3_s20_c10", "SUB_s2": "cluster_libero_seed2_subgoal_eval_s20"}


def episodes(tag):
    out = {}
    for line in open(RES / tag / "console.log", errors="replace"):
        m = PAT.search(line)
        if m:
            out.setdefault(int(m.group(1)), []).append(m.group(3) == "True")
    return out


def ask(name, prompt, schema, timeout=300):
    ev = OUT / name
    ev.mkdir(parents=True, exist_ok=True)
    if (ev / "answer.json").exists():
        return json.loads((ev / "answer.json").read_text())
    if not sol_ver._ledger_reserve():
        return {"error": "daily_cap_reached"}
    env, restr = sol_ver._setup()
    (ev / "schema.json").write_text(json.dumps(schema))
    (ev / "prompt.txt").write_text(prompt)
    argv = [sol_ver.CODEX, "exec", "-", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
            "--output-schema", str(ev / "schema.json"), "--output-last-message", str(ev / "last.txt"),
            "--color", "never", "--json", "--model", sol_ver.MODEL,
            "--config", "model_reasoning_effort=" + json.dumps(sol_ver.EFFORT),
            "--config", "service_tier=" + json.dumps(sol_ver.TIER), *restr]
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=prompt, env=env, cwd=ev, capture_output=True, text=True, timeout=timeout)
        err = (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        err = "timeout"
    try:
        ans = json.loads((ev / "last.txt").read_text())
        ans["_seconds"] = round(time.time() - t0, 1)
        (ev / "answer.json").write_text(json.dumps(ans, indent=1))
        return ans
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse_failed {type(e).__name__}", "stderr": err}


# ------------------------------------------------------------------ A: noise vs real regression
SCHEMA_A = {"type": "object", "additionalProperties": False, "required": ["regressed_tasks", "reason"],
            "properties": {"regressed_tasks": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}},
                           "reason": {"type": "string", "maxLength": 400}}}
PROMPT_A = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy on 10 tasks.
Below are per-task success counts from two batches of episodes: a reference batch R (an earlier checkpoint) and a new batch N (the current checkpoint). Each batch has only about 13 episodes per task, so the counts are noisy.
Decide for which tasks the policy's TRUE success rate has really dropped from R to N (a real regression, not sampling noise). Only list tasks you believe truly regressed; an empty list is fine.

task | R successes/episodes | N successes/episodes
{rows}

Answer with JSON only."""


def counts(eps, n, rng):
    sel = rng.sample(eps, min(n, len(eps)))
    return len(sel), sum(sel)


def build_A(seed=0):
    rng = random.Random(seed)
    ep = {k: episodes(v) for k, v in LOGS.items()}
    full = {k: {t: sum(v[t]) / len(v[t]) for t in v} for k, v in ep.items()}
    cases = []
    # same policy, two disjoint-ish subsamples: truth = no regression on any task
    for pol in ("SFT", "PPO_s1", "PPO_s2", "SUB_s2"):
        for r in range(2):
            R, N = {}, {}
            for t in range(10):
                eps = ep[pol][t][:]
                rng.shuffle(eps)
                R[t] = (13, sum(eps[:13]))
                N[t] = (13, sum(eps[13:26]))
            cases.append({"name": f"A_same_{pol}_{r}", "R": R, "N": N, "truth": {t: False for t in range(10)}})
    # different policies with known drops: truth per task from the full 50-episode rates
    for ref, new in (("SFT", "PPO_s2"), ("SFT", "PPO_s3"), ("SUB_s2", "PPO_s2"), ("PPO_s1", "PPO_s3")):
        for r in range(2):
            R = {t: counts(ep[ref][t], 13, rng) for t in range(10)}
            N = {t: counts(ep[new][t], 13, rng) for t in range(10)}
            truth = {}
            for t in range(10):
                d = full[ref][t] - full[new][t]
                truth[t] = True if d >= 0.25 else (False if d <= 0.06 else None)  # None = ambiguous, not scored
            cases.append({"name": f"A_diff_{ref}_to_{new}_{r}", "R": R, "N": N, "truth": truth})
    # real training-rollout re-measurements of the same anchor policy (AR v2.3 seed 3): truth = no regression
    rows = [json.loads(l) for l in open("/mnt/data1/faye/rsi_ar_logs/ppo_balanced_seed3_arv23/ar_decisions.jsonl")]
    anchor = rows[6]["tasks"]  # check 7 (12:51, 52/128) became the anchor
    for i in (9, 11, 13, 15, 18):  # re-measurements after rollbacks (remeasure=1)
        if rows[i].get("ar/remeasure") == 1.0:
            cases.append({"name": f"A_real_remeasure_{i + 1}", "R": {int(t): tuple(v) for t, v in anchor.items()},
                          "N": {int(t): tuple(v) for t, v in rows[i]["tasks"].items()}, "truth": {t: False for t in range(10)}})
    return cases


def rule_flags(R, N, kind):
    out = []
    K = sum(1 for t in range(10) if R.get(t, (0, 0))[0] >= 8 and N.get(t, (0, 0))[0] >= 8)
    zc = {"z164": 1.64, "bonf": 2.05 if K >= 10 else 1.96}.get(kind, None)
    for t in range(10):
        (n1, s1), (n2, s2) = R.get(t, (0, 0)), N.get(t, (0, 0))
        if n1 < 1 or n2 < 1:
            continue
        p1, p2 = s1 / n1, s2 / n2
        d = p1 - p2
        if kind == "delta":
            if d > 0.15:
                out.append(t)
            continue
        p = (s1 + s2) / (n1 + n2)
        se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
        if d > 0.15 and d / se > zc:
            out.append(t)
    return out


# ------------------------------------------------------------------ B: learnability
SFT = [23, 40, 32, 49, 24, 0, 31, 15, 8, 16]
FINAL = {"s1": [20, 39, 31, 50, 25, 28, 35, 29, 8, 16], "s2": [25, 20, 35, 49, 30, 48, 12, 15, 2, 8],
         "s3": [27, 22, 19, 37, 35, 49, 23, 2, 8, 7]}
SCHEMA_B = {"type": "object", "additionalProperties": False, "required": ["predicted_change", "at_risk", "reason"],
            "properties": {"predicted_change": {"type": "array", "items": {"type": "number", "minimum": -50, "maximum": 50},
                                                "minItems": 10, "maxItems": 10},
                           "at_risk": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}},
                           "reason": {"type": "string", "maxLength": 500}}}
PROMPT_B = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy (a vision-language-action model) on the 10 LIBERO-Long tasks below.
The starting policy (after supervised fine-tuning) has these success counts out of 50 evaluation episodes per task:
{rows}
Planned training: multi-task PPO with a sparse success reward, 20 iterations x 128 training episodes in total (about 13 episodes per task per iteration), one policy for all tasks.
Predict, for each task 0..9, the change in successes out of 50 after training (positive = improves, negative = gets worse), and list the tasks most at risk of getting worse.
Answer with JSON only."""

# ------------------------------------------------------------------ C: learning-rate choice
SCHEMA_C = {"type": "object", "additionalProperties": False, "required": ["choice", "reason"],
            "properties": {"choice": {"type": "string", "enum": ["1.25e-6", "5e-6", "2e-5"]},
                           "reason": {"type": "string", "maxLength": 500}}}
PROMPT_C = """You help manage reinforcement-learning fine-tuning of a robot manipulation policy (PPO, 8 iterations x 128 episodes).
Three learning rates were tried on three source tasks (all three are "put two objects in the basket" tasks):
- before training: 56.2% success on the 48 validation episodes of the source tasks
- lr 1.25e-6: 60.4%
- lr 5e-6 (the default): 60.4%
- lr 2e-5: 16.7% (training collapsed)
You must now pick ONE learning rate to train, with the same budget, on four NEW tasks whose starting success counts out of 50 are:
- turn on the stove and put the moka pot on it: 32/50
- put the black bowl in the bottom drawer of the cabinet and close it: 49/50
- pick up the book and place it in the back compartment of the caddy: 0/50
- put the yellow and white mug in the microwave and close it: 16/50
Which learning rate do you choose for the new tasks? Answer with JSON only."""


def main():
    jobs = []
    casesA = build_A()
    for c in casesA:
        rows = "\n".join(f"{t} | {c['R'].get(t, (0, 0))[1]}/{c['R'].get(t, (0, 0))[0]} | {c['N'].get(t, (0, 0))[1]}/{c['N'].get(t, (0, 0))[0]}"
                         for t in range(10))
        jobs.append((c["name"], PROMPT_A.format(rows=rows), SCHEMA_A))
    rowsB = "\n".join(f"task {t}: {sol_ver.LIBERO10[t]} -> {SFT[t]}/50" for t in range(10))
    for r in range(3):
        jobs.append((f"B_learnability_{r}", PROMPT_B.format(rows=rowsB), SCHEMA_B))
    for r in range(3):
        jobs.append((f"C_lr_choice_{r}", PROMPT_C, SCHEMA_C))
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: ask(*j), jobs)))
    # ---- score A
    def score(flags_of):
        fp = tn = tp = fn = 0
        same_any = same_n = 0
        for c in casesA:
            flags = set(flags_of(c))
            for t, truth in c["truth"].items():
                if truth is None:
                    continue
                if truth:
                    tp += t in flags
                    fn += t not in flags
                else:
                    fp += t in flags
                    tn += t not in flags
            if all(v is False for v in c["truth"].values()):
                same_n += 1
                same_any += bool(flags)
        return {"false_alarm_per_task": round(fp / max(fp + tn, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn, "no-change cases with any alarm": f"{same_any}/{same_n}"}
    agent_flags = lambda c: res.get(c["name"], {}).get("regressed_tasks", [])  # noqa: E731
    reportA = {"agent (gpt-6.1-sol)": score(agent_flags),
               "rule: drop > 0.15": score(lambda c: rule_flags(c["R"], c["N"], "delta")),
               "rule: drop > 0.15 and z > 1.64": score(lambda c: rule_flags(c["R"], c["N"], "z164")),
               "rule: AR v2.1 Bonferroni": score(lambda c: rule_flags(c["R"], c["N"], "bonf"))}
    # ---- score B
    truth = [sum(FINAL[s][t] for s in FINAL) / 3 - SFT[t] for t in range(10)]

    def spearman(a, b):
        ra = {i: r for r, i in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
        rb = {i: r for r, i in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
        n = len(a)
        return round(1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1)), 3)
    reportB = {"truth_mean_change": [round(x, 1) for x in truth],
               "heuristic headroom (50-SFT)": spearman([50 - s for s in SFT], truth),
               "heuristic p(1-p)": spearman([s * (50 - s) for s in SFT], truth)}
    for r in range(3):
        a = res.get(f"B_learnability_{r}", {})
        if "predicted_change" in a:
            reportB[f"agent run {r}"] = {"spearman": spearman(a["predicted_change"], truth),
                                         "pred": [round(x, 1) for x in a["predicted_change"]], "at_risk": a.get("at_risk")}
    reportC = {f"agent run {r}": res.get(f"C_lr_choice_{r}", {}).get("choice") for r in range(3)}
    reportC["truth"] = "5e-6 (held-out tasks: 58.5% vs 49.0% for 1.25e-6, p=0.023)"
    report = {"A_noise_vs_regression": reportA, "B_learnability": reportB, "C_lr_choice": reportC,
              "calls": len(jobs), "errors": [k for k, v in res.items() if "error" in v], "minutes": round((time.time() - t0) / 60, 1)}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize an agent-managed RL run (2026-11-05) from its manager log (am/decisions.jsonl):
per iteration the training success per task (s/n), the weights the manager set, the lr multiplier / cumulative scale,
the agent's regression probabilities and its reason, plus call failures. Optionally the final paired eval per task
against reference evals (console logs with "[libero eval] task_id=T, trial_id=K, success=B" lines).
Usage: analyze_manager_run.py <decisions.jsonl> [--eval run_console.log] [--ref name=console.log ...] [--out summary.md]"""
import json
import re
import sys
from pathlib import Path

PAT = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")


def per_task(console):
    res = {}
    for m in PAT.finditer(Path(console).read_text(errors="replace")):
        res[(int(m.group(1)), int(m.group(2)))] = m.group(3) == "True"
    out = [0] * 10
    for (t, _), s in res.items():
        out[t] += s
    return out, len(res)


def main():
    a = sys.argv[1:]
    dec = Path(a[0])
    ev = a[a.index("--eval") + 1] if "--eval" in a else None
    refs = [a[i + 1] for i, x in enumerate(a) if x == "--ref"]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else None
    rows = [json.loads(line) for line in dec.read_text().splitlines() if line.strip()]
    L = [f"# Manager run: {dec.parent.parent.name} ({rows[0]['mode'] if rows else '?'}), {len(rows)} decisions", ""]
    L.append("| it | time | train success per task (s/n) | total | weights after | lr x | lr scale | agent |")
    L.append("|---|---|---|---|---|---|---|---|")
    fails = 0
    for r in rows:
        c = r.get("counts", {})
        cells = " ".join(f"{c.get(str(t), [0, 0])[1]}/{c.get(str(t), [0, 0])[0]}" for t in range(10))
        tot_n = sum(v[0] for v in c.values()); tot_s = sum(v[1] for v in c.values())
        ag = r.get("agent")
        if ag is None:
            note = ""
        elif "error" in ag:
            fails += 1
            note = "ERROR " + str(ag["error"])
        else:
            pr = ag.get("prob_regressed", [])
            hi = [f"t{t}:{p:.2f}" for t, p in enumerate(pr) if p >= 0.5]
            note = (f"p>=0.5 {' '.join(hi)}; " if hi else "") + str(ag.get("reason", ""))[:120].replace("|", "/")
        w = " ".join(f"{x:.2f}" for x in r.get("weights", []))
        L.append(f"| {r.get('iteration')} | {r.get('time')} | {cells} | {tot_s}/{tot_n} | {w} | {r.get('lr_multiplier')} | "
                 f"{r.get('lr_scale')} | {note} |")
    L += ["", f"agent call failures: {fails}/{sum(1 for r in rows if 'agent' in r)}"]
    if ev:
        e, n = per_task(ev)
        L += ["", f"## Final eval ({n} episodes): {sum(e)}/{n} = {100 * sum(e) / max(n, 1):.1f}%", "",
              "| run | total | " + " | ".join(f"t{t}" for t in range(10)) + " |", "|---|---|" + "---|" * 10,
              f"| this run | {sum(e)} | " + " | ".join(str(x) for x in e) + " |"]
        for spec in refs:
            name, path = spec.split("=", 1)
            r, _ = per_task(path)
            L.append(f"| {name} | {sum(r)} | " + " | ".join(str(x) for x in r) + " |")
        L.append("")
        for spec in refs:
            name, path = spec.split("=", 1)
            r, _ = per_task(path)
            diff = [x - y for x, y in zip(e, r)]
            L.append(f"vs {name}: total {sum(e) - sum(r):+d}; tasks worse by >=5: "
                     f"{[t for t, d in enumerate(diff) if d <= -5]}; better by >=5: {[t for t, d in enumerate(diff) if d >= 5]}")
    text = "\n".join(L)
    if out:
        out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

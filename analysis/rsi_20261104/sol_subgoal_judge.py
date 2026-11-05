#!/usr/bin/env python3
"""Capability tests 2/3 (2026-11-05): can the agent (gpt-6.1-sol) judge SUBGOAL completion from frames?

Input: an eval frame dump made with +env.eval.dump_goal_preds=True (step<t>.png + step<t>.json per frame; the json
holds every BDDL goal conjunct with its simulator truth value at that frame). For each episode the agent sees 6 frames
(same picker as sol_ver.py), the instruction and the numbered subgoal list (written in plain English), and answers
which subgoals hold in the FINAL frame, whether the task succeeded, and the progress. Truth = the final frame's json.
Metrics: per-subgoal accuracy and kappa, exact localization of what is missing on partial failures, progress
correlation with the true satisfied fraction, success agreement.
Success truth = sim success at any time (eval runs with ignore_terminations) from --console, else any dumped frame
with every conjunct true. Baseline for localization: the most common truth vector of the other failures of the task.
Usage: sol_subgoal_judge.py <frames_root> <out_dir> [--workers 8] [--console <eval console.log>]"""
import concurrent.futures as cf
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sol_ver  # noqa: E402

TEXT = {
    "in alphabet_soup_1 basket_1_contain_region": "the alphabet soup can is inside the basket",
    "in tomato_sauce_1 basket_1_contain_region": "the tomato sauce is inside the basket",
    "in cream_cheese_1 basket_1_contain_region": "the cream cheese box is inside the basket",
    "in butter_1 basket_1_contain_region": "the butter is inside the basket",
    "turnon flat_stove_1": "the stove is turned on",
    "on moka_pot_1 flat_stove_1_cook_region": "a moka pot is on the stove",
    "on moka_pot_2 flat_stove_1_cook_region": "the second moka pot is on the stove",
    "close white_cabinet_1_bottom_region": "the bottom drawer of the cabinet is closed",
    "in akita_black_bowl_1 white_cabinet_1_bottom_region": "the black bowl is inside the bottom drawer of the cabinet",
    "on porcelain_mug_1 plate_1": "the white mug is on the (left) plate",
    "on white_yellow_mug_1 plate_2": "the yellow-and-white mug is on the right plate",
    "in black_book_1 desk_caddy_1_back_contain_region": "the book is in the back compartment of the caddy",
    "on chocolate_pudding_1 living_room_table_plate_right_region": "the chocolate pudding is to the right of the plate",
    "in white_yellow_mug_1 microwave_1_heating_region": "the yellow-and-white mug is inside the microwave",
    "close microwave_1": "the microwave door is closed",
}
PROMPT = """You are a strict evaluator of robot manipulation episodes recorded in a tabletop simulator.
The robot was given this instruction: "{instr}"
The task is complete when ALL of these conditions hold at the same time:
{conds}
The {n} attached images are frames from a fixed third-person camera, in chronological order; the last image is the final state.
{note}Using only the images, report for EACH condition (same order) whether it holds in the FINAL image, whether the whole task was completed at some point, and the task progress at the end from 0 to 1. Answer with JSON only."""


def schema(k):
    return {"type": "object", "additionalProperties": False, "required": ["conditions_hold", "success", "progress", "reason"],
            "properties": {"conditions_hold": {"type": "array", "items": {"type": "boolean"}, "minItems": k, "maxItems": k},
                           "success": {"type": "boolean"},
                           "progress": {"type": "number", "minimum": 0, "maximum": 1},
                           "reason": {"type": "string", "maxLength": 300}}}


def judge(ep: Path, out: Path):
    m = re.match(r"task(\d+)_trial(\d+)", ep.name)
    tid = int(m.group(1))
    pngs = sorted(ep.glob("step*.png"))
    if not pngs:
        return {"episode": ep.name, "error": "no_frames"}
    if not pngs[-1].with_suffix(".json").exists():
        return {"episode": ep.name, "error": "no_goal_preds"}
    frames = sol_ver.pick_frames(ep)  # same 6-frame picker as the success verifier
    final = json.loads(pngs[-1].with_suffix(".json").read_text())
    preds = final["preds"]
    truth = [bool(v) for _, v in preds]
    conds = "\n".join(f"{i + 1}. {TEXT.get(p, p)}" for i, (p, _) in enumerate(preds))
    ev = out / ep.name
    ev.mkdir(parents=True, exist_ok=True)
    if (ev / "answer.json").exists():
        ans = json.loads((ev / "answer.json").read_text())
    else:
        if not sol_ver._ledger_reserve():
            return {"episode": ep.name, "error": "daily_cap_reached"}
        env, restr = sol_ver._setup()
        sch = schema(len(preds))
        (ev / "schema.json").write_text(json.dumps(sch))
        prompt = PROMPT.format(instr=sol_ver.LIBERO10[tid], conds=conds, n=len(frames), note=NOTE[0])
        (ev / "prompt.txt").write_text(prompt)
        argv = [sol_ver.CODEX, "exec", "-", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                "--output-schema", str(ev / "schema.json"), "--output-last-message", str(ev / "last.txt"),
                "--color", "never", "--json", "--model", sol_ver.MODEL,
                "--config", "model_reasoning_effort=" + json.dumps(sol_ver.EFFORT),
                "--config", "service_tier=" + json.dumps(sol_ver.TIER), *restr]
        for f in frames:
            argv += ["--image", str(f)]
        t0 = time.time()
        try:
            subprocess.run(argv, input=prompt, env=env, cwd=ev, capture_output=True, text=True, timeout=300)
            ans = json.loads((ev / "last.txt").read_text())
            if len(ans.get("conditions_hold", [])) != len(preds):
                return {"episode": ep.name, "error": "wrong_length"}
            ans["_seconds"] = round(time.time() - t0, 1)
            (ev / "answer.json").write_text(json.dumps(ans, indent=1))
        except Exception as e:  # noqa: BLE001
            return {"episode": ep.name, "error": f"{type(e).__name__}"}
    # success = completed at ANY time (eval runs with ignore_terminations, so the final frame can be after a success):
    # sim success from the console when given, else any dumped frame with every goal conjunct true
    if ep.name in SIM_SUCCESS:
        succ_truth = SIM_SUCCESS[ep.name]
    else:
        succ_truth = any(all(v for _, v in json.loads(j.read_text())["preds"]) for j in sorted(ep.glob("step*.json")))
    return {"episode": ep.name, "task": tid, "final_step": int(pngs[-1].stem[4:]), "preds": [p for p, _ in preds],
            "truth": truth, "pred": [bool(x) for x in ans["conditions_hold"]],
            "success_truth": bool(succ_truth), "success_pred": bool(ans["success"]), "progress": float(ans["progress"])}


SIM_SUCCESS = {}
NOTE = [""]
VIEW_NOTE = ("Left and right in the instruction and in the conditions are from the robot's point of view. The camera faces "
             "the robot, so the robot's left appears on the RIGHT side of the images and its right on the LEFT side.\n")
CONSOLE_PAT = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")


def score(res):
    ok = [r for r in res if "error" not in r]
    if not ok:
        return {"episodes": 0, "errors": res[:5]}
    pairs = [(t, p) for r in ok for t, p in zip(r["truth"], r["pred"])]
    n = len(pairs)
    acc = sum(t == p for t, p in pairs) / max(n, 1)
    pt = sum(t for t, _ in pairs) / max(n, 1)
    pp = sum(p for _, p in pairs) / max(n, 1)
    pe = pt * pp + (1 - pt) * (1 - pp)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    tp = sum(t and p for t, p in pairs); fp = sum((not t) and p for t, p in pairs)
    fn = sum(t and (not p) for t, p in pairs)
    partial = [r for r in ok if any(r["truth"]) and not all(r["truth"])]
    loc = sum(r["truth"] == r["pred"] for r in partial)
    fails = [r for r in ok if not all(r["truth"])]
    loc_all = sum(r["truth"] == r["pred"] for r in fails)

    def loo_mode(r):
        """Baseline without images: the most common truth vector among the OTHER failures of the same task."""
        from collections import Counter

        c = Counter(tuple(o["truth"]) for o in fails if o["task"] == r["task"] and o is not r)
        return list(c.most_common(1)[0][0]) if c else [False] * len(r["truth"])

    loc_base = sum(r["truth"] == loo_mode(r) for r in fails)
    fr = [sum(r["truth"]) / len(r["truth"]) for r in ok]
    pg = [r["progress"] for r in ok]
    mx, my = sum(fr) / len(fr), sum(pg) / len(pg)
    cov = sum((a - mx) * (b - my) for a, b in zip(fr, pg))
    corr = cov / math.sqrt(max(sum((a - mx) ** 2 for a in fr) * sum((b - my) ** 2 for b in pg), 1e-12))
    succ = sum(r["success_truth"] == r["success_pred"] for r in ok) / max(len(ok), 1)
    by_pred = {}
    for r in ok:
        for p, t, q in zip(r["preds"], r["truth"], r["pred"]):
            v = by_pred.setdefault(p, [0, 0])
            v[0] += 1
            v[1] += t == q
    rep = {"episodes": len(ok), "errors": [r for r in res if "error" in r][:5], "subgoal_instances": n,
           "subgoal_accuracy": round(acc, 3), "subgoal_kappa": round(kappa, 3),
           "precision(holds)": round(tp / max(tp + fp, 1), 3), "recall(holds)": round(tp / max(tp + fn, 1), 3),
           "partial_failures": len(partial), "exact_localization_on_partial": f"{loc}/{len(partial)}",
           "failures": len(fails), "exact_localization_on_failures": f"{loc_all}/{len(fails)}",
           "baseline_localization_on_failures(task mode, no images)": f"{loc_base}/{len(fails)}",
           "progress_vs_true_fraction_pearson": round(corr, 3), "success_agreement": round(succ, 3),
           "accuracy_by_condition": {TEXT.get(p, p): f"{v[1]}/{v[0]}" for p, v in sorted(by_pred.items())}}
    return rep


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()  # absolute: codex runs with cwd=<episode dir>
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    if "--console" in sys.argv:  # per-episode sim success (success at any time) from the eval console log
        for m in CONSOLE_PAT.finditer(Path(sys.argv[sys.argv.index("--console") + 1]).read_text(errors="replace")):
            SIM_SUCCESS[f"task{m.group(1)}_trial{m.group(2)}"] = m.group(3) == "True"
    out.mkdir(parents=True, exist_ok=True)
    eps = sorted(d for d in root.iterdir() if d.is_dir() and re.match(r"task\d+_trial\d+", d.name))
    if "--tasks" in sys.argv:  # e.g. --tasks 4,6
        keep = {int(t) for t in sys.argv[sys.argv.index("--tasks") + 1].split(",")}
        eps = [d for d in eps if int(re.match(r"task(\d+)", d.name).group(1)) in keep]
    if "--viewpoint-note" in sys.argv:
        NOTE[0] = VIEW_NOTE
    sol_ver._setup()  # login check + restriction flags once, before the threads
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda e: judge(e, out), eps))
    (out / "episodes.json").write_text(json.dumps(res, indent=1))
    if "--merge-with" in sys.argv:  # replace these episodes in an earlier run's episodes.json and score the union
        base = json.loads(Path(sys.argv[sys.argv.index("--merge-with") + 1]).read_text())
        new = {r["episode"]: r for r in res}
        res = [new.get(r["episode"], r) for r in base]
    rep = score(res)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

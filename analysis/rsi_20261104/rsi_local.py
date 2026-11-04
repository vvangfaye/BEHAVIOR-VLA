#!/usr/bin/env python3
"""Local (2 x RTX 6000 Ada) sequential executor of the Physical-RSI outer loop, same protocol as rsi_loop.py
(folds, knob, MULT, Ver = mean training success of the last 3 iterations, seeds 42/0) but every run executes on the
local machine, one RLinf job at a time (RLinf attaches to any running Ray cluster). Used when the cluster Spot queue
does not schedule the cluster loop. All runs of one loop stay on one platform (no hardware confound).

Order: round-1 candidates -> Ver -> Phi_1 -> EE(Phi_0), EE(Phi_1) + evals (L4 result) -> round-2 candidates -> Ver ->
       Phi_2 -> EE(Phi_2) + eval (L5 evidence). Resumable: finished runs (wall.json + checkpoint) are skipped.
Usage: rsi_local.py run --base ppo [--state rsi_state_ppo_local.json] [--iters 8]
"""
import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LF = HERE.parent / "libero_feasibility_20261103"
spec = importlib.util.spec_from_file_location("rsi_loop", HERE / "rsi_loop.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
VENV_PY = R.VENV_PY


def busy():
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    return any(l.startswith(("python examples/embodiment/train_embodied_agent.py", "python evaluations/eval_embodied_agent.py"))
               for l in out.splitlines())


def wait_idle():
    # HOLD_LOCAL: another local experiment (e.g. espo_c2_local.sh) owns the GPUs; wait until it removes the file
    while busy() or (HERE / "HOLD_LOCAL").exists():
        time.sleep(60)
    time.sleep(20)
    subprocess.run([str(Path(VENV_PY).parent / "ray"), "stop", "--force"], capture_output=True)
    time.sleep(5)


def ckpt(tag, step):
    hits = list((LF / "results" / tag).glob(f"**/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt"))
    return str(hits[0]) if hits else None


def local_curve(tag):
    r = subprocess.run([VENV_PY, str(R.CP / "tb_last.py"), str(LF / "results" / tag)], capture_output=True, text=True)
    parts = r.stdout.split()
    return [float(x) / 100 for x in parts[1:]] if parts and parts[0] != "0" else []


def train(st, tag, fold, knob_value):
    rec = st["runs"].setdefault(tag, {"fold": fold, "knob": R.fmt(knob_value)})
    if ckpt(tag, st["iters"]) and (LF / "results" / tag / "wall.json").exists():
        rec["curve"] = rec.get("curve") or local_curve(tag)
        return rec
    ov = R.train_overrides(st, fold, knob_value)
    if st["base"] == "ppo":  # local launcher layout keeps the local micro batch; drop the cluster-only cache cap
        ov = [o for o in ov if not o.startswith("+actor.prefix_cache_max_gib")]
    wait_idle()
    if ckpt(tag, st["iters"]) and (LF / "results" / tag / "wall.json").exists():  # finished while we waited
        rec["curve"] = rec.get("curve") or local_curve(tag)
        R.save(st)
        return rec
    R.log(st, f"local START {tag} fold {fold} knob {R.fmt(knob_value)}")
    R.save(st)
    env = dict(os.environ, CONFIG="libero_10_ppo_pi05_local2gpu_all")
    t0 = time.time()
    rc = subprocess.run(["bash", str(LF / "run_ppo_local_libero2.sh"), tag] + ov, env=env, cwd=LF,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    rec.update(rc=rc, minutes=round((time.time() - t0) / 60), curve=local_curve(tag), ckpt=ckpt(tag, st["iters"]))
    R.log(st, f"local END {tag} rc={rc} after {rec['minutes']} min, curve {rec['curve']}")
    R.save(st)
    return rec


def evaluate(st, tag, train_tag):
    rec = st["evals"].setdefault(tag, {"train": train_tag})
    log = LF / "results" / tag / "console.log"
    if (LF / "results" / tag / "wall.json").exists() and log.exists():
        res = R.ee_eval_result_path(log)
        rec.update(n=len(res), succ=sum(res.values()))
        return rec
    ck = ckpt(train_tag, st["iters"])
    if not ck:
        R.log(st, f"local eval {tag}: no checkpoint for {train_tag}")
        return rec
    wait_idle()
    R.log(st, f"local START eval {tag} of {train_tag}")
    R.save(st)
    env = dict(os.environ, EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10 +env.eval.task_id_filter=["
               + ",".join(map(str, R.FOLDS["EE"])) + "]")
    rc = subprocess.run(["bash", str(LF / "run_eval_ckpt.sh"), tag, ck, "all", "20", "10"], env=env, cwd=LF,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    res = R.ee_eval_result_path(log)
    rec.update(rc=rc, n=len(res), succ=sum(res.values()))
    R.log(st, f"local END eval {tag} rc={rc}: EE {rec['succ']}/{rec['n']}")
    R.save(st)
    return rec


def ver(st, tag, fold):
    """External in-loop verifier: eval the candidate's final policy on 30 source-fold episodes with frame dumping,
    then gpt-6.1-sol judges each episode from the frames (sol_ver.py). Simulator success of the same episodes is kept
    only for the verifier-agreement analysis; selection never reads it."""
    import importlib.util as iu
    rec = st["runs"][tag]
    if rec.get("ver_sol") is not None:
        return rec
    vtag = f"{tag}_ver"
    frames = Path("/mnt/data1/faye/rsi_frames") / vtag  # no spaces: run_eval_ckpt.sh word-splits EVAL_EXTRA_OVERRIDES
    if not (LF / "results" / vtag / "wall.json").exists():
        ck = ckpt(tag, st["iters"])
        if not ck:
            R.log(st, f"ver {tag}: no checkpoint")
            return rec
        wait_idle()
        R.log(st, f"local START ver-eval {vtag} ({fold}, 48 episodes, frame dump)")
        R.save(st)
        env = dict(os.environ, EVAL_EXTRA_OVERRIDES=" ".join([
            "rollout.model.num_action_chunks=10", "+env.eval.task_id_filter=[" + ",".join(map(str, R.FOLDS[fold])) + "]",
            f"+env.eval.dump_frames_dir={frames}", "+env.eval.dump_frames_every=70"]))
        subprocess.run(["bash", str(LF / "run_eval_ckpt.sh"), vtag, ck, "all", "24", "2"], env=env, cwd=LF,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sim = R.ee_eval_result_path(LF / "results" / vtag / "console.log")
    spec_ = iu.spec_from_file_location("sol_ver", HERE / "sol_ver.py")
    SV = iu.module_from_spec(spec_)
    spec_.loader.exec_module(SV)
    summary, res = SV.judge_all(frames, LF / "results" / vtag / "sol", workers=6)
    agree = [int(r["success"]) == int(sim.get(f"{r['task_id']}:{r['episode'].split('trial')[1]}", -1))
             for r in res if "error" not in r and f"{r['task_id']}:{r['episode'].split('trial')[1]}" in sim]
    rec.update(ver_sol=summary["success_rate"], ver_sol_progress=summary["progress_mean"], ver_sol_n=summary["n_ok"],
               ver_sol_errors=summary["errors"], ver_sim=(sum(sim.values()) / len(sim)) if sim else None,
               ver_agreement=(sum(agree) / len(agree)) if agree else None)
    R.log(st, f"ver {tag}: Sol success {rec['ver_sol']} (progress {rec['ver_sol_progress']}, n={rec['ver_sol_n']}), "
              f"sim {rec['ver_sim']}, agreement {rec['ver_agreement']}, errors {rec['ver_sol_errors']}")
    R.save(st)
    return rec


def select(st, k, cands):
    train_scores = [R.ver_score(c.get("curve", [])) for c in cands]
    sol = [c.get("ver_sol") for c in cands]
    if all(v is not None for v in sol):
        key = [(0.5 * c["ver_sol"] + 0.5 * (c.get("ver_sol_progress") or 0.0), c["ver_sol"]) for c in cands]
        rule = "external verifier (gpt-6.1-sol: 0.5*success + 0.5*progress, then success)"
    else:
        key = [(v, 0.0) for v in train_scores]
        rule = "fallback: training success of the last 3 iterations (verifier unavailable)"
    best = max(range(len(cands)), key=lambda i: (key[i], R.MULT[k][i] == 1.0))
    best_train = max(range(len(cands)), key=lambda i: (train_scores[i], R.MULT[k][i] == 1.0))
    st["rounds"][str(k)] = {"rule": rule, "ver_keys": key, "train_scores": train_scores, "selected": best,
                            "selected_by_train_ver": best_train, "knobs": [c["knob"] for c in cands]}
    st["phi"].append(float(cands[best]["knob"]))
    R.log(st, f"round {k}: {rule}: {key} -> Phi_{k} = {st['phi'][-1]} (training-success Ver would pick "
              f"{cands[best_train]['knob']})")
    R.save(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--base", choices=["ppo", "espo"], default="ppo")
    ap.add_argument("--espo-variant", default="v3")
    ap.add_argument("--knob", default=None)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--state", default="rsi_state_ppo_local.json")
    a = ap.parse_args()
    R.STATE = HERE / a.state
    if R.STATE.exists():
        st = R.load()
    else:
        knob = R.KNOB_CHOICES[a.knob] if (a.base == "espo" and a.knob) else R.KNOB[a.base]
        st = {"created": R.now(), "platform": "local 2x RTX 6000 Ada (libero_10_ppo_pi05_local2gpu_all)", "base": a.base,
              "espo_variant": a.espo_variant, "iters": a.iters, "prefix": f"rsil_{a.base}", "folds": R.FOLDS,
              "mult": {str(k): v for k, v in R.MULT.items()}, "knob": knob[0], "phi": [knob[1]], "runs": {},
              "rounds": {}, "evals": {}, "log": []}
        R.save(st)
    P = st["prefix"]
    # round 1 -> Phi_1
    c1 = [train(st, f"{P}_r1_c{i}", "S1", st["phi"][0] * m) for i, m in enumerate(R.MULT[1])]
    c1 = [ver(st, f"{P}_r1_c{i}", "S1") for i in range(len(c1))]
    if len(st["phi"]) < 2:
        select(st, 1, c1)
    # EE with Phi_0 and Phi_1, then evals (L4)
    for k in (0, 1):
        train(st, f"{P}_ee_phi{k}", "EE", st["phi"][k])
        evaluate(st, f"{P}_ee_phi{k}_eval", f"{P}_ee_phi{k}")
    # round 2 -> Phi_2, EE with Phi_2 (L5 evidence)
    c2 = [train(st, f"{P}_r2_c{i}", "S2", st["phi"][1] * m) for i, m in enumerate(R.MULT[2])]
    c2 = [ver(st, f"{P}_r2_c{i}", "S2") for i in range(len(c2))]
    if len(st["phi"]) < 3:
        select(st, 2, c2)
    train(st, f"{P}_ee_phi2", "EE", st["phi"][2])
    evaluate(st, f"{P}_ee_phi2_eval", f"{P}_ee_phi2")
    R.log(st, "local RSI loop finished")
    R.save(st)


if __name__ == "__main__":
    main()

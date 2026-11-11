#!/usr/bin/env python3
"""Physical-RSI outer loop (2026-11-04): a meta-level update of the RL improver Phi, inherited across LIBERO-10 folds.

Protocol (after arXiv 2610.18842: compare Phi_k with Phi_{k-1} from the same starting generation, experience and budget,
measured on tasks the improvement procedure never reads):
  folds  S1 = [0, 1, 7]     two objects into the basket
         S2 = [4, 6, 8]     two placements (plates / stove)
         EE = [2, 3, 5, 9]  articulated + place (stove knob, drawer, microwave door) and book into caddy;
                            skill families disjoint from S1 and S2; EE is only scored, never used for selection.
  Phi    base RL algorithm (ESPO or PPO) + one configuration knob (ESPO: action-space trust region prox_coef;
         PPO: learning rate). Ver = in-loop verifier on the agent's own training rollouts: mean training success of the
         last 3 iterations on the source fold (all candidates of a round share seeds: common random numbers).
  round k = 1, 2: from the SFT policy, run Phi_{k-1} with the knob scaled by MULT[k] on S_k; Ver keeps the best -> Phi_k.
  EE     from the SFT policy, RL with Phi_0, Phi_1, Phi_2 on EE (same budget, same seeds), then the deployment-protocol
         eval (ODE, chunk 10, 4 tasks x 50 init states). L4: Phi_1 > Phi_0 on EE; L5 evidence: Phi_2 > Phi_1 > Phi_0.
State: rsi_state.json next to this file (receipts, curves, Ver scores, decisions). `step` is idempotent.

Usage: rsi_loop.py [--state rsi_state_ppo.json] init --base espo|ppo [--espo-variant v3|v4a|v4b] [--iters 8]
       rsi_loop.py step            # submit/advance whatever is ready
       rsi_loop.py status          # print the state
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP = HERE.parent / "cluster_probe_20261104"
STATE = HERE / "rsi_state.json"  # overridden by --state
VENV_PY = "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python"
SCO = "/home/faye/.local/bin/sco"
FOLDS = {"S1": [0, 1, 7], "S2": [4, 6, 8], "EE": [2, 3, 5, 9]}
MULT = {1: [0.25, 1.0, 4.0], 2: [0.5, 1.0, 2.0]}
CACHE = ["+actor.prefix_cache=True", "+actor.prefix_cache_pin=False", "+actor.prefix_cache_verify_every=64",
         "+actor.prefix_cache_max_gib=20"]
SEEDS = ["actor.seed=42", "env.train.seed=0"]
ESPO_COMMON = {"algorithm.loss_type": "espo", "+algorithm.espo.model_samples": "4", "+algorithm.espo.eta": "3.0",
               "+algorithm.espo.coef": "1.0", "+algorithm.espo.stop_frac": "0.5", "actor.model.openpi.noise_level": "0.0",
               "actor.optim.lr": "3e-6", "actor.optim.critic_warmup_steps": "12", "actor.micro_batch_size": "4"}
ESPO_VARIANT = {"v3": {}, "v4a": {"+algorithm.espo.adv_mode": "positive"},
                "v4b": {"+algorithm.espo.adv_mode": "positive", "+algorithm.espo.feature": "action"}}
KNOB = {"espo": ("+algorithm.espo.prox_coef", 1.0), "ppo": ("actor.optim.lr", 5e-6)}
KNOB_CHOICES = {"prox": ("+algorithm.espo.prox_coef", 1.0), "lr": ("actor.optim.lr", 3e-6)}  # ESPO knob options
ESPO_PROX_DEFAULT = "1.0"


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load():
    return json.loads(STATE.read_text())


def save(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def log(st, msg):
    st.setdefault("log", []).append(f"{now()} {msg}")
    print(msg, flush=True)


def fmt(v):
    return f"{v:.6g}"


def train_overrides(st, fold, knob_value):
    base = st["base"]
    ov = {"runner.max_epochs": str(st["iters"]), "runner.save_interval": str(st["iters"]),
          "+env.train.task_id_filter": "[" + ",".join(map(str, FOLDS[fold])) + "]"}
    if base == "espo":
        ov.update(ESPO_COMMON)
        ov.update(ESPO_VARIANT[st["espo_variant"]])
    key = st.get("knob") or KNOB[base][0]
    if base == "espo" and key != "+algorithm.espo.prox_coef":
        ov["+algorithm.espo.prox_coef"] = ESPO_PROX_DEFAULT
    ov[key] = fmt(knob_value)
    return CACHE + SEEDS + [f"{k}={v}" for k, v in ov.items()]


def submit(st, tag, config, overrides, entry="train", eval_ckpt=None, hours=12, frames=False, gpus=4):
    # 2026-11-04 platform notice: this orchestrator must never submit on its own. It only records proposals; a human
    # (or the agent, deliberately and within submission_policy.md limits) submits them with submit_rlinf.py.
    st.setdefault("proposed", []).append({"time": now(), "tag": tag, "config": config, "entry": entry, "gpus": gpus,
                                          "eval_ckpt": eval_ckpt, "frames": frames, "overrides": overrides})
    log(st, f"PROPOSED (not submitted) {tag} ({entry}, {gpus} GPU)")
    return False
    cmd = ["uv", "run", "--offline", "--with", "boto3", "python", "submit_rlinf.py", "--tag", tag, "--pool", "5090",
           "--gpus", str(gpus), "--quota", "spot", "--venv", ".venv", "--config", config, "--entry", entry, "--hours", str(hours)]
    if eval_ckpt:
        cmd += ["--eval-ckpt-tag", eval_ckpt[0], "--eval-ckpt-step", str(eval_ckpt[1])]
    if frames:
        cmd += ["--frames"]
    cmd += ["--"] + overrides
    r = subprocess.run(cmd, cwd=CP, capture_output=True, text=True, timeout=600)
    ok = r.returncode == 0 and '"submit_rc": 0' in r.stdout
    log(st, f"submit {tag} ({entry}) rc={r.returncode} ok={ok} {r.stdout.strip()[-160:]}")
    return ok


def job_state(tag):
    try:
        name = json.loads((CP / "runs" / f"{tag}.json").read_text())["name"]
        r = subprocess.run([SCO, "acp", "jobs", "describe", name, "--workspace-name=share-space", "-o", "json"],
                           capture_output=True, text=True, timeout=60)
        s = json.loads(r.stdout).get("state")
        return s.get("state") if isinstance(s, dict) else s
    except Exception:
        return "?"


def mirror_state(tag):
    subprocess.run(["uv", "run", "--offline", "--with", "boto3", "python", "fetch_run.py", tag], cwd=CP,
                   capture_output=True, text=True, timeout=300)
    try:
        return json.loads((CP / "cluster_runs" / tag / "status.json").read_text())["state"]
    except Exception:
        return "-"


def curve(tag):
    r = subprocess.run([VENV_PY, str(CP / "tb_last.py"), str(CP / "cluster_runs" / tag)], capture_output=True, text=True,
                       timeout=300)
    parts = r.stdout.split()
    return [float(x) / 100 for x in parts[1:]] if parts and parts[0] != "0" else []


def ee_eval_result_path(p):
    pat = re.compile(r"\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)")
    out = {}
    p = Path(p)
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if m:
                out[f"{m.group(1)}:{m.group(2)}"] = m.group(3) == "True"
    return out


def ee_eval_result(tag):
    return ee_eval_result_path(CP / "cluster_runs" / tag / "console.log")


def run_done(st, run):
    """Advance one run record; returns True when it finished successfully. Resubmits once after a failure."""
    if run.get("done"):
        return True
    js, ms = job_state(run["tag"]), mirror_state(run["tag"])
    run["job_state"], run["mirror_state"] = js, ms
    if ms == "finished_rc_0":
        run["done"] = True
        if run["entry"] == "train":
            run["curve"] = curve(run["tag"])
        else:
            res = ee_eval_result(run["tag"])
            run["n_eval"], run["succ"] = len(res), sum(res.values())
            run["per_episode"] = res
        log(st, f"done {run['tag']}: " + (f"curve {run.get('curve')}" if run["entry"] == "train"
                                          else f"EE {run['succ']}/{run['n_eval']}"))
        return True
    if js in ("STOPPED", "SUSPENDED", "STOPPING"):  # stopped on purpose (by us): never resubmit
        run["cancelled"] = run["dead"] = True
        return False
    failed = ms.startswith("finished_rc_") or ms.startswith("fetch_failed") or ms.startswith("ckpt_") or js == "FAILED"
    if failed and run.get("retries", 0) < 0:  # automatic retries disabled (platform notice 2026-11-04)
        run["retries"] = run.get("retries", 0) + 1
        run["tag"] = run["tag_base"] + f"_r{run['retries'] + 1}"
        log(st, f"retry {run['tag_base']} -> {run['tag']} (job {js}, mirror {ms})")
        if run["entry"] == "train":
            submit(st, run["tag"], run["config"], run["overrides"])
        else:
            submit(st, run["tag"], run["config"], run["overrides"], entry="eval", eval_ckpt=run["eval_ckpt"], hours=3,
                   frames=run.get("frames", False), gpus=run.get("gpus", 4))
    elif failed:
        run["dead"] = True
    return False


def new_train_run(st, tag, fold, knob_value):
    ov = train_overrides(st, fold, knob_value)
    run = {"tag": tag, "tag_base": tag, "entry": "train", "fold": fold, "knob": fmt(knob_value),
           "config": "libero_10_ppo_pi05_5090x4", "overrides": ov, "submitted": now()}
    submit(st, tag, run["config"], ov)
    return run


def new_eval_run(st, tag, train_tag):
    ov = ["rollout.model.num_action_chunks=10", "+env.eval.task_id_filter=[" + ",".join(map(str, FOLDS["EE"])) + "]",
          "env.eval.total_num_envs=10", "env.eval.rollout_epoch=20"]
    run = {"tag": tag, "tag_base": tag, "entry": "eval", "config": "libero_10_pi05_eval_5090_1gpu", "overrides": ov,
           "eval_ckpt": [train_tag, st["iters"]], "gpus": 1, "submitted": now()}
    submit(st, tag, run["config"], ov, entry="eval", eval_ckpt=run["eval_ckpt"], hours=3, gpus=1)
    return run


def new_ver_run(st, cand):
    """30+ episode source-fold eval of a finished candidate, with frame dump for the external verifier."""
    fold = cand["fold"]
    ov = ["rollout.model.num_action_chunks=10", "+env.eval.task_id_filter=[" + ",".join(map(str, FOLDS[fold])) + "]",
          "env.eval.total_num_envs=12", "env.eval.rollout_epoch=4"]
    tag = cand["tag_base"] + "_ver"
    run = {"tag": tag, "tag_base": tag, "entry": "eval", "config": "libero_10_pi05_eval_5090_1gpu", "overrides": ov,
           "eval_ckpt": [cand["tag"], st["iters"]], "frames": True, "gpus": 1, "submitted": now()}
    submit(st, tag, run["config"], ov, entry="eval", eval_ckpt=run["eval_ckpt"], hours=2, frames=True, gpus=1)
    return run


def sol_judge(st, cand):
    """Fetch the verifier eval's frames and let gpt-6.1-sol judge them (sol_ver.py); sim success only for agreement."""
    import importlib.util as iu
    v = cand["ver"]
    subprocess.run(["uv", "run", "--offline", "--with", "boto3", "python", "fetch_run.py", v["tag"], "--frames"], cwd=CP,
                   capture_output=True, text=True, timeout=900)
    frames = CP / "cluster_runs" / v["tag"] / "frames"
    if not frames.exists():
        log(st, f"ver {cand['tag_base']}: frames missing for {v['tag']}")
        return
    spec_ = iu.spec_from_file_location("sol_ver", HERE / "sol_ver.py")
    SV = iu.module_from_spec(spec_)
    spec_.loader.exec_module(SV)
    summary, res = SV.judge_all(frames, CP / "cluster_runs" / v["tag"] / "sol", workers=6)
    sim = ee_eval_result(v["tag"])
    agree = [int(r["success"]) == int(sim[f"{r['task_id']}:{r['episode'].split('trial')[1]}"])
             for r in res if "error" not in r and f"{r['task_id']}:{r['episode'].split('trial')[1]}" in sim]
    cand.update(ver_sol=summary["success_rate"], ver_sol_progress=summary["progress_mean"], ver_sol_n=summary["n_ok"],
                ver_sol_errors=summary["errors"], ver_sim=(sum(sim.values()) / len(sim)) if sim else None,
                ver_agreement=(sum(agree) / len(agree)) if agree else None)
    log(st, f"ver {cand['tag_base']}: Sol success {cand['ver_sol']} (progress {cand['ver_sol_progress']}, "
            f"n={cand['ver_sol_n']}), sim {cand['ver_sim']}, agreement {cand['ver_agreement']}")


def ver_score(c):
    return sum(c[-3:]) / max(len(c[-3:]), 1) if c else float("-inf")


def step(st):
    P = st["prefix"]
    phi = st["phi"]  # list of knob values, phi[k] = Phi_k
    # EE runs for every Phi_k already decided
    for k, v in enumerate(phi):
        ee = st["ee"].setdefault(str(k), {})
        if "train" not in ee:
            ee["train"] = new_train_run(st, f"{P}_ee_phi{k}", "EE", v)
        elif run_done(st, ee["train"]) and "eval" not in ee:
            ee["eval"] = new_eval_run(st, f"{P}_ee_phi{k}_eval", ee["train"]["tag"])
        elif "eval" in ee:
            run_done(st, ee["eval"])
    # source rounds
    for k in (1, 2):
        if len(phi) < k:
            break
        rnd = st["rounds"].setdefault(str(k), {"fold": f"S{k}", "cands": []})
        if not rnd["cands"]:
            for i, m in enumerate(MULT[k]):
                rnd["cands"].append(new_train_run(st, f"{P}_r{k}_c{i}", rnd["fold"], phi[k - 1] * m))
        if len(phi) == k:
            done = [run_done(st, c) for c in rnd["cands"]]  # poll every candidate (no short-circuit)
            for d, c in zip(done, rnd["cands"]):
                if d and "ver" not in c:
                    c["ver"] = new_ver_run(st, c)
                elif d and c.get("ver_sol") is None and run_done(st, c["ver"]):
                    sol_judge(st, c)
            vdone = [d and (c.get("ver_sol") is not None or c.get("ver", {}).get("dead", False))
                     for d, c in zip(done, rnd["cands"])]
            settled = [vd or c.get("dead", False) for vd, c in zip(vdone, rnd["cands"])]
            if all(settled) and sum(done) >= 2:
                train_scores = [ver_score(c.get("curve", [])) if d else float("-inf") for d, c in zip(done, rnd["cands"])]
                if all(c.get("ver_sol") is not None for d, c in zip(done, rnd["cands"]) if d):
                    keys = [(0.5 * c["ver_sol"] + 0.5 * (c.get("ver_sol_progress") or 0.0), c["ver_sol"]) if d
                            else (float("-inf"), 0.0) for d, c in zip(done, rnd["cands"])]
                    rule = "external verifier (gpt-6.1-sol: 0.5*success + 0.5*progress, then success)"
                else:
                    keys = [(t, 0.0) for t in train_scores]
                    rule = "fallback: training success of the last 3 iterations"
                best = max(range(len(keys)), key=lambda i: (keys[i], MULT[k][i] == 1.0))
                best_train = max(range(len(train_scores)), key=lambda i: (train_scores[i], MULT[k][i] == 1.0))
                rnd.update(rule=rule, ver_keys=keys, train_scores=train_scores, selected=best,
                           selected_by_train_ver=best_train)
                phi.append(float(rnd["cands"][best]["knob"]))
                log(st, f"round {k}: {rule}: {keys} -> Phi_{k} knob {phi[-1]} (training-success Ver would pick "
                        f"{rnd['cands'][best_train]['knob']})")
            elif all(settled):
                log(st, f"round {k}: fewer than 2 candidates finished; no selection possible")
    save(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None, help="state file name next to this script (one per loop)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--base", choices=["espo", "ppo"], required=True)
    i.add_argument("--espo-variant", choices=list(ESPO_VARIANT), default="v3")
    i.add_argument("--iters", type=int, default=8)
    i.add_argument("--knob", choices=list(KNOB_CHOICES), default=None, help="ESPO only: prox (default) or lr")
    i.add_argument("--prefix", default=None)
    sub.add_parser("step")
    sub.add_parser("status")
    a = ap.parse_args()
    global STATE
    if a.state:
        STATE = HERE / a.state
    if a.cmd == "init":
        if STATE.exists():
            sys.exit(f"{STATE} exists; refusing to overwrite")
        prefix = a.prefix or f"rsi_{a.base}" + (f"_{a.espo_variant}" if a.base == "espo" else "")
        knob = KNOB_CHOICES[a.knob] if (a.base == "espo" and a.knob) else KNOB[a.base]
        st = {"created": now(), "base": a.base, "espo_variant": a.espo_variant, "iters": a.iters, "prefix": prefix,
              "folds": FOLDS, "mult": {str(k): v for k, v in MULT.items()}, "knob": knob[0],
              "phi": [knob[1]], "ee": {}, "rounds": {}, "log": []}
        save(st)
        print(json.dumps({k: st[k] for k in ("base", "espo_variant", "iters", "prefix", "knob", "phi")}))
    elif a.cmd == "step":
        if Path(str(STATE) + ".disabled").exists():
            return  # loop handed over (e.g. to rsi_local.py); do not touch its jobs
        st = load()
        step(st)
    else:
        st = load()
        print(json.dumps({k: v for k, v in st.items() if k != "log"}, indent=1, ensure_ascii=False)[:6000])
        print("\n".join(st.get("log", [])[-15:]))


if __name__ == "__main__":
    main()

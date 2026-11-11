#!/usr/bin/env python3
"""In-job sequential stage runner (2026-11-04): one cluster allocation runs several RLinf stages back to back, so one
submission replaces several (platform notice: fewer, deliberate submissions). GPUs stay busy; no idle waiting.

Env: B1K_OUT (AFS output dir), B1K_STAGES_B64 (base64 JSON list of stages). Stage fields:
  name, entry (train|eval), config, overrides (list), optional:
  ckpt_stage + ckpt_step   evaluate a checkpoint written by an earlier stage of this job
  ckpt_afs = "tag:step"    evaluate a checkpoint of an earlier cluster run on AFS
  frames = true            LIBERO eval frame dump for the external verifier (/opt/b1k_frames/<name>)
  entry = script           run s["script"] (path under RLinf) with s["args"] + --out <stage dir> (2026-11-05)
  timeout_s                kill the stage (process group) after this many seconds; rc = "timeout"
  requires = [stage, ...]  skip unless these earlier stages finished with rc 0
Writes <OUT>/<name>/console.log per stage and <OUT>/stages_status.json; a failed stage does not stop later stages
(an eval of a missing checkpoint is recorded as ckpt_missing).
"""
import base64
import glob
import json
import os
import subprocess
import time

OUT = os.environ["B1K_OUT"]
STAGES = json.loads(base64.b64decode(os.environ["B1K_STAGES_B64"]).decode())
REPO = "/mnt/data1/faye/code/libero_rlinf_20261103/RLinf"
CFGDIR = "/mnt/data1/faye/code/libero_rlinf_20261103/configs/"
STRIP = ("import sys, torch; sd = torch.load(sys.argv[1], map_location='cpu'); d = [k for k in sd if k.startswith('value_head.')]; "
         "[sd.pop(k) for k in d]; torch.save(sd, sys.argv[2]); print('dropped', len(d), 'value_head tensors')")
status = {"stages": [s["name"] for s in STAGES], "results": {}}
# Resume after a Spot preemption (B1K_RESUME_FROM = tag of the interrupted multi job): stages that finished there are
# skipped, their outputs are used by later stages, and an unfinished train stage resumes from its latest checkpoint.
PREV = os.environ.get("B1K_RESUME_FROM") or None
PREV_OUT = f"/mnt/afs/260010165/rlinf_runs_20261104/{PREV}" if PREV else None
prev_results = {}
if PREV_OUT and os.path.exists(os.path.join(PREV_OUT, "stages_status.json")):
    prev_results = json.load(open(os.path.join(PREV_OUT, "stages_status.json"))).get("results", {})
status["resume_from"] = PREV


def save():
    with open(os.path.join(OUT, "stages_status.json"), "w") as f:
        json.dump(status, f, indent=1)


def run_logged(cmd, log_path, env, timeout_s=None):
    """Run a stage command (own process group) with stdout/stderr to log_path. On timeout the whole group is killed
    and 'timeout' is returned instead of an exit code (2026-11-05: diagnostic stages must not hold the job)."""
    import signal
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            log.write(f"\n[multi_runner] stage killed after {timeout_s} s\n")
            return "timeout"


def find_ckpt(root, step):
    hits = glob.glob(f"{root}/**/checkpoints/global_step_{step}/actor/model_state_dict/full_weights.pt", recursive=True)
    return hits[0] if hits else None


def latest_ckpt_dir(root):
    """Latest checkpoints/global_step_N directory under root (for runner.resume_dir), or None."""
    best = None
    for d in glob.glob(f"{root}/**/checkpoints/global_step_*", recursive=True):
        try:
            n = int(d.rsplit("_", 1)[1])
        except ValueError:
            continue
        if os.path.exists(os.path.join(d, "actor")) and (best is None or n > best[0]):
            best = (n, d)
    return best


for s in STAGES:
    name = s["name"]
    py = f"{REPO}/{s['venv']}/bin/python" if s.get("venv") else "python"  # optional per-stage venv (e.g. .venv-calvin)
    ray = f"{REPO}/{s['venv']}/bin/ray" if s.get("venv") else "ray"
    env = dict(os.environ)
    if s.get("venv"):  # activate the stage's venv for child processes too (PATH lookups of python/ray)
        env.update(PATH=f"{REPO}/{s['venv']}/bin:" + env.get("PATH", ""), VIRTUAL_ENV=f"{REPO}/{s['venv']}")
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    ov = list(s.get("overrides", []))
    res = {"entry": s["entry"], "config": s["config"], "start": time.strftime("%H:%M:%S")}
    status["current"] = name
    save()  # visible to the log sync from the first stage on
    bad = [r for r in s.get("requires", []) if status["results"].get(r, {}).get("rc") not in (0, "skipped_done_in_" + str(PREV))]
    if bad:  # 2026-11-05: e.g. no training on a node whose render diagnostic found the driver stuck
        res.update(rc="skipped_requires_" + "+".join(bad))
        status["results"][name] = res
        save()
        continue
    if prev_results.get(name, {}).get("rc") == 0:  # finished in the interrupted job: skip, reuse its outputs
        res.update(rc="skipped_done_in_" + PREV)
        status["results"][name] = res
        save()
        continue
    if s["entry"] == "train" and PREV_OUT and not any(o.startswith("runner.resume_dir=") for o in ov):
        lc = latest_ckpt_dir(os.path.join(PREV_OUT, name))
        if lc:
            ov.append(f"runner.resume_dir={lc[1]}")
            res["resumed_from_step"] = lc[0]
    if s["entry"] == "eval" and (s.get("ckpt_stage") or s.get("ckpt_afs")):
        if s.get("ckpt_stage"):
            root, step = os.path.join(OUT, s["ckpt_stage"]), s["ckpt_step"]
            if PREV_OUT and find_ckpt(root, step) is None:
                root = os.path.join(PREV_OUT, s["ckpt_stage"])  # train stage finished in the interrupted job
        else:
            tag, step = s["ckpt_afs"].split(":")
            root = f"/mnt/afs/260010165/rlinf_runs_20261104/{tag}"
        ck = find_ckpt(root, step)
        if ck is None:
            res["rc"] = "ckpt_missing"
            status["results"][name] = res
            save()
            continue
        dst = f"/opt/b1k_ckpt_{name}.pt"
        subprocess.run([py, "-c", STRIP, ck, dst], cwd=REPO, env=env, check=False)
        res["ckpt"] = ck
        ov.append(f"runner.ckpt_path={dst}")
    if s.get("frames"):
        fd = f"/opt/b1k_frames/{name}"
        os.makedirs(fd, exist_ok=True)
        ov += [f"+env.eval.dump_frames_dir={fd}", "+env.eval.dump_frames_every=70"]
    t0 = time.time()
    tmo = float(s["timeout_s"]) if s.get("timeout_s") else None
    if s["entry"] == "script":
        # diagnostic stage (2026-11-05): a python script shipped in the overlay (RLinf/rlinf/...), args + stage out dir
        rc = run_logged([py, s["script"], *s.get("args", []), "--out", out], os.path.join(out, "console.log"), env,
                        tmo or 3600.0)
        res.update(rc=rc, minutes=round((time.time() - t0) / 60, 1))
        status["results"][name] = res
        save()
        continue
    script = "evaluations/eval_embodied_agent.py" if s["entry"] == "eval" else "examples/embodiment/train_embodied_agent.py"
    rc = run_logged([py, script, "--config-path", CFGDIR, "--config-name", s["config"], f"runner.logger.log_path={out}",
                     *ov], os.path.join(out, "console.log"), env, tmo)
    res.update(rc=rc, minutes=round((time.time() - t0) / 60, 1))
    status["results"][name] = res
    save()
    subprocess.run([ray, "stop", "--force"], capture_output=True, env=env)  # RLinf attaches to any running Ray cluster
    time.sleep(5)
status["current"] = None
save()

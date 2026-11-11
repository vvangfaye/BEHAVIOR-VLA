#!/usr/bin/env python3
"""Submit an RLinf run to the SenseCore ACP cluster (2026-11-04).

Run with:  uv run --offline --with boto3 python submit_rlinf.py --tag T --pool h100|5090 --gpus 4 --quota reserved|spot \
               --venv .venv-calvin --config calvin_abc_d_ppo_pi05_cluster4 [--entry eval] [--robocasa] [--hours 30] -- OVERRIDES...
Bundle parts come from aoss://260010165/rlinf_bundle_20261104/ (see make_and_upload_bundle.sh). A per-run overlay with the
current configs and locally changed RLinf files is uploaded too. Outputs: AFS /mnt/afs/260010165/rlinf_runs_20261104/<tag>;
console.log, tensorboard and status are mirrored to aoss://260010165/rlinf_runs_20261104/<tag>/ every 5 minutes.
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import aoss_io

HERE = Path(__file__).resolve().parent
BUNDLE_LOCAL = Path("/mnt/data1/faye/rlinf_bundle_20261104")
BASE = Path("/mnt/data1/faye/code/libero_rlinf_20261103")
CLI = "/home/faye/.local/bin/sco"
POOLS = {
    "h100": {"template": "b1k-update-composition-1022-plumbing06h100", "spec": {1: "n6ls.iu.i40.1", 4: "n6ls.iu.i40.4"}},
    "5090": {"template": "b1k-repeat1688-1019-batch-00", "spec": {1: "n12lp.nn.i10a.1", 4: "n12lp.nn.i10a.4"}},
}


def shas():
    out = {}
    for f in BUNDLE_LOCAL.glob("SHA256SUMS_part*"):
        for line in f.read_text().splitlines():
            h, name = line.split()
            out[name] = h
    return out


def overlay_bytes():
    """Current configs + every RLinf file that differs from the bundle's base commit (tracked diffs and new files)."""
    repo = BASE / "RLinf"
    changed = subprocess.run(["git", "-C", str(repo), "diff", "--name-only"], capture_output=True, text=True, check=True).stdout.split()
    new = subprocess.run(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "rlinf"], capture_output=True,
                         text=True, check=True).stdout.split()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted((BASE / "configs").glob("*.yaml")):
            tar.add(p, arcname=f"configs/{p.name}")
        for rel in sorted(set(changed + new)):
            if (repo / rel).is_file():
                tar.add(repo / rel, arcname=f"RLinf/{rel}")
    return buf.getvalue(), sorted(set(changed + new))


POLICY = dict(min_interval_s=15 * 60, max_per_hour=3, max_per_day=50, max_queued=2)  # submission_policy.md (daily 10 -> 50: user, 2026-11-06 12:2x)
LEDGER = HERE / "submission_ledger.json"


def enforce_submission_policy(tag):
    """Rate limits for deliberate, human-reviewed submissions (platform notice 2026-11-04). Raises SystemExit."""
    now = datetime.datetime.now()
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    times = [datetime.datetime.fromisoformat(e["time"]) for e in led]
    if times and (now - max(times)).total_seconds() < POLICY["min_interval_s"]:
        raise SystemExit(f"policy: last submission {max(times):%H:%M}; wait {POLICY['min_interval_s'] // 60} min between submissions")
    if sum((now - t).total_seconds() < 3600 for t in times) >= POLICY["max_per_hour"]:
        raise SystemExit(f"policy: already {POLICY['max_per_hour']} submissions in the last hour")
    if sum(t.date() == now.date() for t in times) >= POLICY["max_per_day"]:
        raise SystemExit(f"policy: daily limit of {POLICY['max_per_day']} submissions reached")
    queued = 0
    for state in ("STARTING", "QUEUEING", "INIT", "PENDING"):
        r = subprocess.run([CLI, "acp", "jobs", "list", "--workspace-name=share-space", f"--state={state}", "--page-size=500",
                            "-o", "json"], capture_output=True, text=True, timeout=120)
        try:
            d = json.loads(r.stdout)
        except Exception:
            continue
        for j in (d if isinstance(d, list) else (d.get("training_jobs") or [])):
            if (j.get("ownership") or {}).get("user_name") == "260010165" or str(j.get("display_name", "")).startswith("b1k-"):
                queued += 1
    if queued >= POLICY["max_queued"]:
        raise SystemExit(f"policy: {queued} jobs of this account are still queued (max {POLICY['max_queued']}); wait")
    led.append({"time": now.isoformat(timespec="seconds"), "tag": tag})
    LEDGER.write_text(json.dumps(led, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool", choices=POOLS, required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--quota", choices=["reserved", "spot"], required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--entry", choices=["train", "eval", "multi"], default="train")
    ap.add_argument("--stages", default="", help="multi entry: JSON file with the stage list (see multi_runner.py)")
    ap.add_argument("--resume-from", default="", help="multi entry: tag of a preempted multi job to resume (skip done stages)")
    ap.add_argument("--robocasa", action="store_true", help="also fetch the RoboCasa model and assets")
    ap.add_argument("--hours", type=float, default=30.0, help="wall-clock limit (GPU-h cap is 128 per job)")
    ap.add_argument("--nccl-p2p-disable", default="1")
    ap.add_argument("--ray-mem-threshold", default="0.95", help="RAY_memory_usage_threshold in the job (11/5)")
    ap.add_argument("--eval-ckpt-tag", default="", help="eval entry: tag of an earlier cluster run whose checkpoint to evaluate")
    ap.add_argument("--eval-ckpt-step", default="", help="global step of that checkpoint")
    ap.add_argument("--frames", action="store_true", help="eval entry: dump per-episode frames (LIBERO) and upload them")
    ap.add_argument("--no-validate", action="store_true", help="skip the pre-submit Hydra/divisibility check")
    ap.add_argument("--relay", default="", help="comma-separated stage names whose agent manager runs in mode=relay: presigned PUT/GET URLs for <run>/relay/<stage>/{req,ans}.json go into B1K_RELAY_URLS_B64 (local side: analysis/rsi_20261104/sol_relay.py) (11/5)")
    ap.add_argument("--raw-script", default="", help="run this shell script instead of the RLinf bootstrap (no bundle download; maintenance jobs, e.g. afs_cleanup_20261106.sh); presigned B1K_PUT_* URLs are still exported (11/6)")
    ap.add_argument("--presign-get", action="append", default=[], help="NAME=aoss_key: export a presigned GET URL as env NAME (never written to the receipt) (11/6)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    if a.quota != "spot" and not a.dry_run:
        raise SystemExit("User policy from 2026-11-08 requires Spot resources for all project experiments")
    # 2026-11-04: platform notice about automated, high-frequency submissions from this account. Hard stop while the
    # freeze marker exists; submissions must be deliberate, human-reviewed and rate-limited (see submission_policy.md).
    if (HERE / "SUBMISSION_FROZEN").exists() and not a.dry_run:
        raise SystemExit("SUBMISSION_FROZEN: cluster submissions are frozen (platform notice 2026-11-04); not submitting")
    assert a.gpus * a.hours <= 128, "per-job cap: 128 GPU-hours"
    for o in a.overrides:
        assert " " not in o, f"override must not contain spaces: {o}"
    stages = json.loads(Path(a.stages).read_text()) if a.entry == "multi" else None
    checks = ([(st["config"], st["entry"], st.get("overrides", [])) for st in stages if st["entry"] != "script"]
              if stages else [(a.config, a.entry, a.overrides)])  # script stages (diagnostics) have no Hydra config
    for st in stages or []:
        if st["entry"] == "script":
            sp = Path("/mnt/data1/faye/code/libero_rlinf_20261103/RLinf") / st["script"]
            assert sp.exists() and st["script"].startswith("rlinf/"), f"script stage must ship in the overlay: {sp}"
        assert st["entry"] in ("train", "eval", "script"), st["entry"]
    if not a.no_validate:  # compose the config(s) as the job will and check RLinf's runtime divisibility asserts
        for cfg_name, entry, ovs in checks:
            v = subprocess.run(["/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python",
                                str(HERE / "validate_cfg.py"), cfg_name, str(a.gpus), entry, *ovs],
                               capture_output=True, text=True, timeout=300)
            tail = (v.stdout.strip().splitlines() or [""])[-1]
            if v.returncode != 0:
                raise SystemExit(f"pre-submit validation failed ({cfg_name}): {tail or v.stderr[-500:]}")
            print(f"validated {cfg_name}: {tail}")
    for _, _, ovs in checks:
        for o in ovs:
            assert " " not in o, f"override must not contain spaces: {o}"
    if not a.dry_run:  # rate limits / queue check only after validation passed (no ledger entry for rejected configs)
        enforce_submission_policy(a.tag)
    sha = shas()
    parts = ["sys_extra.tar.gz", "uv_python.tar.gz", "hf_cache.tar.gz", "rlinf_code_venvs.tar.gz", "models_libero_calvin.tar"]
    if a.robocasa:
        parts += ["models_robocasa.tar", "robocasa_assets.tar"]
    s3 = aoss_io.local()
    for p in parts:
        s3.head_object(Bucket=aoss_io.BUCKET, Key=f"rlinf_bundle_20261104/{p}")  # raises if not uploaded yet
    ov, ov_files = overlay_bytes()
    ov_name = f"overlay_{a.tag}.tar.gz"
    s3.put_object(Bucket=aoss_io.BUCKET, Key=f"rlinf_bundle_20261104/overlays/{ov_name}", Body=ov)
    sha[ov_name] = hashlib.sha256(ov).hexdigest()
    keys = {p: f"rlinf_bundle_20261104/{p}" for p in parts}
    keys[ov_name] = f"rlinf_bundle_20261104/overlays/{ov_name}"
    parts.append(ov_name)
    env = {"B1K_TAG": a.tag, "B1K_PARTS": " ".join(parts), "B1K_VENV": a.venv, "B1K_ENTRY": a.entry,
           "B1K_CONFIG": a.config, "B1K_OVERRIDES": " ".join(a.overrides), "B1K_NCCL_P2P_DISABLE": a.nccl_p2p_disable, "B1K_RAY_MEM_THRESHOLD": a.ray_mem_threshold}
    if a.eval_ckpt_tag:
        assert a.entry == "eval" and a.eval_ckpt_step, "--eval-ckpt-tag needs --entry eval and --eval-ckpt-step"
        env["B1K_EVAL_CKPT_TAG"], env["B1K_EVAL_CKPT_STEP"] = a.eval_ckpt_tag, str(a.eval_ckpt_step)
    if stages:
        if a.resume_from:
            env["B1K_RESUME_FROM"] = a.resume_from
        env["B1K_STAGES_B64"] = base64.b64encode(json.dumps(stages).encode()).decode()
        env["B1K_MULTI_B64"] = base64.b64encode((HERE / "multi_runner.py").read_bytes()).decode()
    if a.frames:
        assert a.entry == "eval", "--frames needs --entry eval"
        env["B1K_FRAMES_DIR"] = "/opt/b1k_frames"
        env["B1K_OVERRIDES"] += " +env.eval.dump_frames_dir=/opt/b1k_frames +env.eval.dump_frames_every=70"
    for i, p in enumerate(parts):
        env[f"B1K_GET_{i}"] = aoss_io.presign("get_object", keys[p], seconds=3 * 86400)
        env[f"B1K_SHA_{i}"] = sha[p]
    run_prefix = f"rlinf_runs_20261104/{a.tag}/"
    for k, f in [("B1K_PUT_LOG", "console.log"), ("B1K_PUT_TB", "tensorboard.tgz"), ("B1K_PUT_STATUS", "status.json"),
                 ("B1K_PUT_FRAMES", "frames.tgz")]:
        env[k] = aoss_io.presign("put_object", run_prefix + f, seconds=3 * 86400)
    relay_keys = [k for k in a.relay.split(",") if k]
    if relay_keys:  # agent manager in mode=relay (agent_manager.ask_relay); URLs are never written to the receipt
        assert all(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", k) for k in relay_keys), relay_keys
        urls = {k: {"req_put": aoss_io.presign("put_object", f"{run_prefix}relay/{k}/req.json", seconds=3 * 86400),
                    "ans_get": aoss_io.presign("get_object", f"{run_prefix}relay/{k}/ans.json", seconds=3 * 86400)}
                for k in relay_keys}
        env["B1K_RELAY_URLS_B64"] = base64.b64encode(json.dumps(urls).encode()).decode()
    for spec in a.presign_get:
        nm, key = spec.split("=", 1)
        assert re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", nm), nm
        env[nm] = aoss_io.presign("get_object", key, seconds=3 * 86400)
    boot = base64.b64encode((Path(a.raw_script) if a.raw_script else HERE / "cluster_bootstrap.sh").read_bytes()).decode()
    exports = "; ".join(f"export {k}='{v}'" for k, v in env.items())
    cmd = f"{exports}; echo {boot} | base64 -d > /tmp/b1k.sh; timeout {int(a.hours * 3600)} bash /tmp/b1k.sh"
    assert len(cmd) < 64000, len(cmd)
    name = f"b1k-rl-{a.tag}".replace("_", "-").lower()[:60]
    receipt = {"name": name, "tag": a.tag, "pool": a.pool, "gpus": a.gpus, "quota": a.quota, "venv": a.venv, "config": a.config,
               "entry": a.entry, "overrides": a.overrides, "parts": parts, "overlay_files": ov_files, "hours": a.hours,
               "eval_ckpt": f"{a.eval_ckpt_tag}:{a.eval_ckpt_step}" if a.eval_ckpt_tag else "",
               "frames": bool(a.frames), "relay_keys": relay_keys, "raw_script": a.raw_script, "presign_get_names": [x.split("=", 1)[0] for x in a.presign_get], "effective_overrides": env["B1K_OVERRIDES"], "stages": stages,
               "aoss_run_prefix": run_prefix, "afs_out": f"/mnt/afs/260010165/rlinf_runs_20261104/{a.tag}",
               "command_bytes": len(cmd), "submitted": datetime.datetime.now().isoformat()}
    if a.dry_run:
        print(json.dumps(receipt, indent=1))
        return
    pool = POOLS[a.pool]
    r = subprocess.run([CLI, "acp", "jobs", "copy", "--workspace-name=share-space", f"--copy-job-name={pool['template']}",
                        f"--name={name}", f"--job-name={name}", f"--worker-spec={pool['spec'][a.gpus]}", "--worker-nodes=1",
                        f"--quota-type={a.quota}", "--retry-times=0", "--enable-fault-tolerance=false",
                        "--enable-anomaly-detection=false", f"--command={cmd}"], capture_output=True, text=True, timeout=120)
    receipt["submit_rc"] = r.returncode
    tail = (r.stdout + r.stderr).strip().splitlines()
    receipt["submit_tail"] = tail[-1][:200] if tail else ""
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / f"{a.tag}.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
    print(json.dumps({k: receipt[k] for k in ("name", "submit_rc", "submit_tail", "command_bytes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Deliberate, single CPU job: compact existing AFS trajectories, no simulation.

The command and presigned artifact URLs are never saved or printed locally.
Run without --submit to inspect the public request. No retries are performed.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCO = "/home/faye/.local/bin/sco"
NAME = "b1k-supermimic-export-1106-a1"
PREFIX = "behavior1k/supermimic-feasibility-20261106/export-a1/"
TEMPLATE = "b1k-rl-mechanism-1102-a1"
LEDGER = ROOT / "analysis/cluster_probe_20261104/submission_ledger.json"


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def run_cli(args):
    return subprocess.run([SCO, *args], capture_output=True, text=True, timeout=60)


def parse_job_list(stdout):
    # SCO's JSON mode still emits this exact plain-text sentinel for an empty
    # result. Arbitrary empty/invalid output must remain a preflight failure.
    if stdout.strip() == "No jobs found":
        return []
    data = json.loads(stdout)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("training_jobs"), list):
        return data["training_jobs"]
    raise ValueError("Unknown job-list response schema")


def queue_snapshot():
    rows = []
    for state in ("STARTING", "QUEUEING", "INIT", "WAITING", "CREATING"):
        page = 1
        while True:
            result = run_cli(["acp", "jobs", "list", "--workspace-name=share-space",
                              "--user-name=260010165", f"--state={state}",
                              "--page-size=20", f"--page-token={page}", "-o", "json"])
            if result.returncode:
                raise RuntimeError(f"Cannot verify {state} queue (rc={result.returncode})")
            items = parse_job_list(result.stdout)
            rows.extend(items)
            if len(items) < 20:
                break
            page += 1
    unique = {r["uid"]: r for r in rows}
    return [{"uid": r["uid"], "name": r.get("display_name"), "state": r["state"]}
            for r in unique.values()]


def build_manifest():
    entries = []
    for path in sorted((ROOT / "analysis/rl_multiround_20261102/receipts/round_01").glob("episode_*.json")):
        receipt = json.loads(path.read_text())
        entries.append({"path": receipt["path"], "task": "installing_a_modem",
                        "instance": receipt["instance"], "seed": receipt["seed"],
                        "sha256": receipt["sha256"],
                        "reward_contract_sha256": receipt.get("reward_contract_sha256", receipt.get("reward_contract", receipt.get("contract")))})
    if len(entries) != 64:
        raise ValueError(f"Expected 64 original round-one receipts, found {len(entries)}")
    return {"episodes": entries}


def worker_source(exporter, manifest, urls):
    # The parent needs only stdlib. Torch/numpy run in the same existing
    # environment that wrote these original trajectories.
    return f'''import json,os,pathlib,subprocess,tempfile,time,urllib.request
root=pathlib.Path(tempfile.mkdtemp(prefix="b1k-wm-export-"))
urls={urls!r}
status={{"state":"running","started_unix":time.time(),"gpu_count":0,"new_environment_steps":0}}
def put(name,data):
    request=urllib.request.Request(urls[name],data=data,method="PUT")
    with urllib.request.urlopen(request,timeout=300) as response:
        if response.status not in (200,201,204): raise RuntimeError("Artifact upload failed")
try:
    (root/"export_transitions.py").write_text({exporter!r})
    manifest={manifest!r}
    present=sum(pathlib.Path(e["path"]).is_file() for e in manifest["episodes"])
    status["existing_episodes"]=present
    if present!=len(manifest["episodes"]): raise FileNotFoundError("Original AFS trajectories are unavailable")
    (root/"input.json").write_text(json.dumps(manifest))
    env={{**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"}}
    python="/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python"
    run=subprocess.run([python,str(root/"export_transitions.py"),"--manifest",str(root/"input.json"),"--output",str(root/"transitions.npz"),"--max-episodes","64"],env=env,capture_output=True,text=True,timeout=3300)
    (root/"worker.log").write_text(run.stdout+run.stderr)
    if run.returncode: raise RuntimeError("Export process failed: "+run.stderr[-1500:])
    for filename in ("transitions.npz","transitions.json","worker.log"):
        data=(root/filename).read_bytes()
        if len(data)>256*1024**2: raise RuntimeError("Export artifact exceeds frozen size cap")
        put(filename,data)
    status["state"]="complete"
    status["metadata"]=json.loads((root/"transitions.json").read_text())
except Exception as error:
    status.update(state="failed",error_type=type(error).__name__,error=str(error)[:2000])
finally:
    status["finished_unix"]=time.time()
    put("status.json",json.dumps(status).encode())
print(json.dumps({{k:v for k,v in status.items() if k!="metadata"}}),flush=True)
if status["state"]!="complete":raise SystemExit(1)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    exporter = (HERE / "data/export_transitions.py").read_text()
    manifest = build_manifest()
    request = {"name": NAME, "template": TEMPLATE, "scope": "read original 64 completed trajectories; no simulator, no policy training",
               "spec": "n6ls.iu.i40.2c8g", "gpu_count": 0, "max_seconds": 3600,
               "quota": "reserved", "aoss_prefix": PREFIX, "new_afs_bytes": 0,
               "exporter_sha256": hashlib.sha256(exporter.encode()).hexdigest(), "episodes": len(manifest["episodes"])}
    (HERE / "data/input_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not args.submit:
        print(json.dumps(request, indent=2))
        return
    receipt_path = HERE / "data/export_submission.json"
    if receipt_path.exists():
        raise RuntimeError("Submission already attempted; inspect exact existing job before any manual recovery")
    if (ROOT / "analysis/cluster_probe_20261104/SUBMISSION_FROZEN").exists():
        raise RuntimeError("Project cluster submission freeze is enabled")
    records = json.loads(LEDGER.read_text())
    cst = dt.timezone(dt.timedelta(hours=8))
    current = dt.datetime.now(cst).replace(tzinfo=None)
    times = [dt.datetime.fromisoformat(r["time"]) for r in records]
    times = [t.astimezone(cst).replace(tzinfo=None) if t.tzinfo else t for t in times]
    if times and (current - max(times)).total_seconds() < 900:
        raise RuntimeError("Project minimum 15-minute interval has not elapsed")
    if sum((current - t).total_seconds() < 3600 for t in times) >= 3:
        raise RuntimeError("Project 3 submissions/hour cap reached")
    if sum(t.date() == current.date() for t in times) >= 50:
        raise RuntimeError("Project 50 submissions/day cap reached")
    queued = queue_snapshot()
    if len(queued) >= 2:
        raise RuntimeError("Two queued jobs already exist; no CPU job submitted")
    sys.path.insert(0, str(ROOT / "analysis/cluster_probe_20261104"))
    import aoss_io
    signer = aoss_io.internal()
    urls = {name: signer.generate_presigned_url("put_object", Params={"Bucket": "260010165", "Key": PREFIX+name}, ExpiresIn=7200)
            for name in ("transitions.npz", "transitions.json", "worker.log", "status.json")}
    code = worker_source(exporter, manifest, urls)
    payload = base64.b64encode(zlib.compress(code.encode(), 9)).decode()
    python_code = f"import base64,zlib;exec(zlib.decompress(base64.b64decode({payload!r})))"
    command = "timeout --signal=TERM --kill-after=30s 3600s python3 -u -c " + shlex.quote(python_code)
    if json.loads(LEDGER.read_text()) != records:
        raise RuntimeError("Another task changed the shared submission ledger during preflight; no submission made")
    request.update(attempted_at_cst=now(), state="submission_unknown", command_bytes=len(command.encode()), queue_snapshot=queued)
    request["startup_sha256"] = hashlib.sha256(command.encode()).hexdigest()
    receipt_path.write_text(json.dumps(request, indent=2))
    # This ledger contains only public metadata; record intent before the API.
    records.append({"time": current.isoformat(timespec="seconds"), "tag": "supermimic_export_1106_a1"})
    LEDGER.write_text(json.dumps(records, indent=1))
    result = run_cli(["acp", "jobs", "copy", "--workspace-name=share-space", f"--copy-job-name={TEMPLATE}",
                      f"--name={NAME}", f"--job-name={NAME}", "--worker-spec=n6ls.iu.i40.2c8g", "--worker-nodes=1",
                      "--quota-type=reserved", "--retry-times=0", "--enable-fault-tolerance=false",
                      "--enable-anomaly-detection=false", "--command="+command])
    request["returncode"] = result.returncode
    if result.returncode == 0:
        described = run_cli(["acp", "jobs", "describe", NAME, "--workspace-name=share-space", "-o", "json"])
        if described.returncode:
            raise RuntimeError("Copy returned success; exact job description is not available yet. Do not resubmit.")
        data = json.loads(described.stdout)
        if hashlib.sha256(data["roles"][0]["startup_script"].encode()).hexdigest() != request["startup_sha256"]:
            raise RuntimeError("Created startup command does not match its recorded digest")
        request.update(uid=data.get("uid"), state=data.get("state", "submitted"))
    else:
        request["error"] = "Submission did not return success; query the exact name before any recovery"
        reason = re.sub(r'https?://[^\s"\']+', '<url>', result.stderr or result.stdout)
        request["cli_error_redacted"] = re.sub(r'[A-Za-z0-9+/=]{80,}', '<encoded-payload>', reason)[-1000:]
    receipt_path.write_text(json.dumps(request, indent=2))
    print(json.dumps(request, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

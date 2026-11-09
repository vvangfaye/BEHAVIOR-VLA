#!/usr/bin/env python3
"""Advance the nine authorized runs, with no automatic resubmission of failures."""
import datetime as dt
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "cluster_probe_20261104"
sys.path.insert(0, str(LEGACY))
import aoss_io

STATES = ("RUNNING", "STARTING", "QUEUEING", "CREATING", "INIT", "WAITING",
          "RESTARTING", "RECOVERING", "SUSPENDING", "DELETING")
CLI = "/home/faye/.local/bin/sco"
PYTHON = str(ROOT.parent / "rlinf_adaptation_20261020/venv/bin/python")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def command_json(command):
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if process.returncode:
        state = next((arg for arg in command if arg.startswith("--state=")), "state unspecified")
        raise RuntimeError(f"Read command failed for {state} with code {process.returncode}")
    if process.stdout.strip() == "No jobs found":
        return {}
    return json.loads(process.stdout)


def current_jobs():
    jobs = {}
    for state in STATES:
        page = 1
        while True:
            response = command_json([CLI, "acp", "jobs", "list", "--workspace-name=share-space",
                "--user-name=260010165", f"--state={state}", "--page-size=20", f"--page-token={page}", "-o", "json"])
            rows = response if isinstance(response, list) else response.get("training_jobs", [])
            for row in rows:
                gpu_count = 0
                for role in row.get("roles", []):
                    specifications = role.get("resource_spec", [])
                    if not specifications:
                        raise RuntimeError("Active role omitted its resource specification")
                    per_worker = max(int(spec.get("limits", {}).get("nvidia.com/gpu", 0)) for spec in specifications)
                    gpu_count += per_worker * max(1, int(role.get("total_replicas", 1)))
                jobs[row["uid"]] = {"uid": row["uid"], "name": row.get("display_name") or row.get("name", ""),
                    "state": state, "gpus": gpu_count,
                    "quota": row.get("scheduling", {}).get("quota_type", "UNKNOWN")}
            if len(rows) < 20:
                break
            page += 1
            if page > 50:
                raise RuntimeError("Unexpectedly large active job listing")
    return list(jobs.values())


def storage_ready(required_gib=192):
    process = subprocess.run([CLI, "afs", "dir-quota", "list",
        "--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5", "--dirPath=/260010165"],
        capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise RuntimeError("Unable to read AFS quota")
    line = next(line for line in process.stdout.splitlines() if "/260010165" in line)
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s+(TB|GB|MB)", line)
    if len(sizes) != 2:
        raise RuntimeError("Unexpected AFS capacity response")
    factors = {"TB": 10**12, "GB": 10**9, "MB": 10**6}
    capacity, used = [float(size) * factors[unit] for size, unit in sizes]
    # The CLI rounds its output. Reserve an additional 0.1 TB for rounding and old runs.
    return capacity - used - 10**11 >= required_gib * 2**30


def required_storage_gib(job, active):
    if job.get("storage_required_gib"):
        active_runs = sum(row["name"].startswith("b1k-rl-pc-") for row in active)
        return int(job["storage_required_gib"]) * (active_runs + 1)
    return 64 if job.get("diagnostic", False) else 192


def remote_progress(client, tag):
    prefix = f"rlinf_runs_20261104/{tag}/"
    try:
        state = json.loads(client.get_object(Bucket=aoss_io.BUCKET, Key=prefix + "status.json")["Body"].read())
    except client.exceptions.NoSuchKey:
        return {"state": "submitted"}, {}
    stages = {}
    try:
        body = client.get_object(Bucket=aoss_io.BUCKET, Key=prefix + "tensorboard.tgz")["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
            member = next((m for m in archive.getmembers() if m.name.lstrip("./") == "stages_status.json"), None)
            if member:
                stages = json.load(archive.extractfile(member))
    except client.exceptions.NoSuchKey:
        pass
    return {"state": state.get("state", "unknown"), "time": state.get("time")}, stages


def failed(remote):
    if any(value not in (0, None) for value in remote["stages"].values()):
        return True
    state = str(remote["state"])
    if state.startswith("finished_rc_"):
        return state != "finished_rc_0"
    return state.startswith(("fetch_failed", "sha_failed", "ckpt_missing", "ckpt_strip_failed"))


def rate_ready():
    ledger_path = LEGACY / "submission_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    now = dt.datetime.now()
    times = [dt.datetime.fromisoformat(row["time"]) for row in ledger]
    return (not times or (now - max(times)).total_seconds() >= 900) and (
        sum((now - when).total_seconds() < 3600 for when in times) < 3
        and sum(when.date() == now.date() for when in times) < 50)


def relay_ready():
    path = ROOT / "relay_health.json"
    return path.exists() and time.time() - json.loads(path.read_text())["time"] < 60


def submit(job, quota):
    hours = job.get("wall_hours", 8) if job.get("resume_from") or job.get("checkpoint_path") else 8
    command = [PYTHON, str(ROOT / "submit.py"), "--tag", job["tag"], "--pool", "5090",
               "--gpus", "4", "--quota", quota, "--venv", ".venv", "--entry", "multi",
               "--config", "libero_10_ppo_pi05_5090x4", "--stages", str(ROOT / "stages" / (job["tag"] + ".json")),
               "--hours", str(hours)]
    if job.get("raw_script"):
        command.extend(["--raw-script", job["raw_script"]])
    with (ROOT / f"submit_{job['tag']}.log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=360)
    if process.returncode:
        raise RuntimeError(f"Submission failed for {job['tag']}; manual diagnosis required")
    path = LEGACY / "monitor_tags.txt"
    tags = path.read_text().splitlines()
    if job["tag"] not in tags:
        with path.open("a") as handle:
            handle.write(job["tag"] + "\n")


def main():
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    jobs = json.loads((ROOT / "jobs.json").read_text())
    continuation_file = ROOT / "reviewed_continuations.json"
    continuations = json.loads(continuation_file.read_text()) if continuation_file.exists() else []
    continuations = [job for job in continuations if not job.get("diagnostic", False)]
    jobs = continuations + jobs
    terminal_file = ROOT / "known_terminal_jobs.json"
    known_terminal = json.loads(terminal_file.read_text()) if terminal_file.exists() else {}
    client = aoss_io.local()
    deadline = time.monotonic() + 72 * 3600
    with (ROOT / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.monotonic() < deadline:
            status = {"at": dt.datetime.now().isoformat(), "jobs": {}}
            for job in jobs:
                receipt = LEGACY / "runs" / f"{job['tag']}.json"
                if not receipt.exists():
                    continue
                receipt_data = json.loads(receipt.read_text())
                if receipt_data.get("submit_rc") != 0:
                    raise RuntimeError(f"{job['tag']} has an unsuccessful submission receipt")
                try:
                    remote, stages = remote_progress(client, job["tag"])
                except Exception as error:
                    status["jobs"][job["tag"]] = {"state": "observation_unavailable", "stages": {}}
                    status.setdefault("observation_errors", {})[job["tag"]] = type(error).__name__
                    continue
                remote["stages"] = {name: value.get("rc") for name, value in stages.get("results", {}).items()}
                if job["tag"] in known_terminal:
                    remote.update(known_terminal[job["tag"]])
                status["jobs"][job["tag"]] = remote
                if job["tag"] in known_terminal:
                    continue
                if failed(remote):
                    status["attention"] = f"{job['tag']} failed; automatic resubmission is disabled"
                    write_json(ROOT / "dispatch_state.json", status)
                    return
                if job["tag"] == "pc_fixed_s1" and remote["stages"].get("smoke") == 0:
                    write_json(ROOT / "engineering_gate.json", {"passed": True, "source": job["tag"], "at": status["at"]})
            write_json(ROOT / "dispatch_state.json", status)
            if status.get("observation_errors"):
                time.sleep(1200)
                continue
            required = [job["tag"] for job in jobs
                        if not known_terminal.get(job["tag"], {}).get("continuation_tag")]
            if all(status["jobs"].get(tag, {}).get("stages", {}).get("eval_s20") == 0 for tag in required):
                status["complete"] = True
                write_json(ROOT / "dispatch_state.json", status)
                return
            pending = [job for job in jobs if job["tag"] not in status["jobs"]]
            unfinished = any(tag not in known_terminal and row.get("stages", {}).get("eval_s20") != 0
                             for tag, row in status["jobs"].items())
            if (pending or unfinished) and (ROOT / "engineering_gate.json").exists():
                try:
                    active = current_jobs()
                    space_ready = not pending or storage_ready(required_storage_gib(pending[0], active))
                except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    status["scheduler_observation_error"] = str(error)[:300]
                    write_json(ROOT / "dispatch_state.json", status)
                    time.sleep(1200)
                    continue
                status["active_account_jobs"] = active
                active_names = {row["name"] for row in active}
                for tag, row in status["jobs"].items():
                    if tag in known_terminal or row.get("stages", {}).get("eval_s20") == 0:
                        continue
                    receipt = json.loads((LEGACY / "runs" / f"{tag}.json").read_text())
                    if receipt["name"] in active_names:
                        continue
                    try:
                        detail = command_json([CLI, "acp", "jobs", "describe", receipt["name"],
                                               "--workspace-name=share-space", "-o", "json"])
                    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError):
                        continue
                    if detail.get("state") in ("FAILED", "DELETED", "SUSPENDED"):
                        entry = {"state": detail["state"], "verified_at": status["at"]}
                        known_terminal[tag] = entry
                        row.update(entry)
                        write_json(terminal_file, known_terminal)
                        write_json(ROOT / "dispatch_attention.json", {
                            "at": status["at"], "tag": tag,
                            "reason": "Platform job ended before final results; manual recovery review required"})
                own = [row for row in active if row["name"].startswith("b1k-rl-pc-")]
                queued = [row for row in active if row["state"] != "RUNNING"]
                if pending and sum(row["gpus"] for row in own) + 4 <= 8 and len(queued) < 2 and sum(row["gpus"] for row in active) + 4 <= 32:
                    if space_ready and rate_ready() and (pending[0]["arm"] != "agent" or relay_ready()):
                        submit(pending[0], "spot")
                write_json(ROOT / "dispatch_state.json", status)
            time.sleep(1200)
        write_json(ROOT / "dispatch_attention.json", {"reason": "72-hour dispatcher deadline reached"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(ROOT / "dispatch_attention.json", {"at": dt.datetime.now().isoformat(),
            "type": type(error).__name__, "reason": str(error)[:300]})
        raise

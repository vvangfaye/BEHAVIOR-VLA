"""Bounded continuation: preserve completed work, then submit exactly one job.

Runs locally. It never overlaps the old and new GPU workers and never relaxes
the experiment's numerical or closed-loop gates.
"""

import hashlib
import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "cloud_smoke_retry06_20261023"
ROOT = HERE.parents[2]
PYTHON = str(HERE.parent / "venv/bin/python")
OLD_UID = "19f023b244544fd787eb449c546b5ed4"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def state(stage, **fields):
    value = {"stage": stage, "updated_at": time.time(), **fields}
    target = HERE / "continuation_state.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(args, timeout):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(args[1]).name} returned {result.returncode}: {result.stderr[-1600:]}")
    return result


def main():
    with (HERE / "continuation_intent.json").open("x") as stream:
        json.dump({"old_uid": OLD_UID, "new_job": "b1k-rlinf-1023-r07", "gpu_count": 1, "max_new_gpu_hours": 6}, stream)
    state("waiting_for_first_complete_episode")
    deadline = time.monotonic() + 3 * 3600
    first = None
    while time.monotonic() < deadline:
        receipts = list(OLD.glob("live_[0-9]*.json"))
        if receipts:
            latest = max(receipts, key=lambda p: int(p.stem.split("_")[1]))
            first = read(latest).get("files", {}).get("results/jax/301/result.json")
            if first is not None:
                break
        status = read(OLD / "latest_job_status.json")
        if status and status["state"] in {"FAILED", "TERMINATED", "CANCELED", "SUCCEEDED"}:
            raise RuntimeError("Old job ended before the planned controlled continuation")
        time.sleep(15)
    else:
        raise TimeoutError("No completed first episode before continuation deadline")
    assert first["backend"] == "jax" and first["instance"] == 301
    state("preserving_first_episode", first_episode=first)
    run(
        [PYTHON, str(HERE.parent / "forensics_20261023/exec_probe.py"),
         "--job", "b1k-rlinf-1023-r06", "--uid", OLD_UID,
         "--script", str(OLD / "finish_stage_for_resume.py"),
         "--output", str(HERE / "controlled_resume_receipt.json"), "--timeout", "60"],
        110,
    )
    state("waiting_for_old_worker_and_archive", first_episode=first)
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        status = read(OLD / "latest_job_status.json")
        monitor = read(OLD / "monitor_state.json")
        if status and status["uid"] == OLD_UID and status["state"] in {"FAILED", "TERMINATED", "CANCELED", "STOPPED"} and monitor and monitor.get("state") == "results_retrieved":
            break
        time.sleep(15)
    else:
        raise TimeoutError("Old worker/archive not confirmed; no new job was submitted")
    with tarfile.open(OLD / "result.tar.gz", "r:gz") as archive:
        for member in archive:
            rel = Path(member.name)
            assert member.isfile() and not rel.is_absolute() and ".." not in rel.parts
            if rel.suffix != ".npz":
                continue
            target = OLD / "retrieved" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                target.write_bytes(stream.read())
    returned = OLD / "retrieved"
    assert read(returned / "jax/301/result.json") == first
    assert (returned / "jax/301/initial.json").is_file()
    assert (returned / "jax/301/actions.json").is_file()
    assert read(returned / "controlled_resume.json")["second_has_no_control_steps"]
    assert read(returned / "precision_gate.json")["passed"]
    assert not (returned / "jax/302/result.json").exists()
    files = {}
    for path in returned.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(returned)
        take = ((str(rel).startswith("jax/301/") and rel.suffix in {".json", ".npz"})
                or (str(rel).startswith("precheck_reference/") and rel.suffix == ".npz")
                or str(rel) in {"precision_gate.json", "reference_progress.json"})
        if take:
            files[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sum(name.startswith("precheck_reference/") for name in files) == 12
    plan = read(HERE / "plan.json")
    plan["reuse_files_sha256"] = files
    plan["recovery"].update(planned_only=False, previous_state=status["state"], first_episode=first)
    (HERE / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    for name in ("common.py", "simulator.py", "helpers.py", "gate.py", "ppo.py"):
        assert (HERE / name).read_bytes() == (OLD / name).read_bytes(), name
    for source in (OLD / "b1k_rlinf").glob("*.py"):
        assert source.read_bytes() == (HERE / "b1k_rlinf" / source.name).read_bytes()
    (HERE / "local_preflight.json").write_text(json.dumps({
        "passed": True, "same_model_and_simulator_source": True,
        "reused_first_episode": first, "reused_files": files,
        "old_worker_terminal": status, "gpu_count": 1,
        "max_new_gpu_hours": 6, "remaining_jax_instances": [302],
        "new_torch_instances": [301, 302], "ppo_gate_unchanged": True,
    }, indent=2) + "\n")
    state("submitting_one_continuation", first_episode=first)
    result = run([PYTHON, str(HERE / "deploy.py")], 240)
    print(result.stdout, flush=True)
    intent = read(HERE / "submission_intent.json")
    assert intent and intent.get("uid"), "Submission outcome needs exact-name verification"
    uid = intent["uid"]
    observer = (OLD / "observe_progress.py").read_text().replace("b1k-rlinf-1023-r06", "b1k-rlinf-1023-r07").replace(OLD_UID, uid)
    (HERE / "observe_progress.py").write_text(observer)
    with (HERE / "observe_progress.log").open("a") as log:
        child = subprocess.Popen([PYTHON, str(HERE / "observe_progress.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    state("continuation_submitted", job=intent["job_name"], uid=uid, observer_pid=child.pid, first_episode=first)
    with (HERE / "monitor.log").open("a") as log:
        result = subprocess.run([PYTHON, str(HERE / "monitor.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7.5*3600)
    assert result.returncode == 0
    audit = run([PYTHON, str(HERE / "audit_result.py")], 60)
    report = read(HERE / "acceptance_report.json")
    state("complete" if report["passed"] else "needs_attention", job=intent["job_name"], uid=uid, passed=report["passed"], audit=json.loads(audit.stdout))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        state("needs_attention", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise

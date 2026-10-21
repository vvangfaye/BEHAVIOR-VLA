"""Finish the existing frozen study, then run its separately fixed native panel.

No training, checkpoint selection, retries, or changes to either study protocol.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / "closed_loop_exploratory_ada01"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # Linux stat field 22: process start time.
    except FileNotFoundError:
        return None


def main():
    transition = read(ROOT / "transition_plan.json")
    for name, expected in transition["inputs_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Transition input changed: " + name)
    parent_pid = transition["training_controller_pid"]
    expected_identity = transition["training_controller_starttime"]
    completion = HERE / "pilot_ada01/completion.json"
    if not completion.exists() and process_identity(parent_pid) != expected_identity:
        raise RuntimeError("The expected training controller is not live")
    with (ROOT / "transition_launch.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started_at=stamp(),
                       plan_sha256=sha(ROOT / "transition_plan.json")), stream, indent=2)
    state = dict(stage="waiting_for_existing_training", updated_at=stamp(),
                 parent_pid=parent_pid, parent_starttime=expected_identity)
    write(ROOT / "transition_state.json", state)
    deadline = time.monotonic() + transition["max_wait_seconds"]
    child = None

    def interrupted(*_):
        raise InterruptedError("Transition supervisor interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while not completion.exists() or process_identity(parent_pid) == expected_identity:
            identity = process_identity(parent_pid)
            if identity != expected_identity and not completion.exists():
                raise RuntimeError("Training ended without a completion receipt; no retry")
            if time.monotonic() >= deadline:
                raise TimeoutError("Training wait cap reached; no dependent launch")
            time.sleep(20)
        if read(completion).get("complete") is not True:
            raise ValueError("Training completion is not accepted")
        for name, expected in transition["inputs_sha256"].items():
            if sha(HERE / name) != expected:
                raise ValueError("Transition input changed while waiting: " + name)
        state.update(stage="analyzing_frozen_four_arm_study", updated_at=stamp())
        write(ROOT / "transition_state.json", state)
        with (ROOT / "four_arm_analysis.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, str(HERE / "analyze.py")],
                                     cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait(timeout=600)
        child = None
        if code:
            raise RuntimeError("Frozen four-arm analysis failed")
        report = read(HERE / "pilot_ada01/reanalysis.json")
        if report.get("ready") is not True:
            raise ValueError("Four-arm analysis incomplete")
        state.update(stage="running_separate_exploratory_native_study", updated_at=stamp(),
                     original_offline_screen_passed=report["screen_passed"],
                     offline_report_sha256=sha(HERE / "pilot_ada01/reanalysis.json"))
        write(ROOT / "transition_state.json", state)
        print(json.dumps(state), flush=True)
        child = subprocess.Popen([sys.executable, "-u", str(HERE / "run_native_study.py"),
                                  "--execute"], cwd=HERE)
        code = child.wait(timeout=transition["max_native_seconds"])
        child = None
        if code or not (ROOT / "completion.json").exists():
            raise RuntimeError("Native study did not complete; no automatic retry")
        state.update(stage="complete", updated_at=stamp(), complete=True)
        write(ROOT / "transition_state.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(stage="stopped", updated_at=stamp(), complete=False,
                     error_type=type(error).__name__, error=str(error))
        write(ROOT / "transition_state.json", state)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()

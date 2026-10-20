"""One bounded, CPU-only subprocess; refuses reusing a run identity."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--name", choices=("baseline1688977", "expert1000"), required=True)
args = parser.parse_args()
plan = json.loads((HERE / "plan.json").read_text())
spec = plan["runs"][args.name]
runtime = PROJECT / "analysis/organizer_rehearsal_1688977_20261016/runtime/policy-bundle/rootfs/mnt/afs/260010165/behavior_1k/runtime_20261007_1306"
environment = os.environ.copy()
environment.update(PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="0",
    CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="16", OPENBLAS_NUM_THREADS="16", MKL_NUM_THREADS="16",
    LD_LIBRARY_PATH=str(runtime / "envs/b1k-pi05/lib"), OPENPI_DATA_HOME=str(runtime / "openpi-cache"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    PYTHONPATH=str(PROJECT / "BEHAVIOR-VLAs/openpi/src") + ":" + str(PROJECT / "BEHAVIOR-VLAs/openpi/packages/openpi-client/src"))
entry = ("import asyncio,runpy,sys;sys.path.insert(0," + repr(str(HERE.parent / "flow_endpoint_screen_20261020")) + ");"
         "from run_local_cpu import PeriodicWakeupPolicy;asyncio.set_event_loop_policy(PeriodicWakeupPolicy());"
         "sys.argv=" + repr([str(HERE / "probe.py"), "--name", args.name, "--checkpoint", spec["checkpoint"],
                               "--cpus", ",".join(str(x) for x in spec["cpus"])]) + ";"
         "runpy.run_path(" + repr(str(HERE / "probe.py")) + ",run_name='__main__')")
status_path = HERE / f"{args.name}.status.json"
status = {"name": args.name, "status": "starting", "max_seconds": plan["maximum_seconds_per_checkpoint"],
          "cpus": spec["cpus"], "training": False, "simulation": False}
with status_path.open("x") as stream:
    json.dump(status, stream)
start = time.monotonic()
with (HERE / f"{args.name}.log").open("x") as log:
    process = subprocess.Popen([str(runtime / "envs/b1k-pi05/bin/python3.11"), "-u", "-c", entry],
        cwd=PROJECT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status.update(status="running", pid=process.pid)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        code = process.wait(timeout=plan["maximum_seconds_per_checkpoint"])
        status.update(status="complete" if code == 0 else "failed", exit_code=code)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        status.update(status="timed_out", exit_code=process.returncode)
status.update(elapsed_seconds=time.monotonic() - start, result_exists=(HERE / f"{args.name}.json").exists())
status_path.write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status))
sys.exit(0 if status["status"] == "complete" else 1)

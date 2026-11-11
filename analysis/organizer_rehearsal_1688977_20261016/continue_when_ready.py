"""Finite local continuation: verify preparation, then admit one idle-GPU episode."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(json.loads((ROOT / "RUNTIME_ALIAS.json").read_text())["runtime"]) if (ROOT / "RUNTIME_ALIAS.json").exists() else ROOT / "runtime"


def status(state, **extra):
    value = {"state": state, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}
    temp = ROOT / "SETUP_STATUS.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(ROOT / "SETUP_STATUS.json")
    print(state, flush=True)


def cpu_checks():
    python = RUNTIME / "b1k-eval/bin/python"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMNIGIBSON_NO_OMNIVERSE="1", OMNI_KIT_ACCEPT_EULA="YES",
               OMNIGIBSON_DATA_PATH=str(RUNTIME / "datasets"), TMPDIR=str(RUNTIME / "tmp"), PYTHONNOUSERSITE="1",
               LD_LIBRARY_PATH=str(RUNTIME / "b1k-eval/lib"),
               PYTHONHOME=str(RUNTIME / "b1k-eval"),
               TORCHINDUCTOR_CACHE_DIR=str(RUNTIME / "cache/torchinductor"),
               TRITON_CACHE_DIR=str(RUNTIME / "cache/triton"))
    with (RUNTIME / "logs/evaluator-preflight.log").open("w") as output:
        subprocess.run([str(python), "-m", "omnigibson.eval.eval", "--help"], env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=300)
    subprocess.run([str(python), str(ROOT / "unpack_image.py")], check=True, timeout=3600)
    bundle = RUNTIME / "policy-bundle"
    spec = json.loads((RUNTIME / "rootless_probe/config.json").read_text())
    image_manifest = json.loads((RUNTIME / "image/oci/blobs/sha256/469ddc6a55ee7e2f1a9486c3d366b1de7c44469750562bc628b099c4f88c2dea").read_text())
    image = json.loads((RUNTIME / "image/oci/blobs/sha256" / image_manifest["config"]["digest"].split(":")[1]).read_text())
    image_env = dict(e.split("=", 1) for e in image["config"]["Env"])
    image_python = image_env["PI05_RUNTIME"] + "/envs/b1k-pi05/bin/python"
    spec["process"]["args"] = [image_python, "-c", "import sys,importlib.metadata as m;print(sys.version);print('jax',m.version('jax'))"]
    spec["process"]["env"] = ["PATH=/usr/bin:/bin", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1"]
    spec["process"]["capabilities"] = {k: [] for k in ["bounding", "effective", "permitted"]}
    spec["linux"]["namespaces"].append({"type": "network"})
    (bundle / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
    with (RUNTIME / "logs/policy-cpu-preflight.log").open("w") as output:
        subprocess.run(["runc", "--root", "/tmp/b1k-rehearsal-cpu-preflight", "--rootless=true", "run",
                        "--bundle", str(bundle), "b1k-policy-cpu-preflight"],
                       stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120)
    (ROOT / "CPU_PREFLIGHT_PASSED.json").write_text(json.dumps({
        "evaluator_cli": True, "rootless_policy_python": True, "gpu_work_performed": False,
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-timeout", type=int, default=14400)
    parser.add_argument("--gpu-wait-seconds", type=int, default=21600)
    args = parser.parse_args()
    workers = json.loads((ROOT / "ONGOING_WORKERS.json").read_text())
    deadline = time.monotonic() + args.preparation_timeout
    while True:
        missing = []
        for worker in workers:
            if (ROOT / worker["marker"]).exists():
                continue
            missing.append(worker["marker"])
            proc = Path("/proc") / str(worker["pid"])
            stats = proc.joinpath("stat").read_text().split() if proc.exists() else []
            if not stats or stats[2] == "Z" or stats[21] != worker["start_ticks"]:
                raise RuntimeError("Preparation worker exited without readiness marker: " + worker["marker"])
        if not missing:
            break
        status("preparing", awaiting=missing, preparation_deadline_epoch=time.time() + deadline - time.monotonic())
        if time.monotonic() >= deadline:
            raise TimeoutError("Preparation exceeded the finite four-hour wait")
        time.sleep(30)
    status("cpu_verification")
    if not ((ROOT / "CPU_PREFLIGHT_PASSED.json").exists() and
            (ROOT / "ISOLATED_CPU_PREFLIGHT_PASSED.json").exists()):
        cpu_checks()
    status("waiting_for_idle_gpu", gpu_wait_seconds=args.gpu_wait_seconds)
    subprocess.run([str(RUNTIME / "b1k-eval/bin/python"), str(ROOT / "run_one_episode.py"),
                    "--wait-seconds", str(args.gpu_wait_seconds)], check=True)
    result = json.loads((ROOT / "RUN_STATE.json").read_text())
    status(result["state"], episode_state=result)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("needs_attention", error_type=type(error).__name__, error=str(error))
        raise

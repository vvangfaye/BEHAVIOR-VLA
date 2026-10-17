"""One independent public episode. Never stop or preempt another user's GPU work."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(json.loads((ROOT / "RUNTIME_ALIAS.json").read_text())["runtime"]) if (ROOT / "RUNTIME_ALIAS.json").exists() else ROOT / "runtime"
PYTHON = RUNTIME / "b1k-eval/bin/python"
RUNC_STATE = Path("/tmp") / f"b1k-rehearsal-runc-{os.getuid()}"
CONTAINER = "b1k-policy-1688977-rehearsal"
PORT = 18080
TASK = "turning_on_radio"
DIGEST = "sha256:469ddc6a55ee7e2f1a9486c3d366b1de7c44469750562bc628b099c4f88c2dea"


def save(state, **kwargs):
    data = {"state": state, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task": TASK, "instance_id": 301, "rollout_id": 0, "image_digest": DIGEST, **kwargs}
    temporary = ROOT / "RUN_STATE.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(ROOT / "RUN_STATE.json")


def gpu_snapshot(preferred_index=None):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,pci.bus_id", "--format=csv,noheader,nounits"
    ], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    occupied = {r[0].strip() for r in csv.reader(io.StringIO(processes)) if r}
    snapshots = [{"index": int(row[0]), "uuid": row[1].strip(), "total_MiB": int(row[2]),
                  "used_MiB": int(row[3]), "utilization": int(row[4]),
                  "has_compute_process": row[1].strip() in occupied, "pci_bus_id": row[5].strip()}
                 for row in csv.reader(io.StringIO(output))]
    if preferred_index is not None:
        return next(gpu for gpu in snapshots if gpu["index"] == preferred_index)
    available = [gpu for gpu in snapshots if not gpu["has_compute_process"] and
                 gpu["total_MiB"] - gpu["used_MiB"] >= 44000 and gpu["utilization"] == 0]
    return (available or snapshots)[0]


def bind(config, source, destination, readonly=True):
    config["mounts"].append({"source": str(source), "destination": destination, "type": "bind",
                             "options": ["bind", "nosuid", "ro" if readonly else "rw"]})


def configure_policy(bundle, gpu):
    config = json.loads((RUNTIME / "rootless_probe/config.json").read_text())
    image_manifest = json.loads((RUNTIME / "image/oci/blobs/sha256" / DIGEST.split(":")[1]).read_text())
    image = json.loads((RUNTIME / "image/oci/blobs/sha256" / image_manifest["config"]["digest"].split(":")[1]).read_text())
    meta = image["config"]
    config["root"] = {"path": "rootfs", "readonly": False}
    config["hostname"] = "b1k-policy-rehearsal"
    config["process"]["args"] = [env_python := dict(s.split("=", 1) for s in meta.get("Env", []))["PI05_RUNTIME"]
                                  + "/envs/b1k-pi05/bin/python", str(ROOT / "loopback_and_policy.py")]
    config["process"]["cwd"] = meta.get("WorkingDir") or "/"
    env = dict(s.split("=", 1) for s in meta.get("Env", []))
    env.update(TASK=TASK, PORT=str(PORT), XLA_PYTHON_CLIENT_PREALLOCATE="false",
               CUDA_VISIBLE_DEVICES=gpu["uuid"], NVIDIA_VISIBLE_DEVICES=gpu["uuid"],
               LD_LIBRARY_PATH="/usr/local/nvidia/lib64:" + env.get("LD_LIBRARY_PATH", ""),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    config["process"]["env"] = [f"{k}={v}" for k, v in env.items()]
    # NET_ADMIN exists only inside the new private network namespace. The
    # bootstrap drops it before starting the submitted policy entrypoint.
    config["process"]["capabilities"] = {k: ["CAP_NET_ADMIN"] for k in ["bounding", "effective", "permitted"]}
    config["linux"]["namespaces"].append({"type": "network"})
    config["process"]["rlimits"] = [{"type": "RLIMIT_NOFILE", "hard": 65536, "soft": 65536}]
    for mount in config["mounts"]:
        if mount["destination"] == "/dev/shm":
            mount["options"] = ["nosuid", "nodev", "mode=1777", "size=8g"]
    for name in [f"nvidia{gpu['index']}", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset"]:
        device = Path("/dev") / name
        if device.exists():
            bind(config, device, str(device), readonly=False)
    for node in Path("/dev/dri/by-path").glob("pci-" + gpu["pci_bus_id"][-12:].lower() + "-*"):
        bind(config, node.resolve(), str(node.resolve()), readonly=False)
    library_root = Path("/usr/lib/x86_64-linux-gnu")
    libraries = set()
    for pattern in ["libcuda.so*", "libnvidia-*.so*", "libEGL_nvidia.so*", "libGLX_nvidia.so*", "libGLES*_nvidia.so*"]:
        libraries.update(library_root.glob(pattern))
    for library in sorted(libraries):
        if library.is_file():
            bind(config, library.resolve(), "/usr/local/nvidia/lib64/" + library.name)
    hosts = RUNTIME / "policy-hosts"
    hosts.write_text("127.0.0.1 localhost b1k-policy-rehearsal\n::1 localhost\n")
    bind(config, hosts, "/etc/hosts")
    bind(config, ROOT, str(ROOT), readonly=False)
    if RUNTIME.parent != ROOT:
        bind(config, ROOT, str(RUNTIME.parent), readonly=False)
    # The simulator environment is external to the submitted policy paths.
    bind(config, "/usr/lib/x86_64-linux-gnu", "/rehearsal-host-libs")
    # The external simulator's Torch compiler uses the host toolchain. These
    # read-only mounts do not replace the submitted model's source or runtime.
    for source in ["/usr/include", "/usr/lib/gcc", "/usr/bin/g++", "/usr/bin/gcc",
                   "/usr/bin/c++", "/usr/bin/cc", "/usr/bin/git", "/usr/bin/as", "/usr/bin/ld",
                   "/usr/bin/ld.bfd", "/usr/lib/x86_64-linux-gnu/libc.so",
                   "/usr/lib/x86_64-linux-gnu/libc_nonshared.a",
                   "/usr/lib/x86_64-linux-gnu/libpthread.a", "/usr/lib/x86_64-linux-gnu/libdl.a",
                   "/usr/lib/x86_64-linux-gnu/librt.a", "/usr/lib/x86_64-linux-gnu/libutil.a",
                   "/usr/lib/x86_64-linux-gnu/libm.so", "/usr/lib/x86_64-linux-gnu/libmvec.so",
                   "/usr/lib/x86_64-linux-gnu/libm.so.6", "/usr/lib/x86_64-linux-gnu/libmvec.so.1",
                   "/usr/lib/x86_64-linux-gnu/Scrt1.o", "/usr/lib/x86_64-linux-gnu/crt1.o",
                   "/usr/lib/x86_64-linux-gnu/crti.o", "/usr/lib/x86_64-linux-gnu/crtn.o"]:
        if Path(source).exists():
            bind(config, Path(source).resolve(), source)
    for source in ["/usr/share/vulkan/icd.d/nvidia_icd.json", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"]:
        if Path(source).exists():
            bind(config, source, source)
    (bundle / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    for prerequisite in ["IMAGE_READY.json", "DATASETS_READY.json", "ENVIRONMENT_READY.json"]:
        if not (ROOT / prerequisite).exists():
            raise SystemExit("Missing prerequisite: " + prerequisite)
    marker = ROOT / "ONE_EPISODE_STARTED.json"
    if marker.exists():
        raise SystemExit("An episode has already been started; inspect its artifacts before any further action.")
    bundle = RUNTIME / "policy-bundle"
    if not (bundle / "ROOTFS_READY.json").exists():
        save("unpacking_verified_image")
        subprocess.run([str(PYTHON), str(ROOT / "unpack_image.py")], check=True)
    end = time.monotonic() + max(0, args.wait_seconds)
    stable = 0
    last_uuid = None
    while True:
        gpu = gpu_snapshot()
        available = not gpu["has_compute_process"] and gpu["total_MiB"] - gpu["used_MiB"] >= 44000 and gpu["utilization"] == 0
        stable = (stable + 1 if last_uuid == gpu["uuid"] else 1) if available else 0
        last_uuid = gpu["uuid"]
        if stable >= 3:
            break
        save("waiting_for_free_gpu", gpu=gpu, stable_observations=stable)
        if time.monotonic() >= end and not available:
            save("blocked_gpu_busy", gpu=gpu, other_processes_untouched=True)
            return
        time.sleep(10)
    configure_policy(bundle, gpu)
    # Admission recheck immediately before starting either GPU workload.
    final_gpu = gpu_snapshot(gpu["index"])
    if final_gpu["has_compute_process"] or final_gpu["total_MiB"] - final_gpu["used_MiB"] < 44000:
        save("blocked_gpu_changed_during_admission", gpu=final_gpu)
        return
    output = RUNTIME.parent / "episode_turning_on_radio_301"
    output.mkdir(exist_ok=False)
    marker.write_text(json.dumps({"task": TASK, "instance_id": 301, "gpu": gpu, "time": time.time()}) + "\n")
    policy = evaluator = None
    try:
        with (output / "policy.log").open("w") as log:
            policy = subprocess.Popen(["runc", "--root", str(RUNC_STATE), "--rootless=true", "run",
                                       "--bundle", str(bundle), CONTAINER], stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        save("starting_policy", gpu=gpu, policy_pid=policy.pid, output=str(output))
        deadline = time.monotonic() + 900
        while True:
            if policy.poll() is not None:
                raise RuntimeError("Policy process exited during startup")
            ready = subprocess.run(["runc", "--root", str(RUNC_STATE), "exec", CONTAINER,
                                    str(PYTHON), "-c", "import socket; socket.create_connection(('127.0.0.1',18080),1).close()"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            if ready.returncode == 0:
                break
            else:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Policy startup exceeded 15 minutes")
                time.sleep(2)
        env = {"PATH": str(PYTHON.parent) + ":/usr/local/bin:/usr/bin:/bin", "HOME": str(RUNTIME / "home")}
        env.update(OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_GPU_ID="0",
                   CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1",
                   OMNIGIBSON_DATA_PATH=str(RUNTIME / "datasets"), OMNIGIBSON_APPDATA_PATH=str(RUNTIME / "appdata"),
                   TMPDIR=str(RUNTIME / "tmp"), LD_LIBRARY_PATH=str(RUNTIME / "b1k-eval/lib") + ":/usr/local/nvidia/lib64:/rehearsal-host-libs",
                   VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json",
                   TORCHINDUCTOR_CACHE_DIR=str(RUNTIME / "cache/torchinductor"),
                   TRITON_CACHE_DIR=str(RUNTIME / "cache/triton"))
        env["PYTHONHOME"] = str(RUNTIME / "b1k-eval")
        command = [str(PYTHON), "-m", "omnigibson.eval.eval", "--task-name", TASK, "--host", "127.0.0.1",
                   "--port", str(PORT), "--mode", "public_test", "--prompt-mode", "whole_task",
                   "--policy", "websocket", "--instance-indices", "0", "--num-rollouts", "1",
                   "--env-wrapper", "omnigibson.eval.wrappers.DefaultWrapper", "--output-dir", str(output), "--write-video"]
        (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        (RUNTIME / "home").mkdir(exist_ok=True)
        process_spec = {"terminal": False, "user": {"uid": 0, "gid": 0}, "args": command,
                        "env": [f"{k}={v}" for k, v in env.items()],
                        "cwd": str(RUNTIME.parent / "frozen_source/BEHAVIOR-1K/OmniGibson"), "noNewPrivileges": True,
                        "capabilities": {"bounding": [], "effective": [], "permitted": []}}
        process_file = output / "evaluator-process.json"
        process_file.write_text(json.dumps(process_spec, indent=2) + "\n")
        with (output / "evaluator.log").open("w") as log:
            evaluator = subprocess.Popen(["runc", "--root", str(RUNC_STATE), "exec", "--process",
                                          str(process_file), CONTAINER],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save("running_episode", gpu=gpu, policy_pid=policy.pid, evaluator_pid=evaluator.pid, output=str(output))
        code = evaluator.wait(timeout=21600)
        videos = [str(p) for p in output.rglob("*.mp4") if p.stat().st_size > 0]
        results = []
        for path in output.rglob("*.json"):
            try:
                metrics = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(metrics, dict) and "q_score" in metrics and "success" in metrics:
                results.append({"path": str(path), "success": metrics["success"], "q_score": metrics["q_score"]})
        if code != 0 or not videos or len(results) != 1:
            raise RuntimeError(f"Evaluator exit={code}, videos={len(videos)}, JSONs={len(results)}")
        video_artifacts = []
        for name in videos:
            path = Path(name)
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            video_artifacts.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
        save("episode_finished_needs_artifact_review", gpu=gpu, output=str(output), videos=video_artifacts, results=results)
    except BaseException as error:
        save("episode_infrastructure_error", error_type=type(error).__name__, error=str(error), output=str(output))
        raise
    finally:
        for process in [evaluator, policy]:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        subprocess.run(["runc", "--root", str(RUNC_STATE), "delete", "--force", CONTAINER],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()

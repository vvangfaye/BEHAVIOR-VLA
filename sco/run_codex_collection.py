"""Finite SCO worker driver for one isolated 311--320 Codex collection attempt.

The caller stages a separate code tree and supplies hashes of every new/changed
file. Immutable source/checkpoint receipts are verified without editing them.
No queue, cloud submission, credentials, or official score is created here.
"""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROBOT_CONFIG = "BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml"


def robot_contract(config):
    """Static check; the correction adapter repeats this against live controllers."""
    if config.get("model") != "r1pro" or config.get("action_normalize") is not False:
        raise ValueError("Robot must use unnormalized R1Pro actions")
    controllers = config["controller_config"]
    for name in ("trunk", "arm_left", "arm_right"):
        ctrl = controllers[name]
        if (ctrl.get("name") != "JointController" or ctrl.get("motor_type") != "position"
                or ctrl.get("use_delta_commands") is not False or ctrl.get("use_impedances") is not False
                or ctrl.get("command_input_limits") is not None or ctrl.get("command_output_limits") is not None):
            raise ValueError(name + " must accept absolute unscaled joint positions")
    if controllers["camera"].get("name") != "NullJointController":
        raise ValueError("Camera controller must be zero-dimensional")
    for name in ("gripper_left", "gripper_right"):
        ctrl = controllers[name]
        if (ctrl.get("name") != "MultiFingerGripperController" or ctrl.get("mode") != "smooth"
                or ctrl.get("motor_type", "position") != "position"
                or any(ctrl.get(key) != "default" for key in ("command_input_limits", "command_output_limits"))):
            raise ValueError("Grippers must retain smooth position control")
    base = controllers["base"]
    if base.get("name") != "HolonomicBaseJointController" or base.get("motor_type") != "velocity":
        raise ValueError("Base must retain holonomic velocity control")
    return {"model": "r1pro", "action_dim": 23, "arm_and_trunk_mode": "absolute_joint_position",
            "gripper_mode": "smooth_position", "base_mode": "holonomic_velocity"}


def cpu_preflight(payload, env):
    """Import CPU helpers / decode demo PNG before starting the GPU policy."""
    script = '''import importlib,json,sys
from pathlib import Path
code,robot,demo,task = map(str,sys.argv[1:])
sys.path[:0]=[str(Path(code)/"scripts"),str(Path(code)/"sco")]
names=("numpy","h5py","PIL.Image","msgpack","websockets.sync.client","yaml","omegaconf",
       "codex_collection_actions","codex_collection_trajectory","codex_collection_demonstrations",
       "codex_teacher_mailbox","collect_b1k_codex")
modules={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in names}
from run_codex_collection import robot_contract
import yaml
contract=robot_contract(yaml.safe_load(Path(robot).read_text()))
from codex_collection_demonstrations import demonstration_context
context,images=demonstration_context(demo,task,limit=1)
from PIL import Image
for path in images:
    with Image.open(path) as image: image.verify()
print(json.dumps(dict(modules=modules,robot_contract=contract,demonstration_fingerprint=context["demonstration_fingerprint"],example_images=len(images))))
'''
    output = subprocess.check_output([str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python"), "-c", script,
                                      payload["code"], payload["robot_config"], payload["demonstration_manifest"], payload["task"]],
                                     env=env, text=True, timeout=90)
    return json.loads(output)

def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validate_payload(payload):
    backend = payload.get("teacher_backend", "mailbox")
    if backend not in ("mailbox", "codex_local"):
        raise ValueError("Invalid teacher backend")
    teacher_paths = (("teacher_mailbox_config",) if backend == "mailbox" else
                     ("codex_binary", "codex_home", "mihomo_binary", "mihomo_config_dir"))
    for key in ("code", "base_code", "runtime", "checkpoint", "run_root", *teacher_paths,
                "demonstration_manifest", "robot_config"):
        if not isinstance(payload.get(key), str) or not Path(payload[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    if Path(payload["code"]).resolve() == Path(payload["base_code"]).resolve():
        raise ValueError("Collection must use an independent code overlay")
    output = Path(payload["run_root"]).resolve()
    if any(output.is_relative_to(Path(payload[key]).resolve())
           for key in ("base_code", "code", "runtime", "checkpoint")):
        raise ValueError("Collection output must be separate from code, runtime and checkpoint")
    if type(payload.get("instance_id")) is not int or not 311 <= payload["instance_id"] <= 320:
        raise ValueError("Teacher collection requires instance 311..320")
    for key in ("task", "attempt_id", "execution_name"):
        if not isinstance(payload.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", payload[key]):
            raise ValueError(f"Invalid {key}")
    mode = payload.get("collection_mode", "teacher_conditioned")
    if mode not in ("teacher_conditioned", "native_with_corrections"):
        raise ValueError("Unknown collection mode")
    values = dict(max_steps=4500, max_decisions=32, max_seconds=7200, teacher_interval=256,
                  step_timeout=120, port=8920, wall_timeout_seconds=8400, teacher_timeout=120)
    for key, default in values.items():
        values[key] = payload.get(key, default)
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["wall_timeout_seconds"] < values["max_seconds"] + 60:
        raise ValueError("Worker wall deadline must leave bounded startup/cleanup time")
    if values["port"] > 65535:
        raise ValueError("Invalid policy port")
    if values["teacher_timeout"] > 180:
        raise ValueError("Teacher timeout must not exceed 180 seconds")
    files = payload.get("source_files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("New collection code requires a frozen file hash inventory")
    required = {"scripts/run_codex_collection.sh", "scripts/collect_b1k_codex.py",
                "scripts/codex_collection_teacher.py", "scripts/codex_teacher_mailbox.py",
                "scripts/codex_collection_actions.py", "scripts/codex_collection_trajectory.py",
                "scripts/codex_collection_demonstrations.py", "sco/run_codex_collection.py", ROBOT_CONFIG}
    if backend == "codex_local":
        required.add("sco/codex_cluster_proxy.py")
    if not required <= files.keys():
        raise ValueError("Collection overlay inventory is missing required scripts")
    for name, digest in files.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid source file hash inventory")
    return {**payload, **values, "teacher_backend": backend, "collection_mode": mode}


def verify_sources(payload):
    code, base = Path(payload["code"]), Path(payload["base_code"])
    runtime, checkpoint = Path(payload["runtime"]), Path(payload["checkpoint"])
    public = json.loads((base / "_PUBLIC_TEST_SOURCE_READY").read_text())
    if public["checkpoint_step"] != 1688977 or sha(base / "_CODE_SNAPSHOT_READY") != public["base_source_sha256"]:
        raise ValueError("Original source receipt does not match immutable checkpoint")
    original = json.loads((base / "_CODE_SNAPSHOT_READY").read_text())
    base_files = {**original["overlay_files_sha256"], **public["files_sha256"]}
    for name, digest in base_files.items():
        if sha(base / name) != digest:
            raise ValueError("Immutable original code changed: " + name)
        if name not in payload["source_files_sha256"] and sha(code / name) != digest:
            raise ValueError("Collection copy changed an undeclared source file: " + name)
    for name, digest in payload["source_files_sha256"].items():
        if sha(code / name) != digest:
            raise ValueError("Collection overlay hash differs: " + name)
    if (sha(payload["robot_config"]) != sha(base / ROBOT_CONFIG)
            or sha(payload["robot_config"]) != payload["source_files_sha256"][ROBOT_CONFIG]):
        raise ValueError("Collection robot config differs from the pinned public evaluator config")
    if json.loads((checkpoint / "_READY").read_text())["checkpoint_step"] != 1688977:
        raise ValueError("Checkpoint step differs")
    for row in public["checkpoint_files"]:
        if (checkpoint / row["relative_path"]).stat().st_size != row["bytes"]:
            raise ValueError("Checkpoint file size differs")
    if not (runtime / "_RUNTIME_READY").is_file():
        raise ValueError("Runtime is not ready")
    if sha(runtime / "graphics-runtime/_READY") != original["graphics_runtime_manifest_sha256"]:
        raise ValueError("Graphics runtime receipt differs")
    graphics = json.loads((runtime / "graphics-runtime/_READY").read_text())
    supplement = public["supplemental_library"]
    supplement_root = Path(supplement["root"])
    if sha(supplement_root / "_READY") != supplement["ready_manifest_sha256"]:
        raise ValueError("Supplemental graphics library receipt differs")
    library = json.loads((supplement_root / "_READY").read_text())
    for ready in (graphics, library):
        for name, digest in ready["libraries_sha256"].items():
            if sha(Path(ready["library_path"]) / name) != digest:
                raise ValueError("Graphics library changed")
    return library, sha(base / "_PUBLIC_TEST_SOURCE_READY")


def environment(payload, cache, vk, egl, library_path, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK", "GOALKEEPER_", "CODEX_")):
            env.pop(key)
    for key in ("PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY"):
        env.pop(key, None)
    runtime, root = Path(payload["runtime"]), Path(payload["run_root"])
    env.update(PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1",
               OMNIGIBSON_DISABLE_TORCH_COMPILE="1", OPENPI_DATA_HOME=str(runtime / "openpi-cache"),
               OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
               PI05_ENV=str(runtime / "envs/b1k-pi05"), EVAL_ENV=str(runtime / "envs/b1k-eval"),
               PI05_CKPT=payload["checkpoint"], PI05_REPO_ID="b1k_full100_joint_subtask_v1_drop_episode_s1",
               PI05_MEM_FRACTION="0.35", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PORT=str(payload["port"]), MAX_STEPS=str(payload["max_steps"]),
               CODEX_COLLECTION_TASK=payload["task"], CODEX_COLLECTION_INSTANCE=str(payload["instance_id"]),
               CODEX_COLLECTION_ATTEMPT=payload["attempt_id"], CODEX_COLLECTION_OUTPUT=str(root / "attempt"),
               CODEX_COLLECTION_MODE=payload.get("collection_mode", "teacher_conditioned"),
               CODEX_COLLECTION_LOG_ROOT=str(root / "logs"), CODEX_CHECKPOINT_ID="pi05_b1k_full100_joint_1688977",
               CODEX_TEACHER_BACKEND=payload["teacher_backend"],
               CODEX_DEMONSTRATION_MANIFEST=payload["demonstration_manifest"], CODEX_ROBOT_CONFIG=payload["robot_config"],
               CODEX_MAX_DECISIONS=str(payload["max_decisions"]), CODEX_MAX_SECONDS=str(payload["max_seconds"]),
               CODEX_TEACHER_INTERVAL=str(payload["teacher_interval"]), CODEX_STEP_TIMEOUT=str(payload["step_timeout"]),
               SCO_EPHEMERAL_RUNTIME_ROOT=str(cache), VK_ICD_FILENAMES=str(vk),
               __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               LD_LIBRARY_PATH=library_path + ":" + str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu")
                   + ":" + str(runtime / "envs/b1k-eval/lib") + ":" + env.get("LD_LIBRARY_PATH", ""))
    if payload["teacher_backend"] == "codex_local":
        env.update(CODEX_HOME=payload["codex_home"], CODEX_TEACHER_BINARY=payload["codex_binary"],
                   CODEX_TEACHER_TIMEOUT=str(payload["teacher_timeout"]),
                   PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", ""))
    else:
        env["CODEX_TEACHER_MAILBOX_CONFIG"] = payload["teacher_mailbox_config"]
    return env


def bounded_process(command, *, env, cwd, log, timeout):
    process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # Only this driver's new process group belongs to us. Do not discover or
        # terminate other evaluators, policy servers or shared workloads.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        raise


def run(payload):
    payload = validate_payload(payload)
    root = Path(payload["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation disallows concurrent or accidental replay of an attempt.
    with (root / "_STARTED").open("x") as stream:
        stream.write(str(time.time()) + "\n")
    status = {"state": "preparing", "task": payload["task"], "instance_id": payload["instance_id"],
              "attempt_id": payload["attempt_id"], "checkpoint_step": 1688977,
              "official_benchmark": False, "training_eligible": False, "started_at": time.time()}
    def publish():
        status["updated_at"] = time.time()
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "status.json")
        print("STATUS_JSON=" + json.dumps(status), flush=True)
    publish()
    resources = ExitStack()
    try:
        library, base_sha = verify_sources(payload)
        input_keys = ["demonstration_manifest", "robot_config"]
        if payload["teacher_backend"] == "mailbox":
            input_keys.append("teacher_mailbox_config")
        for key in input_keys:
            if not Path(payload[key]).is_file():
                raise ValueError("Missing collection input: " + key)
        status["input_files_sha256"] = {key: sha(payload[key]) for key in input_keys}
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                           "--format=csv,noheader"], text=True).strip().splitlines()
        if len(gpu_rows) != 1 or not any(name in gpu_rows[0] for name in ("5090", "N12LP")):
            raise ValueError("This pilot requires one RTX 5090 worker")
        driver = gpu_rows[0].split(",")[2].strip()
        libraries = [path for parent in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/lib/x86_64-linux-gnu", "/lib64")
                     for path in Path(parent).glob("libEGL_nvidia.so." + driver)]
        if not libraries:
            raise ValueError("Matching NVIDIA graphics library is unavailable")
        graphics = root / "graphics"
        graphics.mkdir()
        vk, egl = graphics / "nvidia_icd.json", graphics / "10_nvidia.json"
        vk.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0]), "api_version": "1.3.242"}}))
        egl.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}}))
        cache = Path("/var/tmp/behavior-1k-runtime") / payload["execution_name"]
        cache.mkdir(parents=True, exist_ok=False)
        if cache.stat().st_dev == Path(payload["runtime"]).stat().st_dev or shutil.disk_usage(cache).free < 40 * 1024**3:
            raise ValueError("Need at least 40GiB of separate worker-local runtime storage")
        env = environment(payload, cache, vk, egl, library["library_path"])
        if payload["teacher_backend"] == "codex_local":
            from codex_cluster_proxy import CodexHomeLease, MihomoProxy
            lease = resources.enter_context(CodexHomeLease(Path(payload["codex_home"])))
            proxy = resources.enter_context(MihomoProxy(config_dir=Path(payload["mihomo_config_dir"]),
                run_dir=cache / "teacher-network", mihomo_binary=payload["mihomo_binary"]))
            env = proxy.environment(env, codex_home=lease.home)
            status["teacher_proxy"] = proxy.receipt
        status["cpu_preflight"] = cpu_preflight(payload, env)
        status.update(state="running", gpu_rows=gpu_rows, original_source_manifest_sha256=base_sha,
                      source_files_sha256=payload["source_files_sha256"])
        publish()
        with (root / "driver.log").open("a", buffering=1) as log:
            code = bounded_process(["bash", str(Path(payload["code"]) / "scripts/run_codex_collection.sh")],
                                   env=env, cwd=payload["code"], log=log, timeout=payload["wall_timeout_seconds"])
        status["driver_exit_code"] = code
        if code != 0:
            raise RuntimeError("Collector or policy server failed; retain the attempt evidence")
        completed = json.loads((root / "attempt/completed.json").read_text())
        if completed.get("official_benchmark") is not False or type(completed.get("training_eligible")) is not bool:
            raise ValueError("Collection completion receipt is invalid")
        status.update(state="complete", training_eligible=completed["training_eligible"],
                      result=completed["result"], completed_at=time.time())
        publish()
        (root / "_COMPLETE").write_text(json.dumps(status, indent=2) + "\n")
        return status
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, training_eligible=False)
        publish()
        raise
    finally:
        resources.close()


if __name__ == "__main__":
    import sys
    run(json.loads(Path(sys.argv[1]).read_text()))

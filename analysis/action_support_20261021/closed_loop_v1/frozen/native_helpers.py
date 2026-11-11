"""Finite worker for the fixed 301--310 paired panel, without a teacher.

Launches only the immutable original policy and evaluator. A new checkpoint is
admitted only after a verified training receipt and inference-file inventory.
Each shard starts a fresh server and one evaluator process (seed 0 once), with
the original order and native task budgets. This module never submits SCO jobs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import time

SOURCE_SHA256 = "7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4"
SHARD_IDS = ("radio-pilot", "a000", "a001", "a088", "a089")
HEX = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}


def verify_ref(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
            or not Path(ref["path"]).is_absolute() or not HEX.fullmatch(str(ref.get("sha256")))
            or type(ref.get("size")) is not int or ref["size"] < 1):
        raise ValueError("Expected an absolute artifact path with SHA256 and positive size")
    path = Path(ref["path"])
    if not path.is_file() or path.stat().st_size != ref["size"] or sha(path) != ref["sha256"]:
        raise ValueError("Artifact does not match receipt: " + str(path))
    return path


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_verifier(ref):
    path = verify_ref(ref)
    spec = importlib.util.spec_from_file_location("native_eval_artifact_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(s["id"] for s in module.SHARDS) != SHARD_IDS:
        raise ValueError("Artifact verifier changed the fixed shard schedule")
    return module


def validate_payload(payload):
    for key in ("code", "runtime", "checkpoint", "run_root"):
        if not isinstance(payload.get(key), str) or not Path(payload[key]).is_absolute():
            raise ValueError(key + " must be an absolute path")
    output = Path(payload["run_root"]).resolve()
    if any(output.is_relative_to(Path(payload[key]).resolve()) for key in ("code", "runtime", "checkpoint")):
        raise ValueError("Evaluation output must be separate from immutable inputs")
    if payload.get("source_manifest_sha256") != SOURCE_SHA256:
        raise ValueError("Evaluation requires the exact original 1688977 source manifest")
    for key in ("checkpoint_id", "repo_id", "execution_name"):
        if not isinstance(payload.get(key), str) or not NAME.fullmatch(payload[key]):
            raise ValueError("Invalid " + key)
    if payload["checkpoint_id"] == "1688977":
        raise ValueError("Evaluation requires the new trained checkpoint")
    ids = payload.get("shard_ids", list(SHARD_IDS))
    if (not isinstance(ids, list) or not ids or any(s not in SHARD_IDS for s in ids)
            or ids != [s for s in SHARD_IDS if s in ids]):
        raise ValueError("Shard selection must preserve the fixed unique shard order")
    defaults = {"port": 8930, "startup_timeout_seconds": 900, "shard_timeout_seconds": 28800}
    values = {key: payload.get(key, default) for key, default in defaults.items()}
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError("Worker deadlines and port must be positive integers")
    if values["port"] > 65535 or values["startup_timeout_seconds"] > 1800:
        raise ValueError("Port/startup deadline is outside the bounded runtime contract")
    if values["shard_timeout_seconds"] < values["startup_timeout_seconds"] + 120:
        raise ValueError("Shard deadline must leave time for evaluation and cleanup")
    for key in ("training_receipt", "report_helper"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(key + " must carry an immutable file reference")
    if not HEX.fullmatch(str(payload.get("driver_sha256"))):
        raise ValueError("The new worker driver must be hash-pinned")
    forbidden = {"max_steps", "instance_ids", "task", "seed", "teacher", "teacher_mailbox_config",
                 "demonstration_manifest", "external_subtask", "prompt_mode", "action_horizon"}
    if forbidden & payload.keys():
        raise ValueError("Fixed native evaluation settings cannot be overridden")
    return {**payload, **values, "shard_ids": ids}


def verify_training(payload):
    receipt = load(verify_ref(payload["training_receipt"]))
    if (receipt.get("kind") != "codex_pilot_trained_checkpoint_v1"
            or receipt.get("checkpoint_id") != payload["checkpoint_id"]
            or receipt.get("parent_checkpoint_step") != 1688977
            or type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] < 1
            or any(receipt.get(k) is not True for k in ("ready_for_evaluation", "real_lerobot_readback_verified",
                                                       "base_normalization_preserved"))):
        raise ValueError("Training receipt does not prove completed training/readback")
    checkpoint = Path(payload["checkpoint"]).resolve()
    manifest = verify_ref(receipt["checkpoint_manifest"])
    if manifest.resolve() != checkpoint / "SHA256SUMS.inference":
        raise ValueError("Training receipt describes a different checkpoint directory")
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        relative = Path(name)
        if (not separator or not HEX.fullmatch(digest) or not name or relative.is_absolute()
                or ".." in relative.parts or name in entries):
            raise ValueError("Malformed or duplicate inference manifest entry")
        path = checkpoint / name
        if path.is_symlink() or not path.resolve().is_relative_to(checkpoint) or sha(path) != digest:
            raise ValueError("Trained inference file differs: " + name)
        entries[name] = digest
    required = {"params/_METADATA", "assets/subtask_conditioning.json",
                "assets/" + payload["repo_id"] + "/norm_stats.json"}
    if not required <= entries.keys():
        raise ValueError("Trained checkpoint lacks native joint inference assets")
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()}
    if actual != entries.keys() | {"_READY", "SHA256SUMS.inference"}:
        raise ValueError("Inference checkpoint contains unmanifested or missing files")
    ready = load(checkpoint / "_READY")
    if (ready.get("init_checkpoint_step") != 1688977
            or ready.get("manifest_sha256") != receipt["checkpoint_manifest"]["sha256"]
            or ready.get("restored_train_state_step") != receipt["optimizer_steps"]):
        raise ValueError("Inference readiness does not match actual trained weights")
    norm = required - {"params/_METADATA", "assets/subtask_conditioning.json"}
    if entries[next(iter(norm))] != receipt.get("base_norm_sha256"):
        raise ValueError("Inference normalization differs from the verified base statistics")
    if load(checkpoint / "assets/subtask_conditioning.json").get("action_conditioning") != "task_and_subtask":
        raise ValueError("Trained checkpoint is not a native joint policy")
    return receipt


def verify_sources(payload):
    code, runtime = Path(payload["code"]), Path(payload["runtime"])
    if sha(code / "_PUBLIC_TEST_SOURCE_READY") != SOURCE_SHA256:
        raise ValueError("Evaluator must use the original immutable source snapshot")
    public = load(code / "_PUBLIC_TEST_SOURCE_READY")
    if public["checkpoint_step"] != 1688977 or sha(code / "_CODE_SNAPSHOT_READY") != public["base_source_sha256"]:
        raise ValueError("Original source chain differs")
    base = load(code / "_CODE_SNAPSHOT_READY")
    for name, digest in {**base["overlay_files_sha256"], **public["files_sha256"]}.items():
        if sha(code / name) != digest:
            raise ValueError("Original source changed: " + name)
    if not (runtime / "_RUNTIME_READY").is_file():
        raise ValueError("Runtime is not ready")
    if sha(runtime / "graphics-runtime/_READY") != base["graphics_runtime_manifest_sha256"]:
        raise ValueError("Graphics runtime receipt differs")
    graphics = load(runtime / "graphics-runtime/_READY")
    supplement = public["supplemental_library"]
    supplement_path = Path(supplement["root"]) / "_READY"
    if sha(supplement_path) != supplement["ready_manifest_sha256"]:
        raise ValueError("Supplemental library receipt differs")
    library = load(supplement_path)
    for ready in (graphics, library):
        for name, digest in ready["libraries_sha256"].items():
            if sha(Path(ready["library_path"]) / name) != digest:
                raise ValueError("Pinned graphics library changed")
    return library


def build_env(payload, cache, vk, egl, library_path, *, process, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if (key.startswith(("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK", "GOALKEEPER_",
                            "CODEX_", "TEACHER_"))
                or key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                           "VIRTUAL_ENV", "MAX_STEPS", "PYTHONHASHSEED", "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR"}):
            env.pop(key)
    runtime, code = Path(payload["runtime"]), Path(payload["code"])
    env.update(PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1",
               OMNIGIBSON_DISABLE_TORCH_COMPILE="1", OPENPI_DATA_HOME=str(runtime / "openpi-cache"),
               OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
               VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               LD_LIBRARY_PATH=library_path + ":" + str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu")
                   + ":" + str(runtime / "envs/b1k-eval/lib") + ":" + env.get("LD_LIBRARY_PATH", ""))
    env.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="4")
    if process == "policy":
        env.update(CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="true",
                   XLA_PYTHON_CLIENT_MEM_FRACTION="0.35", HF_HUB_OFFLINE="1",
                   PYTHONPATH=str(code / "openpi/src") + ":" + str(code / "openpi/packages/openpi-client/src"),
                   PATH=str(runtime / "envs/b1k-pi05/bin") + ":" + env.get("PATH", ""))
    elif process == "evaluator":
        env.update(OMNIGIBSON_GPU_ID="0", OMNIGIBSON_APPDATA_PATH=str(cache / "appdata"),
                   XDG_CACHE_HOME=str(cache / "cache"), XDG_CONFIG_HOME=str(cache / "config"),
                   WARP_CACHE_PATH=str(cache / "warp"), HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   PYTHONPATH=":".join(str(code / "BEHAVIOR-1K" / name) for name in ("OmniGibson", "bddl3", "joylo")),
                   LD_LIBRARY_PATH=str(runtime / "envs/b1k-eval/lib") + ":" + env["LD_LIBRARY_PATH"],
                   PATH=str(runtime / "envs/b1k-eval/bin") + ":" + env.get("PATH", ""))
    else:
        raise ValueError("Unknown process environment")
    return env


def commands(payload, shard, output, token):
    runtime, code = Path(payload["runtime"]), Path(payload["code"])
    server = [str(runtime / "envs/b1k-pi05/bin/python"), str(code / "openpi/scripts/b1k/serve_b1k.py"),
              "--robot", "b1k/R1Pro", "--task", "b1k/" + shard["task"], "--repo-id", payload["repo_id"],
              "--policy.config", "pi05_b1k", "--policy.dir", payload["checkpoint"],
              "--control_mode", "receding_horizon", "--action_horizon", "16", "--subtask-prediction",
              "--max-token-len", "512", "--max-subtask-len", "64", "--subtask-temperature", "0.0",
              "--subtask-action-conditioning", "task_and_subtask", "--server-token", token,
              "--port", str(payload["port"])]
    evaluator = [str(runtime / "envs/b1k-eval/bin/python"), "-m", "omnigibson.eval.eval",
                 "--task-name", shard["task"], "--host", "127.0.0.1", "--port", str(payload["port"]),
                 "--mode", "public_test", "--prompt-mode", "whole_task", "--num-rollouts", "1",
                 "--instance-indices", *map(lambda i: str(i - 301), shard["instance_ids"]),
                 "--expected-server-token", token, "--output-dir", str(output),
                 "--write-video" if shard["write_video"] else "--no-write-video", "--headless"]
    return server, evaluator


def stop_process(process):
    if process is None:
        return
    # Only this driver's process group is owned. Never discover/kill other jobs.
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


def cpu_preflight(payload, policy_env, eval_env):
    code, runtime = Path(payload["code"]), Path(payload["runtime"])
    findings = {}
    for label, env, modules, paths in (
        ("policy", policy_env, ["openpi", "openpi_client"],
         [code / "openpi/src/openpi", code / "openpi/packages/openpi-client/src/openpi_client"]),
        ("evaluator", eval_env, ["omnigibson"], [code / "BEHAVIOR-1K/OmniGibson/omnigibson"]),
    ):
        script = '''import importlib.util,json,sys
from pathlib import Path
mods=json.loads(sys.argv[1]); expected=json.loads(sys.argv[2]); result={}
for name,root in zip(mods,expected):
 spec=importlib.util.find_spec(name)
 assert spec and spec.origin and Path(spec.origin).resolve().is_relative_to(Path(root).resolve()),name
 result[name]=str(Path(spec.origin).resolve())
print(json.dumps(result))
'''
        python = runtime / ("envs/b1k-pi05/bin/python" if label == "policy" else "envs/b1k-eval/bin/python")
        findings[label] = json.loads(subprocess.check_output([str(python), "-c", script,
            json.dumps(modules), json.dumps(list(map(str, paths)))], env=env, cwd=code, text=True, timeout=90))
    stats = runtime / "omnigibson-data/2026-challenge-task-instances/metadata/task.jsonl"
    rows = {row["task_name"]: row for line in stats.read_text().splitlines() if line.strip()
            for row in [json.loads(line)]}
    budgets = {task: int(1.5 * rows[task]["length"]) for task in ("turning_on_radio", "putting_shoes_on_rack")}
    if budgets != {"turning_on_radio": 3225, "putting_shoes_on_rack": 11590}:
        raise ValueError("Runtime official task budgets differ from the paired baseline")
    findings.update(task_metadata=file_ref(stats), resolved_step_budgets=budgets)
    return findings


def wait_for_policy(payload, process, token, env, destination):
    script = '''import json,msgpack,os,sys,threading,time
from pathlib import Path
import websockets.sync.client
port,token,pid,seconds,out=sys.argv[1:]; deadline=time.monotonic()+int(seconds)
while time.monotonic()<deadline:
 try:
  os.kill(int(pid),0)
  status=Path(f"/proc/{pid}/stat")
  if status.is_file() and status.read_text().rsplit(")",1)[1].split()[0]=="Z":
   raise SystemExit("Policy exited during startup")
 except ProcessLookupError: raise SystemExit("Policy exited during startup")
 try:
  with websockets.sync.client.connect(f"ws://127.0.0.1:{port}",compression=None,max_size=None,
      open_timeout=2,close_timeout=1) as connection:
   metadata=msgpack.unpackb(connection.recv(timeout=2),raw=False)
  assert metadata.get("server_instance_token")==token
  assert metadata.get("subtask_generation") is True
  assert metadata.get("subtask_action_conditioning")=="task_and_subtask"
  with Path(out).open("x") as stream: json.dump(metadata,stream)
  raise SystemExit(0)
 except Exception: pass
 threading.Event().wait(min(1,max(0,deadline-time.monotonic())))
raise SystemExit("Policy readiness deadline exceeded")
'''
    subprocess.run([str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python"), "-c", script,
                    str(payload["port"]), token, str(process.pid), str(payload["startup_timeout_seconds"]),
                    str(destination)], env=env, check=True, timeout=payload["startup_timeout_seconds"] + 15)
    if process.poll() is not None:
        raise RuntimeError("Policy exited after startup")
    return load(destination)


def verify_video(payload, path, env, root):
    script = '''import cv2,json,math,sys
from pathlib import Path
cap=cv2.VideoCapture(sys.argv[1]); assert cap.isOpened()
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
assert count>1 and math.isfinite(fps) and fps>0
indices=[0,count//2,count-1]
for index in indices:
 cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=cap.read(); assert ok and frame.size>0
 assert cv2.imwrite(str(Path(sys.argv[2])/f"pilot-frame-{index}.png"),frame)
cap.release()
print(json.dumps(dict(frames=count,fps=fps,decoded_frames=indices)))
'''
    output = subprocess.check_output([str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python"),
        "-c", script, str(path), str(root)], env=env, text=True, timeout=90)
    return {**json.loads(output), "video_sha256": sha(path)}


def run_shard(payload, shard, helper, training, library, cache, vk, egl, gpu_rows):
    root = Path(payload["run_root"]) / shard["id"]
    root.mkdir()
    output = root / "results/pi05" / shard["task"]
    output.mkdir(parents=True)
    shard_cache = cache / shard["id"]
    for name in ("appdata", "cache", "config", "warp"):
        (shard_cache / name).mkdir(parents=True)
    policy_env = build_env(payload, shard_cache, vk, egl, library["library_path"], process="policy")
    eval_env = build_env(payload, shard_cache, vk, egl, library["library_path"], process="evaluator")
    preflight = cpu_preflight(payload, policy_env, eval_env)
    token = secrets.token_hex(32)
    server_command, eval_command = commands(payload, shard, output, token)
    started, deadline = time.time(), time.monotonic() + payload["shard_timeout_seconds"]
    server = evaluator = None
    receipt = {"kind": "codex_pilot_native_eval_run_v1", "state": "running", "shard_id": shard["id"],
               "checkpoint_id": payload["checkpoint_id"],
               "checkpoint_manifest_sha256": training["checkpoint_manifest"]["sha256"],
               "runtime_contract": helper.runtime_contract(SOURCE_SHA256),
               "external_agent": False, "teacher_imported": False,
               "environment_keys": sorted(set(policy_env) | set(eval_env)), "started_at": started,
               "cpu_preflight": preflight, "gpu_rows": gpu_rows, "driver_sha256": payload["driver_sha256"],
               "report_helper": payload["report_helper"], "training_receipt": payload["training_receipt"],
               "server_command": server_command, "evaluator_command": eval_command}
    try:
        with (root / "policy.log").open("x") as policy_log, (root / "eval.log").open("x") as eval_log:
            server = subprocess.Popen(server_command, env=policy_env, cwd=Path(payload["code"]) / "openpi",
                                      stdout=policy_log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt["policy_metadata"] = wait_for_policy(payload, server, token, eval_env, root / "policy_metadata.json")
            evaluator = subprocess.Popen(eval_command, env=eval_env, cwd=Path(payload["code"]) / "BEHAVIOR-1K",
                                         stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True)
            evaluator.wait(timeout=max(1, deadline - time.monotonic()))
            if evaluator.returncode != 0 or server.poll() is not None:
                raise RuntimeError("Original evaluator or native policy failed; preserve partial artifacts")
        results = []
        expected_jsons = {f"{shard['task']}_{instance}_0.json" for instance in shard["instance_ids"]}
        if {p.name for p in (output / "json").glob("*.json")} != expected_jsons:
            raise ValueError("Evaluator did not produce the exact ordered shard")
        for instance in shard["instance_ids"]:
            path = output / "json" / f"{shard['task']}_{instance}_0.json"
            helper.validate_result(load(path), (shard["task"], instance, 0))
            video = output / "videos" / f"{shard['task']}_{instance}_0.mp4"
            results.append({"task": shard["task"], "instance_id": instance, "rollout_id": 0,
                            "shard_id": shard["id"], "json": file_ref(path),
                            "video": file_ref(video) if shard["write_video"] else None})
        if shard["write_video"]:
            receipt["video_validation"] = verify_video(payload, Path(results[0]["video"]["path"]), eval_env, root)
        receipt.update(state="complete", results=results, completed_at=time.time(),
                       logs={name: file_ref(root / name) for name in ("policy.log", "eval.log")})
    except BaseException as error:
        receipt.update(state="failed", error_type=type(error).__name__, completed_at=time.time())
        write_new(root / "failed_run_receipt.json", receipt)
        raise
    finally:
        stop_process(evaluator)
        stop_process(server)
    write_new(root / "run_receipt.json", receipt)
    return file_ref(root / "run_receipt.json")


def assemble_panel(payload, run_receipts):
    helper = load_verifier(payload["report_helper"])
    receipts = [load(verify_ref(ref)) for ref in run_receipts]
    if [row["shard_id"] for row in receipts] != list(SHARD_IDS):
        raise ValueError("Panel requires all five receipts in the declared order")
    panel = {"kind": "codex_pilot_native_eval_panel_v1", "role": "finetuned",
             "checkpoint": {"id": payload["checkpoint_id"], "kind": "trained",
                            "training_receipt": payload["training_receipt"]},
             "runtime_contract": helper.runtime_contract(SOURCE_SHA256),
             "results": [row for receipt in receipts for row in receipt["results"]],
             "run_receipts": run_receipts}
    helper._panel_rows(panel)
    helper._validate_candidate(panel, panel["runtime_contract"])
    return panel


def run(payload):
    payload = validate_payload(payload)
    if sha(__file__) != payload["driver_sha256"]:
        raise ValueError("Running worker differs from its frozen source hash")
    helper = load_verifier(payload["report_helper"])
    training = verify_training(payload)
    library = verify_sources(payload)
    root = Path(payload["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "payload.json", payload)
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu_rows) != 1 or not any(name in gpu_rows[0] for name in ("5090", "N12LP")):
        raise ValueError("Paired pilot requires exactly one RTX 5090 worker")
    driver = gpu_rows[0].split(",")[2].strip()
    libraries = [p for parent in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                                 "/lib/x86_64-linux-gnu", "/lib64")
                 for p in Path(parent).glob("libEGL_nvidia.so." + driver)]
    if not libraries:
        raise ValueError("Matching NVIDIA graphics library is missing")
    graphics = root / "graphics"
    graphics.mkdir()
    vk, egl = graphics / "nvidia_icd.json", graphics / "10_nvidia.json"
    write_new(vk, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0]), "api_version": "1.3.242"}})
    write_new(egl, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}})
    cache = Path("/var/tmp/behavior-1k-runtime") / payload["execution_name"]
    cache.mkdir(parents=True, exist_ok=False)
    if cache.stat().st_dev == Path(payload["runtime"]).stat().st_dev or shutil.disk_usage(cache).free < 40 * 1024**3:
        raise ValueError("Need at least 40GiB of separate worker-local runtime storage")
    receipts = []
    try:
        for shard in helper.SHARDS:
            if shard["id"] in payload["shard_ids"]:
                receipts.append(run_shard(payload, shard, helper, training, library, cache, vk, egl, gpu_rows))
                print(json.dumps({"completed_shard": shard["id"], "receipt": receipts[-1]}), flush=True)
        if payload["shard_ids"] == list(SHARD_IDS):
            write_new(root / "candidate_panel.json", assemble_panel(payload, receipts))
        summary = {"state": "complete", "run_receipts": receipts, "external_agent": False,
                   "checkpoint_id": payload["checkpoint_id"], "completed_at": time.time()}
        write_new(root / "_COMPLETE", summary)
        return summary
    except BaseException as error:
        write_new(root / "_FAILED", {"state": "failed", "error_type": type(error).__name__, "completed_run_receipts": receipts})
        raise


if __name__ == "__main__":
    import sys
    run(load(sys.argv[1]))

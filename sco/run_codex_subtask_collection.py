"""Finite, isolated worker for one Codex-assisted training subtask attempt.

This driver does not submit resources or modify prior collection plans. It pins
the existing 1688977 model, starts only its own policy/collector process groups,
and leases the already staged Codex home through a private mihomo instance.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

from run_codex_collection import ROBOT_CONFIG, bounded_process, robot_contract, sha, verify_sources

REPO_ID = "b1k_full100_joint_subtask_v1_drop_episode_s1"
NORM_RELATIVE = f"assets/{REPO_ID}/norm_stats.json"
NORM_SHA256 = "52d63e89f8359143d793ca80d0661c7facb0daff43c7d8c08f0d05572dd528fb"
CONDITIONING_SHA256 = "64c2a1fadca27d42715c88ef7c12e2f8a9ef23f37b2bf886327aa4fc639b4932"
REQUIRED_SOURCES = {
    "sco/run_codex_subtask_collection.py", "sco/run_codex_collection.py", "sco/codex_cluster_proxy.py",
    "scripts/collect_b1k_codex_subtask.py", "scripts/codex_subtask_collection.py",
    "scripts/codex_subtask_qualification.py",
    "scripts/codex_preparation_checkpoint.py",
    "scripts/codex_gripper_feedback.py",
    "scripts/codex_robot_guidance.py",
    "scripts/codex_scene_ground_truth.py",
    "scripts/codex_jacobian_validation.py",
    "scripts/codex_subtask_teacher.py", "scripts/codex_collection_teacher.py",
    "scripts/codex_collection_actions.py", "scripts/codex_collection_demonstrations.py",
    "scripts/codex_collection_trajectory.py", "scripts/subtask_trajectory.py", ROBOT_CONFIG,
    "scripts/accepted_subtask_collection.py", "scripts/additional_training_panel.py",
    "sco/audit_history_assignment_artifacts.py", "scripts/codex_teacher_bridge.py",
    "scripts/codex_teacher_mailbox.py", "scripts/collect_b1k_codex.py",
    "scripts/collect_b1k_subtasks.py", "scripts/collection_checkpoint.py",
    "scripts/diagnose_b1k_grasp_history.py", "scripts/diagnose_b1k_grasp_restore.py",
    "scripts/eval_b1k_dynamic.py", "scripts/eval_b1k_queue.py",
    "scripts/eval_b1k_subtask_dynamic.py", "scripts/eval_b1k_subtask_samples.py",
    "sco/full100_shards.py", "scripts/full_episode_checkpoint.py",
    "scripts/full_episode_multipart.py", "scripts/full_episode_trajectory.py",
    "scripts/full_subtask_collection.py", "scripts/gen_subtask_prompts.py",
    "scripts/legacy_history_recovery.py", "scripts/pi05_eval_queue.py",
    "scripts/recorded_fixture_state.py", "scripts/recorded_native_state.py",
    "scripts/stage_subtask_replay.py", "scripts/strip_depth_from_info.py",
    "scripts/subtask_completion.py", "scripts/subtask_improvement.py",
    "scripts/subtask_paired_retest.py", "scripts/subtask_sample_panel.py",
    "scripts/subtask_source_exclusion.py", "scripts/summarize_subtask_closed_loop.py",
    "scripts/validate_b1k_grasp_reconstruction.py", "scripts/validate_b1k_native_snapshot_model.py",
    "openpi/scripts/b1k/serve_b1k.py", "openpi/src/openpi/shared/collection_seeding.py",
    "openpi/src/openpi/shared/eval_b1k_wrapper.py", "openpi/src/openpi/serving/websocket_b1k_server.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/policies.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/network_utils.py",
}


def validate_payload(payload):
    paths = ("code", "base_code", "runtime", "checkpoint", "run_root", "plan_path", "raw_data_root",
             "demonstration_manifest", "robot_config", "codex_binary", "codex_home", "mihomo_binary",
             "mihomo_config_dir", "checkpoint_upload_plan")
    for key in paths:
        if not isinstance(payload.get(key), str) or not Path(payload[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    code, base, output = (Path(payload[key]).resolve() for key in ("code", "base_code", "run_root"))
    if code == base or code.is_relative_to(base) or base.is_relative_to(code):
        raise ValueError("Subtask collection requires an independent code overlay")
    for key in ("code", "base_code", "runtime", "checkpoint", "raw_data_root", "codex_home", "mihomo_config_dir"):
        source = Path(payload[key]).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be separate from source, runtime, data and private inputs")
    for key in ("execution_name",):
        if not isinstance(payload.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", payload[key]):
            raise ValueError(f"Invalid {key}")
    if payload.get("teacher_backend", "codex_local") != "codex_local":
        raise ValueError("This training-subtask worker requires the existing local Codex account")
    if not isinstance(payload.get("plan_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", payload["plan_sha256"]):
        raise ValueError("A frozen subtask plan SHA256 is required")
    values = {"port": 8940, "wall_timeout_seconds": 9000, "teacher_timeout": 120,
              "policy_startup_timeout": 900, "teacher_capacity_retries": 0}
    limits = {"port": (1024, 65535), "wall_timeout_seconds": (120, 10800),
              "teacher_timeout": (1, 180), "policy_startup_timeout": (1, 900),
              "teacher_capacity_retries": (0, 2)}
    for key, default in values.items():
        value = payload.get(key, default)
        lower, upper = limits[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid bounded {key}")
        values[key] = value
    files = payload.get("source_files_sha256")
    if not isinstance(files, dict) or not REQUIRED_SOURCES <= files.keys():
        raise ValueError("Subtask overlay inventory is missing required sources")
    for name, digest in files.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Invalid frozen source inventory")
    if payload.get('reuse_qualification_root'):
        source = Path(payload['reuse_qualification_root'])
        if not source.is_absolute() or source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError('Qualification reuse requires a separate absolute source root')
        if payload.get('persist_preparation') is not True:
            raise ValueError('Qualification reuse must persist the recovered scene')
    if payload.get('resume_preparation') and not payload.get('reuse_qualification_root'):
        raise ValueError('Preparation resume requires its verified qualification')
    if 'persist_preparation' in payload and type(payload['persist_preparation']) is not bool:
        raise ValueError('persist_preparation must be boolean')
    return {**payload, **values, "teacher_backend": "codex_local"}


def verify_checkpoint_inventory(checkpoint, inventory, identity):
    """Verify every inference parameter, not just a saved-step label or size."""
    checkpoint = Path(checkpoint).resolve()
    if inventory.get("checkpoint_step") != 1688977:
        raise ValueError("Wrong checkpoint inventory step")
    source = identity.get("checkpoint_source")
    if isinstance(source, str) and source.startswith("oss://"):
        canonical = "oss://" + inventory["bucket"] + "/" + inventory["prefix"]
        if source.rstrip("/") != canonical.rstrip("/"):
            raise ValueError("Checkpoint source differs from its upload inventory")
    entries = inventory.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint inventory is empty")
    seen = set()
    for row in entries:
        name, digest, size = row.get("relative_path"), row.get("sha256"), row.get("size")
        if (not isinstance(name, str) or not name or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or type(size) is not int or size <= 0):
            raise ValueError("Invalid checkpoint inventory entry")
        relative = Path(name)
        target = checkpoint / relative
        if (relative.is_absolute() or ".." in relative.parts or name in seen
                or target.resolve() != target or not target.is_file()
                or target.stat().st_size != size or sha(target) != digest):
            raise ValueError("Actual checkpoint bytes differ from its inventory")
        seen.add(name)
    required = {"params/_METADATA", "_CHECKPOINT_METADATA", "assets/subtask_conditioning.json", NORM_RELATIVE}
    params = {path.relative_to(checkpoint).as_posix() for path in (checkpoint / "params").rglob("*") if path.is_file()}
    if (not required <= seen or not params or not params <= seen
            or inventory.get("file_count") != len(entries)
            or inventory.get("total_bytes") != sum(row["size"] for row in entries)
            or (checkpoint / "model.safetensors").exists()):
        raise ValueError("Checkpoint inventory does not completely verify the JAX model")


def verify_inputs(payload):
    """Validate the new plan against actual checkpoint assets and raw source."""
    if sha(payload["plan_path"]) != payload["plan_sha256"]:
        raise ValueError("Frozen subtask plan changed")
    scripts = str(Path(payload["code"]) / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from codex_subtask_collection import load_plan, SELECTED_SUBTASKS
    plan = load_plan(payload["plan_path"])
    sample = plan["sample"]
    if ((sample["task"], sample["subtask"]) not in SELECTED_SUBTASKS
            or type(sample["task_instance_id"]) is not int or not 0 <= sample["task_instance_id"] <= 300):
        raise ValueError("This bounded worker only admits selected training subtasks")
    if Path(plan["raw_data_root"]).resolve() != Path(payload["raw_data_root"]).resolve():
        raise ValueError("Raw data root differs from the frozen subtask plan")
    identity = plan["checkpoint_identity"]
    if identity["checkpoint_step"] != 1688977:
        raise ValueError("The selected trained checkpoint must be 1688977")
    checkpoint = Path(payload["checkpoint"])
    if sha(checkpoint / NORM_RELATIVE) != NORM_SHA256:
        raise ValueError("Checkpoint normalization differs from the verified 1688977 assets")
    if sha(checkpoint / "assets/subtask_conditioning.json") != CONDITIONING_SHA256:
        raise ValueError("Checkpoint action conditioning differs")
    conditioning = json.loads((checkpoint / "assets/subtask_conditioning.json").read_text())
    if conditioning.get("action_conditioning") != "task_and_subtask":
        raise ValueError("Checkpoint must preserve joint task/subtask conditioning")
    # The plan's checkpoint manifest is the immutable full upload inventory, not
    # the new fine-tune's saved-step count or the old 1375000 collection plan.
    manifest_path = payload.get("checkpoint_upload_plan")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_absolute():
        raise ValueError("An actual 1688977 checkpoint upload inventory is required")
    if sha(manifest_path) != identity["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint inventory differs from the frozen plan")
    inventory = json.loads(Path(manifest_path).read_text())
    verify_checkpoint_inventory(checkpoint, inventory, identity)
    budgets = plan["budgets"]
    if payload["wall_timeout_seconds"] < budgets["max_seconds"] + payload["policy_startup_timeout"] + 60:
        raise ValueError("Worker deadline must cover startup, collection and bounded cleanup")
    if not Path(payload["demonstration_manifest"]).is_file():
        raise ValueError("Prepared successful demonstration is missing")
    return plan


def environment(payload, cache, vk, egl, library_path, *, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith(("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK", "GOALKEEPER_", "CODEX_", "B1K_")):
            env.pop(key)
    for key in ("PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    runtime = Path(payload["runtime"])
    env.update(PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1",
               OMNIGIBSON_DISABLE_TORCH_COMPILE="1", OPENPI_DATA_HOME=str(runtime / "openpi-cache"),
               OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
               SCO_EPHEMERAL_RUNTIME_ROOT=str(cache), VK_ICD_FILENAMES=str(vk),
               __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1",
               CODEX_HOME=payload["codex_home"],
               PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", ""),
               LD_LIBRARY_PATH=library_path + ":" + str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu")
                   + ":" + str(runtime / "envs/b1k-eval/lib") + ":" + env.get("LD_LIBRARY_PATH", ""))
    return env


def policy_command(payload, task, server_token):
    code, runtime = Path(payload["code"]), Path(payload["runtime"])
    return [str(runtime / "envs/b1k-pi05/bin/python"), str(code / "openpi/scripts/b1k/serve_b1k.py"),
            "--robot", "b1k/R1Pro", "--task", "b1k/" + task, "--repo-id", REPO_ID,
            "--policy.config", "pi05_b1k", "--policy.dir", payload["checkpoint"],
            "--control_mode", "receding_horizon", "--action_horizon", "16", "--subtask-prediction",
            "--max-token-len", "512", "--max-subtask-len", "64", "--subtask-temperature", "0.0",
            "--subtask-action-conditioning", "task_and_subtask", "--collection-seeding",
            "--server-token", server_token, "--port", str(payload["port"])]


def collector_command(payload, server_token):
    code, runtime = Path(payload["code"]), Path(payload["runtime"])
    command = [str(runtime / "envs/b1k-eval/bin/python"), str(code / "scripts/collect_b1k_codex_subtask.py"),
            "--plan", payload["plan_path"], "--output-dir", str(Path(payload["run_root"]) / "attempt"),
            "--policy-host", "127.0.0.1", "--policy-port", str(payload["port"]),
            "--policy-server-token", server_token, "--teacher-codex", "--codex-binary", payload["codex_binary"],
            "--codex-timeout", str(payload["teacher_timeout"]),
            "--codex-capacity-retries", str(payload.get("teacher_capacity_retries", 0)),
            "--demonstration-manifest", payload["demonstration_manifest"], "--robot-config", payload["robot_config"]]
    if payload.get('reuse_qualification_root'):
        command.extend(['--reuse-qualification-root', payload['reuse_qualification_root']])
    if payload.get('persist_preparation'):
        command.append('--persist-preparation')
    if payload.get('resume_preparation'):
        command.extend(['--resume-preparation', payload['resume_preparation']])
    return command


def cpu_preflight(payload, env):
    script = '''import importlib,json,sys
from pathlib import Path
code,robot,demo,plan=map(str,sys.argv[1:])
sys.path[:0]=[str(Path(code)/"scripts"),str(Path(code)/"sco")]
names=("numpy","h5py","PIL.Image","msgpack","websockets.sync.client","yaml","omegaconf",
       "codex_collection_actions","codex_subtask_teacher","codex_subtask_collection","codex_gripper_feedback",
       "codex_robot_guidance","codex_scene_ground_truth","codex_jacobian_validation","collect_b1k_codex_subtask")
modules={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in names}
from codex_subtask_collection import load_plan
plan=load_plan(plan)
from run_codex_subtask_collection import robot_contract
import yaml
contract=robot_contract(yaml.safe_load(Path(robot).read_text()))
from codex_collection_demonstrations import demonstration_context
sample=plan["sample"]
context,images=demonstration_context(demo,sample["task"],sample["subtask"],limit=1)
if plan.get("teacher_robot_guidance"):
 from codex_robot_guidance import prepare_reference_capture
 relative=f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
 prepare_reference_capture(demo,plan,{"sha256":plan["source_plan"]["raw_source_sha256"][relative]})
from PIL import Image
for path in images:
    with Image.open(path) as image:image.verify()
print(json.dumps(dict(modules=modules,robot_contract=contract,plan_fingerprint=plan["fingerprint"],
    demonstration_fingerprint=context["demonstration_fingerprint"],example_images=len(images),
    teacher_execution_protocol=plan.get("teacher_execution"),
    teacher_robot_guidance=plan.get("teacher_robot_guidance"),
    teacher_scene_ground_truth=plan.get("teacher_scene_ground_truth"),
    teacher_configuration=plan.get("teacher_configuration"),
    teacher_access=plan.get("teacher_access"), teacher_strategy=plan.get("teacher_strategy"),
    teacher_motion_validation=plan.get("teacher_motion_validation"))))
'''
    output = subprocess.check_output([str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python"), "-c", script,
        payload["code"], payload["robot_config"], payload["demonstration_manifest"], payload["plan_path"]],
        env=env, text=True, timeout=90)
    return json.loads(output)


def stop_owned(process):
    """Terminate this driver's new process group, including any owned children."""
    if process is None:
        return
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


def wait_policy(payload, server_token, process, env, timeout):
    script = '''import msgpack,os,sys,threading,time
import websockets.sync.client
port,token,pid,seconds=sys.argv[1:]
deadline=time.monotonic()+float(seconds)
while time.monotonic()<deadline:
    try:os.kill(int(pid),0)
    except ProcessLookupError:raise SystemExit("Owned policy exited during startup")
    try:
        with websockets.sync.client.connect(f"ws://127.0.0.1:{int(port)}",compression=None,max_size=None,
                open_timeout=2,close_timeout=1) as connection:
            metadata=msgpack.unpackb(connection.recv(timeout=2),raw=False)
        if (isinstance(metadata,dict) and metadata.get("server_instance_token")==token
                and metadata.get("subtask_generation") is True
                and metadata.get("subtask_action_conditioning")=="task_and_subtask"
                and metadata.get("collection_seeding") is True):raise SystemExit(0)
    except Exception:pass
    threading.Event().wait(min(1,max(0,deadline-time.monotonic())))
raise SystemExit("Owned policy readiness deadline exceeded")
'''
    subprocess.run([str(Path(payload["runtime"]) / "envs/b1k-eval/bin/python"), "-c", script,
                    str(payload["port"]), server_token, str(process.pid), str(timeout)],
                   env=env, check=True, timeout=timeout + 10)


def validate_completion(completed, plan):
    if (completed.get("plan_fingerprint") != plan["fingerprint"]
            or completed.get("official_benchmark") is not False
            or type(completed.get("training_eligible")) is not bool):
        raise ValueError("Subtask completion receipt does not match its frozen plan")
    result = completed.get("result", {})
    if (result.get("official_benchmark") is not False or type(result.get("success")) is not bool
            or type(result.get("metric_eligible")) is not bool or type(result.get("steps")) is not int
            or not 0 <= result["steps"] <= plan["budgets"]["max_steps"]):
        raise ValueError("Invalid subtask completion result")
    eligible = result["success"] and result["metric_eligible"] and result["steps"] > 0
    if completed["training_eligible"] is not eligible:
        raise ValueError("Training eligibility must follow actual subtask success")
    return result


def run(payload):
    payload = validate_payload(payload)
    os.umask(0o077)
    root = Path(payload["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + payload["wall_timeout_seconds"]
    status = {"kind": "codex_subtask_worker_v1", "state": "preparing", "checkpoint_step": 1688977,
              "official_benchmark": False, "training_eligible": False, "started_at": time.time()}
    def publish():
        status["updated_at"] = time.time()
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "status.json")
        print("STATUS_JSON=" + json.dumps(status, allow_nan=False), flush=True)
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Subtask worker deadline exhausted")
        return seconds
    publish()
    resources, policy = ExitStack(), None
    previous_signals = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Owned subtask worker interrupted")
    for number in previous_signals:
        signal.signal(number, interrupted)
    try:
        library, base_sha = verify_sources(payload)
        plan = verify_inputs(payload)
        sample = plan["sample"]
        status.update(task=sample["task"], instance_id=sample["task_instance_id"], subtask=sample["subtask"],
                      attempt_id=plan["attempt_id"], plan_fingerprint=plan["fingerprint"],
                      checkpoint_identity=plan["checkpoint_identity"], plan_sha256=payload["plan_sha256"])
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True, timeout=30).strip().splitlines()
        if len(rows) != 1 or not any(name in rows[0] for name in ("5090", "N12LP")):
            raise ValueError("The bounded pilot requires one RTX 5090 worker")
        driver = rows[0].split(",")[2].strip()
        libraries = [path for parent in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                                         "/lib/x86_64-linux-gnu", "/lib64")
                     for path in Path(parent).glob("libEGL_nvidia.so." + driver)]
        if not libraries:
            raise ValueError("Matching NVIDIA graphics library is unavailable")
        graphics = root / "graphics"
        graphics.mkdir()
        vk, egl = graphics / "nvidia_icd.json", graphics / "10_nvidia.json"
        vk.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
            "library_path": str(libraries[0]), "api_version": "1.3.242"}}))
        egl.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}}))
        cache = Path("/var/tmp/behavior-1k-runtime") / payload["execution_name"]
        cache.mkdir(parents=True, exist_ok=False)
        if cache.stat().st_dev == Path(payload["runtime"]).stat().st_dev or shutil.disk_usage(cache).free < 40 * 1024**3:
            raise ValueError("Need 40GiB of separate worker-local runtime storage")
        env = environment(payload, cache, vk, egl, library["library_path"])
        from codex_cluster_proxy import CodexHomeLease, MihomoProxy
        home = Path(payload["codex_home"]).resolve(strict=True)
        if plan.get("teacher_access") != "shared_home_per_call_v1":
            resources.enter_context(CodexHomeLease(home))
        proxy = resources.enter_context(MihomoProxy(config_dir=Path(payload["mihomo_config_dir"]),
            run_dir=cache / "teacher-network", mihomo_binary=payload["mihomo_binary"]))
        env = proxy.environment(env, codex_home=home)
        status["teacher_proxy"] = proxy.receipt
        status["cpu_preflight"] = cpu_preflight(payload, env)
        status.update(state="starting_policy", original_source_manifest_sha256=base_sha,
                      source_files_sha256=payload["source_files_sha256"], gpu_rows=rows)
        publish()
        code, runtime = Path(payload["code"]), Path(payload["runtime"])
        server_token = secrets.token_hex(32)
        policy_env = {**env, "PYTHONPATH": str(code / "openpi/src") + ":" + str(code / "openpi/packages/openpi-client/src"),
                      "XLA_PYTHON_CLIENT_PREALLOCATE": "true", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
                      "LD_LIBRARY_PATH": str(runtime / "envs/b1k-pi05/lib") + ":" + env["LD_LIBRARY_PATH"]}
        logs = root / "logs"
        logs.mkdir()
        policy_log = resources.enter_context((logs / "policy.log").open("x", buffering=1))
        policy = subprocess.Popen(policy_command(payload, sample["task"], server_token), env=policy_env,
            cwd=code / "openpi", stdout=policy_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_policy(payload, server_token, policy, env, min(payload["policy_startup_timeout"], remaining()))
        sim = cache / "sim"
        for directory in ("appdata", "cache", "config", "warp"):
            (sim / directory).mkdir(parents=True)
        collector_env = {**env, "PYTHONPATH": ":".join(str(code / path) for path in (
            "BEHAVIOR-1K/OmniGibson", "BEHAVIOR-1K/bddl3", "BEHAVIOR-1K/joylo", "openpi/packages/openpi-client/src")),
            "OMNIGIBSON_APPDATA_PATH": str(sim / "appdata"), "OMNIGIBSON_GPU_ID": "0",
            "XDG_CACHE_HOME": str(sim / "cache"), "XDG_CONFIG_HOME": str(sim / "config"),
            "WARP_CACHE_PATH": str(sim / "warp")}
        from codex_subtask_collection import HISTORY, NATIVE
        if plan["source_plan"].get("source_protocol_by_sample", {}).get(sample["sample_id"], NATIVE) == HISTORY:
            collector_env["B1K_LEGACY_HISTORY_RECOVERY"] = HISTORY
        status["state"] = "collecting"
        publish()
        with (logs / "collector.log").open("x", buffering=1) as log:
            exit_code = bounded_process(collector_command(payload, server_token), env=collector_env,
                cwd=code / "BEHAVIOR-1K", log=log, timeout=remaining())
        status["collector_exit_code"] = exit_code
        attempt = root / "attempt"
        failures = [attempt / "failed.json", attempt / "collector_failure.json",
                    root / ("collector_failure_" + plan["attempt_id"] + ".json")]
        if exit_code or policy.poll() is not None or any(path.exists() for path in failures):
            raise RuntimeError("Collector or owned policy failed; preserve subtask evidence")
        qualification_path = root / "qualification/qualification.json"
        if plan["schema_version"] >= 2:
            qualification = json.loads(qualification_path.read_text())
            if (qualification.get("plan_fingerprint") != plan["fingerprint"]
                    or type(qualification.get("qualified")) is not bool):
                raise ValueError("Qualification does not match the frozen plan")
            status["qualification"] = qualification
            if qualification["qualified"] is False:
                if attempt.exists():
                    raise ValueError("A rejected qualification must not start assisted collection")
                status.update(state="complete", training_eligible=False,
                              collection_skipped=qualification["rejection_reason"], completed_at=time.time())
                publish()
                (root / "_COMPLETE").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
                return status
        completed = json.loads((attempt / "completed.json").read_text())
        result = validate_completion(completed, plan)
        status.update(state="complete", training_eligible=completed["training_eligible"], result=result,
                      completion_receipt_sha256=sha(attempt / "completed.json"), completed_at=time.time())
        publish()
        (root / "_COMPLETE").write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        return status
    except BaseException as error:
        status.update(failure_phase=status["state"], state="failed", error_type=type(error).__name__,
                      training_eligible=False)
        publish()
        raise
    finally:
        try:
            stop_owned(policy)
        finally:
            resources.close()
            for number, handler in previous_signals.items():
                signal.signal(number, handler)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))

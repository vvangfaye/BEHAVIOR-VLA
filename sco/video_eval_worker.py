"""One immutable checkpoint-2000000 episode, with paired video and bounded execution.

The controller supplies signed PUT URLs in ``artifact_puts``. They are deliberately
excluded from all persistent metadata and command arguments. This worker never
retries a policy rollout: an infrastructure error produces a failed receipt.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request


SCHEMA_VERSION = 1
CHECKPOINT_STEP = 2000000
INIT_TIMEOUT = 1800
STALL_TIMEOUT = 900
MAX_JOB_SECONDS = 96 * 3600
REQUIRED_SOURCE_FILES = {
    "scripts/eval_b1k.sh", "scripts/eval_b1k_queue.py", "scripts/pi05_eval_queue.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/__init__.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/default_wrapper.py",
    "BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/eval_utils.py",
}

# This observer changes neither observations nor actions. It records completed
# native Evaluator.step calls, so a live logger cannot conceal a stuck simulator.
PROGRESS_HOOK = r'''
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

if os.environ.get("VIDEO_EVAL_PROGRESS") and sys.argv[0].endswith("eval_b1k_queue.py"):
    _path = Path(os.environ["VIDEO_EVAL_PROGRESS"])
    def _write(value):
        value.update(pid=os.getpid(), observed_at=time.time(), run_id=os.environ["VIDEO_EVAL_RUN_ID"])
        temporary = _path.with_name(_path.name + ".tmp")
        temporary.write_text(json.dumps(value))
        os.replace(temporary, _path)
    class _Loader(importlib.abc.Loader):
        def __init__(self, original): self.original = original
        def create_module(self, spec): return self.original.create_module(spec)
        def exec_module(self, module):
            self.original.exec_module(module)
            original = module.Evaluator.step
            def observed_step(evaluator, *args, **kwargs):
                result = original(evaluator, *args, **kwargs)
                now = time.monotonic()
                previous = getattr(evaluator, "_video_eval_last_observed", None)
                terminal = bool(result[0] or result[1])
                if previous is None or now - previous >= 5 or terminal:
                    _write(dict(state="progress", steps=int(evaluator._rollout_steps), terminal=terminal))
                    evaluator._video_eval_last_observed = now
                return result
            module.Evaluator.step = observed_step
            _write(dict(state="hook_ready", steps=0, terminal=False))
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "omnigibson.eval.evaluator": return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec
    sys.meta_path.insert(0, _Finder())
'''

VIDEO_CHECK = r'''
import cv2
import json
import sys
from pathlib import Path
path, expected, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
cap = cv2.VideoCapture(str(path))
if not cap.isOpened(): raise RuntimeError("Cannot open rollout video")
fps = float(cap.get(cv2.CAP_PROP_FPS))
advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
dimensions = None
samples = []
while True:
    ok, frame = cap.read()
    if not ok: break
    if frame is None or frame.size == 0: raise RuntimeError("Empty decoded frame")
    shape = (int(frame.shape[1]), int(frame.shape[0]))
    if shape != (672, 448): raise RuntimeError("Unexpected native three-camera video dimensions")
    if dimensions is not None and dimensions != shape: raise RuntimeError("Changing video dimensions")
    dimensions = shape
    if count in (0, expected // 2, expected - 1):
        sample = output / ("frame-" + str(count) + ".png")
        if not cv2.imwrite(str(sample), frame): raise RuntimeError("Cannot write verification frame")
        samples.append(sample.name)
    count += 1
cap.release()
if count != expected or advertised != expected: raise RuntimeError("Video does not contain exactly one frame per simulator step")
if abs(fps - 30.0) > 0.02: raise RuntimeError("Unexpected native video FPS")
print(json.dumps(dict(decoded_frames=count, advertised_frames=advertised, fps=fps,
                     width=dimensions[0], height=dimensions[1], samples=samples,
                     full_sequential_decode=True, frames_equal_simulator_steps=True)))
'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def contained_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing or escaping immutable input")
    return path


def validate_payload(payload):
    if payload.get("checkpoint_step") != CHECKPOINT_STEP:
        raise ValueError("Unexpected checkpoint step")
    instance = payload.get("instance_id")
    if isinstance(instance, bool) or not isinstance(instance, int) or not 301 <= instance <= 310:
        raise ValueError("Only official public-test IDs 301..310 are permitted")
    for key in ("task", "run_id", "execution_name"):
        if not isinstance(payload.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", payload[key]):
            raise ValueError("Invalid immutable identity field: " + key)
    for key in ("source_manifest_sha256", "checkpoint_ready_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, ""))):
            raise ValueError("Invalid input digest: " + key)
    for key in ("checkpoint", "code", "runtime", "assignment_root"):
        if not Path(payload[key]).is_absolute():
            raise ValueError("Worker paths must be absolute")
    if not isinstance(payload.get("pilot"), bool):
        raise ValueError("Explicit pilot flag required")
    for key in ("archive", "receipt"):
        value = payload["artifact_puts"][key]
        if not isinstance(value, str) or not value.startswith("https://") or any(c in value for c in "\r\n"):
            raise ValueError("Signed artifact PUT must use HTTPS")
    get_url = payload["artifact_gets"]["archive"]
    if not isinstance(get_url, str) or not get_url.startswith("https://") or any(c in get_url for c in "\r\n"):
        raise ValueError("Signed archive readback must use HTTPS")
    for key, default in (("init_timeout_seconds", INIT_TIMEOUT), ("stall_timeout_seconds", STALL_TIMEOUT),
                         ("job_timeout_seconds", MAX_JOB_SECONDS)):
        value = payload.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_JOB_SECONDS:
            raise ValueError("Invalid bounded timeout: " + key)
    return {k: v for k, v in payload.items() if k not in ("artifact_puts", "artifact_gets")}


def verify_inputs(payload):
    code, checkpoint, runtime = (Path(payload[k]) for k in ("code", "checkpoint", "runtime"))
    source_file = code / "_PUBLIC_TEST_SOURCE_READY"
    ready_file = checkpoint / "_READY"
    if sha256(source_file) != payload["source_manifest_sha256"] or sha256(ready_file) != payload["checkpoint_ready_sha256"]:
        raise ValueError("Immutable source or checkpoint manifest changed")
    source, ready = json.loads(source_file.read_bytes()), json.loads(ready_file.read_bytes())
    if source["checkpoint_step"] != CHECKPOINT_STEP or ready["checkpoint_step"] != CHECKPOINT_STEP:
        raise ValueError("Checkpoint manifest identity mismatch")
    if not REQUIRED_SOURCE_FILES <= set(source["files_sha256"]):
        raise ValueError("Source manifest omits required evaluation/configuration files")
    for relative, expected in source["files_sha256"].items():
        if sha256(contained_file(code, relative)) != expected:
            raise ValueError("Frozen source hash mismatch: " + relative)
    checkpoint_files = source["checkpoint_files"]
    if not checkpoint_files:
        raise ValueError("Checkpoint manifest is empty")
    for row in checkpoint_files:
        path = contained_file(checkpoint, row["relative_path"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Downloaded checkpoint file failed full readback")
    if not (runtime / "_RUNTIME_READY").is_file():
        raise ValueError("Missing validated runtime")
    version_file = code / "BEHAVIOR-1K/OmniGibson/omnigibson/__init__.py"
    if not re.search(r'__version__\s*=\s*[\"\']3\.9\.3[\"\']', version_file.read_text()):
        raise ValueError("Frozen evaluator must identify as OmniGibson 3.9.3")
    return source


def validate_result(value, payload, progress):
    expected = (payload["task"], payload["instance_id"], 0)
    if (value.get("task"), value.get("instance_id"), value.get("rollout_id")) != expected:
        raise ValueError("Result identity mismatch")
    steps = value.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("Result has no actual simulator steps")
    if not isinstance(value.get("success"), bool) or not (value.get("terminated") or value.get("truncated")):
        raise ValueError("Result is not a completed episode")
    if progress.get("run_id") != payload["run_id"] or progress.get("steps") != steps or progress.get("terminal") is not True:
        raise ValueError("Native terminal step heartbeat does not match result")
    conditioning = value["conditioning"]
    if conditioning.get("mode") != "whole_task" or conditioning.get("oracle") is not False or conditioning.get("privileged") is not False:
        raise ValueError("Privileged or non-whole-task evaluation rejected")
    events = value["subtask_events"]
    if not events or not any(x.get("source") == "generated" for x in events) or any(x.get("source") not in ("generated", "cached") for x in events):
        raise ValueError("Subtasks must come from the policy itself")
    if value.get("selector_events"):
        raise ValueError("External task selector events rejected")
    q = value["q_score"]["final"]
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Invalid Q score")
    if value["success"] != (q == 1):
        raise ValueError("Success disagrees with final task Q score")
    if value["time"]["simulator_steps"] != steps:
        raise ValueError("Simulator metrics disagree with step count")
    for metric in (value["time"]["simulator_time"], value["time"]["normalized_time"],
                   *(value["normalized_agent_distance"][side] for side in ("base", "left", "right"))):
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric) or metric < 0:
            raise ValueError("Invalid official efficiency metric")
    return steps, q


class ProgressWatchdog:
    def __init__(self, started, init_timeout=INIT_TIMEOUT, stall_timeout=STALL_TIMEOUT, job_timeout=MAX_JOB_SECONDS):
        self.started = self.last_progress = started
        self.steps = 0
        self.init_timeout, self.stall_timeout, self.job_timeout = init_timeout, stall_timeout, job_timeout

    def observe(self, now, progress):
        steps = progress.get("steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < self.steps:
            raise ValueError("Native progress reset or invalid step count")
        if steps > self.steps:
            self.steps, self.last_progress = steps, now
        if now - self.started >= self.job_timeout:
            return "job_walltime_limit"
        if self.steps == 0 and now - self.started >= self.init_timeout:
            return "no_native_steps_within_initialization_grace"
        if self.steps > 0 and now - self.last_progress >= self.stall_timeout:
            return "native_simulator_progress_stalled"
        return None


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def put_file(path, signed_url):
    # The URL stays on stdin, not in ps output or persisted command logs.
    config = "url = " + json.dumps(signed_url) + "\n"
    result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--retry", "2",
                             "--connect-timeout", "30", "--max-time", "1200",
                             "--upload-file", str(path), "--config", "-"],
                            input=config, text=True, capture_output=True, timeout=3700)
    if result.returncode:
        raise RuntimeError("Artifact upload failed (curl exit " + str(result.returncode) + ")")


def verify_remote_archive(signed_url, expected_size, expected_sha256, *, opener=urllib.request.urlopen):
    """Read the complete uploaded object without retaining another disk copy."""
    for attempt in range(3):
        digest, size, started = hashlib.sha256(), 0, time.monotonic()
        try:
            with opener(signed_url, timeout=60) as stream:
                while True:
                    block = stream.read(8 * 1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > expected_size or time.monotonic() - started > 1200:
                        raise ValueError("Uploaded archive size or readback time exceeds bound")
                    digest.update(block)
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError("Uploaded archive failed complete size/SHA256 readback")
            return True
        except Exception:
            if attempt == 2:
                # Do not propagate transport exception messages containing signed URLs.
                raise RuntimeError("Uploaded archive failed complete readback after three attempts") from None
            time.sleep(2)


def _environment(payload, source, root, cache, hook_dir):
    runtime = Path(payload["runtime"])
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu_rows) != 1 or not any(name in gpu_rows[0] for name in ("5090", "N12LP")):
        raise ValueError("Exactly one compatible RTX5090 required")
    driver = gpu_rows[0].split(",")[2].strip()
    libraries = [p for parent in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/lib/x86_64-linux-gnu", "/lib64")
                 for p in Path(parent).glob("libEGL_nvidia.so." + driver)]
    if not libraries:
        raise ValueError("Matching host graphics driver missing")
    graphics = root / "graphics"
    graphics.mkdir()
    vk, egl = graphics / "nvidia_icd.json", graphics / "10_nvidia.json"
    write_json(vk, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0]), "api_version": "1.3.242"}})
    write_json(egl, {"file_format_version": "1.0.0", "ICD": {"library_path": str(libraries[0])}})
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("PI05_", "SUBTASK_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK", "GOALKEEPER_", "VIDEO_EVAL_")):
            env.pop(name)
    for name in ("PYTHONHOME", "MAX_STEPS", "EVAL_EXTRA_ARGS", "ENV_WRAPPER", "ROBOT_CONFIG"):
        env.pop(name, None)
    library_paths = [str(runtime / "graphics-runtime/usr/lib/x86_64-linux-gnu"), str(runtime / "envs/b1k-eval/lib")]
    if "supplemental_library" in source:
        library = source["supplemental_library"]
        ready = Path(library["root"]) / "_READY"
        if sha256(ready) != library["ready_manifest_sha256"]:
            raise ValueError("Supplemental runtime manifest changed")
        library_ready = json.loads(ready.read_bytes())
        for name, expected in library_ready["libraries_sha256"].items():
            if sha256(Path(library_ready["library_path"]) / name) != expected:
                raise ValueError("Supplemental library changed")
        library_paths.insert(0, library_ready["library_path"])
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(hook_dir),
               OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1", OMNIGIBSON_DISABLE_TORCH_COMPILE="1",
               OPENPI_DATA_HOME=str(runtime / "openpi-cache"), OMNIGIBSON_DATA_PATH=str(runtime / "omnigibson-data"),
               PI05_ENV=str(runtime / "envs/b1k-pi05"), EVAL_ENV=str(runtime / "envs/b1k-eval"),
               PI05_CKPT=payload["checkpoint"], PI05_REPO_ID="b1k_full100_joint_subtask_v1_drop_episode_s1",
               SUBTASK_PREDICTION="1", MAX_TOKEN_LEN="512", MAX_SUBTASK_LEN="64", EXECUTE_HORIZON="16",
               PI05_MEM_FRACTION="0.35", XLA_PYTHON_CLIENT_PREALLOCATE="false", MODEL="pi05",
               SERVER_GPU="0", EVAL_GPU="0", PORT="8910", MODE="public_test", PROMPT_MODE="whole_task",
               NUM_ROLLOUTS="1", INSTANCE_INDICES=str(payload["instance_id"] - 301), WRITE_VIDEO="1",
               OUTPUT_ROOT=str(root / "results"), SCO_EPHEMERAL_RUNTIME_ROOT=str(cache),
               VK_ICD_FILENAMES=str(vk), __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
               LD_LIBRARY_PATH=":".join(library_paths) + ":" + env.get("LD_LIBRARY_PATH", ""),
               PI05_EVAL_QUEUE_DB=str(root / "queue.sqlite3"), PI05_EVAL_QUEUE_WORKER="gpu0",
               PI05_EVAL_QUEUE_RUN_ID=payload["run_id"], PI05_EVAL_QUEUE_GENERATION="0",
               PI05_EVAL_QUEUE_ATTEMPT_ROOT=str(root / "attempts"),
               VIDEO_EVAL_PROGRESS=str(cache / "native_progress.json"), VIDEO_EVAL_RUN_ID=payload["run_id"])
    return env, gpu_rows


def run(payload):
    safe_payload = validate_payload(payload)
    root = Path(payload["assignment_root"])
    root.mkdir(parents=True, exist_ok=False)  # Never rerun/overwrite an existing attempt.
    write_json(root / "payload.json", safe_payload)
    process = None
    receipt = dict(schema_version=SCHEMA_VERSION, checkpoint_step=CHECKPOINT_STEP,
                   task=payload["task"], instance_id=payload["instance_id"], rollout_id=0,
                   run_id=payload["run_id"], execution_name=payload["execution_name"], pilot=payload["pilot"],
                   source_manifest_sha256=payload["source_manifest_sha256"],
                   checkpoint_ready_sha256=payload["checkpoint_ready_sha256"], started_at=time.time(), state="preflight")
    try:
        source = verify_inputs(payload)
        code, runtime = Path(payload["code"]), Path(payload["runtime"])
        cache = Path("/var/tmp/behavior-1k-runtime") / payload["execution_name"]
        cache.mkdir(parents=True, exist_ok=False)
        if cache.stat().st_dev == runtime.stat().st_dev or shutil.disk_usage(cache).free < 40 * 1024**3:
            raise ValueError("Worker cache must be local with at least 40 GiB free")
        hook_dir = cache / "progress_hook"
        hook_dir.mkdir()
        (hook_dir / "sitecustomize.py").write_text(PROGRESS_HOOK)
        env, gpu_rows = _environment(payload, source, root, cache, hook_dir)
        receipt["gpu_rows"] = gpu_rows
        sys.path.insert(0, str(code / "scripts"))
        from pi05_eval_queue import EvaluationQueue
        queue = EvaluationQueue(root / "queue.sqlite3")
        queue.initialize([(payload["task"], payload["instance_id"])], max_attempts=1,
                         metadata=dict(run_id=payload["run_id"], mode="public_test", num_rollouts=1,
                                       require_videos=True, checkpoint_step=CHECKPOINT_STEP,
                                       source_manifest_sha256=payload["source_manifest_sha256"]))
        command = ["bash", str(code / "scripts/eval_b1k.sh"), payload["task"]]
        write_json(root / "execution.json", dict(command=command, env={k: env[k] for k in (
            "MODE", "PROMPT_MODE", "NUM_ROLLOUTS", "INSTANCE_INDICES", "WRITE_VIDEO", "PI05_CKPT",
            "SUBTASK_PREDICTION", "EXECUTE_HORIZON", "MAX_TOKEN_LEN", "MAX_SUBTASK_LEN", "PI05_REPO_ID",
            "PI05_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "OMNIGIBSON_DISABLE_TORCH_COMPILE",
            "SERVER_GPU", "EVAL_GPU", "PORT", "OMNIGIBSON_DATA_PATH", "PI05_ENV", "EVAL_ENV",
            "PYTHONPATH", "PYTHONNOUSERSITE", "LD_LIBRARY_PATH", "VK_ICD_FILENAMES", "__EGL_VENDOR_LIBRARY_FILENAMES")},
            wrapper="omnigibson.eval.wrappers.DefaultWrapper", robot_config="omnigibson/eval/r1pro.yaml",
            max_steps=None, video_fps=30, native_progress_hook_sha256=sha256(hook_dir / "sitecustomize.py")))
        watcher = ProgressWatchdog(time.monotonic(), payload.get("init_timeout_seconds", INIT_TIMEOUT),
                                   payload.get("stall_timeout_seconds", STALL_TIMEOUT), payload.get("job_timeout_seconds", MAX_JOB_SECONDS))
        progress = {}
        receipt["state"] = "running"
        write_json(root / "status.json", receipt)
        with (root / "driver.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, env=env, cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                progress_path = cache / "native_progress.json"
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_bytes())
                    if progress.get("run_id") != payload["run_id"]:
                        raise ValueError("Progress belongs to another execution")
                    write_json(root / "native_progress.json", progress)
                reason = watcher.observe(time.monotonic(), progress)
                if reason:
                    terminate_group(process)
                    raise TimeoutError(reason)
                time.sleep(5)
        if process.returncode:
            raise RuntimeError("Evaluator failed (exit " + str(process.returncode) + ")")
        progress = json.loads((cache / "native_progress.json").read_bytes())
        write_json(root / "native_progress.json", progress)
        snapshot = queue.snapshot(readonly=True)
        if len(snapshot["jobs"]) != 1 or snapshot["jobs"][0]["state"] != "succeeded" or snapshot["jobs"][0]["attempts"] != 1:
            raise ValueError("Exactly one successful infrastructure attempt required")
        artifact = queue.create_result_manifest(root / "results", root / "RESULT_MANIFEST.json", model="pi05", num_rollouts=1, require_videos=True)
        if artifact["rollout_count"] != 1:
            raise ValueError("Single-episode output mismatch")
        row = artifact["rollouts"][0]
        result_path = contained_file(root / "results", row["json"]["path"])
        video_path = contained_file(root / "results", row["video"]["path"])
        value = json.loads(result_path.read_bytes())
        steps, q = validate_result(value, payload, progress)
        attempt = snapshot["attempts"]
        if len(attempt) != 1 or attempt[0]["state"] != "succeeded":
            raise ValueError("Unexpected attempt history")
        attempt_root = Path(attempt[0]["attempt_dir"])
        if sha256(attempt_root / "json" / result_path.name) != sha256(result_path) or sha256(attempt_root / "videos" / video_path.name) != sha256(video_path):
            raise ValueError("JSON and video must both match the single original attempt")
        frames_dir = root / "video_verification"
        frames_dir.mkdir()
        video_validation = json.loads(subprocess.check_output(
            [str(runtime / "envs/b1k-eval/bin/python"), "-c", VIDEO_CHECK, str(video_path), str(steps), str(frames_dir)],
            env=env, text=True, timeout=1800))
        write_json(frames_dir / "validation.json", video_validation)
        bundle = root / "bundle"
        bundle.mkdir()
        originals = {"json/" + result_path.name: result_path, "videos/" + video_path.name: video_path,
                     "metadata/source_manifest.json": code / "_PUBLIC_TEST_SOURCE_READY",
                     "metadata/checkpoint_ready.json": Path(payload["checkpoint"]) / "_READY",
                     "metadata/execution.json": root / "execution.json", "metadata/payload.json": root / "payload.json",
                     "metadata/native_progress.json": root / "native_progress.json", "metadata/RESULT_MANIFEST.json": root / "RESULT_MANIFEST.json",
                     "config/r1pro.yaml": code / "BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml",
                     "wrapper/default_wrapper.py": code / "BEHAVIOR-1K/OmniGibson/omnigibson/eval/wrappers/default_wrapper.py",
                     "code/progress_sitecustomize.py": hook_dir / "sitecustomize.py", "logs/driver.log": root / "driver.log"}
        write_json(root / "queue_snapshot.json", snapshot)
        originals["metadata/queue_snapshot.json"] = root / "queue_snapshot.json"
        for directory in (root / "logs", frames_dir):
            if directory.is_dir():
                for path in directory.rglob("*"):
                    if path.is_file():
                        originals[path.relative_to(root).as_posix()] = path
        for relative in ("scripts/eval_b1k.sh", "scripts/eval_b1k_queue.py", "scripts/pi05_eval_queue.py",
                         "BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py",
                         "BEHAVIOR-1K/OmniGibson/omnigibson/eval/eval.py"):
            originals["code/" + relative] = contained_file(code, relative)
        files = []
        for relative, path in originals.items():
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            original_hash = sha256(path)
            shutil.copyfile(path, target)
            if sha256(target) != original_hash:
                raise ValueError("Artifact changed during packaging")
            files.append(dict(path=relative, bytes=target.stat().st_size, sha256=original_hash))
        receipt.update(state="artifacts_verified", completed_at=time.time(), model_success=value["success"], q_score=q,
                       verified=False, metrics=dict(success=value["success"], q_score=q),
                       steps=steps, video_validation=video_validation, rollout_count=1,
                       canonical_results_kept_regardless_of_score=True, same_attempt_json_and_video=True,
                       full_video_decode_verified=True, files=files)
        write_json(bundle / "ARTIFACT_MANIFEST.json", receipt)
        archive = root / "artifact.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(bundle, arcname="episode", recursive=True)
        receipt["archive"] = dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive))
        receipt["archive_sha256"] = receipt["archive"]["sha256"]
        receipt["archive_size"] = receipt["archive"]["bytes"]
        write_json(root / "receipt.json", receipt)
        put_file(archive, payload["artifact_puts"]["archive"])
        verify_remote_archive(payload["artifact_gets"]["archive"], receipt["archive_size"], receipt["archive_sha256"])
        receipt.update(archive_readback_verified=True, verified=True, state="complete")
        write_json(root / "receipt.json", receipt)
        put_file(root / "receipt.json", payload["artifact_puts"]["receipt"])
        write_json(root / "_SUCCESS", receipt)
        print("STATUS_JSON=" + json.dumps({k: v for k, v in receipt.items() if k not in ("files", "gpu_rows")}), flush=True)
        return receipt
    except BaseException as error:
        if process is not None:
            terminate_group(process)
        # Preserve complete artifacts if only transport failed. Never overwrite
        # their archive with a failure log, or rerun the model to repair transport.
        if receipt.get("state") in ("artifacts_verified", "complete"):
            receipt.update(state="delivery_failed", verified=False, error_type=type(error).__name__)
            write_json(root / "delivery_error.json", receipt)
        else:
            receipt.update(state="failed", verified=False, failed_at=time.time(), error_type=type(error).__name__, error=str(error))
            write_json(root / "receipt.json", receipt)
            try:
                put_file(root / "receipt.json", payload["artifact_puts"]["receipt"])
            except Exception:
                pass
        write_json(root / "status.json", receipt)
        raise

"""Finite four-GPU worker for an already verified, frozen Codex training plan.

No cloud submission, network client, model teacher, data export, or automatic
retry. Run this driver with the existing b1k-eval Python. Training remains in
the existing b1k-pi05 environment through finetune_codex_collection.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

KIND = "codex_pilot_training_worker_v1"
PLAN_KIND = "codex_teacher_collection_finetuning"
SOURCE_SHA = "7f2a16738961dfb1ca5b95d280ee61450f996757c5514b8c144e86dce3c982d4"
HEX = re.compile(r"[0-9a-f]{64}")


class WorkerDeadline(BaseException):
    """Cannot be swallowed by a dependency's ordinary error handler."""


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), size=path.stat().st_size, sha256=sha(path))


def verify_ref(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
            or not Path(ref["path"]).is_absolute() or not HEX.fullmatch(str(ref.get("sha256")))
            or type(ref.get("size")) is not int or ref["size"] < 1):
        raise ValueError("Artifact requires an absolute path, positive size and SHA256")
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != ref["size"] or sha(path) != ref["sha256"]:
        raise ValueError("Pinned artifact changed: " + str(path))
    return path


def write_new(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def redact(value):
    text = re.sub(r"(?:https?|oss)://[^\s<>\"']+", "<URL redacted>", str(value or ""), flags=re.I)
    text = re.sub(r"(?i)(\b(?:[A-Z_]*ACCESS_KEY[A-Z_]*|[A-Z_]*API_KEY[A-Z_]*|[A-Z_]*SECRET[A-Z_]*|[A-Z_]*TOKEN[A-Z_]*|PASSWORD|SIGNATURE)\b[\"']?\s*[:=]\s*)([^\s,;]+)", r"\1<redacted>", text)
    return text[-8192:]


def disjoint(paths):
    resolved = [Path(path).resolve() for path in paths]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(resolved) for b in resolved[i + 1:]):
        raise ValueError("Training outputs and immutable input directories must be separate")


def validate_payload(payload):
    if payload.get("kind") != KIND:
        raise ValueError("Unknown training worker contract")
    for key in ("code", "runtime", "report_root", "destination"):
        if not isinstance(payload.get(key), str) or not Path(payload[key]).is_absolute():
            raise ValueError(key + " must be absolute")
    for key in ("training_plan", "code_ready"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(key + " must be a pinned file reference")
    if not HEX.fullmatch(str(payload.get("driver_sha256"))):
        raise ValueError("Training worker source must be hash-pinned")
    defaults = dict(wall_timeout_seconds=13800, preflight_timeout_seconds=1800,
                    smoke_timeout_seconds=3600, full_timeout_seconds=8400, stage_timeout_seconds=1500)
    values = {key: payload.get(key, value) for key, value in defaults.items()}
    if (any(type(value) is not int or value < 60 or value > 13800 for value in values.values())
            or values["wall_timeout_seconds"] < 180
            or any(value > values["wall_timeout_seconds"] for key, value in values.items() if key != "wall_timeout_seconds")):
        raise ValueError("All phases need finite limits within the total worker wall deadline")
    if {"teacher", "teacher_backend", "devices", "batch_size", "steps", "resume", "overwrite"} & payload.keys():
        raise ValueError("Do not override the fixed training plan through the worker")
    disjoint(payload[key] for key in ("code", "runtime", "report_root", "destination"))
    return {**payload, **values}


def verify_plan(payload):
    path = verify_ref(payload["training_plan"]);plan = load(path)
    if plan.get("fingerprint") != digest({key: value for key, value in plan.items() if key != "fingerprint"}):
        raise ValueError("Training plan fingerprint changed")
    exact = dict(schema_version=1, kind=PLAN_KIND, conditioning="task_and_subtask", devices=4, batch_size=4, seed=42,
                 smoke_steps=2, steps=256, training_action_horizon=32, execution_horizon=16,
                 optimizer_resume=False, video_backend="pyav", evaluation_agent_enabled=False,
                 collection_checkpoint_id="pi05_b1k_full100_joint_1688977",
                 collection_instances=list(range(311, 321)), evaluation_instances=list(range(301, 311)))
    if any(plan.get(key) != value or (isinstance(value, bool) and plan.get(key) is not value)
           or (type(value) is int and type(plan.get(key)) is not int) for key, value in exact.items()):
        raise ValueError("Training plan differs from the fixed 4-GPU smoke/full pilot")
    if plan.get("learning_rate") != dict(peak=1e-5, floor=1e-6, warmup=64, decay=256):
        raise ValueError("Training plan learning rate differs from the canonical helper")
    if (plan.get("init_checkpoint_identity", {}).get("checkpoint_step") != 1688977
            or not HEX.fullmatch(str(plan.get("base_norm_sha256")))
            or not isinstance(plan.get("dataset_files"), dict) or not plan["dataset_files"]):
        raise ValueError("Training plan lacks pinned base checkpoint, normalization or dataset")
    run_id = plan.get("training_run_id")
    if (not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id) is None
            or plan.get("repo_id") != "b1k_codex_" + run_id
            or plan.get("base_repo_id") != "b1k_full100_joint_subtask_v1_drop_episode_s1"
            or not isinstance(plan.get("tasks"), list) or not plan["tasks"]
            or len(set(plan["tasks"])) != len(plan["tasks"])
            or not set(plan["tasks"]) <= {"turning_on_radio", "putting_shoes_on_rack"}):
        raise ValueError("Training plan task or checkpoint asset identity differs")
    proof = plan.get("export_verification", {})
    if (any(proof.get(key) is not True for key in ("complete", "real_lerobot_reader", "real_lerobot_writer", "label_coverage"))
            or proof.get("action_horizon") != 32 or proof.get("reader_tolerance_s") != 0.005
            or proof.get("export_fingerprint") != plan.get("dataset_fingerprint")):
        raise ValueError("Mixed dataset lacks real codec/source/training-reader verification")
    for key in ("run_root", "dataset_root", "checkpoint"):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise ValueError("Plan runtime roots must be absolute")
    root = Path(plan["run_root"])
    if (root.is_symlink() or path.resolve() != root.resolve() / "training_plan.json"
            or set(item.name for item in root.iterdir()) != {"training_plan.json"}):
        raise ValueError("Training run root must contain only its new canonical plan")
    disjoint([payload[key] for key in ("code", "runtime", "report_root", "destination")]
             + [plan[key] for key in ("run_root", "dataset_root", "checkpoint")])
    return plan


def verify_code(payload):
    code = Path(payload["code"]);path = verify_ref(payload["code_ready"]);ready = load(path)
    if (path.resolve() != code.resolve() / "_CODEX_PILOT_CODE_READY" or ready.get("state") != "code_verified"
            or ready.get("code") != str(code) or ready.get("base_source_manifest_sha256") != SOURCE_SHA):
        raise ValueError("Code overlay differs from its pinned installation receipt")
    inventory = ready.get("source_files_sha256", {})
    required = {"scripts/finetune_codex_collection.py", "scripts/prepare_codex_training_data.py",
                "scripts/export_codex_collection.py", "scripts/finetune_subtask_collection.py",
                "scripts/train_pi05_subtask_b1k.sh", "scripts/train_pi05_b1k.sh",
                "sco/run_codex_pilot_evaluation.py"}
    if not required <= inventory.keys():
        raise ValueError("Code receipt omits required training helpers")
    base = Path(ready["base_code"])
    if sha(base / "_PUBLIC_TEST_SOURCE_READY") != SOURCE_SHA:
        raise ValueError("Immutable base source receipt changed")
    public = load(base / "_PUBLIC_TEST_SOURCE_READY")
    if sha(base / "_CODE_SNAPSHOT_READY") != public["base_source_sha256"]:
        raise ValueError("Immutable base snapshot receipt changed")
    original = load(base / "_CODE_SNAPSHOT_READY")
    files = {**original["overlay_files_sha256"], **public["files_sha256"], **inventory}
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not HEX.fullmatch(str(expected)) or sha(code / relative) != expected:
            raise ValueError("Frozen training source differs: " + str(name))
    return dict(bundle_sha256=ready["bundle_sha256"], source_files=len(files), code_ready=payload["code_ready"])


def environment(payload, inherited=None):
    inherited = os.environ if inherited is None else inherited
    keep = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH",
            "CUDA_HOME", "CUDA_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in inherited.items() if key in keep}
    code, runtime = Path(payload["code"]), Path(payload["runtime"])
    env.update(PI05_ENV=str(runtime / "envs/b1k-pi05"), EVAL_ENV=str(runtime / "envs/b1k-eval"),
               OPENPI_DATA_HOME=str(runtime / "openpi-cache"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0,1,2,3", WANDB_MODE="disabled",
               PYTHONPATH=os.pathsep.join(str(code / path) for path in ("scripts", "sco", "openpi/src", "openpi/packages/openpi-client/src")))
    return env


def gpu_inventory(env):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                            env=env, capture_output=True, text=True, check=True, timeout=20)
    rows = [row.strip().split(",") for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 4 or any(len(row) != 5 for row in rows) or {row[0].strip() for row in rows} != {"0", "1", "2", "3"}:
        raise ValueError("Training requires exactly four allocated GPUs indexed 0..3")
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          env=env, capture_output=True, text=True, check=True, timeout=20)
    if busy.stdout.strip():
        raise ValueError("Allocated GPUs already have compute processes; no process was stopped")
    return [dict(index=int(row[0]), uuid=row[1].strip(), name=row[2].strip(),
                 memory_total_mib=int(row[3]), memory_free_mib=int(row[4])) for row in rows]


def terminate_group(process):
    # Popen(start_new_session=True) made this PID the leader of our own group.
    # Kill the group even when the direct child exited, to reap its descendants.
    try:os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait(timeout=5)


def bounded_process(command, *, env, cwd, log_path, seconds, deadline, heartbeat):
    end = min(time.monotonic() + seconds, deadline - 30)
    if end <= time.monotonic():
        raise TimeoutError("Training worker cleanup reserve reached")
    with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as log:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                remaining = end - time.monotonic()
                if remaining <= 0:raise TimeoutError("Training phase exceeded its finite deadline")
                try:process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:pass
                heartbeat()
            if process.returncode:
                raise RuntimeError("Training subprocess failed with return code " + str(process.returncode))
        finally:
            terminate_group(process)


DATA_PREFLIGHT = '''import json,sys
from pathlib import Path
from finetune_codex_collection import audit_export,verify_checkpoint
plan=json.loads(Path(sys.argv[1]).read_text())
report,provenance,proof=audit_export(plan['dataset_root'],collection_checkpoint_id=plan['collection_checkpoint_id'])
if proof!=plan['export_verification'] or report['fingerprint']!=plan['dataset_fingerprint']:
 raise ValueError('Actual mixed reader evidence differs from the frozen training plan')
verify_checkpoint(plan['checkpoint'],plan['init_checkpoint_identity'],plan['base_repo_id'])
with Path(sys.argv[2]).open('x') as stream:json.dump({'state':'verified','export_verification':proof},stream)
'''

JAX_PREFLIGHT = '''import json,sys
from pathlib import Path
import jax
import jax.numpy as jnp
from openpi.training import config
devices=jax.devices()
if len(devices)!=4 or any(device.platform!='gpu' for device in devices) or jax.process_count()!=1:
 raise ValueError('Training runtime must expose exactly four GPU devices in one process')
cfg=config.get_config('pi05_b1k')
if cfg.data.base_config.dataset_kwargs.get('tolerance_s')!=0.005:
 raise ValueError('Actual training config reader tolerance differs')
result=jax.pmap(lambda x:jax.lax.psum(x,'device'),axis_name='device')(jnp.ones(4))
if result.block_until_ready().tolist()!=[4.0]*4:raise ValueError('Four-device collective readback failed')
with Path(sys.argv[1]).open('x') as stream:json.dump({'state':'verified','devices':4,'processes':1,'collective_verified':True,'reader_tolerance_s':0.005},stream)
'''


def phase_receipt(plan, phase):
    root = Path(plan["run_root"]) / phase
    receipt = load(root / "phase_receipt.json");proof = load(root / "checkpoint_restore_verified.json")
    expected_steps = 2 if phase == "smoke" else 256
    if ((root / "_RUNNING").exists() or (root / "_FAILED").exists() or receipt.get("complete") is not True
            or receipt.get("kind") != PLAN_KIND or receipt.get("phase") != phase
            or receipt.get("training_plan_fingerprint") != plan["fingerprint"]
            or receipt.get("checkpoint_restore_evidence") != proof
            or proof.get("restored_step") != expected_steps or proof.get("global_devices") != 4
            or proof.get("processes") != 1 or proof.get("conditioning_mode") != "joint"
            or proof.get("action_horizon") != 32 or receipt.get("base_norm_sha256") != plan["base_norm_sha256"]
            or receipt.get("training_log_sha256") != sha(root / "train.log")):
        raise ValueError("Training phase lacks actual matching checkpoint restore evidence")
    return file_ref(root / "phase_receipt.json")


def verify_final(payload, plan):
    path = Path(payload["code"]) / "sco/run_codex_pilot_evaluation.py"
    spec = importlib.util.spec_from_file_location("training_output_native_verifier", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    receipt = file_ref(Path(plan["run_root"]) / "training_receipt.json")
    checkpoint_id = "pi05_b1k_codex_" + plan["training_run_id"] + "_256"
    result = module.verify_training(dict(training_receipt=receipt, checkpoint=payload["destination"],
                                         checkpoint_id=checkpoint_id, repo_id=plan["repo_id"]))
    if (result.get("optimizer_steps") != 256 or result.get("training_plan") != payload["training_plan"]
            or result.get("base_norm_sha256") != plan["base_norm_sha256"]):
        raise ValueError("Staged training receipt differs from this worker's exact plan")
    return receipt


def run(payload):
    payload = validate_payload(payload)
    # Reject overlapping/used plan roots before creating any worker output.
    plan = verify_plan(payload)
    code_ready = load(verify_ref(payload["code_ready"]))
    if not isinstance(code_ready.get("base_code"), str) or not Path(code_ready["base_code"]).is_absolute():
        raise ValueError("Code installation receipt needs an absolute immutable base")
    disjoint([code_ready["base_code"], plan["run_root"], payload["report_root"], payload["destination"]])
    report_root = Path(payload["report_root"])
    if report_root.exists() or report_root.is_symlink() or Path(payload["destination"]).exists() or Path(payload["destination"]).is_symlink():
        raise ValueError("Worker report and inference destinations must be fresh")
    os.umask(0o077);sys.dont_write_bytecode = True
    report_root.mkdir(parents=True, mode=0o700)
    report = dict(kind=KIND, state="starting", stage="verify_inputs", ready_for_evaluation=False,
                  model_teacher_enabled=False, training_plan=payload["training_plan"], phases={})
    started = time.monotonic();deadline = started + payload["wall_timeout_seconds"]
    def publish():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = report_root / "status.json.tmp"
        write_new(temporary, report);temporary.replace(report_root / "status.json")
    def stop(*_):raise WorkerDeadline("Finite training worker wall deadline reached")
    old_alarm = signal.signal(signal.SIGALRM, stop);old_term = signal.signal(signal.SIGTERM, stop)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, min(payload["wall_timeout_seconds"], old_timer[0]) if old_timer[0] > 0 else payload["wall_timeout_seconds"])
    marker = report_root / "_RUNNING";marker.touch(mode=0o600, exist_ok=False)
    log_path = None
    try:
        publish()
        if sha(__file__) != payload["driver_sha256"]:raise ValueError("Training worker source hash differs")
        report["code"] = verify_code(payload)
        runtime = Path(payload["runtime"]);python = runtime / "envs/b1k-eval/bin/python"
        pi05 = runtime / "envs/b1k-pi05/bin/python"
        if (not (runtime / "_RUNTIME_READY").is_file() or not python.is_file() or not pi05.is_file()
                or not (pi05.parent / "activate").is_file() or not (runtime / "openpi-cache").is_dir()
                or not os.path.samefile(sys.executable, python)):
            raise ValueError("Run with the pinned b1k-eval Python and existing b1k-pi05/cache runtime")
        env = environment(payload);report["gpus"] = gpu_inventory(env)
        commands = [
            ("data_preflight", [str(python), "-c", DATA_PREFLIGHT, payload["training_plan"]["path"], str(report_root / "data_preflight.json")], payload["preflight_timeout_seconds"]),
            ("jax_preflight", [str(pi05), "-c", JAX_PREFLIGHT, str(report_root / "jax_preflight.json")], min(300, payload["preflight_timeout_seconds"])),
            ("smoke", [str(python), str(Path(payload["code"]) / "scripts/finetune_codex_collection.py"), "run", "--plan", payload["training_plan"]["path"], "--phase", "smoke"], payload["smoke_timeout_seconds"]),
            ("full", [str(python), str(Path(payload["code"]) / "scripts/finetune_codex_collection.py"), "run", "--plan", payload["training_plan"]["path"], "--phase", "full"], payload["full_timeout_seconds"]),
            ("stage", [str(python), str(Path(payload["code"]) / "scripts/finetune_codex_collection.py"), "stage", "--plan", payload["training_plan"]["path"], "--destination", payload["destination"]], payload["stage_timeout_seconds"])]
        for phase, command, seconds in commands:
            verify_ref(payload["training_plan"])
            report.update(state="running", stage=phase);publish()
            log_path = report_root / (phase + ".log")
            phase_env = {**env, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"} if phase == "jax_preflight" else env
            bounded_process(command, env=phase_env, cwd=report_root, log_path=log_path,
                            seconds=seconds, deadline=deadline, heartbeat=publish)
            if phase in ("smoke", "full"):
                evidence = phase_receipt(plan, phase)
            elif phase == "stage":
                evidence = verify_final(payload, plan)
            else:
                evidence = file_ref(report_root / (phase + ".json"))
                if load(evidence["path"]).get("state") != "verified":raise ValueError("Runtime preflight did not verify")
            report["phases"][phase] = dict(state="verified", evidence=evidence, log=file_ref(log_path));publish()
        report.update(state="complete", stage="complete", ready_for_evaluation=True,
                      training_receipt=report["phases"]["stage"]["evidence"], checkpoint=payload["destination"])
    except BaseException as error:
        report.update(state="failed", ready_for_evaluation=False, error_type=type(error).__name__, error_detail=redact(error))
        if log_path is not None and log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 12000))
                report["failure_tail"] = redact(stream.read().decode("utf-8", errors="replace"))
        write_new(report_root / "failure.json", report)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        marker.unlink(missing_ok=True);publish()
        signal.signal(signal.SIGALRM, old_alarm);signal.signal(signal.SIGTERM, old_term)
        if old_timer[0] > 0:
            remaining = old_timer[0] - (time.monotonic() - started)
            if remaining > 0:signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(load(args.payload))
    print(json.dumps({key: report[key] for key in ("state", "stage", "ready_for_evaluation")}, indent=2))
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pi05_eval_queue import (
    ArtifactValidationError,
    EvaluationQueue,
    JobLease,
    LeaseLostError,
    atomic_write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("carrying_in_groceries", "make_rose_centerpieces")
INSTANCE_IDS = tuple(range(301, 321))
ROLLOUT_IDS = (0, 1, 2)
WORKER_COUNT = 8
EXPECTED_ROLLOUTS = len(TASKS) * len(INSTANCE_IDS) * len(ROLLOUT_IDS)
CHILD_ENV_ALLOWLIST = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SHELL",
    "TMPDIR",
    "USER",
    "OPENPI_DATA_HOME",
    "PI05_ENV",
    "EVAL_ENV",
    "OMNIGIBSON_DATA_PATH",
    "PI05_MEM_FRACTION",
    "SUBTASK_PREDICTION",
    "PROVIDED_SUBTASK_AS_PROMPT",
    "EXTERNAL_SUBTASK_SELECTOR",
    "MAX_TOKEN_LEN",
    "MAX_SUBTASK_LEN",
    "SUBTASK_INTERVAL",
    "SUBTASK_TEMPERATURE",
    "SUBTASK_ACTION_CONDITIONING",
    "EXECUTE_HORIZON",
    "OMNIGIBSON_DISABLE_TORCH_COMPILE",
    "HF_ENDPOINT",
    "LD_LIBRARY_PATH",
}


@dataclass(frozen=True)
class WorkerSpec:
    slot: int
    gpu: int
    port: int
    tasks: tuple[str, str]


@dataclass
class WorkerProcess:
    spec: WorkerSpec
    generation: int
    process: subprocess.Popen
    log_stream: object
    started_at: float

    @property
    def queue_worker(self) -> str:
        return f"slot-{self.spec.slot}"


class SupervisorInterrupted(RuntimeError):
    pass


def worker_specs(gpus: list[int], base_port: int) -> list[WorkerSpec]:
    if len(gpus) != WORKER_COUNT or len(gpus) != len(set(gpus)):
        raise ValueError(
            f"--gpus must contain exactly {WORKER_COUNT} unique GPU indices"
        )
    if any(isinstance(gpu, bool) or gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be non-negative")
    if base_port <= 0 or base_port + WORKER_COUNT - 1 > 65535:
        raise ValueError("base port does not leave room for eight valid ports")
    return [
        WorkerSpec(
            slot=slot,
            gpu=gpu,
            port=base_port + slot,
            tasks=TASKS if slot < 4 else tuple(reversed(TASKS)),
        )
        for slot, gpu in enumerate(gpus)
    ]


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_conditioning(conditioning: object) -> dict:
    if not isinstance(conditioning, dict):
        raise TypeError("conditioning must be a JSON object")
    subtask_prediction = conditioning.get("subtask_prediction")
    provided = conditioning.get("provided_subtask_as_prompt")
    selector = conditioning.get("external_subtask_selector")
    max_token_len = conditioning.get("max_token_len")
    if not isinstance(subtask_prediction, bool) or not isinstance(provided, bool):
        raise TypeError("subtask conditioning modes must be booleans")
    if subtask_prediction and provided:
        raise ValueError(
            "SUBTASK_PREDICTION and PROVIDED_SUBTASK_AS_PROMPT are mutually exclusive"
        )
    if selector not in (None, "state_machine_v1"):
        raise ValueError("EXTERNAL_SUBTASK_SELECTOR must be state_machine_v1")
    if selector is not None and not (subtask_prediction or provided):
        raise ValueError(
            "EXTERNAL_SUBTASK_SELECTOR requires a subtask conditioning mode"
        )
    if provided and selector is None:
        raise ValueError(
            "provided-subtask formal evaluation requires EXTERNAL_SUBTASK_SELECTOR"
        )
    if (
        isinstance(max_token_len, bool)
        or not isinstance(max_token_len, int)
        or max_token_len <= 0
    ):
        raise ValueError("MAX_TOKEN_LEN must be a positive integer")
    if provided and max_token_len != 200:
        raise ValueError("PROVIDED_SUBTASK_AS_PROMPT requires MAX_TOKEN_LEN=200")
    return conditioning


def _conditioning_from_args(args: argparse.Namespace) -> dict:
    provided = bool(args.provided_subtask_as_prompt)
    max_token_len = (
        args.max_token_len
        if args.max_token_len is not None
        else (200 if provided else 512)
    )
    return validate_conditioning(
        {
            "subtask_prediction": bool(args.subtask_prediction),
            "provided_subtask_as_prompt": provided,
            "external_subtask_selector": args.external_subtask_selector,
            "max_token_len": max_token_len,
        }
    )


def _manifest_payload(args: argparse.Namespace) -> dict:
    run_root = args.run_root.expanduser().resolve()
    runtime_root = (
        (args.runtime_root or (run_root.parent / f".{run_root.name}-runtime"))
        .expanduser()
        .resolve()
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    eval_env = Path(args.eval_env).expanduser().resolve()
    pi05_env = Path(args.pi05_env).expanduser().resolve()
    data_path = Path(args.omnigibson_data_path).expanduser().resolve()
    conditioning = _conditioning_from_args(args)
    passthrough = {
        name: os.environ[name]
        for name in sorted(CHILD_ENV_ALLOWLIST)
        if name in os.environ
    }
    passthrough.update(
        {
            "PI05_ENV": str(pi05_env),
            "EVAL_ENV": str(eval_env),
            "OMNIGIBSON_DATA_PATH": str(data_path),
            "PI05_MEM_FRACTION": "0.35",
            "SUBTASK_PREDICTION": "1" if conditioning["subtask_prediction"] else "0",
            "PROVIDED_SUBTASK_AS_PROMPT": "1"
            if conditioning["provided_subtask_as_prompt"]
            else "0",
            "EXTERNAL_SUBTASK_SELECTOR": conditioning["external_subtask_selector"]
            or "",
            "MAX_TOKEN_LEN": str(conditioning["max_token_len"]),
        }
    )
    return {
        "schema_version": 1,
        "run_id": args.run_id or run_root.name,
        "profile": "two_tasks",
        "mode": "public_test",
        "prompt_mode": "whole_task",
        "conditioning": conditioning,
        "checkpoint": str(checkpoint),
        "repo_id": args.repo_id,
        "gpus": args.gpus,
        "base_port": args.base_port,
        "max_attempts": args.max_attempts,
        "max_restarts_per_slot": args.max_restarts_per_slot,
        "runtime_root": str(runtime_root),
        "results_root": str(run_root / "results"),
        "attempt_root": str(run_root / "attempts"),
        "queue_db": str(run_root / "queue.sqlite3"),
        "eval_env": str(eval_env),
        "pi05_env": str(pi05_env),
        "omnigibson_data_path": str(data_path),
        "tasks": list(TASKS),
        "instance_ids": list(INSTANCE_IDS),
        "rollout_ids": list(ROLLOUT_IDS),
        "num_rollouts": len(ROLLOUT_IDS),
        "expected_rollouts": EXPECTED_ROLLOUTS,
        "require_videos": True,
        "child_environment": passthrough,
    }


def create_manifest(args: argparse.Namespace) -> dict:
    immutable = _manifest_payload(args)
    return {**immutable, "fingerprint": _fingerprint(immutable)}


def validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a JSON object")
    fingerprint = manifest.get("fingerprint")
    immutable = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(immutable):
        raise ValueError("manifest fingerprint mismatch")
    expected = {
        "profile": "two_tasks",
        "mode": "public_test",
        "prompt_mode": "whole_task",
        "tasks": list(TASKS),
        "instance_ids": list(INSTANCE_IDS),
        "rollout_ids": list(ROLLOUT_IDS),
        "num_rollouts": 3,
        "expected_rollouts": EXPECTED_ROLLOUTS,
        "require_videos": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest is not the formal two_tasks profile: {mismatches}")
    worker_specs(list(manifest.get("gpus", [])), manifest.get("base_port", 0))
    if (
        not isinstance(manifest.get("max_attempts"), int)
        or manifest["max_attempts"] <= 0
    ):
        raise ValueError("manifest max_attempts must be positive")
    if (
        not isinstance(manifest.get("max_restarts_per_slot"), int)
        or manifest["max_restarts_per_slot"] < 0
    ):
        raise ValueError("manifest max_restarts_per_slot must be non-negative")
    conditioning = validate_conditioning(manifest.get("conditioning"))
    environment = manifest.get("child_environment")
    if not isinstance(environment, dict) or set(environment) - CHILD_ENV_ALLOWLIST:
        raise ValueError(
            "manifest child environment contains non-whitelisted variables"
        )
    expected_environment = {
        "PI05_MEM_FRACTION": "0.35",
        "SUBTASK_PREDICTION": "1" if conditioning["subtask_prediction"] else "0",
        "PROVIDED_SUBTASK_AS_PROMPT": "1"
        if conditioning["provided_subtask_as_prompt"]
        else "0",
        "EXTERNAL_SUBTASK_SELECTOR": conditioning["external_subtask_selector"] or "",
        "MAX_TOKEN_LEN": str(conditioning["max_token_len"]),
    }
    mismatches = {
        key: (environment.get(key), value)
        for key, value in expected_environment.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"manifest child conditioning environment does not match settings: {mismatches}"
        )
    return manifest


def _require_nonempty(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{description} missing or empty: {path}")


def _query_nvidia_smi(query: str) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        row
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        if row
    ]


def _gpu_compute_occupancy() -> tuple[set[int], dict[int, list[dict]]]:
    uuid_to_index = {}
    for row in _query_nvidia_smi("gpu=index,uuid"):
        if len(row) != 2:
            raise ValueError(f"could not parse nvidia-smi GPU row: {row!r}")
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise ValueError(
                f"could not parse nvidia-smi GPU index: {row[0]!r}"
            ) from exc
        uuid = row[1].strip()
        if not uuid or uuid in uuid_to_index:
            raise ValueError(f"invalid or duplicate nvidia-smi GPU UUID: {uuid!r}")
        uuid_to_index[uuid] = index

    processes = {index: [] for index in uuid_to_index.values()}
    for row in _query_nvidia_smi("compute-apps=gpu_uuid,pid,process_name"):
        if len(row) != 3:
            raise ValueError(f"could not parse nvidia-smi compute process row: {row!r}")
        uuid, pid_text, process_name = (value.strip() for value in row)
        if uuid not in uuid_to_index:
            raise ValueError(
                f"nvidia-smi reported a process for unknown GPU UUID {uuid!r}"
            )
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise ValueError(
                f"could not parse nvidia-smi compute process PID: {pid_text!r}"
            ) from exc
        processes[uuid_to_index[uuid]].append(
            {"pid": pid, "process_name": process_name}
        )
    return set(uuid_to_index.values()), processes


def _assert_selected_gpus_idle(
    gpus: list[int], processes: dict[int, list[dict]]
) -> None:
    occupied = {gpu: processes.get(gpu, []) for gpu in gpus if processes.get(gpu)}
    if occupied:
        raise ValueError(f"selected GPUs already have compute processes: {occupied}")


def verify_preflight(manifest: dict, *, resume: bool) -> None:
    checkpoint = Path(manifest["checkpoint"])
    repo_id = manifest["repo_id"]
    eval_env = Path(manifest["eval_env"])
    pi05_env = Path(manifest["pi05_env"])
    data_path = Path(manifest["omnigibson_data_path"])
    runtime_root = Path(manifest["runtime_root"])
    _require_nonempty(
        checkpoint / "assets" / repo_id / "norm_stats.json", "checkpoint norm stats"
    )
    if (checkpoint / "model.safetensors").is_file():
        _require_nonempty(checkpoint / "model.safetensors", "Torch checkpoint")
    else:
        _require_nonempty(
            checkpoint / "params" / "_METADATA", "JAX checkpoint metadata"
        )
    _require_nonempty(eval_env / "bin" / "python", "evaluation Python")
    _require_nonempty(pi05_env / "bin" / "activate", "policy environment activation")
    for path, description in (
        (data_path / "behavior-1k-assets" / "VERSION", "BEHAVIOR assets"),
        (data_path / "omnigibson-robot-assets" / "VERSION", "robot assets"),
        (
            data_path / "2026-challenge-task-instances" / "metadata" / "task.jsonl",
            "challenge task instances",
        ),
        (data_path / "omnigibson.key", "OmniGibson key"),
    ):
        _require_nonempty(path, description)
    available, processes = _gpu_compute_occupancy()
    requested = set(manifest["gpus"])
    if not requested.issubset(available):
        raise ValueError(
            f"requested GPUs {sorted(requested)} are not all available; found {sorted(available)}"
        )
    _assert_selected_gpus_idle(manifest["gpus"], processes)
    for spec in worker_specs(manifest["gpus"], manifest["base_port"]):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", spec.port))
            except OSError as exc:
                raise ValueError(f"port {spec.port} is unavailable: {exc}") from exc
    if not resume and runtime_root.exists():
        raise ValueError(f"runtime root already exists: {runtime_root}")
    parent = runtime_root.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"runtime root parent is not writable: {parent}")


@contextlib.contextmanager
def exclusive_run_lock(run_root: Path):
    lock_path = run_root / ".supervisor.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another supervisor is active for {run_root}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def supervisor_is_active(run_root: Path) -> bool:
    lock_path = run_root / ".supervisor.lock"
    if not lock_path.is_file():
        return False
    descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _worker_status_path(run_root: Path, slot: int) -> Path:
    return run_root / "status" / f"worker-{slot}.json"


def _write_worker_status(
    run_root: Path, worker: WorkerProcess, state: str, exit_code: int | None = None
) -> None:
    atomic_write_json(
        _worker_status_path(run_root, worker.spec.slot),
        {
            "worker": worker.spec.slot,
            "gpu": worker.spec.gpu,
            "port": worker.spec.port,
            "tasks": list(worker.spec.tasks),
            "generation": worker.generation,
            "state": state,
            "pid": worker.process.pid,
            "started_at": worker.started_at,
            "exit_code": exit_code,
        },
    )


def child_environment(
    manifest: dict, spec: WorkerSpec, generation: int
) -> dict[str, str]:
    environment = {
        str(name): str(value) for name, value in manifest["child_environment"].items()
    }
    runtime = (
        Path(manifest["runtime_root"])
        / f"slot-{spec.slot}"
        / f"generation-{generation}"
    )
    environment.update(
        {
            "MODEL": "pi05",
            "PI05_CKPT": manifest["checkpoint"],
            "PI05_REPO_ID": manifest["repo_id"],
            "PI05_MEM_FRACTION": "0.35",
            "PROMPT_MODE": "whole_task",
            "MODE": "public_test",
            "SERVER_GPU": str(spec.gpu),
            "EVAL_GPU": str(spec.gpu),
            "PORT": str(spec.port),
            "EVAL_RUNTIME_ROOT": str(runtime),
            "OUTPUT_ROOT": manifest["results_root"],
            "INSTANCE_INDICES": "",
            "NUM_ROLLOUTS": "3",
            "MAX_STEPS": "",
            "WRITE_VIDEO": "1",
            "PI05_EVAL_QUEUE_DB": manifest["queue_db"],
            "PI05_EVAL_QUEUE_WORKER": f"slot-{spec.slot}",
            "PI05_EVAL_QUEUE_RUN_ID": manifest["run_id"],
            "PI05_EVAL_QUEUE_GENERATION": str(generation),
            "PI05_EVAL_QUEUE_ATTEMPT_ROOT": manifest["attempt_root"],
            "PI05_EVAL_QUEUE_LOG_ROOT": str(Path(manifest["queue_db"]).parent / "logs"),
        }
    )
    return environment


def spawn_worker(
    manifest: dict,
    spec: WorkerSpec,
    generation: int,
    *,
    popen_factory: Callable = subprocess.Popen,
) -> WorkerProcess:
    run_root = Path(manifest["queue_db"]).parent
    runtime = (
        Path(manifest["runtime_root"])
        / f"slot-{spec.slot}"
        / f"generation-{generation}"
    )
    runtime.mkdir(parents=True, exist_ok=False)
    log_path = (
        run_root
        / "logs"
        / f"slot-{spec.slot}"
        / f"supervisor-generation-{generation}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab", buffering=0)
    command = ["bash", str(REPO_ROOT / "scripts" / "eval_b1k.sh"), *spec.tasks]
    try:
        process = popen_factory(
            command,
            cwd=REPO_ROOT,
            env=child_environment(manifest, spec, generation),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_stream.close()
        raise
    worker = WorkerProcess(spec, generation, process, log_stream, time.time())
    _write_worker_status(run_root, worker, "running")
    return worker


def terminate_process_group(worker: WorkerProcess, grace_seconds: float = 20.0) -> None:
    process = worker.process
    if process.poll() is not None:
        process.wait()
        worker.log_stream.close()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    worker.log_stream.close()


def _canonical_task_dir(manifest: dict, task: str) -> Path:
    return Path(manifest["results_root"]) / "pi05" / task


def reconcile_lease(queue: EvaluationQueue, manifest: dict, lease: JobLease) -> bool:
    return queue.reconcile(
        lease,
        _canonical_task_dir(manifest, lease.task),
        num_rollouts=manifest["num_rollouts"],
        require_videos=manifest["require_videos"],
        provenance={"run_id": manifest["run_id"], "supervisor_reconciled": True},
    )


def _settle_leases(
    queue: EvaluationQueue, manifest: dict, leases: list[JobLease], error: str
) -> None:
    for lease in leases:
        try:
            if reconcile_lease(queue, manifest, lease):
                continue
        except (ArtifactValidationError, LeaseLostError, OSError, ValueError):
            pass
        try:
            queue.fail(lease, error)
        except LeaseLostError:
            pass


def reconcile_worker(
    queue: EvaluationQueue, manifest: dict, worker: WorkerProcess, exit_code: int
) -> None:
    expected_worker = (
        f"{manifest['run_id']}:{worker.queue_worker}:generation-{worker.generation}"
    )
    _settle_leases(
        queue,
        manifest,
        queue.running_leases(expected_worker),
        f"worker process exited with code {exit_code}",
    )


def requeue_orphaned_leases(queue: EvaluationQueue, manifest: dict) -> None:
    _settle_leases(
        queue,
        manifest,
        queue.running_leases(),
        "orphaned lease recovered during supervisor resume",
    )


def _next_generation(run_root: Path, runtime_root: Path, slot: int) -> int:
    generations = []
    log_dir = run_root / "logs" / f"slot-{slot}"
    if log_dir.is_dir():
        for path in log_dir.glob("supervisor-generation-*.log"):
            try:
                generations.append(int(path.stem.rsplit("-", 1)[1]))
            except ValueError:
                continue
    runtime_dir = runtime_root / f"slot-{slot}"
    if runtime_dir.is_dir():
        for path in runtime_dir.glob("generation-*"):
            try:
                generations.append(int(path.name.rsplit("-", 1)[1]))
            except ValueError:
                continue
    return max(generations, default=-1) + 1


def _conditioning_flags(manifest: dict) -> list[str]:
    conditioning = validate_conditioning(manifest["conditioning"])
    subtask_prediction = conditioning["subtask_prediction"]
    selector = conditioning["external_subtask_selector"]
    if selector:
        return ["--require-subtask-events", "--require-selector-events"]
    if subtask_prediction:
        return ["--require-subtask-events", "--require-generated-subtasks"]
    return []


def run_summarizer(manifest: dict, wall_time_seconds: float) -> None:
    run_root = Path(manifest["queue_db"]).parent
    command = [
        sys.executable,
        str(REPO_ROOT / "nebula" / "summarize_pi05_eval.py"),
        "--results-root",
        manifest["results_root"],
        "--output",
        str(run_root / "summary.json"),
        "--run-id",
        manifest["run_id"],
        "--profile",
        "two_tasks",
        "--checkpoint",
        manifest["checkpoint"],
        "--expected-rollouts",
        str(manifest["expected_rollouts"]),
        "--expected-tasks",
        " ".join(manifest["tasks"]),
        "--expected-instance-ids",
        " ".join(str(value) for value in manifest["instance_ids"]),
        "--expected-rollout-ids",
        " ".join(str(value) for value in manifest["rollout_ids"]),
        "--wall-time-seconds",
        str(max(wall_time_seconds, 0.001)),
        "--require-videos",
        "--require-complete",
        *_conditioning_flags(manifest),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"strict result summarizer failed with code {completed.returncode}"
        )


def finalize_results(
    manifest: dict, queue: EvaluationQueue, wall_time_seconds: float
) -> None:
    run_root = Path(manifest["queue_db"]).parent
    queue.create_result_manifest(
        manifest["results_root"],
        run_root / "result_manifest.json",
        model="pi05",
        num_rollouts=manifest["num_rollouts"],
        require_videos=manifest["require_videos"],
    )
    run_summarizer(manifest, wall_time_seconds)
    atomic_write_json(
        run_root / "_SUCCESS",
        {
            "completed_at": time.time(),
            "expected_rollouts": manifest["expected_rollouts"],
        },
    )


def supervise(manifest: dict, queue: EvaluationQueue) -> None:
    run_root = Path(manifest["queue_db"]).parent
    runtime_root = Path(manifest["runtime_root"])
    specs = worker_specs(manifest["gpus"], manifest["base_port"])
    workers: dict[int, WorkerProcess] = {}
    next_generations = {
        spec.slot: _next_generation(run_root, runtime_root, spec.slot) for spec in specs
    }
    initial_counts = queue.counts(readonly=True)["counts"]
    if initial_counts["exhausted"]:
        raise RuntimeError(
            f"{initial_counts['exhausted']} queue jobs exhausted their attempts"
        )
    started_at = time.time()
    atomic_write_json(
        run_root / "_RUNNING", {"pid": os.getpid(), "started_at": started_at}
    )
    (run_root / "_FAILED").unlink(missing_ok=True)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def interrupt(signum, _frame):
        raise SupervisorInterrupted(f"supervisor received signal {signum}")

    for signum in previous_handlers:
        signal.signal(signum, interrupt)
    try:
        while True:
            for slot, worker in list(workers.items()):
                exit_code = worker.process.poll()
                if exit_code is None:
                    continue
                worker.process.wait()
                worker.log_stream.close()
                _write_worker_status(run_root, worker, "exited", exit_code)
                reconcile_worker(queue, manifest, worker, exit_code)
                del workers[slot]

            queue_state = queue.counts(readonly=True)
            counts = queue_state["counts"]
            if counts["exhausted"]:
                raise RuntimeError(
                    f"{counts['exhausted']} queue jobs exhausted their attempts"
                )
            if counts["succeeded"] == queue_state["total"]:
                break

            if counts["pending"]:
                for spec in specs:
                    if spec.slot in workers:
                        continue
                    generation = next_generations[spec.slot]
                    if generation >= manifest["max_restarts_per_slot"] + 1:
                        continue
                    workers[spec.slot] = spawn_worker(manifest, spec, generation)
                    next_generations[spec.slot] = generation + 1
                    if len(workers) >= min(WORKER_COUNT, counts["pending"]):
                        break
                if not workers:
                    raise RuntimeError(
                        "pending jobs remain but every worker slot exceeded its restart limit"
                    )
            elif not workers:
                raise RuntimeError("queue has unfinished jobs but no active workers")
            time.sleep(1.0)

        for worker in workers.values():
            terminate_process_group(worker)
            _write_worker_status(
                run_root, worker, "stopped_after_completion", worker.process.returncode
            )
        workers.clear()
        finalize_results(manifest, queue, time.time() - started_at)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for worker in workers.values():
            terminate_process_group(worker)
            _write_worker_status(
                run_root, worker, "terminated", worker.process.returncode
            )
        (run_root / "_RUNNING").unlink(missing_ok=True)


def _load_manifest(run_root: Path) -> dict:
    path = run_root / "manifest.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load run manifest {path}: {exc}") from exc
    return validate_manifest(value)


def _run_start(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    if run_root.exists():
        raise ValueError(f"start requires a fresh run root: {run_root}")
    manifest = create_manifest(args)
    verify_preflight(manifest, resume=False)
    run_root.mkdir(parents=True)
    with exclusive_run_lock(run_root):
        try:
            Path(manifest["runtime_root"]).mkdir(parents=True)
            for name in ("results", "attempts", "logs", "status"):
                (run_root / name).mkdir()
            atomic_write_json(run_root / "manifest.json", manifest)
            queue = EvaluationQueue(manifest["queue_db"])
            queue.initialize(
                (
                    (task, instance_id)
                    for task in manifest["tasks"]
                    for instance_id in manifest["instance_ids"]
                ),
                max_attempts=manifest["max_attempts"],
                metadata={
                    "run_id": manifest["run_id"],
                    "fingerprint": manifest["fingerprint"],
                    "mode": manifest["mode"],
                    "num_rollouts": manifest["num_rollouts"],
                    "require_videos": manifest["require_videos"],
                },
            )
            supervise(manifest, queue)
        except BaseException as exc:
            atomic_write_json(
                run_root / "_FAILED",
                {"failed_at": time.time(), "error": f"{type(exc).__name__}: {exc}"},
            )
            raise


def _run_resume(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    manifest = _load_manifest(run_root)
    with exclusive_run_lock(run_root):
        verify_preflight(manifest, resume=True)
        queue = EvaluationQueue(manifest["queue_db"])
        metadata = queue.metadata(readonly=True)
        if (
            metadata.get("fingerprint") != manifest["fingerprint"]
            or metadata.get("run_id") != manifest["run_id"]
        ):
            raise ValueError("queue metadata does not match the immutable run manifest")
        try:
            requeue_orphaned_leases(queue, manifest)
            if queue.counts(readonly=True)["counts"]["succeeded"] == len(
                manifest["tasks"]
            ) * len(manifest["instance_ids"]):
                finalize_results(manifest, queue, 0.001)
                (run_root / "_FAILED").unlink(missing_ok=True)
                return
            supervise(manifest, queue)
            (run_root / "_FAILED").unlink(missing_ok=True)
        except BaseException as exc:
            atomic_write_json(
                run_root / "_FAILED",
                {"failed_at": time.time(), "error": f"{type(exc).__name__}: {exc}"},
            )
            raise


def _canonical_progress(manifest: dict) -> dict:
    results_root = Path(manifest["results_root"])
    expected = {
        (task, instance_id, rollout_id)
        for task in manifest["tasks"]
        for instance_id in manifest["instance_ids"]
        for rollout_id in manifest["rollout_ids"]
    }
    observed = {}
    invalid = []
    for path in (
        sorted(results_root.rglob("json/*.json")) if results_root.is_dir() else []
    ):
        try:
            result = json.loads(path.read_text())
            key = (result["task"], result["instance_id"], result["rollout_id"])
            if key not in expected or key in observed:
                raise ValueError(f"unexpected or duplicate rollout key {key!r}")
            if not isinstance(result.get("success"), bool):
                raise TypeError("success must be boolean")
            q_score = result.get("q_score", 0.0)
            if isinstance(q_score, dict):
                q_score = q_score.get("final", 0.0)
            observed[key] = (bool(result["success"]), float(q_score or 0.0))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    successes = sum(value[0] for value in observed.values())
    return {
        "completed_rollouts": len(observed),
        "expected_rollouts": manifest["expected_rollouts"],
        "remaining_rollouts": manifest["expected_rollouts"] - len(observed),
        "successes": successes,
        "success_rate": successes / len(observed) if observed else 0.0,
        "mean_q_score": sum(value[1] for value in observed.values()) / len(observed)
        if observed
        else 0.0,
        "invalid_results": invalid,
    }


def _run_status(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    manifest = _load_manifest(run_root)
    queue = EvaluationQueue(manifest["queue_db"])
    snapshot = queue.snapshot(readonly=True)
    workers = []
    status_root = run_root / "status"
    if status_root.is_dir():
        for path in sorted(status_root.glob("worker-*.json")):
            try:
                workers.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError) as exc:
                workers.append({"path": str(path), "error": str(exc)})
    result_manifest_path = run_root / "result_manifest.json"
    result_manifest = None
    if result_manifest_path.is_file():
        try:
            result_manifest = json.loads(result_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            result_manifest = {"error": str(exc)}
    value = {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "active_supervisor": supervisor_is_active(run_root),
        "markers": {
            "running": (run_root / "_RUNNING").is_file(),
            "failed": (run_root / "_FAILED").is_file(),
            "success": (run_root / "_SUCCESS").is_file(),
        },
        "result_manifest": {
            "present": result_manifest is not None,
            "rollout_count": result_manifest.get("rollout_count")
            if isinstance(result_manifest, dict)
            else None,
            "error": result_manifest.get("error")
            if isinstance(result_manifest, dict)
            else None,
        },
        "queue_counts": snapshot["counts"],
        "active_workers": sorted(
            {job["worker"] for job in snapshot["jobs"] if job["state"] == "running"}
        ),
        "retries": sum(max(job["attempts"] - 1, 0) for job in snapshot["jobs"]),
        "workers": workers,
        **_canonical_progress(manifest),
    }
    print(json.dumps(value, indent=2, sort_keys=True))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local formal two-task pi05 evaluation queue."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-root", type=Path, required=True)
    start.add_argument("--run-id")
    start.add_argument("--profile", choices=("two_tasks",), default="two_tasks")
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--repo-id", required=True)
    start.add_argument("--gpus", type=int, nargs="+", required=True)
    start.add_argument("--base-port", type=_positive_int, required=True)
    start.add_argument("--max-attempts", type=_positive_int, required=True)
    start.add_argument("--max-restarts-per-slot", type=_nonnegative_int, default=8)
    start.add_argument(
        "--subtask-prediction", action=argparse.BooleanOptionalAction, default=False
    )
    start.add_argument(
        "--provided-subtask-as-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    start.add_argument("--external-subtask-selector", choices=("state_machine_v1",))
    start.add_argument("--max-token-len", type=_positive_int)
    start.add_argument("--runtime-root", type=Path)
    start.add_argument(
        "--eval-env",
        default=os.environ.get("EVAL_ENV", "/dev/shm/wangfaye/conda_envs/b1k-eval"),
    )
    checkpoint_default_pi05_env = os.environ.get(
        "PI05_ENV", str(REPO_ROOT / "openpi" / ".venv")
    )
    start.add_argument("--pi05-env", default=checkpoint_default_pi05_env)
    start.add_argument(
        "--omnigibson-data-path",
        default=os.environ.get(
            "OMNIGIBSON_DATA_PATH", str(REPO_ROOT / "BEHAVIOR-1K" / "omnigibson-data")
        ),
    )

    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-root", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "start":
        _run_start(args)
    elif args.command == "resume":
        _run_resume(args)
    else:
        _run_status(args)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

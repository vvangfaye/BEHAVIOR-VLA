#!/usr/bin/env python3
"""Run the deterministic 20-episode training-subtask diagnostic on local GPUs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import eval_b1k_dynamic as dynamic
from pi05_eval_queue import (
    ArtifactValidationError,
    EvaluationQueue,
    JobLease,
    LeaseLostError,
    atomic_write_json,
)
from subtask_sample_panel import (
    build_panel,
    load_panel,
    preflight_panel,
    validate_frozen_production_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "training_subtask_closed_loop_episode_segments"
EVALUATION_KIND = "training_subtask_closed_loop"
PRODUCTION_EPISODES_PER_TASK = 10
PRODUCTION_EPISODE_JOBS = 20
MAX_WORKERS = 8


def worker_specs(gpus: list[int], base_port: int) -> list[dynamic.WorkerSpec]:
    if not gpus or len(gpus) > MAX_WORKERS or len(gpus) != len(set(gpus)):
        raise ValueError(f"--gpus must contain 1-{MAX_WORKERS} unique GPU indices")
    if any(isinstance(gpu, bool) or gpu < 0 for gpu in gpus):
        raise ValueError("GPU indices must be non-negative")
    if base_port <= 0 or base_port + len(gpus) - 1 > 65535:
        raise ValueError("base port does not leave room for all worker ports")
    split = (len(gpus) + 1) // 2
    return [
        dynamic.WorkerSpec(
            slot=slot,
            gpu=gpu,
            port=base_port + slot,
            tasks=dynamic.TASKS if slot < split else tuple(reversed(dynamic.TASKS)),
        )
        for slot, gpu in enumerate(gpus)
    ]


def _panel_jobs(panel: dict) -> list[dict]:
    episodes = panel.get("episodes")
    samples = panel.get("samples")
    if not isinstance(episodes, list) or not isinstance(samples, list):
        raise ValueError("panel must contain episode and sample lists")
    sample_by_id = {
        sample.get("sample_id"): sample for sample in samples if isinstance(sample, dict)
    }
    if len(sample_by_id) != len(samples) or None in sample_by_id:
        raise ValueError("panel sample IDs must be present and unique")

    jobs = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("panel episode must be an object")
        sample_ids = episode.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError("panel episode must contain ordered sample IDs")
        rollout_specs = []
        for rollout_id, sample_id in enumerate(sample_ids):
            sample = sample_by_id.get(sample_id)
            if sample is None:
                raise ValueError(f"panel episode references unknown sample {sample_id!r}")
            if (
                sample.get("task") != episode.get("task")
                or sample.get("task_instance_id")
                != episode.get("task_instance_id")
                or sample.get("episode_index") != episode.get("episode_index")
                or sample.get("raw_episode_id") != episode.get("raw_episode_id")
                or sample.get("rollout_id") != rollout_id
            ):
                raise ValueError(
                    f"panel sample {sample_id!r} does not match its episode or rollout"
                )
            rollout_specs.append(
                {
                    "rollout_id": rollout_id,
                    "sample_id": sample_id,
                    "video": "required_if_eligible",
                }
            )
        jobs.append(
            {
                "task": episode["task"],
                "instance_id": int(episode["task_instance_id"]),
                "episode_index": int(episode["episode_index"]),
                "raw_episode_id": int(episode["raw_episode_id"]),
                "rollout_count": len(rollout_specs),
                "sample_ids": list(sample_ids),
                "rollout_specs": rollout_specs,
            }
        )
    return jobs


def _rollout_specs_by_job(manifest: dict) -> dict[tuple[str, int], list[dict]]:
    return {
        (job["task"], job["instance_id"]): job["rollout_specs"]
        for job in manifest["jobs"]
    }


def _queue_metadata(manifest: dict) -> dict:
    return {
        "run_id": manifest["run_id"],
        "fingerprint": manifest["fingerprint"],
        "panel_fingerprint": manifest["panel_fingerprint"],
        "profile": manifest["profile"],
        "evaluation_kind": manifest["evaluation_kind"],
        "mode": "train",
        "num_rollouts": manifest["num_rollouts"],
        "expected_episode_jobs": manifest["expected_episode_jobs"],
        "expected_segment_rollouts": manifest["expected_segment_rollouts"],
        "expected_annotated_segments": manifest["expected_annotated_segments"],
        "expected_exclusions": manifest["expected_exclusions"],
        "require_videos": manifest["require_videos"],
    }


def _manifest_payload(
    args: argparse.Namespace, panel: dict, raw_preflight: dict
) -> dict:
    run_root = args.run_root.expanduser().resolve()
    runtime_root = (
        (args.runtime_root or run_root.parent / f".{run_root.name}-runtime")
        .expanduser()
        .resolve()
    )
    child_environment = {
        name: os.environ[name]
        for name in sorted(dynamic.CHILD_ENV_ALLOWLIST)
        if name in os.environ
    }
    child_environment.update(
        {
            "PI05_ENV": str(Path(args.pi05_env).expanduser().resolve()),
            "EVAL_ENV": str(Path(args.eval_env).expanduser().resolve()),
            "OMNIGIBSON_DATA_PATH": str(
                Path(args.omnigibson_data_path).expanduser().resolve()
            ),
            "PI05_MEM_FRACTION": "0.35",
            "SUBTASK_PREDICTION": "0",
            "PROVIDED_SUBTASK_AS_PROMPT": "1",
            "EXTERNAL_SUBTASK_SELECTOR": "",
            "MAX_TOKEN_LEN": "200",
        }
    )
    jobs = _panel_jobs(panel)
    counts = panel["counts"]
    episodes_per_task = panel["selection"]["episode_anchors"][
        "episodes_per_task"
    ]
    return {
        "schema_version": 2,
        "profile": PROFILE,
        "evaluation_kind": EVALUATION_KIND,
        "run_id": args.run_id or run_root.name,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "repo_id": args.repo_id,
        "gpus": list(args.gpus),
        "base_port": args.base_port,
        "max_attempts": args.max_attempts,
        "max_restarts_per_slot": args.max_restarts_per_slot,
        "sample_max_steps": args.sample_max_steps,
        "require_videos": bool(args.write_video),
        "runtime_root": str(runtime_root),
        "results_root": str(run_root / "results"),
        "attempt_root": str(run_root / "attempts"),
        "queue_db": str(run_root / "queue.sqlite3"),
        "panel_path": str(run_root / "panel.json"),
        "panel_fingerprint": panel["fingerprint"],
        "raw_data_root": str(args.raw_data_root.expanduser().resolve()),
        "raw_preflight": raw_preflight,
        "eval_env": str(Path(args.eval_env).expanduser().resolve()),
        "pi05_env": str(Path(args.pi05_env).expanduser().resolve()),
        "omnigibson_data_path": str(
            Path(args.omnigibson_data_path).expanduser().resolve()
        ),
        "tasks": list(panel["selection"]["tasks"]),
        "episodes_per_task": episodes_per_task,
        "jobs": jobs,
        # The shell parser still requires NUM_ROLLOUTS=1. Artifact cardinality is
        # defined exclusively by each episode job's rollout_specs.
        "num_rollouts": 1,
        "expected_episode_jobs": len(jobs),
        "expected_segment_rollouts": len(panel["samples"]),
        "expected_annotated_segments": counts["annotated_segments"],
        "expected_exclusions": counts["exclusions"],
        "child_environment": child_environment,
    }


def create_manifest(args: argparse.Namespace, panel: dict, raw_preflight: dict) -> dict:
    payload = _manifest_payload(args, panel, raw_preflight)
    return {**payload, "fingerprint": dynamic._fingerprint(payload)}


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_manifest(manifest: object, panel: dict | None = None) -> dict:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a JSON object")
    fingerprint = manifest.get("fingerprint")
    immutable = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if not isinstance(fingerprint, str) or fingerprint != dynamic._fingerprint(
        immutable
    ):
        raise ValueError("manifest fingerprint mismatch")
    expected = {
        "schema_version": 2,
        "profile": PROFILE,
        "evaluation_kind": EVALUATION_KIND,
        "num_rollouts": 1,
        "tasks": sorted(dynamic.TASKS),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest is not the subtask episode profile: {mismatches}")
    worker_specs(list(manifest.get("gpus", [])), manifest.get("base_port", 0))
    if (
        not _is_nonnegative_int(manifest.get("max_attempts"))
        or manifest["max_attempts"] == 0
    ):
        raise ValueError("manifest max_attempts must be positive")
    if not _is_nonnegative_int(manifest.get("max_restarts_per_slot")):
        raise ValueError("manifest max_restarts_per_slot must be non-negative")
    if (
        not _is_nonnegative_int(manifest.get("sample_max_steps"))
        or manifest["sample_max_steps"] == 0
    ):
        raise ValueError("manifest sample_max_steps must be positive")
    if manifest.get("require_videos") is not True:
        raise ValueError("subtask episode evaluation requires videos")

    episodes_per_task = manifest.get("episodes_per_task")
    expected_episode_jobs = manifest.get("expected_episode_jobs")
    expected_segment_rollouts = manifest.get("expected_segment_rollouts")
    expected_annotated_segments = manifest.get("expected_annotated_segments")
    expected_exclusions = manifest.get("expected_exclusions")
    if not _is_nonnegative_int(episodes_per_task) or episodes_per_task == 0:
        raise ValueError("manifest episodes_per_task must be positive")
    for name, value in (
        ("expected_episode_jobs", expected_episode_jobs),
        ("expected_segment_rollouts", expected_segment_rollouts),
        ("expected_annotated_segments", expected_annotated_segments),
        ("expected_exclusions", expected_exclusions),
    ):
        if not _is_nonnegative_int(value):
            raise ValueError(f"manifest {name} must be a non-negative integer")
    if expected_episode_jobs != len(manifest["tasks"]) * episodes_per_task:
        raise ValueError("manifest episode count does not match episodes_per_task")
    if expected_annotated_segments != expected_segment_rollouts + expected_exclusions:
        raise ValueError("manifest annotated segment counts are inconsistent")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != expected_episode_jobs:
        raise ValueError(
            f"manifest must contain exactly {expected_episode_jobs} episode jobs"
        )
    job_keys = []
    sample_ids = []
    task_counts = {task: 0 for task in manifest["tasks"]}
    rollout_total = 0
    required_job_keys = {
        "task",
        "instance_id",
        "episode_index",
        "raw_episode_id",
        "rollout_count",
        "sample_ids",
        "rollout_specs",
    }
    for job in jobs:
        if not isinstance(job, dict) or not required_job_keys.issubset(job):
            raise ValueError("manifest contains an invalid episode job")
        task = job["task"]
        if task not in task_counts:
            raise ValueError(f"manifest job has unexpected task {task!r}")
        for name in ("instance_id", "episode_index", "raw_episode_id"):
            if not _is_nonnegative_int(job[name]):
                raise ValueError(f"manifest job {name} must be non-negative")
        rollout_count = job["rollout_count"]
        job_sample_ids = job["sample_ids"]
        specs = job["rollout_specs"]
        if (
            not _is_nonnegative_int(rollout_count)
            or rollout_count == 0
            or not isinstance(job_sample_ids, list)
            or not isinstance(specs, list)
            or len(job_sample_ids) != rollout_count
            or len(specs) != rollout_count
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in job_sample_ids)
            or len(job_sample_ids) != len(set(job_sample_ids))
        ):
            raise ValueError("manifest job rollout counts or sample IDs are invalid")
        expected_specs = [
            {
                "rollout_id": rollout_id,
                "sample_id": sample_id,
                "video": "required_if_eligible",
            }
            for rollout_id, sample_id in enumerate(job_sample_ids)
        ]
        if specs != expected_specs:
            raise ValueError("manifest job rollout_specs are not dense or do not match sample_ids")
        job_keys.append((task, job["instance_id"]))
        sample_ids.extend(job_sample_ids)
        task_counts[task] += 1
        rollout_total += rollout_count
    if len(job_keys) != len(set(job_keys)):
        raise ValueError("manifest episode jobs are not unique")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample IDs are not unique")
    if any(count != episodes_per_task for count in task_counts.values()):
        raise ValueError("manifest episode jobs are not balanced by task")
    if rollout_total != expected_segment_rollouts:
        raise ValueError("manifest segment rollout count does not match its jobs")

    if panel is not None:
        try:
            panel_episodes_per_task = panel["selection"]["episode_anchors"][
                "episodes_per_task"
            ]
            panel_counts = panel["counts"]
            panel_expected = {
                "panel_fingerprint": panel["fingerprint"],
                "tasks": panel["selection"]["tasks"],
                "episodes_per_task": panel_episodes_per_task,
                "expected_episode_jobs": len(panel["episodes"]),
                "expected_segment_rollouts": len(panel["samples"]),
                "expected_annotated_segments": panel_counts["annotated_segments"],
                "expected_exclusions": panel_counts["exclusions"],
                "jobs": _panel_jobs(panel),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid panel manifest contract: {error}") from error
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in panel_expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"manifest does not match panel: {mismatches}")
    return manifest


def _base_preflight_manifest(manifest: dict) -> dict:
    return {
        "checkpoint": manifest["checkpoint"],
        "repo_id": manifest["repo_id"],
        "eval_env": manifest["eval_env"],
        "pi05_env": manifest["pi05_env"],
        "omnigibson_data_path": manifest["omnigibson_data_path"],
        "runtime_root": manifest["runtime_root"],
        "gpus": manifest["gpus"],
        "base_port": manifest["base_port"],
    }


def verify_preflight(manifest: dict, panel: dict, *, resume: bool) -> None:
    specs = worker_specs(manifest["gpus"], manifest["base_port"])
    dynamic.verify_preflight(
        _base_preflight_manifest(manifest), resume=resume, specs=specs
    )
    report = preflight_panel(panel, manifest["raw_data_root"])
    if not report["valid"]:
        missing = "\n".join(
            f"{item['path']}: {item['error']}" for item in report["errors"]
        )
        raise ValueError(f"raw HDF5 preflight failed:\n{missing}")
    expected = manifest.get("raw_preflight")
    if resume and report != expected:
        raise ValueError("raw HDF5 preflight no longer matches the immutable manifest")


def child_environment(
    manifest: dict, spec: dynamic.WorkerSpec, generation: int
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
            "MODE": "train",
            "SERVER_GPU": str(spec.gpu),
            "EVAL_GPU": str(spec.gpu),
            "PORT": str(spec.port),
            "EVAL_RUNTIME_ROOT": str(runtime),
            "OUTPUT_ROOT": manifest["results_root"],
            "INSTANCE_INDICES": "",
            "NUM_ROLLOUTS": "1",
            "MAX_STEPS": str(manifest["sample_max_steps"]),
            "WRITE_VIDEO": "1" if manifest["require_videos"] else "0",
            "PI05_EVAL_QUEUE_DB": manifest["queue_db"],
            "PI05_EVAL_QUEUE_WORKER": f"slot-{spec.slot}",
            "PI05_EVAL_QUEUE_RUN_ID": manifest["run_id"],
            "PI05_EVAL_QUEUE_GENERATION": str(generation),
            "PI05_EVAL_QUEUE_ATTEMPT_ROOT": manifest["attempt_root"],
            "PI05_EVAL_QUEUE_LOG_ROOT": str(Path(manifest["queue_db"]).parent / "logs"),
            "PI05_SUBTASK_SAMPLE_PANEL": manifest["panel_path"],
            "PI05_SUBTASK_RAW_DATA_ROOT": manifest["raw_data_root"],
            "SUBTASK_SAMPLE_MAX_STEPS": str(manifest["sample_max_steps"]),
        }
    )
    return environment


def spawn_worker(
    manifest: dict,
    spec: dynamic.WorkerSpec,
    generation: int,
    *,
    popen_factory=subprocess.Popen,
) -> dynamic.WorkerProcess:
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
    worker = dynamic.WorkerProcess(spec, generation, process, log_stream, time.time())
    dynamic._write_worker_status(run_root, worker, "running")
    return worker


def _canonical_task_dir(manifest: dict, task: str) -> Path:
    return Path(manifest["results_root"]) / "pi05" / task


def _job_for_lease(manifest: dict, lease: JobLease) -> dict:
    matches = [
        job
        for job in manifest["jobs"]
        if (job["task"], job["instance_id"])
        == (lease.task, lease.instance_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"lease has no unique manifest job: {lease.task}/{lease.instance_id}"
        )
    return matches[0]


def reconcile_lease(queue: EvaluationQueue, manifest: dict, lease: JobLease) -> bool:
    job = _job_for_lease(manifest, lease)
    return queue.reconcile(
        lease,
        _canonical_task_dir(manifest, lease.task),
        rollout_specs=job["rollout_specs"],
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
    queue: EvaluationQueue,
    manifest: dict,
    worker: dynamic.WorkerProcess,
    exit_code: int,
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


def finalize_results(manifest: dict, queue: EvaluationQueue) -> None:
    run_root = Path(manifest["queue_db"]).parent
    queue.create_result_manifest(
        manifest["results_root"],
        run_root / "result_manifest.json",
        model="pi05",
        rollout_specs_by_job=_rollout_specs_by_job(manifest),
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_subtask_closed_loop.py"),
        "--panel",
        manifest["panel_path"],
        "--manifest",
        str(run_root / "manifest.json"),
        "--results-root",
        manifest["results_root"],
        "--output",
        str(run_root / "summary.json"),
        "--checkpoint",
        manifest["checkpoint"],
        "--require-complete",
    ]
    if manifest["require_videos"]:
        command.append("--require-videos")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"strict subtask summarizer failed with code {completed.returncode}"
        )
    atomic_write_json(
        run_root / "_SUCCESS",
        {
            "completed_at": time.time(),
            "expected_episode_jobs": manifest["expected_episode_jobs"],
            "expected_segment_rollouts": manifest["expected_segment_rollouts"],
            "expected_annotated_segments": manifest["expected_annotated_segments"],
            "expected_exclusions": manifest["expected_exclusions"],
        },
    )


def supervise(manifest: dict, queue: EvaluationQueue) -> None:
    run_root = Path(manifest["queue_db"]).parent
    runtime_root = Path(manifest["runtime_root"])
    specs = worker_specs(manifest["gpus"], manifest["base_port"])
    workers: dict[int, dynamic.WorkerProcess] = {}
    generations = {
        spec.slot: dynamic._next_generation(run_root, runtime_root, spec.slot)
        for spec in specs
    }
    restart_counts = {spec.slot: 0 for spec in specs}
    counts = queue.counts(readonly=True)["counts"]
    if counts["exhausted"]:
        raise RuntimeError(f"{counts['exhausted']} queue jobs exhausted their attempts")
    atomic_write_json(
        run_root / "_RUNNING", {"pid": os.getpid(), "started_at": time.time()}
    )
    (run_root / "_FAILED").unlink(missing_ok=True)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def interrupt(signum, _frame):
        raise dynamic.SupervisorInterrupted(f"supervisor received signal {signum}")

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
                dynamic._write_worker_status(run_root, worker, "exited", exit_code)
                reconcile_worker(queue, manifest, worker, exit_code)
                if exit_code != 0:
                    restart_counts[slot] += 1
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
                    generation = generations[spec.slot]
                    if restart_counts[spec.slot] > manifest["max_restarts_per_slot"]:
                        continue
                    workers[spec.slot] = spawn_worker(manifest, spec, generation)
                    generations[spec.slot] = generation + 1
                    if len(workers) >= min(len(specs), counts["pending"]):
                        break
                if not workers:
                    raise RuntimeError(
                        "pending jobs remain but every worker slot exceeded its restart limit"
                    )
            elif not workers:
                raise RuntimeError("queue has unfinished jobs but no active workers")
            time.sleep(1.0)

        for worker in workers.values():
            dynamic.terminate_process_group(worker)
            dynamic._write_worker_status(
                run_root, worker, "stopped_after_completion", worker.process.returncode
            )
        workers.clear()
        finalize_results(manifest, queue)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for worker in workers.values():
            dynamic.terminate_process_group(worker)
            dynamic._write_worker_status(
                run_root, worker, "terminated", worker.process.returncode
            )
        (run_root / "_RUNNING").unlink(missing_ok=True)


def _load_run(run_root: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_root / "manifest.json").read_text())
    panel = load_panel(run_root / "panel.json")
    return validate_manifest(manifest, panel), panel


def _run_start(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    if run_root.exists():
        raise ValueError(f"start requires a fresh run root: {run_root}")
    if not args.write_video:
        raise ValueError("subtask episode evaluation requires videos")
    panel = validate_frozen_production_panel(
        build_panel(
            args.dataset_root,
            args.segments,
            args.train_episodes,
            args.split_manifest,
            seed=1,
            episodes_per_task=PRODUCTION_EPISODES_PER_TASK,
            reserve_per_task=100,
        )
    )
    raw_report = preflight_panel(panel, args.raw_data_root)
    if not raw_report["valid"]:
        missing = "\n".join(
            f"{item['path']}: {item['error']}" for item in raw_report["errors"]
        )
        raise ValueError(f"raw HDF5 preflight failed:\n{missing}")
    manifest = create_manifest(args, panel, raw_report)
    if manifest["expected_episode_jobs"] != PRODUCTION_EPISODE_JOBS:
        raise ValueError(
            f"production subtask evaluation requires {PRODUCTION_EPISODE_JOBS} episode jobs"
        )
    validate_manifest(manifest, panel)
    verify_preflight(manifest, panel, resume=False)

    run_root.mkdir(parents=True)
    with dynamic.exclusive_run_lock(run_root):
        try:
            Path(manifest["runtime_root"]).mkdir(parents=True)
            for name in ("results", "attempts", "logs", "status"):
                (run_root / name).mkdir()
            atomic_write_json(run_root / "panel.json", panel)
            atomic_write_json(run_root / "manifest.json", manifest)
            queue = EvaluationQueue(manifest["queue_db"])
            queue.initialize(
                ((job["task"], job["instance_id"]) for job in manifest["jobs"]),
                max_attempts=manifest["max_attempts"],
                metadata=_queue_metadata(manifest),
            )
            supervise(manifest, queue)
        except BaseException as error:
            atomic_write_json(
                run_root / "_FAILED",
                {"failed_at": time.time(), "error": f"{type(error).__name__}: {error}"},
            )
            raise


def _run_resume(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    manifest, panel = _load_run(run_root)
    with dynamic.exclusive_run_lock(run_root):
        verify_preflight(manifest, panel, resume=True)
        queue = EvaluationQueue(manifest["queue_db"])
        metadata = queue.metadata(readonly=True)
        expected_metadata = _queue_metadata(manifest)
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"queue metadata does not match immutable manifest: {mismatches}"
            )
        try:
            requeue_orphaned_leases(queue, manifest)
            if (
                queue.counts(readonly=True)["counts"]["succeeded"]
                == manifest["expected_episode_jobs"]
            ):
                finalize_results(manifest, queue)
                (run_root / "_FAILED").unlink(missing_ok=True)
                return
            supervise(manifest, queue)
            (run_root / "_FAILED").unlink(missing_ok=True)
        except BaseException as error:
            atomic_write_json(
                run_root / "_FAILED",
                {"failed_at": time.time(), "error": f"{type(error).__name__}: {error}"},
            )
            raise


def _segment_progress(manifest: dict) -> dict:
    expected = {
        (job["task"], job["instance_id"], spec["rollout_id"]): spec[
            "sample_id"
        ]
        for job in manifest["jobs"]
        for spec in job["rollout_specs"]
    }
    observed = {}
    invalid = []
    results_root = Path(manifest["results_root"])
    paths = (
        sorted(results_root.glob("pi05/*/json/*.json"))
        if results_root.is_dir()
        else []
    )
    for path in paths:
        try:
            result = json.loads(path.read_text())
            key = (result["task"], result["instance_id"], result["rollout_id"])
            if key not in expected or key in observed:
                raise ValueError(f"unexpected or duplicate result key {key!r}")
            if result.get("sample_id") != expected[key]:
                raise ValueError(
                    f"result sample_id does not match {expected[key]!r} for {key!r}"
                )
            if not isinstance(result.get("success"), bool):
                raise TypeError("success must be boolean")
            eligible = EvaluationQueue._result_is_metric_eligible(result)
            setup_excluded = result.get("result_state") == "setup_excluded"
            observed[key] = {
                "eligible": eligible,
                "success": eligible and result["success"],
                "setup_excluded": setup_excluded,
            }
        except (
            ArtifactValidationError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            invalid.append({"path": str(path), "error": str(error)})
    eligible = sum(value["eligible"] for value in observed.values())
    successes = sum(value["success"] for value in observed.values())
    completed = len(observed)
    return {
        "completed_segment_rollouts": completed,
        "expected_segment_rollouts": manifest["expected_segment_rollouts"],
        "remaining_segment_rollouts": manifest["expected_segment_rollouts"]
        - completed,
        "eligible_segment_rollouts": eligible,
        "successful_segment_rollouts": successes,
        "setup_excluded_segment_rollouts": sum(
            value["setup_excluded"] for value in observed.values()
        ),
        "success_rate": successes / eligible if eligible else 0.0,
        "invalid_results": invalid,
    }


def _run_status(args: argparse.Namespace) -> None:
    run_root = args.run_root.expanduser().resolve()
    manifest, panel = _load_run(run_root)
    queue = EvaluationQueue(manifest["queue_db"])
    snapshot = queue.snapshot(readonly=True)
    value = {
        "run_id": manifest["run_id"],
        "profile": manifest["profile"],
        "evaluation_kind": manifest["evaluation_kind"],
        "panel_fingerprint": panel["fingerprint"],
        "active_supervisor": dynamic.supervisor_is_active(run_root),
        "expected_episode_jobs": manifest["expected_episode_jobs"],
        "completed_episode_jobs": snapshot["counts"]["succeeded"],
        "episode_queue_counts": snapshot["counts"],
        "queue_counts": snapshot["counts"],
        "expected_annotated_segments": manifest["expected_annotated_segments"],
        "expected_exclusions": manifest["expected_exclusions"],
        **_segment_progress(manifest),
        "markers": {
            "running": (run_root / "_RUNNING").is_file(),
            "failed": (run_root / "_FAILED").is_file(),
            "success": (run_root / "_SUCCESS").is_file(),
        },
    }
    print(json.dumps(value, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-root", type=Path, required=True)
    start.add_argument("--run-id")
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--repo-id", required=True)
    start.add_argument("--dataset-root", type=Path, required=True)
    start.add_argument("--segments", type=Path, required=True)
    start.add_argument("--train-episodes", type=Path, required=True)
    start.add_argument("--split-manifest", type=Path, required=True)
    start.add_argument("--raw-data-root", type=Path, required=True)
    start.add_argument("--gpus", type=int, nargs="+", required=True)
    start.add_argument("--base-port", type=dynamic._positive_int, required=True)
    start.add_argument("--max-attempts", type=dynamic._positive_int, default=2)
    start.add_argument(
        "--max-restarts-per-slot", type=dynamic._nonnegative_int, default=8
    )
    start.add_argument("--sample-max-steps", type=dynamic._positive_int, default=2048)
    start.add_argument(
        "--write-video", action=argparse.BooleanOptionalAction, default=True
    )
    start.add_argument("--runtime-root", type=Path)
    start.add_argument(
        "--eval-env",
        default=os.environ.get("EVAL_ENV", "/dev/shm/wangfaye/conda_envs/b1k-eval"),
    )
    start.add_argument(
        "--pi05-env",
        default=os.environ.get("PI05_ENV", str(REPO_ROOT / "openpi" / ".venv")),
    )
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
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

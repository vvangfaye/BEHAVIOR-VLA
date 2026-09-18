#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

STATE_MACHINE_V1 = "state_machine_v1"
SELECTOR_LABELS_SHA256 = (
    "fd785101426544b24ce756a84e72e2f5971776cfe6ebf7bde18e5ff7fb58ac04"
)
LEGACY_SEGMENTS_SHA256 = (
    "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d"
)
TRAINING_ANNOTATION_SHA256 = (
    "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d"
)


def final_q_score(result: dict) -> float:
    value = result.get("q_score", 0.0)
    if isinstance(value, dict):
        value = value.get("final", 0.0)
    score = float(value or 0.0)
    if not math.isfinite(score):
        raise ValueError("q_score must be finite")
    return score


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split()]
    except ValueError as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")
    return values


def validate_subtask_events(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("subtask_events must be a nonempty list")
    has_text = False
    previous_step = -1
    for event in value:
        if not isinstance(event, dict) or "step" not in event or "subtask" not in event:
            raise ValueError("each subtask event must contain step and subtask")
        step = event["step"]
        subtask = event["subtask"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("subtask event steps must be non-negative integers")
        if step < previous_step:
            raise ValueError("subtask event steps must be ordered")
        if subtask is not None and (
            not isinstance(subtask, str) or not subtask.strip()
        ):
            raise ValueError("subtask event text must be null or a nonempty string")
        has_text = has_text or isinstance(subtask, str) and bool(subtask.strip())
        previous_step = step
    if not has_text:
        raise ValueError("subtask_events contains no predicted subtask text")


def validate_generated_subtask_rollout(result: dict) -> None:
    conditioning = result.get("conditioning")
    if not isinstance(conditioning, dict):
        raise ValueError("conditioning must be an object")
    expected = {
        "mode": "whole_task",
        "oracle": False,
        "privileged": False,
        "comparable_to_whole_task": True,
    }
    for name, value in expected.items():
        actual = conditioning.get(name)
        matches = actual is value if isinstance(value, bool) else actual == value
        if not matches:
            raise ValueError(f"conditioning.{name} is {actual!r}, expected {value!r}")
    if conditioning.get("selector") is not None:
        raise ValueError("conditioning.selector must be absent")
    if result.get("selector_events"):
        raise ValueError("selector_events must be absent or empty")
    if result.get("selector_summary"):
        raise ValueError("selector_summary must be absent or empty")
    validate_subtask_events(result.get("subtask_events"))
    if any(event.get("source") != "generated" for event in result["subtask_events"]):
        raise ValueError("all subtask event sources must be 'generated'")


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def validate_selector_rollout(result: dict) -> dict:
    conditioning = result.get("conditioning")
    if not isinstance(conditioning, dict):
        raise ValueError("conditioning must be an object")
    expected_conditioning = {
        "selector": STATE_MACHINE_V1,
        "label_format": "legacy",
        "selector_labels_sha256": SELECTOR_LABELS_SHA256,
        "legacy_segments_sha256": LEGACY_SEGMENTS_SHA256,
        "training_annotation_sha256": TRAINING_ANNOTATION_SHA256,
        "source": "provided",
        "subtask_source": "provided",
        "oracle": True,
        "privileged": True,
        "official_benchmark": False,
    }
    for name, expected in expected_conditioning.items():
        actual = conditioning.get(name)
        if isinstance(expected, bool):
            matches = actual is expected
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"conditioning.{name} is {actual!r}, expected {expected!r}"
            )

    selector_events = result.get("selector_events")
    if not isinstance(selector_events, list) or not selector_events:
        raise ValueError("selector_events must be a nonempty list")
    entered: dict[str, int] = {}
    previous_step = -1
    for event in selector_events:
        if not isinstance(event, dict):
            raise ValueError("each selector event must be an object")
        step = _nonnegative_int(event.get("step"), "selector event step")
        if step < previous_step:
            raise ValueError("selector event steps must be ordered")
        phase = _nonempty_text(
            event.get("current_phase"), "selector event current_phase"
        )
        _nonempty_text(event.get("current_text"), "selector event current_text")
        entered[phase] = entered.get(phase, 0) + 1
        previous_step = step

    selector_summary = result.get("selector_summary")
    if not isinstance(selector_summary, dict) or not selector_summary:
        raise ValueError("selector_summary must be a nonempty object")
    if selector_summary.get("valid") is not True:
        raise ValueError("selector_summary.valid must be true")
    if selector_summary.get("mode") != STATE_MACHINE_V1:
        raise ValueError(f"selector_summary.mode must be {STATE_MACHINE_V1!r}")
    if selector_summary.get("task_name") != result["task"]:
        raise ValueError("selector_summary.task_name must match the rollout task")

    validated_action_requests = _nonnegative_int(
        selector_summary.get("validated_action_requests"),
        "selector_summary.validated_action_requests",
    )
    if validated_action_requests != result["steps"]:
        raise ValueError(
            f"selector_summary.validated_action_requests is {validated_action_requests}, expected {result['steps']}"
        )

    validate_subtask_events(result.get("subtask_events"))
    if any(event.get("source") != "provided" for event in result["subtask_events"]):
        raise ValueError("all subtask event sources must be 'provided'")

    recoveries = _nonnegative_int(
        selector_summary.get("recoveries"), "selector_summary.recoveries"
    )
    stalls = selector_summary.get("stalls")
    if not isinstance(stalls, list):
        raise ValueError("selector_summary.stalls must be a list")
    stalled: dict[str, int] = {}
    for stall in stalls:
        if not isinstance(stall, dict):
            raise ValueError("each selector stall must be an object")
        phase = _nonempty_text(stall.get("phase"), "selector stall phase")
        stalled[phase] = stalled.get(phase, 0) + 1

    completion_counts = selector_summary.get("phase_completion_counts")
    if not isinstance(completion_counts, dict):
        raise ValueError("selector_summary.phase_completion_counts must be an object")
    completed = {}
    for raw_phase, raw_count in completion_counts.items():
        phase = _nonempty_text(raw_phase, "selector completion phase")
        completed[phase] = _nonnegative_int(
            raw_count, f"selector completion count for {phase}"
        )

    return {
        "recoveries": recoveries,
        "stalls": len(stalls),
        "entered": entered,
        "completed": completed,
        "stalled": stalled,
    }


def load_worker_metrics(
    results_root: Path,
) -> tuple[list[dict], list[dict], list[dict], list[int]]:
    worker_metrics = []
    invalid_files = []
    failed_metrics = []
    ports = []
    for path in sorted(results_root.rglob("eval_metrics_*.json")):
        try:
            value = json.loads(path.read_text())
            required = {
                "port",
                "server_gpu",
                "eval_gpu",
                "elapsed_seconds",
                "exit_code",
            }
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError("missing required worker metric fields")
            for name in ("port", "server_gpu", "eval_gpu", "exit_code"):
                if isinstance(value[name], bool) or not isinstance(value[name], int):
                    raise ValueError(f"{name} must be an integer")
            elapsed = value["elapsed_seconds"]
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or elapsed <= 0
            ):
                raise ValueError("elapsed_seconds must be positive")
            value["path"] = str(path)
            worker_metrics.append(value)
            ports.append(value["port"])
            if value["exit_code"] != 0:
                failed_metrics.append(
                    {"path": str(path), "exit_code": value["exit_code"]}
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})
    worker_metrics.sort(key=lambda value: value["port"])
    duplicate_ports = sorted({port for port in ports if ports.count(port) > 1})
    return worker_metrics, invalid_files, failed_metrics, duplicate_ports


def load_worker_statuses(
    status_root: Path | None,
    expected_workers: int | None,
    profile: str,
    expected_tasks: list[str],
    expected_rollouts: int,
    require_videos: bool,
    require_subtask_events: bool,
    require_selector_events: bool,
) -> tuple[list[dict], list[str], list[str], list[dict], list[dict]]:
    if expected_workers is None:
        return [], [], [], [], []
    if status_root is None:
        raise ValueError("--status-root is required with --expected-workers")

    expected_paths = {
        status_root / f"worker-{worker}.json" for worker in range(expected_workers)
    }
    existing_paths = (
        set(status_root.glob("worker-*.json")) if status_root.is_dir() else set()
    )
    missing_files = sorted(str(path) for path in expected_paths - existing_paths)
    unexpected_files = sorted(str(path) for path in existing_paths - expected_paths)
    statuses = []
    invalid_files = []
    mismatches = []
    per_worker = (
        expected_rollouts // expected_workers
        if expected_rollouts % expected_workers == 0
        else None
    )

    for worker in range(expected_workers):
        path = status_root / f"worker-{worker}.json"
        if path not in existing_paths:
            continue
        try:
            value = json.loads(path.read_text())
            required = {"worker", "gpu", "port", "exit_code"}
            if not isinstance(value, dict) or not required.issubset(value):
                raise ValueError("missing required worker status fields")
            for name in required:
                if isinstance(value[name], bool) or not isinstance(value[name], int):
                    raise ValueError(f"{name} must be an integer")
            errors = []
            if value["worker"] != worker:
                errors.append(f"worker is {value['worker']}, expected {worker}")
            if value["gpu"] != worker:
                errors.append(f"gpu is {value['gpu']}, expected {worker}")
            if value["exit_code"] != 0:
                errors.append(f"exit_code is {value['exit_code']}, expected 0")
            if profile == "two_tasks":
                required_counts = {
                    "expected_rollouts": per_worker,
                    "completed_rollouts": per_worker,
                    "completed_videos": per_worker if require_videos else None,
                    "rollouts_with_subtask_events": per_worker
                    if require_subtask_events
                    else None,
                    "rollouts_with_selector_events": per_worker
                    if require_selector_events
                    else None,
                }
                if value.get("tasks") != expected_tasks:
                    errors.append(
                        f"tasks are {value.get('tasks')!r}, expected {expected_tasks!r}"
                    )
                indices = value.get("instance_indices")
                if not isinstance(indices, list) or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in indices
                ):
                    errors.append("instance_indices must be a list of integers")
                for name, expected in required_counts.items():
                    if expected is not None and value.get(name) != expected:
                        errors.append(
                            f"{name} is {value.get(name)!r}, expected {expected}"
                        )
            value["path"] = str(path)
            statuses.append(value)
            if errors:
                mismatches.append({"path": str(path), "errors": errors})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})

    if profile == "two_tasks" and not invalid_files and not missing_files:
        observed_indices = [
            index for status in statuses for index in status.get("instance_indices", [])
        ]
        if sorted(observed_indices) != list(range(20)):
            mismatches.append(
                {
                    "path": str(status_root),
                    "errors": [
                        f"worker instance indices are {sorted(observed_indices)}, expected 0..19"
                    ],
                }
            )
    statuses.sort(key=lambda value: value["worker"])
    return statuses, missing_files, unexpected_files, invalid_files, mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--expected-task")
    parser.add_argument("--expected-tasks")
    parser.add_argument("--expected-instance-ids")
    parser.add_argument("--expected-rollout-ids")
    parser.add_argument("--expected-workers", type=int)
    parser.add_argument("--status-root", type=Path)
    parser.add_argument("--wall-time-seconds", type=float)
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--require-subtask-events", action="store_true")
    parser.add_argument("--require-generated-subtasks", action="store_true")
    parser.add_argument("--require-selector-events", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.require_generated_subtasks and args.require_selector_events:
        parser.error(
            "--require-generated-subtasks and --require-selector-events are mutually exclusive"
        )
    if args.expected_task and args.expected_tasks:
        parser.error("--expected-task and --expected-tasks are mutually exclusive")
    expected_task_value = args.expected_tasks or args.expected_task
    exact_expectations = any(
        value is not None
        for value in (
            expected_task_value,
            args.expected_instance_ids,
            args.expected_rollout_ids,
        )
    )
    if exact_expectations and (
        not expected_task_value or not args.expected_instance_ids
    ):
        parser.error(
            "expected tasks and --expected-instance-ids must be provided together"
        )
    expected_tasks = expected_task_value.split() if expected_task_value else []
    if expected_task_value and (
        not expected_tasks or len(expected_tasks) != len(set(expected_tasks))
    ):
        parser.error("expected tasks must contain unique values")
    try:
        expected_instance_ids = (
            parse_int_list(args.expected_instance_ids, "--expected-instance-ids")
            if args.expected_instance_ids
            else []
        )
        expected_rollout_ids = (
            parse_int_list(args.expected_rollout_ids, "--expected-rollout-ids")
            if args.expected_rollout_ids
            else [0]
        )
    except ValueError as exc:
        parser.error(str(exc))
    expected_keys = {
        (task, instance_id, rollout_id)
        for task in expected_tasks
        for instance_id in expected_instance_ids
        for rollout_id in expected_rollout_ids
    }
    if expected_keys and len(expected_keys) != args.expected_rollouts:
        parser.error(
            f"exact expectations contain {len(expected_keys)} rollouts, "
            f"but --expected-rollouts is {args.expected_rollouts}"
        )
    if args.expected_workers is not None and args.expected_workers <= 0:
        parser.error("--expected-workers must be positive")

    parsed_results = []
    invalid_files = []
    duplicate_rollouts = []
    invalid_subtask_events = []
    invalid_generated_subtasks = []
    result_paths = {}
    results_by_key = {}
    for path in sorted(args.results_root.rglob("json/*.json")):
        try:
            value = json.loads(path.read_text())
            required = {"task", "instance_id", "rollout_id", "steps", "success"}
            if not isinstance(value, dict):
                raise ValueError("result must be a JSON object")
            missing = required - value.keys()
            if missing:
                raise ValueError(f"missing required fields: {sorted(missing)}")
            if not isinstance(value["task"], str) or not value["task"].strip():
                raise ValueError("task must be a nonempty string")
            if not isinstance(value["success"], bool):
                raise ValueError("success must be a boolean")
            integer_fields = ("instance_id", "rollout_id", "steps")
            if any(
                isinstance(value[name], bool) or not isinstance(value[name], int)
                for name in integer_fields
            ):
                raise ValueError("instance_id, rollout_id, and steps must be integers")
            if (
                value["instance_id"] < 0
                or value["rollout_id"] < 0
                or value["steps"] < 0
            ):
                raise ValueError(
                    "instance_id, rollout_id, and steps must be non-negative"
                )
            final_q_score(value)
            key = (value["task"], value["instance_id"], value["rollout_id"])
            if key in result_paths:
                duplicate_rollouts.append(
                    {"key": list(key), "paths": [str(result_paths[key]), str(path)]}
                )
            else:
                result_paths[key] = path
                results_by_key[key] = value
            if args.require_subtask_events:
                try:
                    validate_subtask_events(value.get("subtask_events"))
                except ValueError as exc:
                    invalid_subtask_events.append(
                        {"path": str(path), "error": str(exc)}
                    )
            if args.require_generated_subtasks:
                try:
                    validate_generated_subtask_rollout(value)
                except ValueError as exc:
                    invalid_generated_subtasks.append(
                        {"path": str(path), "error": str(exc)}
                    )
            parsed_results.append(value)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})

    actual_keys = set(results_by_key)
    missing_rollout_keys = sorted(expected_keys - actual_keys)
    unexpected_rollout_keys = (
        sorted(actual_keys - expected_keys) if expected_keys else []
    )
    selected_keys = (
        sorted(actual_keys & expected_keys) if expected_keys else sorted(actual_keys)
    )
    selected_results = [results_by_key[key] for key in selected_keys]

    invalid_selector_rollouts = []
    selector_total_recoveries = 0
    selector_total_stalls = 0
    selector_phase_counts: dict[str, dict[str, int]] = {}
    if args.require_selector_events:
        for key in selected_keys:
            try:
                diagnostics = validate_selector_rollout(results_by_key[key])
            except (KeyError, TypeError, ValueError) as exc:
                invalid_selector_rollouts.append(
                    {
                        "key": list(key),
                        "path": str(result_paths[key]),
                        "error": str(exc),
                    }
                )
                continue
            selector_total_recoveries += diagnostics["recoveries"]
            selector_total_stalls += diagnostics["stalls"]
            phases = (
                set(diagnostics["entered"])
                | set(diagnostics["completed"])
                | set(diagnostics["stalled"])
            )
            for phase in phases:
                counts = selector_phase_counts.setdefault(
                    phase, {"entered": 0, "completed": 0, "stalled": 0}
                )
                counts["entered"] += diagnostics["entered"].get(phase, 0)
                counts["completed"] += diagnostics["completed"].get(phase, 0)
                counts["stalled"] += diagnostics["stalled"].get(phase, 0)
    selector_phase_counts = dict(sorted(selector_phase_counts.items()))

    all_video_paths = set(args.results_root.rglob("videos/*.mp4"))
    expected_video_paths = set()
    missing_videos = []
    invalid_videos = []
    completed_videos = 0
    if args.require_videos:
        for key in selected_keys:
            result_path = result_paths[key]
            video_path = (
                result_path.parent.parent / "videos" / f"{result_path.stem}.mp4"
            )
            expected_video_paths.add(video_path)
            if not video_path.exists():
                missing_videos.append(str(video_path))
                continue
            try:
                if video_path.stat().st_size <= 0:
                    invalid_videos.append(
                        {"path": str(video_path), "error": "video is empty"}
                    )
                else:
                    completed_videos += 1
            except OSError as exc:
                invalid_videos.append({"path": str(video_path), "error": str(exc)})
    unexpected_videos = (
        sorted(str(path) for path in all_video_paths - expected_video_paths)
        if args.require_videos
        else []
    )

    (
        worker_metrics,
        invalid_metric_files,
        failed_worker_metrics,
        duplicate_metric_ports,
    ) = load_worker_metrics(args.results_root)
    try:
        (
            worker_statuses,
            missing_worker_status_files,
            unexpected_worker_status_files,
            invalid_worker_status_files,
            worker_status_mismatches,
        ) = load_worker_statuses(
            args.status_root,
            args.expected_workers,
            args.profile,
            expected_tasks,
            args.expected_rollouts,
            args.require_videos,
            args.require_subtask_events,
            args.require_selector_events,
        )
    except ValueError as exc:
        parser.error(str(exc))

    status_ports = {status["port"] for status in worker_statuses}
    metric_ports = {metric["port"] for metric in worker_metrics}
    worker_metric_port_mismatch = (
        {"status_ports": sorted(status_ports), "metric_ports": sorted(metric_ports)}
        if args.expected_workers is not None and status_ports != metric_ports
        else None
    )
    worker_metrics_incomplete = (
        args.expected_workers is not None
        and len(worker_metrics) != args.expected_workers
    )

    per_task = {}
    for result in selected_results:
        task = result["task"]
        stats = per_task.setdefault(
            task, {"rollouts": 0, "successes": 0, "q_score_sum": 0.0}
        )
        stats["rollouts"] += 1
        stats["successes"] += int(result["success"])
        stats["q_score_sum"] += final_q_score(result)
    for stats in per_task.values():
        count = stats["rollouts"]
        stats["success_rate"] = stats["successes"] / count
        stats["mean_q_score"] = stats.pop("q_score_sum") / count

    count = len(selected_results)
    successes = sum(result["success"] for result in selected_results)
    total_steps = sum(result["steps"] for result in selected_results)
    wall_time = args.wall_time_seconds
    summary = {
        "run_id": args.run_id,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "expected_rollouts": args.expected_rollouts,
        "completed_rollouts": count,
        "observed_result_files": len(parsed_results),
        "unique_rollouts": len(results_by_key),
        "missing_rollouts": len(missing_rollout_keys)
        if expected_keys
        else max(args.expected_rollouts - count, 0),
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_q_score": sum(final_q_score(result) for result in selected_results)
        / count
        if count
        else 0.0,
        "total_steps": total_steps,
        "wall_time_seconds": wall_time,
        "rollouts_per_hour": count * 3600 / wall_time
        if wall_time and wall_time > 0
        else None,
        "aggregate_steps_per_second": total_steps / wall_time
        if wall_time and wall_time > 0
        else None,
        "expected_videos": args.expected_rollouts if args.require_videos else 0,
        "completed_videos": completed_videos if args.require_videos else None,
        "worker_metrics": worker_metrics,
        "worker_statuses": worker_statuses,
        "duplicate_rollouts": duplicate_rollouts,
        "missing_rollout_keys": [list(key) for key in missing_rollout_keys],
        "unexpected_rollout_keys": [list(key) for key in unexpected_rollout_keys],
        "missing_videos": missing_videos,
        "invalid_videos": invalid_videos,
        "unexpected_videos": unexpected_videos,
        "invalid_subtask_events": invalid_subtask_events,
        "invalid_generated_subtasks": invalid_generated_subtasks,
        "invalid_files": invalid_files,
        "invalid_metric_files": invalid_metric_files,
        "failed_worker_metrics": failed_worker_metrics,
        "duplicate_metric_ports": duplicate_metric_ports,
        "worker_metrics_incomplete": worker_metrics_incomplete,
        "worker_metric_port_mismatch": worker_metric_port_mismatch,
        "missing_worker_status_files": missing_worker_status_files,
        "unexpected_worker_status_files": unexpected_worker_status_files,
        "invalid_worker_status_files": invalid_worker_status_files,
        "worker_status_mismatches": worker_status_mismatches,
        "tasks": per_task,
    }
    if args.require_selector_events:
        summary["selector_diagnostics"] = {
            "selector": STATE_MACHINE_V1,
            "valid_rollouts": count - len(invalid_selector_rollouts),
            "invalid_rollout_keys": [item["key"] for item in invalid_selector_rollouts],
            "invalid_rollouts": invalid_selector_rollouts,
            "total_recoveries": selector_total_recoveries,
            "total_stalls": selector_total_stalls,
            "per_phase": selector_phase_counts,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    incomplete = (
        count != args.expected_rollouts
        or bool(invalid_files)
        or bool(duplicate_rollouts)
        or bool(missing_rollout_keys)
        or bool(unexpected_rollout_keys)
        or bool(invalid_metric_files)
        or bool(failed_worker_metrics)
        or bool(duplicate_metric_ports)
        or worker_metrics_incomplete
        or worker_metric_port_mismatch is not None
        or bool(missing_worker_status_files)
        or bool(unexpected_worker_status_files)
        or bool(invalid_worker_status_files)
        or bool(worker_status_mismatches)
        or (args.require_videos and completed_videos != args.expected_rollouts)
        or bool(missing_videos)
        or bool(invalid_videos)
        or bool(unexpected_videos)
        or bool(invalid_subtask_events)
        or bool(invalid_generated_subtasks)
        or bool(invalid_selector_rollouts)
    )
    if args.require_complete and incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

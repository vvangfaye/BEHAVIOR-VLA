#!/usr/bin/env python3
"""Strictly validate and summarize the episode-segment subtask diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pi05_eval_queue import atomic_write_json
from subtask_sample_panel import load_panel

EVALUATION_KIND = "training_subtask_closed_loop"
PROFILE = "training_subtask_closed_loop_episode_segments"
DEBOUNCE_STEPS = 3
MAX_SAMPLE_STEPS = 2048


def _artifact_stem(sample: dict[str, Any]) -> str:
    return (
        f"{sample['task']}_{sample['task_instance_id']}_"
        f"{int(sample['rollout_id'])}"
    )


def _result_path(results_root: Path, sample: dict[str, Any]) -> Path:
    return (
        results_root
        / "pi05"
        / sample["task"]
        / "json"
        / f"{_artifact_stem(sample)}.json"
    )


def _video_path(results_root: Path, sample: dict[str, Any]) -> Path:
    return (
        results_root
        / "pi05"
        / sample["task"]
        / "videos"
        / f"{_artifact_stem(sample)}.mp4"
    )


def _result_key(value: object) -> tuple[str, int, int] | None:
    if not isinstance(value, dict):
        return None
    task = value.get("task")
    instance_id = value.get("instance_id")
    rollout_id = value.get("rollout_id")
    if (
        not isinstance(task, str)
        or not task
        or isinstance(instance_id, bool)
        or not isinstance(instance_id, int)
        or instance_id < 0
        or isinstance(rollout_id, bool)
        or not isinstance(rollout_id, int)
        or rollout_id < 0
    ):
        return None
    return task, instance_id, rollout_id


def _q_score(result: dict[str, Any]) -> float:
    value = result.get("q_score", 0.0)
    if isinstance(value, dict):
        value = value.get("final", 0.0)
    return float(value or 0.0)


def _manifest_provenance(
    run_manifest: dict[str, Any] | None,
    panel: dict[str, Any],
    checkpoint: str,
) -> dict[str, dict[str, Any]]:
    if run_manifest is None:
        return {}
    expected_manifest = {
        "profile": PROFILE,
        "evaluation_kind": EVALUATION_KIND,
        "panel_fingerprint": panel["fingerprint"],
        "checkpoint": checkpoint,
        "expected_episode_jobs": len(panel.get("episodes", [])),
        "expected_segment_rollouts": len(panel["samples"]),
        "expected_annotated_segments": len(panel["samples"])
        + len(panel.get("exclusions", [])),
        "expected_exclusions": len(panel.get("exclusions", [])),
    }
    mismatches = {
        name: (run_manifest.get(name), value)
        for name, value in expected_manifest.items()
        if run_manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"run manifest does not match panel: {mismatches}")

    expected_jobs = {
        (episode["task"], int(episode["task_instance_id"])): list(
            episode["sample_ids"]
        )
        for episode in panel.get("episodes", [])
    }
    jobs = run_manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected_jobs):
        raise ValueError("run manifest episode jobs do not cover the panel")
    observed_jobs = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise TypeError("run manifest contains an invalid episode job")
        key = (job.get("task"), job.get("instance_id"))
        sample_ids = job.get("sample_ids")
        if key in observed_jobs or not isinstance(sample_ids, list):
            raise ValueError("run manifest contains duplicate or invalid episode jobs")
        observed_jobs[key] = sample_ids
    if observed_jobs != expected_jobs:
        raise ValueError("run manifest episode membership does not match the panel")

    preflight = run_manifest.get("raw_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != 2
        or preflight.get("valid") is not True
    ):
        raise ValueError("run manifest has no valid schema-v2 raw HDF5 preflight")
    episode_rows = preflight.get("episodes")
    sample_rows = preflight.get("samples")
    if not isinstance(episode_rows, list) or not isinstance(sample_rows, list):
        raise TypeError("run manifest raw preflight sections are invalid")

    episode_hashes = {}
    covered_ids = []
    for row in episode_rows:
        if not isinstance(row, dict):
            raise TypeError("run manifest contains invalid episode provenance")
        digest = row.get("sha256")
        sample_ids = row.get("sample_ids")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(sample_ids, list)
            or any(not isinstance(value, str) or not value for value in sample_ids)
        ):
            raise ValueError("run manifest contains invalid episode provenance")
        for sample_id in sample_ids:
            if sample_id in episode_hashes:
                raise ValueError(f"duplicate raw HDF5 provenance for {sample_id}")
            episode_hashes[sample_id] = digest
            covered_ids.append(sample_id)

    sample_evidence = {}
    for row in sample_rows:
        if not isinstance(row, dict):
            raise TypeError("run manifest contains invalid sample provenance")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_evidence:
            raise ValueError("run manifest contains duplicate or invalid sample provenance")
        sample_evidence[sample_id] = row

    expected_samples = {sample["sample_id"]: sample for sample in panel["samples"]}
    if set(covered_ids) != set(expected_samples) or set(sample_evidence) != set(
        expected_samples
    ):
        raise ValueError("run manifest raw HDF5 provenance does not cover the panel")
    provenance = {}
    for sample_id, sample in expected_samples.items():
        evidence = sample_evidence[sample_id]
        expected = {
            "episode_index": sample["episode_index"],
            "rollout_id": sample["rollout_id"],
            "selected_frame": sample["start"],
        }
        if any(evidence.get(name) != value for name, value in expected.items()):
            raise ValueError(f"raw HDF5 provenance disagrees for {sample_id}")
        provenance[sample_id] = {
            **evidence,
            "sha256": episode_hashes[sample_id],
        }
    return provenance


def _validate_result(
    result: object,
    sample: dict[str, Any],
    fingerprint: str,
    raw_provenance: dict[str, Any] | None,
    max_steps: int,
) -> list[str]:
    if not isinstance(result, dict):
        return ["result is not a JSON object"]
    errors = []
    expected = {
        "task": sample["task"],
        "instance_id": sample["task_instance_id"],
        "rollout_id": sample["rollout_id"],
        "sample_id": sample["sample_id"],
        "episode_index": sample["episode_index"],
        "raw_episode_id": sample["raw_episode_id"],
        "source_segment_index": sample["source_segment_index"],
        "source_interval_index": sample["source_interval_index"],
        "segment_start": sample["start"],
        "segment_end": sample["end"],
        "segment_frames": sample["segment_frames"],
        "subtask": sample["subtask"],
        "predicate": sample["predicate"],
        "panel_fingerprint": fingerprint,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            errors.append(f"{name} is {result.get(name)!r}, expected {value!r}")

    if result.get("result_state") == "source_excluded":
        from subtask_source_exclusion import validate_source_exclusion
        try:
            validate_source_exclusion(result, sample, fingerprint, raw_provenance)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
        return errors

    success = result.get("success")
    eligible = result.get("metric_eligible")
    result_state = result.get("result_state")
    steps = result.get("steps")
    if not isinstance(success, bool):
        errors.append("success is not boolean")
    if not isinstance(eligible, bool):
        errors.append("metric_eligible is not boolean")
    if result_state not in ("evaluated", "setup_excluded"):
        errors.append(f"result_state is invalid: {result_state!r}")
    elif eligible is not (result_state == "evaluated"):
        errors.append("metric_eligible conflicts with result_state")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 0
        or steps > max_steps
        or (eligible is True and steps < 1)
        or (eligible is False and steps != 0)
    ):
        errors.append(f"steps is invalid: {steps!r}")

    setup_reason = result.get("setup_exclusion_reason")
    setup_evidence = result.get("setup_evidence")
    failure_reason = result.get("failure_reason")
    if eligible is False:
        if success is not False:
            errors.append("setup-excluded result cannot succeed")
        if setup_reason not in ("predicate_already_true", "predicate_unobservable"):
            errors.append(f"setup exclusion reason is invalid: {setup_reason!r}")
        if failure_reason != setup_reason:
            errors.append("setup exclusion reason does not match failure_reason")
        if not isinstance(setup_evidence, dict) or not setup_evidence:
            errors.append("setup exclusion lacks evidence")
        elif (
            setup_reason == "predicate_already_true"
            and setup_evidence.get("initial_predicate_value") is not True
        ):
            errors.append("already-true exclusion lacks initial predicate evidence")
        elif setup_reason == "predicate_unobservable" and not isinstance(
            setup_evidence.get("error"), str
        ):
            errors.append("unobservable exclusion lacks error evidence")
    elif eligible is True:
        if setup_reason is not None:
            errors.append("evaluated result has a setup exclusion reason")
        if not isinstance(setup_evidence, dict):
            errors.append("evaluated result lacks setup evidence")
        if success is True and failure_reason is not None:
            errors.append("successful result has a failure reason")
        if success is False and failure_reason not in (
            "sample_timeout",
            "environment_terminated",
            "environment_truncated",
        ):
            errors.append(f"evaluated failure reason is invalid: {failure_reason!r}")

    for name in ("terminated", "truncated", "task_success"):
        if not isinstance(result.get(name), bool):
            errors.append(f"{name} is not boolean")
    if success is False and result.get("terminated") is True and failure_reason != "environment_terminated":
        errors.append("terminated result has inconsistent failure_reason")
    if success is False and result.get("truncated") is True and failure_reason != "environment_truncated":
        errors.append("truncated result has inconsistent failure_reason")

    conditioning = result.get("conditioning")
    if not isinstance(conditioning, dict):
        errors.append("conditioning is missing")
    else:
        required_conditioning = {
            "subtask_mode": "external_fixed",
            "subtask": sample["subtask"],
            "source": "provided",
            "subtask_source": "provided",
            "training_set_diagnostic": True,
            "official_benchmark": False,
        }
        for name, value in required_conditioning.items():
            if conditioning.get(name) != value:
                errors.append(
                    f"conditioning.{name} is {conditioning.get(name)!r}, expected {value!r}"
                )

    observed_subtasks = result.get("observed_subtasks")
    observed_sources = result.get("observed_subtask_sources")
    if not isinstance(observed_subtasks, list) or len(observed_subtasks) != steps:
        errors.append("observed_subtasks does not match steps")
    elif any(value != sample["subtask"] for value in observed_subtasks):
        errors.append("observed_subtasks contains text other than the fixed panel subtask")
    if not isinstance(observed_sources, list) or len(observed_sources) != steps:
        errors.append("observed_subtask_sources does not match steps")
    elif any(value != "provided" for value in observed_sources):
        errors.append("observed_subtask_sources contains a non-provided source")
    if not isinstance(result.get("subtask_events"), list):
        errors.append("subtask_events is not a list")

    monitor = result.get("monitor")
    if not isinstance(monitor, dict):
        errors.append("monitor evidence is missing")
    else:
        if monitor.get("predicate") != sample["predicate"]:
            errors.append("monitor predicate does not match panel")
        if monitor.get("debounce_steps") != DEBOUNCE_STEPS:
            errors.append("monitor debounce contract is invalid")
        if monitor.get("metric_eligible") is not eligible:
            errors.append("monitor metric eligibility does not match result")
        if monitor.get("setup_exclusion_reason") != setup_reason:
            errors.append("monitor setup exclusion reason does not match result")
        if monitor.get("setup_evidence") != setup_evidence:
            errors.append("monitor setup evidence does not match result")
        if not isinstance(monitor.get("initial_snapshot"), dict) or not isinstance(
            monitor.get("final_snapshot"), dict
        ):
            errors.append("monitor snapshots are invalid")
        dwell = monitor.get("dwell")
        satisfied = monitor.get("satisfied")
        terminal_credit = result.get("terminal_credit")
        if not isinstance(terminal_credit, bool):
            errors.append("terminal_credit is not boolean")
        elif terminal_credit is True and eligible is not True:
            errors.append("terminal_credit requires an evaluated result")
        if isinstance(dwell, bool) or not isinstance(dwell, int) or dwell < 0:
            errors.append("monitor dwell is invalid")
        elif eligible is True:
            debounced = dwell >= DEBOUNCE_STEPS
            if terminal_credit is True:
                # The whole-task terminal condition ended the episode on the same
                # step the subtask predicate first held, so the debounce window
                # could not accrue. Require a satisfied predicate at a genuine
                # termination to substantiate the credit.
                if not (
                    success is True
                    and result.get("terminated") is True
                    and satisfied is True
                    and dwell >= 1
                ):
                    errors.append("terminal completion credit is not substantiated")
            elif debounced is not (success is True):
                errors.append("success does not match debounced monitor state")
        recoveries = monitor.get("diagnostic_recoveries")
        if (
            isinstance(recoveries, bool)
            or not isinstance(recoveries, int)
            or recoveries < 0
        ):
            errors.append("monitor diagnostic recovery count is invalid")
        if not isinstance(monitor.get("selector_summary"), dict):
            errors.append("monitor selector summary is invalid")

    restoration = result.get("restoration")
    if not isinstance(restoration, dict):
        errors.append("restoration evidence is missing")
    else:
        expected_restoration = {
            "selected_frame": sample["start"],
            "state_prefix_samples": sample["start"] + 1,
            "full_snapshot": True,
        }
        if raw_provenance is not None:
            expected_restoration["selected_state_size"] = raw_provenance[
                "selected_state_size"
            ]
        for name, value in expected_restoration.items():
            if restoration.get(name) != value:
                errors.append(
                    f"restoration.{name} is {restoration.get(name)!r}, expected {value!r}"
                )
        digest = restoration.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("restoration SHA256 is invalid")
        elif raw_provenance is not None and digest != raw_provenance["sha256"]:
            errors.append(
                f"restoration SHA256 is {digest!r}, expected {raw_provenance['sha256']!r}"
            )
        transitions = restoration.get("topology_transition_steps")
        if (
            not isinstance(transitions, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in transitions)
            or transitions != sorted(set(transitions))
            or any(value >= sample["start"] for value in transitions)
        ):
            errors.append("restoration topology transition steps are invalid")
        elif raw_provenance is not None and len(transitions) != raw_provenance[
            "transition_count_before_frame"
        ]:
            errors.append("restoration topology transition count does not match preflight")

    try:
        q_score = _q_score(result)
    except (TypeError, ValueError):
        errors.append("q_score is not numeric")
    else:
        if not math.isfinite(q_score):
            errors.append("q_score is not finite")
    return errors


def _step_statistics(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
    }


def _group_summary(
    scheduled: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    result_by_sample: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    results = [
        result_by_sample[sample["sample_id"]]
        for sample in scheduled
        if sample["sample_id"] in result_by_sample
    ]
    eligible = [result for result in results if result["metric_eligible"]]
    successes = [result for result in eligible if result["success"]]
    q_scores = [_q_score(result) for result in eligible]
    return {
        "annotated_segments": len(scheduled) + len(exclusions),
        "scheduled_segments": len(scheduled),
        "valid_results": len(results),
        "metric_eligible": len(eligible),
        "setup_excluded": sum(result["result_state"] == "setup_excluded" for result in results),
        "source_excluded": sum(result["result_state"] == "source_excluded" for result in results),
        "unsupported": len(exclusions),
        "successes": len(successes),
        "completion_rate": len(successes) / len(eligible) if eligible else None,
        "successful_steps": _step_statistics(
            [int(result["steps"]) for result in successes]
        ),
        "auxiliary": {
            "whole_task_successes": sum(result["task_success"] for result in eligible),
            "mean_q_score": sum(q_scores) / len(q_scores) if q_scores else None,
        },
    }


def summarize(
    panel: dict[str, Any],
    results_root: str | Path,
    *,
    checkpoint: str,
    require_videos: bool,
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results_root = Path(results_root).expanduser().resolve()
    samples = panel["samples"]
    exclusions = panel.get("exclusions", [])
    raw_provenance = _manifest_provenance(run_manifest, panel, checkpoint)
    max_steps = (
        int(run_manifest.get("sample_max_steps", MAX_SAMPLE_STEPS))
        if run_manifest is not None
        else MAX_SAMPLE_STEPS
    )

    expected_by_path = {_result_path(results_root, sample): sample for sample in samples}
    expected_by_key = {
        (sample["task"], sample["task_instance_id"], sample["rollout_id"]): sample
        for sample in samples
    }
    actual_json = set(results_root.glob("pi05/*/json/*.json"))
    parsed_by_path: dict[Path, object] = {}
    parse_errors = {}
    key_paths: dict[tuple[str, int, int], list[Path]] = defaultdict(list)
    for path in sorted(actual_json):
        try:
            parsed = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            parse_errors[path] = f"invalid JSON: {error}"
            continue
        parsed_by_path[path] = parsed
        key = _result_key(parsed)
        if key is not None:
            key_paths[key].append(path)

    duplicate_results = [
        {"key": list(key), "paths": [str(path) for path in paths]}
        for key, paths in sorted(key_paths.items())
        if len(paths) > 1
    ]
    invalid_results = []
    valid_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path, sample in sorted(expected_by_path.items(), key=lambda item: str(item[0])):
        if path not in actual_json:
            continue
        if path in parse_errors:
            invalid_results.append(
                {
                    "sample_id": sample["sample_id"],
                    "path": str(path),
                    "errors": [parse_errors[path]],
                }
            )
            continue
        result = parsed_by_path[path]
        errors = _validate_result(
            result,
            sample,
            panel["fingerprint"],
            raw_provenance.get(sample["sample_id"]),
            max_steps,
        )
        if not errors and result.get("result_state") != "source_excluded" and run_manifest is not None and run_manifest.get("state_protocol") == "recorded_native_fields_v1":
            from recorded_native_state import validate_baseline_result
            try:
                validate_baseline_result(result, sample, raw_provenance.get(sample["sample_id"]), max_steps)
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                errors.append(str(error))
        if errors:
            invalid_results.append(
                {"sample_id": sample["sample_id"], "path": str(path), "errors": errors}
            )
        else:
            valid_results.append((sample, result))

    result_by_sample = {sample["sample_id"]: result for sample, result in valid_results}
    allowed_videos = {_video_path(results_root, sample) for sample in samples}
    actual_videos = set(results_root.glob("pi05/*/videos/*.mp4"))
    required_videos = {
        _video_path(results_root, sample)
        for sample, result in valid_results
        if result["metric_eligible"]
    }
    empty_videos = sorted(
        str(path)
        for path in actual_videos & allowed_videos
        if path.stat().st_size <= 0
    )
    missing_videos = (
        sorted(str(path) for path in required_videos - actual_videos)
        if require_videos
        else []
    )
    unexpected_videos = sorted(str(path) for path in actual_videos - allowed_videos)

    per_task = {}
    for task in sorted({item["task"] for item in [*samples, *exclusions]}):
        per_task[task] = _group_summary(
            [sample for sample in samples if sample["task"] == task],
            [item for item in exclusions if item["task"] == task],
            result_by_sample,
        )

    per_skill_description = {}
    skill_descriptions = sorted(
        {item["skill_description"] for item in [*samples, *exclusions]}
    )
    for description in skill_descriptions:
        per_skill_description[description] = _group_summary(
            [sample for sample in samples if sample["skill_description"] == description],
            [item for item in exclusions if item["skill_description"] == description],
            result_by_sample,
        )

    per_prompt_label = {}
    prompt_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        prompt_groups[(sample["task"], sample["subtask"])].append(sample)
    prompt_exclusions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in exclusions:
        prompt_exclusions[(item["task"], item["subtask"])].append(item)
    for task, prompt in sorted(set(prompt_groups) | set(prompt_exclusions)):
        per_prompt_label.setdefault(task, {})[prompt] = _group_summary(
            prompt_groups[(task, prompt)], prompt_exclusions[(task, prompt)], result_by_sample
        )

    unsupported_annotations = [
        {
            name: item[name]
            for name in (
                "sample_id",
                "task",
                "episode_index",
                "task_instance_id",
                "source_segment_index",
                "source_interval_index",
                "start",
                "end",
                "skill_description",
                "subtask",
                "reason",
            )
        }
        for item in exclusions
    ]
    setup_exclusions = [
        {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "skill_description": sample["skill_description"],
            "subtask": sample["subtask"],
            "reason": result["setup_exclusion_reason"],
            "evidence": result["setup_evidence"],
        }
        for sample, result in valid_results
        if result["result_state"] == "setup_excluded"
    ]
    source_exclusions = [
        {"sample_id": sample["sample_id"], "task": sample["task"],
         "skill_description": sample["skill_description"], "subtask": sample["subtask"],
         "reason": result["source_exclusion_reason"], "evidence": result["source_evidence"]}
        for sample, result in valid_results if result["result_state"] == "source_excluded"
    ]

    overall = _group_summary(samples, exclusions, result_by_sample)
    missing_results = sorted(
        str(path) for path in set(expected_by_path) - actual_json
    )
    unexpected_results = sorted(str(path) for path in actual_json - set(expected_by_path))
    incomplete = bool(
        missing_results
        or unexpected_results
        or invalid_results
        or duplicate_results
        or unexpected_videos
        or empty_videos
        or (require_videos and missing_videos)
        or len(valid_results) != len(samples)
        or overall["annotated_segments"]
        != len(samples) + len(unsupported_annotations)
    )
    accounted = len(valid_results) + len(unsupported_annotations)
    return {
        "schema_version": 2,
        "evaluation_kind": EVALUATION_KIND,
        "profile": PROFILE,
        "official_benchmark": False,
        "held_out": False,
        "training_set_diagnostic": True,
        "checkpoint": checkpoint,
        "run_id": run_manifest.get("run_id") if run_manifest is not None else None,
        "panel_fingerprint": panel["fingerprint"],
        "expected_episode_jobs": len(panel.get("episodes", [])),
        "expected_annotated_segments": len(samples) + len(exclusions),
        "expected_segment_rollouts": len(samples),
        "valid_segment_rollouts": len(valid_results),
        "accounted_annotations": accounted,
        "coverage": accounted / (len(samples) + len(exclusions)),
        "overall": overall,
        "sample_max_steps": max_steps,
        "state_fidelity": {
            "contract": "recorded_native_fields_v1",
            "required": run_manifest is not None and run_manifest.get("state_protocol") == "recorded_native_fields_v1",
            "verified": not incomplete and run_manifest is not None and run_manifest.get("state_protocol") == "recorded_native_fields_v1",
            "eligible_results_verified": overall["metric_eligible"] if run_manifest is not None and run_manifest.get("state_protocol") == "recorded_native_fields_v1" else 0,
            "source_excluded": overall["source_excluded"],
            "unrecorded_physics_caches_or_timers_verified": False,
        },
        "successes": overall["successes"],
        "success_rate": overall["completion_rate"],
        "per_task": per_task,
        "per_skill_description": per_skill_description,
        "per_prompt_label": per_prompt_label,
        "per_label": per_prompt_label,
        "setup_exclusions": setup_exclusions,
        "source_exclusions": source_exclusions,
        "unsupported_annotations": unsupported_annotations,
        "missing_results": missing_results,
        "unexpected_results": unexpected_results,
        "duplicate_results": duplicate_results,
        "invalid_results": invalid_results,
        "video_completeness": {
            "required": len(required_videos) if require_videos else 0,
            "completed": (
                len(required_videos & actual_videos)
                - sum(path in required_videos for path in map(Path, empty_videos))
                if require_videos
                else None
            ),
            "permitted": len(allowed_videos),
        },
        "missing_videos": missing_videos,
        "unexpected_videos": unexpected_videos,
        "empty_videos": empty_videos,
        "complete": not incomplete,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(
        load_panel(args.panel),
        args.results_root,
        checkpoint=args.checkpoint,
        require_videos=args.require_videos,
        run_manifest=json.loads(args.manifest.read_text()),
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if args.require_complete and not summary["complete"] else 0


if __name__ == "__main__":
    sys.exit(main())

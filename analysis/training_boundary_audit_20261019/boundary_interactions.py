#!/usr/bin/env python3
"""Compute the frozen pilot's descriptive, paired boundary interactions.

Input is sample_differences.csv from compare_probes.py, whose upstream checks
establish input/target/noise/normalization pairing. This script checks the CSV's
internal consistency; it cannot reconstruct those hashes from the CSV alone.
No frame-level significance test or confidence interval is estimated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile


GROUPS = ("base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper", "all")
DISTANCES = {"first_action": {64, 32, 16, 1}, "complete_prefix_16": {64, 32, 16}}
BUCKETS = {64: "far_2H", 32: "edge_H", 16: "inside_K", 1: "last_1"}
PRIMARY = "I = mean(delta at 16 and 1 frames before boundary) minus mean(delta at 64 and 32 frames before boundary)."
SECONDARY = "Complete prefix-16 MAE delta at distance 16 minus mean prefix-16 delta at 32 and 64. Exclude distance 1, which has only one valid target."
IDENTITY = ("root_id", "episode_index", "task_index", "task", "task_instance_id")
REQUIRED = {"candidate", *IDENTITY, "frame_index", "boundary_id", "sample_id", "bucket",
            "distance_to_boundary_frames", "window", "action_group", "compared_action_steps",
            "normalization_matches", "baseline_l1", "candidate_l1", "delta_l1",
            "baseline_rmse", "candidate_rmse", "delta_rmse", "l1_outcome", "rmse_outcome"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get("status") != "frozen_before_retrieval_of_any_gpu_result"
            or plan.get("primary_contrast") != PRIMARY or plan.get("secondary_contrast") != SECONDARY):
        raise ValueError("Plan does not specify the supported, frozen pilot contrasts")


def number(value, label, *, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid number: {label}") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"Nonfinite or negative error: {label}")
    return result


def integer(value, label, *, minimum=0):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer: {label}") from error
    if isinstance(value, bool) or str(result) != str(value) or result < minimum:
        raise ValueError(f"Invalid integer: {label}")
    return result


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate_rows(raw_rows):
    if not raw_rows:
        raise ValueError("Empty sample differences")
    rows, seen = [], set()
    boundary_identity, sample_identity, physical_samples, task_names = {}, {}, {}, {}
    baselines, candidate_keys, strata = {}, defaultdict(set), defaultdict(set)
    for index, source in enumerate(raw_rows, start=2):
        if not isinstance(source, dict) or not REQUIRED <= source.keys():
            raise ValueError(f"Missing CSV fields at row {index}")
        row = dict(source)
        for field in ("candidate", "root_id", "task", "boundary_id", "sample_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"Empty identity {field} at row {index}")
        for field in ("episode_index", "frame_index", "task_index", "task_instance_id",
                      "distance_to_boundary_frames", "compared_action_steps"):
            row[field] = integer(row[field], f"row {index}.{field}")
        window, group, distance = row["window"], row["action_group"], row["distance_to_boundary_frames"]
        if window not in DISTANCES or group not in GROUPS or distance not in DISTANCES[window]:
            raise ValueError(f"Invalid window/group/distance at row {index}")
        if row["bucket"] != BUCKETS[distance] or row["compared_action_steps"] != (1 if window == "first_action" else 16):
            raise ValueError(f"Bucket or action-step mismatch at row {index}")
        if str(row["normalization_matches"]).lower() != "true":
            raise ValueError(f"Normalization must match at row {index}")
        for metric in ("l1", "rmse"):
            for prefix in ("baseline", "candidate", "delta"):
                field = f"{prefix}_{metric}"
                row[field] = number(row[field], f"row {index}.{field}", nonnegative=prefix != "delta")
            if not close(row[f"delta_{metric}"], row[f"candidate_{metric}"] - row[f"baseline_{metric}"]):
                raise ValueError(f"Candidate-baseline delta mismatch at row {index}: {metric}")
            if row[f"{metric}_outcome"] not in {"win", "tie", "loss"}:
                raise ValueError(f"Invalid outcome at row {index}: {metric}")
        for prefix in ("baseline", "candidate"):
            if row[f"{prefix}_rmse"] + 1e-10 < row[f"{prefix}_l1"]:
                raise ValueError(f"RMSE below L1 at row {index}")

        boundary = (row["root_id"], row["boundary_id"])
        identity = tuple(row[field] for field in IDENTITY) + (row["frame_index"] + distance,)
        if boundary in boundary_identity and boundary_identity[boundary] != identity:
            raise ValueError(f"Inconsistent boundary identity: {boundary}")
        boundary_identity[boundary] = identity
        task_key = (row["root_id"], row["task_index"])
        if task_key in task_names and task_names[task_key] != row["task"]:
            raise ValueError(f"Inconsistent task identity: {task_key}")
        task_names[task_key] = row["task"]
        sample = (row["root_id"], row["sample_id"])
        sample_info = identity + (row["boundary_id"], row["frame_index"], distance)
        if sample in sample_identity and sample_identity[sample] != sample_info:
            raise ValueError(f"Inconsistent sample identity: {sample}")
        sample_identity[sample] = sample_info
        physical = (row["root_id"], row["episode_index"], row["frame_index"])
        if physical in physical_samples and physical_samples[physical] != sample:
            raise ValueError(f"Multiple sample identities for one physical frame: {physical}")
        physical_samples[physical] = sample

        key = (row["candidate"], *boundary, window, group, distance)
        if key in seen:
            raise ValueError(f"Duplicate distance: {key}")
        seen.add(key)
        strata[key[:-1]].add(distance)
        logical = (*boundary, window, group, distance)
        candidate_keys[row["candidate"]].add(logical)
        baseline = (row["baseline_l1"], row["baseline_rmse"])
        if logical in baselines and not all(close(a, b) for a, b in zip(baselines[logical], baseline)):
            raise ValueError(f"Inconsistent baseline across candidates: {logical}")
        baselines[logical] = baseline
        rows.append(row)

    expected = {(root, boundary, window, group, distance)
                for root, boundary in boundary_identity for window, distances in DISTANCES.items()
                for group in GROUPS for distance in distances}
    for candidate, keys in candidate_keys.items():
        if keys != expected:
            missing = sorted(expected - keys)
            raise ValueError(f"Missing distances or groups for {candidate}: {missing[:3]}")
    for key, distances in strata.items():
        if distances != DISTANCES[key[-2]]:
            raise ValueError(f"Incomplete distance set: {key}")
    return rows


def describe(values, tolerance):
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values), "range": [min(values), max(values)],
            "positive": sum(value > tolerance for value in values),
            "negative": sum(value < -tolerance for value in values),
            "ties": sum(abs(value) <= tolerance for value in values)}


def analyze(raw_rows, plan, *, tie_tolerance=1e-12):
    validate_plan(plan)
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    rows = validate_rows(raw_rows)
    strata = defaultdict(dict)
    for row in rows:
        strata[(row["candidate"], row["root_id"], row["boundary_id"], row["window"], row["action_group"])][row["distance_to_boundary_frames"]] = row
    boundaries = []
    for key, samples in sorted(strata.items()):
        candidate, root, boundary, window, group = key
        reference = samples[64]
        near = (16, 1) if window == "first_action" else (16,)
        result = {"candidate": candidate, **{field: reference[field] for field in IDENTITY},
                  "boundary_id": boundary, "boundary_frame": reference["frame_index"] + 64,
                  "window": window, "action_group": group,
                  "contrast_role": "primary" if window == "first_action" and group != "all" else "secondary"}
        for prefix in ("baseline", "candidate", "delta"):
            result[f"{prefix}_far_mean_l1"] = statistics.fmean(samples[d][f"{prefix}_l1"] for d in (64, 32))
            result[f"{prefix}_near_mean_l1"] = statistics.fmean(samples[d][f"{prefix}_l1"] for d in near)
        result["interaction"] = result["delta_near_mean_l1"] - result["delta_far_mean_l1"]
        result["direction"] = ("positive" if result["interaction"] > tie_tolerance else
                               "negative" if result["interaction"] < -tie_tolerance else "tie")
        boundaries.append(result)
    by_task = defaultdict(list)
    for row in boundaries:
        by_task[(row["candidate"], row["task"], row["window"], row["action_group"])].append(row)
    tasks = []
    for key, records in sorted(by_task.items()):
        candidate, task, window, group = key
        tasks.append({"candidate": candidate, "task": task, "window": window, "action_group": group,
                      "unit": "boundary", "boundaries": len(records),
                      "episodes": len({(r["root_id"], r["episode_index"]) for r in records}),
                      **describe([r["interaction"] for r in records], tie_tolerance)})
    by_group = defaultdict(list)
    for row in tasks:
        by_group[(row["candidate"], row["window"], row["action_group"])].append(row)
    groups = []
    for key, records in sorted(by_group.items()):
        candidate, window, group = key
        groups.append({"candidate": candidate, "window": window, "action_group": group,
                       "contrast_role": "primary" if window == "first_action" and group != "all" else "secondary",
                       "unit": "equally weighted task mean", "tasks": len(records),
                       "boundaries": sum(r["boundaries"] for r in records),
                       **describe([r["mean"] for r in records], tie_tolerance)})
    return {"schema_version": 1, "analysis_status": "descriptive_preregistered_pilot_contrasts",
            "metric": "physical-action MAE (L1), candidate minus baseline",
            "interaction_sign": "Positive means relatively worse near annotated boundaries on these expert inputs.",
            "summary_weighting": "Average boundary interactions within each task, then weight task means equally.",
            "direction_count_unit": "Boundaries in task summaries; task means in group summaries.",
            "limitations": ["No p-values or confidence intervals; frames are not independent replicates.",
                            "Annotation boundaries are not verified physical completion events.",
                            "These offline contrasts do not establish a causal or closed-loop performance effect.",
                            "CSV internal checks do not independently revalidate upstream input, target or noise hashes."],
            "tie_tolerance": tie_tolerance, "plan": plan,
            "counts": {"input_rows": len(rows), "candidates": len({r["candidate"] for r in rows}),
                       "boundaries": len({(r["root_id"], r["boundary_id"]) for r in rows}),
                       "tasks": len({r["task"] for r in rows})},
            "boundary_interactions": boundaries, "task_interactions": tasks, "group_summaries": groups}


def publish_directory(stage, destination):
    """Atomically publish without replacing even an empty directory (Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-overwrite directory publication requires renameat2")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f"Refusing to overwrite output directory: {destination}")
        raise OSError(code, os.strerror(code), str(destination))


def write_outputs(output_dir, report):
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name in ("boundary_interactions", "task_interactions", "group_summaries"):
            records = report[name]
            with (stage / f"{name}.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("pilot_analysis_plan.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)
    try:
        data, plan_data = args.input_csv.read_bytes(), args.plan.read_bytes()
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicate CSV column names")
        report = analyze(list(reader), json.loads(plan_data), tie_tolerance=args.tie_tolerance)
        report["provenance"] = {"input_csv": {"path": str(args.input_csv.resolve()), "sha256": digest(data)},
                                "plan": {"path": str(args.plan.resolve()), "sha256": digest(plan_data)},
                                "implementation_sha256": digest(Path(__file__).read_bytes())}
        write_outputs(args.output_dir, report)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({**report["counts"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

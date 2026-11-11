#!/usr/bin/env python3
"""Strict paired comparison of cached eval_openloop boundary-probe results.

Example:
  python compare_probes.py --baseline baseline.json --candidate expert=expert.json \
      --candidate mixed=mixed.json --output-dir comparison

All statistics describe offline demonstration action error. They are exploratory
and do not establish a change in closed-loop task success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


GROUP_DIMS = {"all": 23, "base": 3, "torso": 4, "left_arm": 7, "left_gripper": 1,
              "right_arm": 7, "right_gripper": 1}
BUCKETS = {"far_2H", "edge_H", "inside_K", "last_1"}
WINDOWS = {"first_action": 1, "complete_prefix_16": 16}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value, location):
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {location}")
    return value


def require_string(value, location):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid string: {location}")
    return value


def require_int(value, location, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Missing or invalid integer: {location}")
    return value


def require_digest(value, location):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Missing or invalid SHA256: {location}")
    return value


def require_number(value, location):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing, nonfinite or negative error: {location}")
    return float(value)


def metric_at(diagnostics, window, group, steps, location):
    block = require_mapping(diagnostics.get(window), f"{location}.{window}")
    methods = (block.get("metrics") if group == "all" else
               require_mapping(block.get("per_action_group"), f"{location}.{window}.per_action_group").get(group))
    methods = require_mapping(methods, f"{location}.{window}.{group}")
    model = require_mapping(methods.get("model"), f"{location}.{window}.{group}.model")
    l1 = require_number(model.get("l1"), f"{location}.{window}.{group}.l1")
    rmse = require_number(model.get("rmse"), f"{location}.{window}.{group}.rmse")
    if rmse + 1e-10 < l1:
        raise ValueError(f"RMSE is below L1 in {location}.{window}.{group}")
    if model.get("valid_action_steps") != steps or type(model.get("valid_action_steps")) is not int:
        raise ValueError(f"Unequal valid horizon in {location}.{window}.{group}")
    if model.get("scalar_values") != steps * GROUP_DIMS[group] or type(model.get("scalar_values")) is not int:
        raise ValueError(f"Wrong action dimensionality in {location}.{window}.{group}")
    return {"l1": l1, "rmse": rmse}


def validate_result(result, name):
    require_mapping(result, name)
    horizon = require_int(result.get("horizon"), f"{name}.horizon", 16)
    if result.get("conditioning_mode") != "provided":
        raise ValueError(f"{name} requires explicit provided-subtask conditioning")
    require_int(result.get("max_token_len"), f"{name}.max_token_len", 1)
    require_string(result.get("repo_id"), f"{name}.repo_id")
    noise = require_mapping(result.get("action_noise"), f"{name}.action_noise")
    require_int(noise.get("seed"), f"{name}.action_noise.seed")
    require_string(noise.get("algorithm"), f"{name}.action_noise.algorithm")
    artifacts = require_mapping(result.get("artifacts"), f"{name}.artifacts")
    for artifact in ("norm_stats", "checkpoint", "sample_keys_file", "segments_file"):
        record = require_mapping(artifacts.get(artifact), f"{name}.artifacts.{artifact}")
        require_digest(record.get("sha256"), f"{name}.artifacts.{artifact}.sha256")
    norm = require_mapping(result.get("normalization"), f"{name}.normalization")
    require_string(norm.get("asset_id"), f"{name}.normalization.asset_id")
    if type(norm.get("use_quantile_norm")) is not bool:
        raise ValueError(f"{name} must record the actual normalization mode")
    samples = result.get("sample_keys")
    if not isinstance(samples, list) or not samples or result.get("samples") != len(samples):
        raise ValueError(f"{name} sample count disagrees with its nonempty sample list")
    indexed = {}
    for row in samples:
        require_mapping(row, f"{name}.sample")
        key = (require_int(row.get("episode_index"), "sample.episode_index"),
               require_int(row.get("frame_index"), "sample.frame_index"))
        if key in indexed:
            raise ValueError(f"Duplicate episode/frame in {name}: {key}")
        location = f"{name}[{key}]"
        require_int(row.get("task_index"), f"{location}.task_index")
        require_int(row.get("global_frame_index"), f"{location}.global_frame_index")
        require_string(row.get("task"), f"{location}.task")
        require_string(row.get("subtask"), f"{location}.subtask")
        valid = require_int(row.get("valid_action_steps"), f"{location}.valid_action_steps", 1)
        for field in ("policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
            require_digest(row.get(field), f"{location}.{field}")
        probe = require_mapping(row.get("probe_metadata"), f"{location}.probe_metadata")
        for field in ("root_id", "boundary_id", "sample_id", "task_name"):
            require_string(probe.get(field), f"{location}.probe_metadata.{field}")
        require_int(probe.get("task_instance_id"), f"{location}.probe_metadata.task_instance_id")
        if probe.get("task_name") != row["task"] or probe.get("bucket") not in BUCKETS:
            raise ValueError(f"Invalid task or bucket in {location}")
        distance = require_int(probe.get("distance_to_boundary_frames"), f"{location}.distance", 1)
        expected_distance = {"far_2H": 2 * horizon, "edge_H": horizon, "inside_K": 16, "last_1": 1}[probe["bucket"]]
        if distance != expected_distance or valid != min(horizon, distance):
            raise ValueError(f"Bucket distance/valid horizon disagrees in {location}")
        segment_end = require_int(probe.get("segment_end"), f"{location}.probe_metadata.segment_end", 1)
        if segment_end - key[1] != distance or row.get("segment_end") != key[1] + valid:
            raise ValueError(f"Segment bounds disagree in {location}")
        diagnostics = require_mapping(row.get("error_diagnostics"), f"{location}.error_diagnostics")
        prefix = require_mapping(diagnostics.get("complete_prefix_16"), f"{location}.complete_prefix_16")
        eligible = valid >= 16
        if type(prefix.get("samples")) is not int or prefix["samples"] != int(eligible):
            raise ValueError(f"Complete-prefix inclusion disagrees with available targets in {location}")
        metrics = {}
        for group in GROUP_DIMS:
            metrics[("first_action", group)] = metric_at(diagnostics, "first_action", group, 1, location)
            if eligible:
                metrics[("complete_prefix_16", group)] = metric_at(diagnostics, "complete_prefix_16", group, 16, location)
        if not eligible:
            if prefix.get("metrics") != {} or prefix.get("per_action_group") != {group: {} for group in GROUP_DIMS if group != "all"}:
                raise ValueError(f"Incomplete prefix must not contain metrics in {location}")
        indexed[key] = (row, metrics)
    return indexed


def paired_summary(rows, metric, tolerance):
    baseline = [row[f"baseline_{metric}"] for row in rows]
    candidate = [row[f"candidate_{metric}"] for row in rows]
    differences = [row[f"delta_{metric}"] for row in rows]
    return {"baseline_mean": statistics.fmean(baseline), "candidate_mean": statistics.fmean(candidate),
            "baseline_median": statistics.median(baseline), "candidate_median": statistics.median(candidate),
            "mean_delta": statistics.fmean(differences), "median_delta": statistics.median(differences),
            "wins": sum(delta < -tolerance for delta in differences),
            "ties": sum(abs(delta) <= tolerance for delta in differences),
            "losses": sum(delta > tolerance for delta in differences)}


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower) if upper != lower else values[lower]


def bootstrap_episode_deltas(rows, replicates, seed):
    clusters = defaultdict(list)
    for row in rows:
        clusters[(row["root_id"], row["episode_index"])].append(row)
    if replicates == 0 or len(clusters) < 2:
        return {"computed": False, "reason": "disabled" if replicates == 0 else "fewer_than_two_episodes"}
    values = list(clusters.values())
    rng = random.Random(seed)
    samples = {"l1": [], "rmse": []}
    for _ in range(replicates):
        selected = [values[rng.randrange(len(values))] for _ in values]
        denominator = sum(len(cluster) for cluster in selected)
        for metric in samples:
            samples[metric].append(sum(row[f"delta_{metric}"] for cluster in selected for row in cluster) / denominator)
    return {"computed": True, "unit": "(root_id, episode_index)", "replicates": replicates,
            "exploratory": True, "percentile_95_interval_for_mean_delta": {
                metric: [quantile(values, .025), quantile(values, .975)] for metric, values in samples.items()}}


def compare_results(baseline, candidate, *, label="candidate", allow_different_normalization=False,
                    tie_tolerance=1e-12, bootstrap_replicates=0, bootstrap_seed=0):
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0 or bootstrap_replicates < 0:
        raise ValueError("Tie tolerance and bootstrap count must be nonnegative")
    base_rows = validate_result(baseline, "baseline")
    candidate_rows = validate_result(candidate, label)
    if base_rows.keys() != candidate_rows.keys():
        missing = sorted(base_rows.keys() - candidate_rows.keys())
        extra = sorted(candidate_rows.keys() - base_rows.keys())
        raise ValueError(f"Exact sample set mismatch for {label}: missing={missing[:5]}, extra={extra[:5]}")
    for field in ("repo_id", "conditioning_mode", "horizon", "max_token_len"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Evaluation configuration differs for {label}: {field}")
    for field in ("seed", "algorithm"):
        if baseline["action_noise"][field] != candidate["action_noise"][field]:
            raise ValueError(f"Action noise configuration differs for {label}: {field}")
    for artifact in ("sample_keys_file", "segments_file"):
        if baseline["artifacts"][artifact]["sha256"] != candidate["artifacts"][artifact]["sha256"]:
            raise ValueError(f"Frozen {artifact} content differs for {label}")
    norm_hashes = [result["artifacts"]["norm_stats"]["sha256"] for result in (baseline, candidate)]
    modes = [result["normalization"]["use_quantile_norm"] for result in (baseline, candidate)]
    same_normalization = norm_hashes[0] == norm_hashes[1] and modes[0] == modes[1]
    if not same_normalization and not allow_different_normalization:
        raise ValueError(f"Normalization differs for {label}; use --allow-different-normalization only for an explicitly confounded comparison")
    differences = []
    for key in sorted(base_rows):
        base, base_metrics = base_rows[key]
        current, current_metrics = candidate_rows[key]
        for field in ("task_index", "task", "subtask", "global_frame_index", "segment_end", "valid_action_steps", "probe_metadata",
                      "policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
            if base[field] != current[field]:
                raise ValueError(f"Paired sample {key} differs for {label}: {field}")
        probe = base["probe_metadata"]
        for (window, group), left in base_metrics.items():
            right = current_metrics[(window, group)]
            row = {"candidate": label, "root_id": probe["root_id"], "episode_index": key[0], "frame_index": key[1],
                   "task_index": base["task_index"], "task": base["task"], "task_instance_id": probe["task_instance_id"],
                   "boundary_id": probe["boundary_id"], "sample_id": probe["sample_id"], "bucket": probe["bucket"],
                   "distance_to_boundary_frames": probe["distance_to_boundary_frames"], "window": window,
                   "action_group": group, "compared_action_steps": WINDOWS[window],
                   "normalization_matches": same_normalization}
            for metric in ("l1", "rmse"):
                delta = right[metric] - left[metric]
                row.update({f"baseline_{metric}": left[metric], f"candidate_{metric}": right[metric], f"delta_{metric}": delta,
                            f"{metric}_outcome": "win" if delta < -tie_tolerance else "loss" if delta > tie_tolerance else "tie"})
            differences.append(row)
    groups = defaultdict(list)
    for row in differences:
        for scope, task, bucket in (("overall", None, None), ("bucket", None, row["bucket"]),
                                    ("task_bucket", row["task"], row["bucket"])):
            groups[(scope, task, bucket, row["window"], row["action_group"])].append(row)
    grouped = []
    for key in sorted(groups, key=str):
        scope, task, bucket, window, group = key
        rows = groups[key]
        seed = int.from_bytes(hashlib.sha256(f"{bootstrap_seed}:{key}".encode()).digest(), "big")
        grouped.append({"scope": scope, "task": task, "bucket": bucket, "window": window, "action_group": group,
                        "samples": len(rows), "episodes": len({(r["root_id"], r["episode_index"]) for r in rows}),
                        "boundaries": len({r["boundary_id"] for r in rows}),
                        "l1": paired_summary(rows, "l1", tie_tolerance), "rmse": {
                            **paired_summary(rows, "rmse", tie_tolerance),
                            "baseline_pooled_rmse": math.sqrt(statistics.fmean(r["baseline_rmse"] ** 2 for r in rows)),
                            "candidate_pooled_rmse": math.sqrt(statistics.fmean(r["candidate_rmse"] ** 2 for r in rows))},
                        "episode_bootstrap": bootstrap_episode_deltas(rows, bootstrap_replicates, seed)})
    summary = {"candidate": label, "analysis_status": "exploratory_offline_diagnostic",
               "note": "Paired demonstration action errors, not closed-loop scores or causal evidence of task success. Small pilot intervals are exploratory.",
               "delta_sign": "candidate minus baseline; negative indicates lower action error",
               "exact_samples": len(base_rows), "episodes": len({(r["probe_metadata"]["root_id"], r["episode_index"]) for r, _ in base_rows.values()}),
               "tasks": len({r["task"] for r, _ in base_rows.values()}),
               "normalization": {"matches": same_normalization, "confounded": not same_normalization,
                                 "baseline": {**baseline["normalization"], "sha256": norm_hashes[0]},
                                 "candidate": {**candidate["normalization"], "sha256": norm_hashes[1]}},
               "pairing_checks": "exact sample set, frozen identity/config/annotation, raw input, target and noise SHA256 verified",
               "window_notes": {"first_action": "Same position zero for all four distance buckets.",
                                "complete_prefix_16": "Only fully valid first 16 steps; no last_1 samples included."},
               "tie_tolerance": tie_tolerance, "groups": grouped}
    return summary, differences


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty comparison")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="PATH or LABEL=PATH; may be repeated")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-different-normalization", action="store_true")
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = [args.output_dir / name for name in ("summary.json", "sample_differences.csv", "grouped_differences.csv")]
    if any(path.exists() for path in paths):
        parser.error("Refusing to overwrite existing comparison artifacts")
    baseline = json.loads(args.baseline.read_text())
    summaries, samples, flat_groups, labels = [], [], [], set()
    try:
        for argument in args.candidate:
            label, separator, filename = argument.partition("=")
            path = Path(filename if separator else argument)
            label = label if separator else path.stem
            if not label or label in labels:
                raise ValueError("Candidate labels must be nonempty and unique")
            labels.add(label)
            summary, rows = compare_results(baseline, json.loads(path.read_text()), label=label,
                                           allow_different_normalization=args.allow_different_normalization,
                                           tie_tolerance=args.tie_tolerance, bootstrap_replicates=args.bootstrap_replicates,
                                           bootstrap_seed=args.bootstrap_seed)
            summary["inputs"] = {"baseline": {"path": str(args.baseline.resolve()), "sha256": sha256_file(args.baseline)},
                                 "candidate": {"path": str(path.resolve()), "sha256": sha256_file(path)}}
            summaries.append(summary)
            samples.extend(rows)
            for group in summary["groups"]:
                row = {"candidate": label, **{key: group[key] for key in
                                             ("scope", "task", "bucket", "window", "action_group", "samples", "episodes", "boundaries")}}
                for metric in ("l1", "rmse"):
                    row.update({f"{metric}_{key}": value for key, value in group[metric].items()})
                flat_groups.append(row)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x") as stream:
        stream.write(json.dumps({"schema_version": 1, "implementation_sha256": sha256_file(__file__),
                                 "comparisons": summaries}, indent=2, allow_nan=False) + "\n")
    write_csv(paths[1], samples)
    write_csv(paths[2], flat_groups)
    print(json.dumps({"comparisons": len(summaries), "exact_samples_each": [row["exact_samples"] for row in summaries],
                      "sample_difference_rows": len(samples), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

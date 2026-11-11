#!/usr/bin/env python3
"""Diagnose whether 42 task-independent affine parameters explain action changes.

Example:
  python action_change_calibration.py --baseline baseline.json \
      --candidate candidate.json --output calibration.json
  python action_change_calibration.py --baseline baseline.json \
      --output direct_expert_calibration.json

Omitting --candidate runs only direct expert supervision. It reads no candidate
result, reports no candidate reference, and omits the SFT-delta analysis.

Frozen protocol: leave one entire task out; use valid positions within the first
min(16, H); fit delta[d] = k[d] * (baseline[d] - hold[d]) + b[d] for the 21
non-gripper coordinates. No-change and constant-bias models are comparators.
An independent direct-expert control fits the same family to GT - baseline,
using only each training fold's baseline export and expert targets. Candidate
actions enter that control only as a held-out error reference, never as targets.
Each sample first receives a mean over its valid positions, samples are averaged
within each task, and tasks receive equal weight. Thus sample weights are equal
within a task; unequal task sizes cannot have both globally equal sample weights
and equal task weights. No task labels enter the regression features.

SSEs are reported separately for each coordinate, never pooled across units.
Group summaries average dimensionless explained fractions only. A delta RMS of
at most 1e-6 in a coordinate's raw action units gives an NA explained fraction.
Raw chunks must cover the requested evaluation horizon H; truncated policy
outputs are rejected because the runner's target hash precedes truncation.
This is an offline diagnostic, not evidence about closed-loop performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

import compare_probes
from eval_cached_probe import eval_openloop


PREFIX_STEPS = 16
MIN_DELTA_RMS = 1e-6
RAW_NAMES = ("prediction", "ground_truth", "hold_action", "valid_mask")
GROUPS = {name: tuple(range(start, end)) for name, (start, end) in
          eval_openloop.ACTION_GROUPS.items() if "gripper" not in name}
DIMENSIONS = tuple(index for indices in GROUPS.values() for index in indices)
EXCLUDED_DIMENSIONS = (14, 22)
MODELS = ("no_change", "constant_bias", "affine")


@dataclass(frozen=True)
class Sample:
    task: str
    episode_index: int
    frame_index: int
    bucket: str
    x: np.ndarray
    delta: np.ndarray


def _raw_arrays(row, horizon, location):
    raw = row.get("raw_actions")
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: missing raw_actions; rerun evaluation with --save-raw-actions")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError(f"{location}: raw_actions requires schema_version=1")
    dtypes = raw.get("dtypes")
    if not isinstance(dtypes, dict) or set(dtypes) != set(RAW_NAMES):
        raise ValueError(f"{location}: raw_actions.dtypes must describe exactly the four arrays")
    arrays = {}
    for name in RAW_NAMES:
        if not isinstance(dtypes[name], str):
            raise ValueError(f"{location}: invalid dtype for {name}")
        try:
            dtype = np.dtype(dtypes[name])
            uncast = np.asarray(raw[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{location}: missing or invalid raw array {name}") from error
        if name == "valid_mask":
            if dtype.kind != "b" or uncast.dtype.kind != "b":
                raise ValueError(f"{location}: valid_mask must contain JSON booleans with boolean dtype")
            expected_shape = (horizon,)
        else:
            if dtype.kind != "f" or uncast.dtype.kind not in "fiu":
                raise ValueError(f"{location}: {name} requires a real floating dtype and numeric values")
            expected_shape = (horizon, 23)
        if uncast.shape != expected_shape:
            raise ValueError(f"{location}: {name} shape {uncast.shape} != {expected_shape}; "
                             "raw chunks must cover the requested horizon; truncated chunks are unsupported")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(uncast, dtype=dtype)
        if not np.isfinite(array).all():
            raise ValueError(f"{location}: nonfinite raw array {name}")
        arrays[name] = array
    expected_hash = compare_probes.require_digest(raw.get("sha256"), f"{location}.raw_actions.sha256")
    if eval_openloop.input_content_sha256(arrays) != expected_hash:
        raise ValueError(f"{location}: raw_actions SHA256 mismatch")
    if eval_openloop.input_content_sha256(arrays["ground_truth"]) != row["ground_truth_sha256"]:
        raise ValueError(f"{location}: raw ground_truth disagrees with ground_truth_sha256")
    valid = arrays["valid_mask"]
    if int(valid.sum()) != row["valid_action_steps"] or not valid[0]:
        raise ValueError(f"{location}: raw valid_mask disagrees with paired valid horizon")
    expected_prefix = int(len(valid) >= 16 and valid[:16].all())
    if row["error_diagnostics"]["complete_prefix_16"]["samples"] != expected_prefix:
        raise ValueError(f"{location}: raw valid_mask disagrees with complete-prefix inclusion")
    # Bind the raw prediction to the diagnostics that compare_results checked.
    for window, positions in (("first_action", np.arange(horizon) == 0),
                              ("complete_prefix_16", np.arange(horizon) < 16)):
        if window == "complete_prefix_16" and not expected_prefix:
            continue
        error = (arrays["prediction"].astype(np.float64) -
                 arrays["ground_truth"].astype(np.float64))[valid & positions]
        for group, indices in {"all": tuple(range(23)), **{
                name: tuple(range(start, end)) for name, (start, end) in
                eval_openloop.ACTION_GROUPS.items()}}.items():
            actual = error[:, indices]
            recorded = compare_probes.metric_at(row["error_diagnostics"], window, group,
                                                len(actual), location)
            for metric, computed in (("l1", np.abs(actual).mean()),
                                     ("rmse", np.sqrt(np.square(actual).mean()))):
                if not math.isclose(recorded[metric], float(computed), rel_tol=1e-6, abs_tol=1e-10):
                    raise ValueError(f"{location}: raw prediction disagrees with {window}.{group}.{metric}")
    return arrays


def paired_samples(baseline, candidate):
    """Run the existing fail-closed pairing before interpreting raw arrays."""
    pairing, _ = compare_probes.compare_results(baseline, candidate)
    current = {(row["episode_index"], row["frame_index"]): row for row in candidate["sample_keys"]}
    samples = []
    for row in baseline["sample_keys"]:
        key = row["episode_index"], row["frame_index"]
        base = _raw_arrays(row, baseline["horizon"], f"baseline{key}")
        changed = _raw_arrays(current[key], candidate["horizon"], f"candidate{key}")
        for name in ("ground_truth", "hold_action", "valid_mask"):
            if base[name].dtype != changed[name].dtype or not np.array_equal(base[name], changed[name]):
                raise ValueError(f"Paired raw arrays differ for {key}: {name}")
        selected = base["valid_mask"] & (np.arange(baseline["horizon"]) < PREFIX_STEPS)
        if not selected.any():
            raise ValueError(f"No valid first-prefix positions for {key}")
        prediction = base["prediction"][selected].astype(np.float64)[:, DIMENSIONS]
        hold = base["hold_action"][selected].astype(np.float64)[:, DIMENSIONS]
        delta = changed["prediction"][selected].astype(np.float64)[:, DIMENSIONS] - prediction
        if not np.isfinite(prediction - hold).all() or not np.isfinite(delta).all():
            raise ValueError(f"Nonfinite derived action change for {key}")
        samples.append(Sample(row["task"], *key, row["probe_metadata"]["bucket"], prediction - hold, delta))
    samples.sort(key=lambda sample: (sample.task, sample.episode_index, sample.frame_index))
    return pairing, samples


def direct_expert_samples(baseline):
    """Construct direct-supervision inputs without accepting a candidate result."""
    compare_probes.validate_result(baseline, "baseline")
    samples = []
    for row in baseline["sample_keys"]:
        key = row["episode_index"], row["frame_index"]
        raw = _raw_arrays(row, baseline["horizon"], f"baseline{key}")
        selected = raw["valid_mask"] & (np.arange(baseline["horizon"]) < PREFIX_STEPS)
        prediction = raw["prediction"][selected].astype(np.float64)[:, DIMENSIONS]
        hold = raw["hold_action"][selected].astype(np.float64)[:, DIMENSIONS]
        target = raw["ground_truth"][selected].astype(np.float64)[:, DIMENSIONS] - prediction
        samples.append(Sample(row["task"], *key, row["probe_metadata"]["bucket"], prediction - hold, target))
    return sorted(samples, key=lambda sample: (sample.task, sample.episode_index, sample.frame_index))


def _stack(samples):
    counts = Counter(sample.task for sample in samples)
    if not counts:
        raise ValueError("Cannot fit or score an empty sample collection")
    x = np.concatenate([sample.x for sample in samples])
    delta = np.concatenate([sample.delta for sample in samples])
    weights = np.concatenate([np.full(len(sample.x), 1 / (len(counts) * counts[sample.task] * len(sample.x)))
                              for sample in samples])
    return x, delta, weights


def fit_models(samples):
    """Fit one shared 42-parameter family, using training tasks only."""
    x, delta, weights = _stack(samples)
    x_mean = np.sum(weights[:, None] * x, axis=0)
    delta_mean = np.sum(weights[:, None] * delta, axis=0)
    centered_x, centered_delta = x - x_mean, delta - delta_mean
    variance = np.sum(weights[:, None] * np.square(centered_x), axis=0)
    covariance = np.sum(weights[:, None] * centered_x * centered_delta, axis=0)
    # An unvarying training feature identifies only an intercept; use the
    # minimum-slope solution rather than inventing a held-out-task dependency.
    cutoff = np.finfo(np.float64).eps * np.maximum(1.0, np.square(x_mean))
    slope = np.divide(covariance, variance, out=np.zeros_like(covariance), where=variance > cutoff)
    intercept = delta_mean - slope * x_mean
    return {"no_change": (np.zeros(len(DIMENSIONS)), np.zeros(len(DIMENSIONS))),
            "constant_bias": (np.zeros(len(DIMENSIONS)), delta_mean),
            "affine": (slope, intercept)}, [DIMENSIONS[index] for index in np.flatnonzero(variance <= cutoff)]


def score_predictions(samples, predictions):
    """Every returned SSE is normalized by hierarchical task/sample weights."""
    _, delta, weights = _stack(samples)
    denominator = np.sum(weights[:, None] * np.square(delta), axis=0)
    if not np.isfinite(denominator).all():
        raise ValueError("Nonfinite SSE; action magnitudes are too large")
    residuals = {}
    for name in MODELS:
        predicted = np.concatenate(predictions[name])
        residuals[name] = np.sum(weights[:, None] * np.square(delta - predicted), axis=0)
        if not np.isfinite(residuals[name]).all():
            raise ValueError("Nonfinite residual SSE; action magnitudes are too large")
    groups = []
    offset = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
    for group, dimensions in GROUPS.items():
        coordinates = []
        for dimension in dimensions:
            index = offset[dimension]
            signal = float(denominator[index])
            low_signal = signal <= MIN_DELTA_RMS ** 2
            models = {name: {"sse_residual": float(residuals[name][index]),
                             "explained_fraction": None if low_signal else float(1 - residuals[name][index] / signal)}
                      for name in MODELS}
            coordinates.append({"action_dimension": dimension, "sse_delta": signal,
                                "delta_rms": math.sqrt(signal), "status": "NA_low_signal" if low_signal else "ok",
                                "models": models})
        macro = {}
        for name in MODELS:
            values = [row["models"][name]["explained_fraction"] for row in coordinates
                      if row["models"][name]["explained_fraction"] is not None]
            macro[name] = {"explained_fraction": float(np.mean(values)) if values else None,
                           "coordinates_with_signal": len(values)}
        groups.append({"action_group": group, "coordinates": coordinates,
                       "dimensionless_macro": macro})
    return {"samples": len(samples), "tasks": len({sample.task for sample in samples}),
            "valid_prefix_positions": sum(len(sample.x) for sample in samples), "groups": groups}


def cross_validated_predictions(samples):
    """Fit each fold without accessing any held-out target during fitting."""
    tasks = sorted({sample.task for sample in samples})
    if len(tasks) < 2:
        raise ValueError("Held-out-task CV requires at least two distinct tasks; frame splits are forbidden")
    folds, out_of_task = [], {name: {} for name in MODELS}
    for task in tasks:
        train = [sample for sample in samples if sample.task != task]
        test = [sample for sample in samples if sample.task == task]
        models, fallback = fit_models(train)
        predictions = {name: [sample.x * slope + intercept for sample in test]
                       for name, (slope, intercept) in models.items()}
        for name in MODELS:
            for sample, prediction in zip(test, predictions[name]):
                out_of_task[name][(sample.task, sample.episode_index, sample.frame_index)] = prediction
        folds.append({"held_out_task": task, "training_tasks": sorted(set(tasks) - {task}),
                      "training_samples": len(train), "test_samples": len(test),
                      "affine_constant_feature_fallback_dimensions": fallback,
                      "coefficients": {name: [{"action_dimension": dimension, "k": float(slope[index]),
                                               "b": float(intercept[index])}
                                              for index, dimension in enumerate(DIMENSIONS)]
                                       for name, (slope, intercept) in models.items()},
                      "held_out_scores": score_predictions(test, predictions)})
    pooled = {name: [out_of_task[name][(sample.task, sample.episode_index, sample.frame_index)]
                     for sample in samples] for name in MODELS}
    return folds, pooled


def score_expert_predictions(samples, predictions, candidate_deltas=None):
    """Score action corrections against GT, using candidate only as a reference."""
    _, target, weights = _stack(samples)
    corrections = {"baseline": predictions["no_change"],
                   "hold": [-sample.x for sample in samples],
                   "direct_constant_bias": predictions["constant_bias"],
                   "direct_affine": predictions["affine"]}
    if candidate_deltas is not None:
        corrections["candidate"] = candidate_deltas
    metrics = {}
    for name, values in corrections.items():
        error = np.concatenate(values) - target
        mse = np.sum(weights[:, None] * np.square(error), axis=0)
        mae = np.sum(weights[:, None] * np.abs(error), axis=0)
        if not np.isfinite(mse).all() or not np.isfinite(mae).all():
            raise ValueError("Nonfinite expert error; action magnitudes are too large")
        metrics[name] = mse, mae
    offset = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
    groups = []
    for group, dimensions in GROUPS.items():
        coordinates = []
        for dimension in dimensions:
            index = offset[dimension]
            baseline_mse = float(metrics["baseline"][0][index])
            low_signal = baseline_mse <= MIN_DELTA_RMS ** 2
            models = {name: {"gt_mse": float(mse[index]), "gt_mae": float(mae[index]),
                             "mse_reduction_fraction_vs_baseline": None if low_signal else
                             float(1 - mse[index] / baseline_mse)}
                      for name, (mse, mae) in metrics.items()}
            coordinates.append({"action_dimension": dimension,
                                "baseline_gt_rms": math.sqrt(baseline_mse),
                                "status": "NA_low_baseline_gt_error" if low_signal else "ok", "models": models})
        macro = {}
        for name in metrics:
            values = [row["models"][name]["mse_reduction_fraction_vs_baseline"] for row in coordinates
                      if row["models"][name]["mse_reduction_fraction_vs_baseline"] is not None]
            macro[name] = {"mse_reduction_fraction_vs_baseline": float(np.mean(values)) if values else None,
                           "coordinates_with_signal": len(values)}
        groups.append({"action_group": group, "coordinates": coordinates, "dimensionless_macro": macro})
    return {"samples": len(samples), "tasks": len({sample.task for sample in samples}),
            "valid_prefix_positions": sum(len(sample.x) for sample in samples), "groups": groups}


def direct_expert_analysis(baseline, candidate_samples=None):
    # Finish all fitting/prediction from baseline alone before using candidate
    # outputs for the separate reference errors below.
    samples = direct_expert_samples(baseline)
    folds, predictions = cross_validated_predictions(samples)
    candidate_deltas = None
    if candidate_samples is not None:
        by_key = {(sample.task, sample.episode_index, sample.frame_index): sample.delta for sample in candidate_samples}
        candidate_deltas = [by_key[(sample.task, sample.episode_index, sample.frame_index)] for sample in samples]
    for fold in folds:
        indices = [index for index, sample in enumerate(samples) if sample.task == fold["held_out_task"]]
        fold.pop("held_out_scores")
        fold["held_out_gt_scores"] = score_expert_predictions(
            [samples[index] for index in indices],
            {name: [predictions[name][index] for index in indices] for name in MODELS},
            None if candidate_deltas is None else [candidate_deltas[index] for index in indices])
    bucket_scores = []
    for bucket in sorted({sample.bucket for sample in samples}):
        indices = [index for index, sample in enumerate(samples) if sample.bucket == bucket]
        bucket_scores.append({"bucket": bucket, **score_expert_predictions(
            [samples[index] for index in indices],
            {name: [predictions[name][index] for index in indices] for name in MODELS},
            None if candidate_deltas is None else [candidate_deltas[index] for index in indices])})
    return {"protocol": {"target": "ground_truth - baseline_prediction",
                         "calibrated_action": "baseline_prediction + k * (baseline_prediction - hold_action) + b",
                         "fit_inputs": "training-fold baseline prediction, hold_action and ground_truth only",
                         "candidate_usage": ("not supplied or read; no candidate reference" if candidate_samples is None else
                                             "held-out error reference only; excluded from fitting and direct predictions"),
                         "cross_validation": "leave_one_entire_task_out",
                         "prefix_steps": PREFIX_STEPS,
                         "weighting": "same task/sample/valid-position hierarchical weights as the top-level protocol",
                         "parameter_counts": {"no_change": 0, "hold": 0, "constant_bias": 21, "affine": 42},
                         "low_signal_rule": "baseline GT RMS <= 1e-6: reduction fraction is NA; MSE/MAE remain reported"},
            "out_of_task_gt_scores": score_expert_predictions(samples, predictions, candidate_deltas),
            "bucket_gt_scores": bucket_scores, "folds": folds}


def analyze_baseline(baseline):
    """Run the same direct control without requiring or inventing a candidate."""
    direct = direct_expert_analysis(baseline)
    samples = baseline["sample_keys"]
    return {"schema_version": 1, "analysis_status": "offline_direct_expert_diagnostic_only",
            "input_mode": "baseline_only",
            "note": "Direct expert-supervision diagnostic only; no SFT-delta, candidate reference, or closed-loop conclusion.",
            "protocol": {**direct["protocol"],
                         "feature": "baseline_prediction - hold_action",
                         "weighting": "mean valid positions per sample, mean samples within task, mean tasks",
                         "error_units": "separate raw action units per coordinate; never pooled across dimensions",
                         "group_macro": "equal-coordinate mean of dimensionless reduction fractions, excluding NA",
                         "min_delta_rms": MIN_DELTA_RMS,
                         "excluded_dimensions": list(EXCLUDED_DIMENSIONS),
                         "excluded_reason": "gripper hold_action units are incompatible with action command units"},
            "baseline_validation": {"exact_samples": len(samples),
                                    "tasks": len({row["task"] for row in samples}),
                                    "episodes": len({(row["probe_metadata"]["root_id"], row["episode_index"]) for row in samples}),
                                    "checks": "baseline schema, raw dtype/shape/content hash, GT hash, mask, and metric binding"},
            "direct_expert_supervision": direct}


def analyze(baseline, candidate=None):
    if candidate is None:
        return analyze_baseline(baseline)
    pairing, samples = paired_samples(baseline, candidate)
    folds, pooled = cross_validated_predictions(samples)
    bucket_scores = []
    for bucket in sorted({sample.bucket for sample in samples}):
        indices = [index for index, sample in enumerate(samples) if sample.bucket == bucket]
        bucket_scores.append({"bucket": bucket, **score_predictions([samples[index] for index in indices],
                              {name: [pooled[name][index] for index in indices] for name in MODELS})})
    return {"schema_version": 1, "analysis_status": "offline_action_change_diagnostic_only",
            "note": "Paired action-change and direct expert-supervision diagnostics only; no closed-loop performance or mechanism conclusion.",
            "protocol": {"cross_validation": "leave_one_entire_task_out", "prefix_steps": PREFIX_STEPS,
                         "feature": "baseline_prediction - hold_action", "target": "candidate_prediction - baseline_prediction",
                         "model": "per-coordinate k * feature + b; no task-conditioned parameters",
                         "parameter_counts": {"no_change": 0, "constant_bias": 21, "affine": 42},
                         "weighting": "mean valid positions per sample, mean samples within task, mean tasks",
                         "sse_units": "separate squared raw action units for each coordinate; no pooled all-dimension SSE",
                         "group_macro": "equal-coordinate mean of dimensionless explained fractions, excluding NA",
                         "min_delta_rms": MIN_DELTA_RMS,
                         "excluded_dimensions": list(EXCLUDED_DIMENSIONS),
                         "excluded_reason": "gripper hold_action units are incompatible with action command units"},
            "pairing": {field: pairing[field] for field in ("exact_samples", "episodes", "tasks", "pairing_checks")},
            "out_of_task_scores": score_predictions(samples, pooled), "bucket_scores": bucket_scores, "folds": folds,
            "direct_expert_supervision": direct_expert_analysis(baseline, samples)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, help="Optional; omit for baseline-only direct expert supervision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing calibration artifact")
    try:
        candidate = None if args.candidate is None else json.loads(args.candidate.read_text())
        if args.candidate is not None:
            compare_probes.require_mapping(candidate, "candidate")
        result = analyze(json.loads(args.baseline.read_text()), candidate)
        paths = [("baseline", args.baseline)]
        if args.candidate is not None:
            paths.append(("candidate", args.candidate))
        result["inputs"] = {name: {"path": str(path.resolve()), "sha256": compare_probes.sha256_file(path)}
                            for name, path in paths}
        result["implementation_sha256"] = compare_probes.sha256_file(__file__)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(serialized)
    counts = result["pairing"] if "pairing" in result else result["baseline_validation"]
    print(json.dumps({"output": str(args.output), "tasks": counts["tasks"],
                      "samples": counts["exact_samples"], "status": result["analysis_status"]}))


if __name__ == "__main__":
    main()

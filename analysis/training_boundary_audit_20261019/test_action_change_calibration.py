"""Synthetic checks for the frozen action-change diagnostic, not model quality."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import action_change_calibration as calibration

runner = calibration.eval_openloop


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def attach_raw(row, prediction, ground_truth, hold, valid):
    arrays = {"prediction": prediction, "ground_truth": ground_truth,
              "hold_action": hold, "valid_mask": valid}
    row["raw_actions"] = {"schema_version": 1, **{key: value.tolist() for key, value in arrays.items()},
                          "dtypes": {key: value.dtype.str for key, value in arrays.items()},
                          "sha256": runner.input_content_sha256(arrays)}


def fixture(slopes=None, biases=None, task_count=3, short_samples=False, suffix_shift=0.0):
    """Construct actual runner diagnostics and correctly hashed schema-1 arrays."""
    slopes = [np.linspace(-.3, .5, 23)] * task_count if slopes is None else slopes
    biases = [np.linspace(.1, .2, 23)] * task_count if biases is None else biases
    baseline, candidate = [], []
    for task_index in range(task_count):
        for sample_index in range(2):
            episode = task_index * 10 + sample_index
            short = short_samples and sample_index == 1
            distance, bucket = (1, "last_1") if short else (64, "far_2H")
            frame = 1000 - distance
            valid = np.arange(32) < min(32, distance)
            task = f"task_{task_index}"
            probe = {"root_id": "expert_original", "source_kind": "replay", "repo_id": "dataset",
                     "task_name": task, "task_instance_id": episode,
                     "boundary_id": digest(f"boundary-{episode}"), "sample_id": digest(f"sample-{episode}"),
                     "bucket": bucket, "segment_start": 0, "segment_end": 1000,
                     "next_segment_end": 1100, "next_subtask": "place", "distance_to_boundary_frames": distance}
            sample = runner.EvalSample(task_index, episode, frame, episode * 2000 + frame,
                                       0, "grasp", frame + int(valid.sum()), probe)
            hold = np.tile(np.linspace(.2, 1.0, 23), (32, 1))
            x = np.linspace(-1.5, 1.5, 32)[:, None] + np.linspace(-.4, .4, 23)[None, :] + sample_index * .1
            prediction = hold + x
            target = hold + .05
            change = x * np.asarray(slopes[task_index]) + np.asarray(biases[task_index])
            change[16:] += suffix_shift
            for rows, values in ((baseline, prediction), (candidate, prediction + change)):
                record = runner.ErrorRecord(sample, {"model": values - target}, valid)
                row = runner._sample_record(sample, {task_index: task}, int(valid.sum()))
                row["error_diagnostics"] = runner.sample_error_diagnostics(record)
                row["policy_input_sha256"] = digest(f"input-{episode}")
                row["action_noise_sha256"] = digest(f"noise-{episode}")
                row["ground_truth_sha256"] = runner.input_content_sha256(target)
                attach_raw(row, values, target, hold, valid)
                rows.append(row)
    metadata = {"horizon": 32, "conditioning_mode": "provided", "max_token_len": 512, "repo_id": "dataset",
                "action_noise": {"seed": 0, "algorithm": "b1k-openloop-noise-v1/sha256/PCG64/float32"},
                "normalization": {"asset_id": "robot", "use_quantile_norm": False},
                "artifacts": {name: {"sha256": digest(name)} for name in
                              ("norm_stats", "checkpoint", "sample_keys_file", "segments_file")}}
    return ({**copy.deepcopy(metadata), "samples": len(baseline), "sample_keys": baseline},
            {**copy.deepcopy(metadata), "samples": len(candidate), "sample_keys": candidate})


def coordinates(result):
    return [row for group in result["out_of_task_scores"]["groups"] for row in group["coordinates"]]


def direct_coordinates(result):
    return [row for group in result["direct_expert_supervision"]["out_of_task_gt_scores"]["groups"]
            for row in group["coordinates"]]


def replace_ground_truth(result, task, transform):
    """Change real GT and refresh both schema hashes and runner diagnostics."""
    for row in result["sample_keys"]:
        if row["task"] != task:
            continue
        raw = row["raw_actions"]
        arrays = {name: np.asarray(raw[name], dtype=raw["dtypes"][name]) for name in calibration.RAW_NAMES}
        arrays["ground_truth"] = transform(arrays)
        sample = runner.EvalSample(row["task_index"], row["episode_index"], row["frame_index"],
                                   row["global_frame_index"], 0, row["subtask"], row["segment_end"], row["probe_metadata"])
        row["error_diagnostics"] = runner.sample_error_diagnostics(
            runner.ErrorRecord(sample, {"model": arrays["prediction"] - arrays["ground_truth"]}, arrays["valid_mask"]))
        row["ground_truth_sha256"] = runner.input_content_sha256(arrays["ground_truth"])
        attach_raw(row, arrays["prediction"], arrays["ground_truth"], arrays["hold_action"], arrays["valid_mask"])


def direct_fold(result, task):
    return next(fold for fold in result["direct_expert_supervision"]["folds"] if fold["held_out_task"] == task)


def direct_fold_predictions(result, baseline, task):
    coefficients = direct_fold(result, task)["coefficients"]["affine"]
    slope = np.array([row["k"] for row in coefficients])
    intercept = np.array([row["b"] for row in coefficients])
    return np.concatenate([sample.x * slope + intercept for sample in calibration.direct_expert_samples(baseline)
                           if sample.task == task])


class ActionChangeCalibrationTest(unittest.TestCase):
    def test_baseline_only_matches_paired_direct_without_candidate_reference(self):
        baseline, candidate = fixture()
        paired = calibration.analyze(baseline, candidate)
        direct_only = calibration.analyze(baseline)
        self.assertEqual(direct_only["input_mode"], "baseline_only")
        self.assertNotIn("pairing", direct_only)
        for key in ("out_of_task_scores", "bucket_scores", "folds"):
            self.assertNotIn(key, direct_only)
        for task in ("task_0", "task_1", "task_2"):
            self.assertEqual(direct_fold(paired, task)["coefficients"],
                             direct_fold(direct_only, task)["coefficients"])
            np.testing.assert_array_equal(direct_fold_predictions(paired, baseline, task),
                                          direct_fold_predictions(direct_only, baseline, task))
        for left, right in zip(direct_coordinates(paired), direct_coordinates(direct_only)):
            self.assertNotIn("candidate", right["models"])
            for name in ("baseline", "direct_affine", "direct_constant_bias"):
                self.assertEqual(left["models"][name], right["models"][name])

    def test_baseline_only_cli_reads_no_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(fixture()[0]))
            (root / "candidate.json").write_text("invalid candidate JSON must not be read")
            output = root / "direct.json"
            original_read = Path.read_text
            reads = []

            def only_baseline(path, *args, **kwargs):
                self.assertEqual(path, baseline_path)
                reads.append(path)
                return original_read(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", only_baseline), \
                    mock.patch.object(calibration.compare_probes, "compare_results",
                                      side_effect=AssertionError("paired comparison must not run")), \
                    contextlib.redirect_stdout(io.StringIO()):
                calibration.main(["--baseline", str(baseline_path), "--output", str(output)])
            self.assertEqual(reads, [baseline_path])
            result = json.loads(output.read_text())
            self.assertEqual(set(result["inputs"]), {"baseline"})
            self.assertEqual(result["baseline_validation"]["tasks"], 3)
            self.assertEqual(result["analysis_status"], "offline_direct_expert_diagnostic_only")
            self.assertTrue(all("candidate" not in row["models"] for row in direct_coordinates(result)))

    def test_baseline_only_still_validates_schema_and_raw(self):
        baseline = fixture()[0]
        del baseline["sample_keys"][0]["raw_actions"]
        with self.assertRaisesRegex(ValueError, "missing raw_actions"):
            calibration.analyze(baseline)
        baseline = fixture()[0]
        baseline["conditioning_mode"] = "wrong"
        with self.assertRaisesRegex(ValueError, "provided-subtask"):
            calibration.analyze(baseline)

    def test_explicit_null_candidate_cannot_silently_become_baseline_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "baseline.json").write_text(json.dumps(fixture()[0]))
            for value in (None, []):
                with self.subTest(candidate=value):
                    (root / "candidate.json").write_text(json.dumps(value))
                    output = root / "must_not_exist.json"
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        calibration.main(["--baseline", str(root / "baseline.json"),
                                          "--candidate", str(root / "candidate.json"), "--output", str(output)])
                    self.assertFalse(output.exists())

    def test_known_affine_recovered_on_held_out_tasks(self):
        baseline, candidate = fixture()
        result = calibration.analyze(baseline, candidate)
        expected_k, expected_b = np.linspace(-.3, .5, 23), np.linspace(.1, .2, 23)
        self.assertEqual(result["protocol"]["parameter_counts"]["affine"], 42)
        self.assertEqual(len(result["folds"]), 3)
        for fold in result["folds"]:
            self.assertNotIn(fold["held_out_task"], fold["training_tasks"])
            for coefficient in fold["coefficients"]["affine"]:
                dimension = coefficient["action_dimension"]
                self.assertAlmostEqual(coefficient["k"], expected_k[dimension], places=12)
                self.assertAlmostEqual(coefficient["b"], expected_b[dimension], places=12)
        for row in coordinates(result):
            self.assertAlmostEqual(row["models"]["affine"]["explained_fraction"], 1.0)
            self.assertAlmostEqual(row["models"]["no_change"]["explained_fraction"], 0.0)
        self.assertEqual({row["action_dimension"] for row in coordinates(result)}, set(calibration.DIMENSIONS))
        self.assertNotIn("all", {row["action_group"] for row in result["out_of_task_scores"]["groups"]})

    def test_task_dependent_affine_change_fails_out_of_task(self):
        baseline, candidate = fixture(slopes=[1.0, -1.0], biases=[0.0, 0.0], task_count=2)
        result = calibration.analyze(baseline, candidate)
        for row in coordinates(result):
            self.assertAlmostEqual(row["models"]["affine"]["explained_fraction"], -3.0)

    def test_direct_expert_affine_recovers_gt_without_more_capacity(self):
        result = calibration.analyze(*fixture())
        direct = result["direct_expert_supervision"]
        self.assertEqual(direct["protocol"]["parameter_counts"]["affine"], 42)
        for fold in direct["folds"]:
            for coefficient in fold["coefficients"]["affine"]:
                self.assertAlmostEqual(coefficient["k"], -1.0)
                self.assertAlmostEqual(coefficient["b"], .05)
        for row in direct_coordinates(result):
            self.assertLess(row["models"]["direct_affine"]["gt_mse"], 1e-28)
            self.assertGreater(row["models"]["baseline"]["gt_mse"], .01)
            self.assertGreater(row["models"]["direct_constant_bias"]["gt_mse"], .01)
            self.assertEqual(set(row["models"]), {"baseline", "hold", "candidate", "direct_constant_bias", "direct_affine"})

    def test_direct_hold_baseline_matches_hand_calculation(self):
        # base=2, hold=0, GT=.5: target correction=-1.5; holding gives
        # correction=-2, hence MSE=.25, MAE=.5, versus baseline MSE=2.25.
        sample = calibration.Sample("task", 0, 0, "far_2H", np.full((1, 21), 2.0),
                                    np.full((1, 21), -1.5))
        predictions = {name: [np.zeros((1, 21))] for name in calibration.MODELS}
        scores = calibration.score_expert_predictions([sample], predictions)
        row = scores["groups"][0]["coordinates"][0]
        self.assertEqual(row["models"]["baseline"]["gt_mse"], 2.25)
        self.assertEqual(row["models"]["hold"]["gt_mse"], .25)
        self.assertEqual(row["models"]["hold"]["gt_mae"], .5)
        self.assertAlmostEqual(row["models"]["hold"]["mse_reduction_fraction_vs_baseline"], 8 / 9)

    def test_candidate_changes_cannot_change_direct_fit_or_predictions(self):
        baseline, candidate = fixture()
        other_baseline, other_candidate = fixture(slopes=[2.0] * 3, biases=[-3.0] * 3)
        self.assertEqual(baseline, other_baseline)
        first = calibration.analyze(baseline, candidate)
        second = calibration.analyze(baseline, other_candidate)
        for task in ("task_0", "task_1", "task_2"):
            self.assertEqual(direct_fold(first, task)["coefficients"], direct_fold(second, task)["coefficients"])
            np.testing.assert_array_equal(direct_fold_predictions(first, baseline, task),
                                          direct_fold_predictions(second, baseline, task))
        for left, right in zip(direct_coordinates(first), direct_coordinates(second)):
            for name in ("baseline", "direct_affine", "direct_constant_bias"):
                self.assertEqual(left["models"][name], right["models"][name])
        self.assertNotEqual(direct_coordinates(first)[0]["models"]["candidate"],
                            direct_coordinates(second)[0]["models"]["candidate"])

    def test_held_out_gt_cannot_change_its_direct_training_fold(self):
        baseline, candidate = fixture()
        first = calibration.analyze(baseline, candidate)
        changed_baseline, changed_candidate = copy.deepcopy(baseline), copy.deepcopy(candidate)
        for result in (changed_baseline, changed_candidate):
            replace_ground_truth(result, "task_0", lambda arrays: arrays["ground_truth"] + 10.0)
        second = calibration.analyze(changed_baseline, changed_candidate)
        self.assertEqual(direct_fold(first, "task_0")["coefficients"],
                         direct_fold(second, "task_0")["coefficients"])
        np.testing.assert_array_equal(direct_fold_predictions(first, baseline, "task_0"),
                                      direct_fold_predictions(second, changed_baseline, "task_0"))
        self.assertNotEqual(direct_fold(first, "task_0")["held_out_gt_scores"],
                            direct_fold(second, "task_0")["held_out_gt_scores"])
        # The same changed task IS training data for a different held-out fold.
        self.assertNotEqual(direct_fold(first, "task_1")["coefficients"],
                            direct_fold(second, "task_1")["coefficients"])

    def test_direct_low_baseline_error_keeps_mse_but_marks_fraction_na(self):
        baseline, candidate = fixture()
        for task in ("task_0", "task_1", "task_2"):
            base_by_key = {(row["episode_index"], row["frame_index"]): row for row in baseline["sample_keys"]}
            replace_ground_truth(baseline, task, lambda arrays: arrays["prediction"].copy())
            # Candidate must use the identical expert target, not its own prediction.
            for row in candidate["sample_keys"]:
                if row["task"] == task:
                    target = np.asarray(base_by_key[(row["episode_index"], row["frame_index"])]["raw_actions"]["ground_truth"])
                    replace_ground_truth({"sample_keys": [row]}, task, lambda arrays, target=target: target.copy())
        result = calibration.analyze(baseline, candidate)
        for row in direct_coordinates(result):
            self.assertEqual(row["status"], "NA_low_baseline_gt_error")
            self.assertEqual(row["models"]["baseline"]["gt_mse"], 0.0)
            for metrics in row["models"].values():
                self.assertIsNone(metrics["mse_reduction_fraction_vs_baseline"])
                self.assertGreaterEqual(metrics["gt_mse"], 0.0)

    def test_prefix_only_and_short_valid_samples(self):
        first = calibration.analyze(*fixture(short_samples=True))
        second = calibration.analyze(*fixture(short_samples=True, suffix_shift=1000.0))
        self.assertEqual(first["out_of_task_scores"], second["out_of_task_scores"])
        self.assertEqual(first["out_of_task_scores"]["valid_prefix_positions"], 3 * (16 + 1))
        self.assertEqual({row["bucket"] for row in first["bucket_scores"]}, {"far_2H", "last_1"})

    def test_hierarchical_weights_and_constant_feature_fallback(self):
        def sample(task, index, value, steps):
            return calibration.Sample(task, index, 0, "far_2H", np.zeros((steps, 21)), np.full((steps, 21), value))
        # First task has one short sample; second has three much longer samples.
        samples = [sample("a", 0, 1.0, 1)] + [sample("b", index, 3.0, 16) for index in range(3)]
        models, fallback = calibration.fit_models(samples)
        self.assertEqual(fallback, list(calibration.DIMENSIONS))
        np.testing.assert_allclose(models["constant_bias"][1], 2.0)
        np.testing.assert_allclose(models["affine"][0], 0.0)
        predictions = {name: [row.x * k + b for row in samples] for name, (k, b) in models.items()}
        scores = calibration.score_predictions(samples, predictions)
        row = scores["groups"][0]["coordinates"][0]
        self.assertAlmostEqual(row["sse_delta"], 5.0)
        self.assertAlmostEqual(row["models"]["constant_bias"]["sse_residual"], 1.0)

    def test_low_signal_has_explicit_na(self):
        result = calibration.analyze(*fixture(slopes=[0.0] * 3, biases=[1e-8] * 3))
        for row in coordinates(result):
            self.assertEqual(row["status"], "NA_low_signal")
            self.assertTrue(all(model["explained_fraction"] is None for model in row["models"].values()))
        json.dumps(result, allow_nan=False)

    def test_hash_mismatch_rejected(self):
        baseline, candidate = fixture()
        candidate["sample_keys"][0]["raw_actions"]["prediction"][0][0] += .5
        with self.assertRaisesRegex(ValueError, "raw_actions SHA256 mismatch"):
            calibration.analyze(baseline, candidate)

    def test_missing_raw_rejected_with_export_instruction(self):
        baseline, candidate = fixture()
        del baseline["sample_keys"][0]["raw_actions"]
        with self.assertRaisesRegex(ValueError, "missing raw_actions.*--save-raw-actions"):
            calibration.analyze(baseline, candidate)

    def test_truncated_raw_horizon_is_explicitly_rejected(self):
        baseline, candidate = fixture()
        raw = candidate["sample_keys"][0]["raw_actions"]
        for name in calibration.RAW_NAMES:
            raw[name] = raw[name][:16]
        with self.assertRaisesRegex(ValueError, "requested horizon; truncated chunks are unsupported"):
            calibration.analyze(baseline, candidate)

    def test_dtype_and_paired_hold_rejected(self):
        baseline, candidate = fixture()
        candidate["sample_keys"][0]["raw_actions"]["dtypes"]["prediction"] = "<i8"
        with self.assertRaisesRegex(ValueError, "floating dtype"):
            calibration.analyze(baseline, candidate)
        baseline, candidate = fixture()
        row = candidate["sample_keys"][0]
        arrays = {name: np.asarray(row["raw_actions"][name], dtype=row["raw_actions"]["dtypes"][name])
                  for name in calibration.RAW_NAMES}
        arrays["hold_action"][0][0] += 1
        attach_raw(row, arrays["prediction"], arrays["ground_truth"], arrays["hold_action"], arrays["valid_mask"])
        with self.assertRaisesRegex(ValueError, "Paired raw arrays differ.*hold_action"):
            calibration.analyze(baseline, candidate)

    def test_compare_results_pairing_still_mandatory(self):
        baseline, candidate = fixture()
        candidate["sample_keys"][0]["action_noise_sha256"] = digest("different-noise")
        with self.assertRaisesRegex(ValueError, "action_noise_sha256"):
            calibration.analyze(baseline, candidate)
        with self.assertRaisesRegex(ValueError, "at least two distinct tasks"):
            calibration.analyze(*fixture(task_count=1))

    def test_cli_is_json_and_writes_nothing_on_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, candidate = fixture()
            (root / "baseline.json").write_text(json.dumps(baseline))
            (root / "candidate.json").write_text(json.dumps(candidate))
            output = root / "output" / "calibration.json"
            args = ["--baseline", str(root / "baseline.json"), "--candidate", str(root / "candidate.json"),
                    "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                calibration.main(args)
            result = json.loads(output.read_text())
            self.assertEqual(result["analysis_status"], "offline_action_change_diagnostic_only")
            self.assertEqual(set(result["inputs"]), {"baseline", "candidate"})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                calibration.main(args)
            bad_output = root / "rejected" / "calibration.json"
            del candidate["sample_keys"][0]["raw_actions"]
            (root / "candidate.json").write_text(json.dumps(candidate))
            args[-1] = str(bad_output)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                calibration.main(args)
            self.assertFalse(bad_output.parent.exists())


if __name__ == "__main__":
    unittest.main()

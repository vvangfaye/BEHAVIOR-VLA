"""Synthetic action arrays passed through the real eval_openloop output helpers.

These tests validate the result schema and fail-closed pairing, not model quality.
"""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_probes as compare
from eval_cached_probe import eval_openloop as runner


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(scale=1.0):
    rows = []
    for episode in (11, 12):
        for bucket, distance in (("far_2H", 64), ("edge_H", 32), ("inside_K", 16), ("last_1", 1)):
            frame = 1000 - distance
            valid = min(32, distance)
            probe = {"root_id": "expert_original", "source_kind": "replay", "repo_id": "dataset",
                     "task_name": "task", "task_instance_id": episode + 100,
                     "boundary_id": digest(f"boundary-{episode}"), "sample_id": digest(f"sample-{episode}-{frame}"),
                     "bucket": bucket, "segment_start": 0, "segment_end": 1000,
                     "next_segment_end": 1100, "next_subtask": "place", "distance_to_boundary_frames": distance,
                     "full_H_legal_before": distance >= 32, "executed_K_within_original_label": distance >= 16,
                     "recorded_full_H_future_available_by_index": True, "training_seen_source": True}
            sample = runner.EvalSample(3, episode, frame, episode * 2000 + frame, 0, "grasp", frame + valid, probe)
            # Different error magnitudes across episodes distinguish pooled RMSE
            # from the arithmetic mean of per-sample RMSEs.
            error = np.full((32, 23), (episode - 10) * scale, dtype=np.float64)
            record = runner.ErrorRecord(sample, {"model": error}, np.arange(32) < valid)
            row = runner._sample_record(sample, {3: "task"}, valid)
            row["error_diagnostics"] = runner.sample_error_diagnostics(record)
            row.update({key: digest(f"{key}-{episode}-{frame}") for key in
                        ("policy_input_sha256", "ground_truth_sha256", "action_noise_sha256")})
            rows.append(row)
    return {"horizon": 32, "conditioning_mode": "provided", "max_token_len": 512, "repo_id": "dataset",
            "action_noise": {"seed": 0, "algorithm": "b1k-openloop-noise-v1/sha256/PCG64/float32"},
            "normalization": {"asset_id": "robot", "use_quantile_norm": False},
            "artifacts": {name: {"sha256": digest(name)} for name in
                          ("norm_stats", "checkpoint", "sample_keys_file", "segments_file")},
            "samples": len(rows), "sample_keys": rows}


class CompareProbesTest(unittest.TestCase):
    def test_real_helper_schema_and_matched_prefixes(self):
        summary, rows = compare.compare_results(fixture(), fixture(.5))
        self.assertEqual(summary["exact_samples"], 8)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(len(rows), (8 + 6) * 7)
        self.assertTrue(all(row["delta_l1"] < 0 for row in rows))
        self.assertFalse(any(row["bucket"] == "last_1" and row["window"] == "complete_prefix_16" for row in rows))
        whole = next(group for group in summary["groups"] if group["scope"] == "overall" and
                     group["window"] == "first_action" and group["action_group"] == "all")
        self.assertEqual(whole["l1"]["wins"], 8)
        self.assertEqual(whole["l1"]["mean_delta"], -.75)
        self.assertAlmostEqual(whole["rmse"]["baseline_mean"], 1.5)
        self.assertAlmostEqual(whole["rmse"]["baseline_pooled_rmse"], np.sqrt(2.5))

    def test_order_independence_and_exact_sample_set(self):
        baseline = fixture()
        candidate = fixture()
        candidate["sample_keys"].reverse()
        summary, rows = compare.compare_results(baseline, candidate)
        self.assertTrue(all(row["l1_outcome"] == "tie" for row in rows))
        candidate["sample_keys"].pop()
        candidate["samples"] -= 1
        with self.assertRaisesRegex(ValueError, "sample set mismatch"):
            compare.compare_results(baseline, candidate)
        candidate = fixture()
        candidate["sample_keys"][-1] = copy.deepcopy(candidate["sample_keys"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare.compare_results(baseline, candidate)

    def test_each_required_content_hash_is_checked(self):
        for key in ("policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
            with self.subTest(key=key):
                candidate = fixture()
                candidate["sample_keys"][0][key] = digest("changed")
                with self.assertRaisesRegex(ValueError, key):
                    compare.compare_results(fixture(), candidate)
                del candidate["sample_keys"][0][key]
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    compare.compare_results(fixture(), candidate)

    def test_normalization_mismatch_requires_explicit_confounded_flag(self):
        candidate = fixture()
        candidate["artifacts"]["norm_stats"]["sha256"] = digest("different-norm")
        with self.assertRaisesRegex(ValueError, "Normalization differs"):
            compare.compare_results(fixture(), candidate)
        summary, _ = compare.compare_results(fixture(), candidate, allow_different_normalization=True)
        self.assertFalse(summary["normalization"]["matches"])
        self.assertTrue(summary["normalization"]["confounded"])
        candidate = fixture()
        candidate["normalization"]["use_quantile_norm"] = True
        with self.assertRaisesRegex(ValueError, "Normalization differs"):
            compare.compare_results(fixture(), candidate)

    def test_false_prefix_and_nonfinite_errors_are_rejected(self):
        candidate = fixture()
        candidate["sample_keys"][3]["error_diagnostics"]["complete_prefix_16"]["samples"] = 1
        with self.assertRaisesRegex(ValueError, "prefix inclusion"):
            compare.compare_results(fixture(), candidate)
        candidate = fixture()
        candidate["sample_keys"][0]["error_diagnostics"]["first_action"]["metrics"]["model"]["l1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare.compare_results(fixture(), candidate)

    def test_frozen_identity_configuration_and_dimensions_are_checked(self):
        candidate = fixture()
        candidate["sample_keys"][0]["subtask"] = "wrong instruction"
        with self.assertRaisesRegex(ValueError, "subtask"):
            compare.compare_results(fixture(), candidate)
        candidate = fixture()
        candidate["max_token_len"] = 768
        with self.assertRaisesRegex(ValueError, "max_token_len"):
            compare.compare_results(fixture(), candidate)
        candidate = fixture()
        candidate["sample_keys"][0]["error_diagnostics"]["first_action"]["per_action_group"]["base"]["model"]["scalar_values"] = 23
        with self.assertRaisesRegex(ValueError, "dimensionality"):
            compare.compare_results(fixture(), candidate)

    def test_episode_bootstrap_is_reproducible_and_exploratory(self):
        first, _ = compare.compare_results(fixture(), fixture(.5), bootstrap_replicates=25, bootstrap_seed=7)
        second, _ = compare.compare_results(fixture(), fixture(.5), bootstrap_replicates=25, bootstrap_seed=7)
        self.assertEqual(first, second)
        group = next(group for group in first["groups"] if group["scope"] == "overall")
        self.assertEqual(group["episodes"], 2)
        self.assertTrue(group["episode_bootstrap"]["computed"])
        self.assertTrue(group["episode_bootstrap"]["exploratory"])
        self.assertIn("exploratory", first["analysis_status"])

    def test_cli_multiple_candidates_and_no_partial_result_on_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for filename, result in (("base.json", fixture()), ("good.json", fixture(.5))):
                (root / filename).write_text(json.dumps(result))
            bad = fixture()
            bad["sample_keys"][0]["action_noise_sha256"] = digest("wrong")
            (root / "bad.json").write_text(json.dumps(bad))
            args = ["--baseline", str(root / "base.json"), "--candidate", f"good={root / 'good.json'}",
                    "--candidate", f"bad={root / 'bad.json'}", "--output-dir", str(root / "rejected")]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare.main(args)
            self.assertFalse((root / "rejected").exists())
            args[5] = f"same={root / 'base.json'}"
            args[-1] = str(root / "accepted")
            with contextlib.redirect_stdout(io.StringIO()):
                compare.main(args)
            summary = json.loads((root / "accepted/summary.json").read_text())
            self.assertEqual(len(summary["comparisons"]), 2)
            with (root / "accepted/sample_differences.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2 * (8 + 6) * 7)
            self.assertEqual({row["candidate"] for row in rows}, {"good", "same"})


if __name__ == "__main__":
    unittest.main()

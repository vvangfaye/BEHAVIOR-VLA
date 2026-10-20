"""Synthetic checks for the frozen pilot contrasts, not model-quality evidence."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import boundary_interactions as analysis


PLAN = json.loads(Path(__file__).with_name("pilot_analysis_plan.json").read_text())


def fixture(tasks=(("task_a", 1),), delta=None, candidate="candidate"):
    delta = delta or (lambda task, boundary, window, group, distance: 1.0)
    rows, episode = [], 0
    for task_index, (task, count) in enumerate(tasks):
        for boundary in range(count):
            episode += 1
            for window, distances in analysis.DISTANCES.items():
                for group in analysis.GROUPS:
                    for distance in sorted(distances):
                        change = delta(task, boundary, window, group, distance)
                        rows.append({"candidate": candidate, "root_id": "expert", "episode_index": str(episode),
                                     "task_index": str(task_index), "task": task, "task_instance_id": str(100 + episode),
                                     "frame_index": str(1000 - distance), "boundary_id": f"boundary-{episode}",
                                     "sample_id": f"sample-{episode}-{distance}", "bucket": analysis.BUCKETS[distance],
                                     "distance_to_boundary_frames": str(distance), "window": window, "action_group": group,
                                     "compared_action_steps": "1" if window == "first_action" else "16",
                                     "normalization_matches": "True", "baseline_l1": "10.0", "baseline_rmse": "11.0",
                                     "candidate_l1": str(10 + change), "candidate_rmse": str(11 + change),
                                     "delta_l1": str(change), "delta_rmse": str(change),
                                     "l1_outcome": "loss" if change > 0 else "win" if change < 0 else "tie",
                                     "rmse_outcome": "loss" if change > 0 else "win" if change < 0 else "tie"})
    return rows


def group(report, window="first_action", action_group="base", candidate="candidate"):
    return next(row for row in report["group_summaries"] if
                (row["window"], row["action_group"], row["candidate"]) == (window, action_group, candidate))


class BoundaryInteractionsTest(unittest.TestCase):
    def test_general_degradation_cancels_but_boundary_differential_remains(self):
        broad = analysis.analyze(fixture(), PLAN)
        self.assertEqual(group(broad)["mean"], 0)
        self.assertEqual(group(broad)["ties"], 1)
        differential = analysis.analyze(fixture(delta=lambda t, b, w, g, d: 3 if d <= 16 else 1), PLAN)
        self.assertEqual(group(differential)["mean"], 2)
        self.assertEqual(group(differential, "complete_prefix_16")["mean"], 2)
        self.assertTrue(all(row["delta_far_mean_l1"] == 1 and row["delta_near_mean_l1"] == 3
                            for row in differential["boundary_interactions"]))

    def test_first_action_and_prefix_use_their_own_prespecified_distances(self):
        report = analysis.analyze(fixture(delta=lambda t, b, w, g, d:
                                         {64: 1, 32: 3, 16: 5, 1: 9}[d]), PLAN)
        self.assertEqual(group(report)["mean"], 5)  # (5 + 9)/2 - (1 + 3)/2
        self.assertEqual(group(report, "complete_prefix_16")["mean"], 3)
        self.assertEqual(group(report, action_group="all")["contrast_role"], "secondary")
        self.assertEqual(group(report, "complete_prefix_16")["contrast_role"], "secondary")

    def test_groups_are_kept_separate_and_task_means_are_equally_weighted(self):
        def changes(task, boundary, window, action_group, distance):
            near_delta = 2 if task == "many_boundaries" else -2
            if action_group == "torso":
                near_delta = -near_delta
            return near_delta if distance <= 16 else 0
        report = analysis.analyze(fixture((("many_boundaries", 3), ("one_boundary", 1)), changes), PLAN)
        result = group(report)
        self.assertEqual(result["mean"], 0)  # Boundary-pooled mean would incorrectly be +1.
        self.assertEqual(result["median"], 0)
        self.assertEqual(result["range"], [-2, 2])
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (1, 1, 0))
        task = next(row for row in report["task_interactions"] if row["task"] == "many_boundaries"
                    and row["window"] == "first_action" and row["action_group"] == "base")
        self.assertEqual((task["boundaries"], task["positive"], task["mean"]), (3, 3, 2))
        torso = next(row for row in report["task_interactions"] if row["task"] == "many_boundaries"
                     and row["window"] == "first_action" and row["action_group"] == "torso")
        self.assertEqual(torso["mean"], -2)

    def test_missing_duplicate_or_whole_group_rejected(self):
        original = fixture()
        for rows, message in ((original[:-1], "Missing distances"),
                              (original + [copy.deepcopy(original[0])], "Duplicate distance"),
                              ([row for row in original if row["action_group"] != "base"], "Missing distances")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                analysis.analyze(rows, PLAN)

    def test_identity_numbers_and_normalization_fail_closed(self):
        for field, value, message in (("frame_index", "800", "Inconsistent boundary identity"),
                                      ("sample_id", "bad", "Multiple sample identities"),
                                      ("candidate_l1", "nan", "Nonfinite"),
                                      ("delta_l1", "nan", "Nonfinite"),
                                      ("delta_l1", "2", "delta mismatch"),
                                      ("normalization_matches", "False", "Normalization must match")):
            rows = fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                analysis.analyze(rows, PLAN)

    def test_multiple_candidates_require_identical_base_values_and_support(self):
        original, other = fixture(), fixture(candidate="second")
        valid = analysis.analyze(original + other, PLAN)
        self.assertEqual(valid["counts"]["candidates"], 2)
        other[0].update({"baseline_l1": "9", "candidate_l1": "10"})
        with self.assertRaisesRegex(ValueError, "Inconsistent baseline"):
            analysis.analyze(original + other, PLAN)
        with self.assertRaisesRegex(ValueError, "Missing distances"):
            analysis.analyze(original + fixture(candidate="second")[:-1], PLAN)

    def test_reject_changed_plan(self):
        changed = copy.deepcopy(PLAN)
        changed["primary_contrast"] = "a new contrast"
        with self.assertRaisesRegex(ValueError, "frozen pilot contrasts"):
            analysis.analyze(fixture(), changed)

    def test_cli_provenance_atomic_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "samples.csv"
            rows = fixture()
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "result"
            args = ["--input-csv", str(source), "--output-dir", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                analysis.main(args)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["provenance"]["input_csv"]["sha256"], analysis.digest(source.read_bytes()))
            self.assertEqual(report["provenance"]["plan"]["sha256"], analysis.digest(Path(__file__).with_name("pilot_analysis_plan.json").read_bytes()))
            self.assertEqual(len(list(output.iterdir())), 4)
            before = (output / "summary.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analysis.main(args)
            self.assertEqual((output / "summary.json").read_bytes(), before)
            empty_output = root / "empty"
            empty_output.mkdir()
            with self.assertRaises(FileExistsError):
                analysis.write_outputs(empty_output, report)
            # Invalid inputs leave neither a destination nor a staging directory.
            source.write_text(source.read_text().splitlines()[0] + "\n")
            args[-1] = str(root / "invalid")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analysis.main(args)
            self.assertFalse((root / "invalid").exists())
            self.assertFalse(list(root.glob(".*.tmp-*")))


if __name__ == "__main__":
    unittest.main()

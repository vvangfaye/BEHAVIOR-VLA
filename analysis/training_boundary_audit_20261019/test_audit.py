"""Compare audit geometry against exhaustive anchor/target enumeration."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("boundary_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BoundaryAuditTest(unittest.TestCase):
    def test_exact_counts_against_exhaustive_enumeration(self):
        for horizon, execute in [(1, 1), (4, 1), (4, 4), (8, 3), (32, 16)]:
            for length in range(horizon, horizon * 4 + 2):
                anchors = range(length - horizon + 1)
                for distance in range(1, length + 2):
                    frame = length-distance
                    expected = (int(frame >= 0), int(frame in anchors),
                        sum(a <= frame < a+horizon for a in anchors),
                        sum(a <= frame < a+execute for a in anchors))
                    self.assertEqual(audit.coverage_for_length(length, horizon, execute, distance), expected)

    def test_summary_conserves_all_target_pairs_and_support(self):
        group = audit.empty_group()
        for length in [32, 33, 64, 150]:
            group["length_histogram"][length] += 1
            group["interval_count"] += 1
            group["indexed_frames"] += length
            group["legal_H_anchors"] += length-31
        result = audit.summarize(group, 32, 16)
        rows = result["distance_to_interval_end_coverage"]
        self.assertEqual(sum(r["all_H_anchor_target_pairs"] for r in rows), result["legal_H_anchors"]*32)
        self.assertEqual(sum(r["executed_K_anchor_target_pairs"] for r in rows), result["legal_H_anchors"]*16)
        self.assertEqual(sum(r["legal_observation_anchors"] for r in rows), result["legal_H_anchors"])
        self.assertEqual(sum(r["positions_ever_targeted_in_executed_K_slots"] for r in rows), 279-64)
        self.assertEqual(rows[15]["executed_K_anchor_target_pairs"], 0)
        self.assertGreater(rows[16]["executed_K_anchor_target_pairs"], 0)
        self.assertEqual(rows[30]["legal_observation_anchors"], 0)
        self.assertEqual(rows[31]["legal_observation_anchors"], 4)

    def test_boundaries_integrity_and_duplicate_rejection(self):
        document = {"schema_version": 1, "kind": "b1k_valid_intervals", "action_horizon": 4,
            "source_kind": "replay", "root_id": "r", "episodes": [
                {"episode_index": 0, "length": 26, "task_name": "t"}],
            "intervals": [{"episode_index": 0, "start": 0, "end": 4, "subtask": "a"},
                          {"episode_index": 0, "start": 4, "end": 8, "subtask": "a"},
                          {"episode_index": 0, "start": 8, "end": 12, "subtask": "b"},
                          {"episode_index": 0, "start": 14, "end": 18, "subtask": "b"},
                          {"episode_index": 0, "start": 20, "end": 24, "subtask": "b"}],
            "provenance": {}}
        document["fingerprint"] = audit.canonical_digest(document)
        with tempfile.TemporaryDirectory() as location:
            path = Path(location)/"index.json"
            path.write_text(json.dumps(document))
            result = audit.audit_cohort([path], 4, 2)["by_source"]["replay"]
            self.assertEqual(result["boundary_types"], {"contiguous_same_label": 1,
                "contiguous_different_label": 1, "unindexed_gap": 2, "unindexed_episode_tail": 1})
            with self.assertRaisesRegex(ValueError, "Duplicate episode"):
                audit.audit_cohort([path, path], 4, 2)
            corrupted = copy.deepcopy(document)
            corrupted["intervals"][0]["end"] = 5
            path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                audit.audit_cohort([path], 4, 2)

    def test_real_report_independent_counts_and_conservation(self):
        report_path = Path(__file__).with_name("report.json")
        if not report_path.exists():
            self.skipTest("Run real-data CLI once to enable report verification")
        report = json.loads(report_path.read_text())
        expected = {
            ("quality_20261016", "replay"): (15949, 282142, 161068145),
            ("quality_20261016", "full_episode"): (270, 5154, 2506369),
            ("recovered_20261013", "recovered"): (3949, 98841, 48757662),
        }
        for (cohort, source), (episodes, intervals, frames) in expected.items():
            row = report["cohorts"][cohort]["by_source"][source]
            self.assertEqual((row["episodes"], row["intervals"], row["indexed_frames"]),
                             (episodes, intervals, frames))
            self.assertEqual(row["excluded_observation_anchor_positions"], intervals*31)
            self.assertEqual(row["action_positions_never_targeted_in_executed_K_slots"], intervals*16)
            distances = row["distance_to_interval_end_coverage"]
            self.assertEqual(sum(r["all_H_anchor_target_pairs"] for r in distances), row["legal_H_anchors"]*32)
            self.assertEqual(sum(r["executed_K_anchor_target_pairs"] for r in distances), row["legal_H_anchors"]*16)
            self.assertEqual(sum(r["observation_positions"] for r in distances), frames)
            for d in distances[:16]:
                self.assertEqual(d["legal_observation_anchors"], 0)
                self.assertEqual(d["positions_ever_targeted_in_executed_K_slots"], 0)
                self.assertEqual(d["positions_ever_targeted_in_any_H_slot"], intervals)
            for d in distances[16:31]:
                self.assertEqual(d["legal_observation_anchors"], 0)
                self.assertEqual(d["positions_ever_targeted_in_executed_K_slots"], intervals)


if __name__ == "__main__":
    unittest.main()

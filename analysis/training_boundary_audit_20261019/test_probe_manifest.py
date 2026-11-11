import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import probe_manifest as probe


def fixture():
    document = {"kind": "b1k_valid_intervals", "schema_version": 1, "action_horizon": 4,
        "source_kind": "replay", "root_id": "expert", "repo_id": "fixture",
        "dataset_root": "/fixture", "episodes": [], "intervals": []}
    # Three eligible episodes of task A, one of B. The extra same-label and gap
    # intervals in each episode must never enter the probe.
    for episode, task in [(0, "A"), (1, "A"), (2, "A"), (3, "B")]:
        document["episodes"].append({"episode_index": episode, "task_name": task,
            "task_instance_id": 100+episode, "length": 32})
        for start, end, label in [(0, 8, "left"), (8, 12, "right"),
                                  (12, 20, "right"), (22, 26, "gap"), (26, 32, "shortleft")]:
            document["intervals"].append({"episode_index": episode, "start": start,
                                         "end": end, "subtask": label})
    document["fingerprint"] = probe.canonical_digest(document)
    return document


def make(document, **kwargs):
    return probe.build_probe(document, source_path="fixture", source_file_sha256="fixture",
                             horizon=4, execute=2, max_per_task=2, seed=0, **kwargs)


class ProbeManifestTest(unittest.TestCase):
    def test_deterministic_under_index_ordering(self):
        document = fixture()
        original = make(document)[0]
        reordered = copy.deepcopy(document)
        reordered["intervals"].reverse()
        reordered["episodes"].reverse()
        reordered["fingerprint"] = probe.canonical_digest({k: v for k, v in reordered.items() if k != "fingerprint"})
        rerun = make(reordered)[0]
        self.assertEqual(original["boundaries"], rerun["boundaries"])
        self.assertEqual(original["sample_keys"], rerun["sample_keys"])
        self.assertEqual(original["statistics"], rerun["statistics"])

    def test_filter_support_distinct_episodes_and_balanced_buckets(self):
        result, episodes = make(fixture())
        self.assertEqual(result["statistics"]["selected_boundaries"], 3)
        self.assertEqual(result["statistics"]["bucket_counts"],
                         {"far_2H": 3, "edge_H": 3, "inside_K": 3, "last_1": 3})
        selected = result["boundaries"]
        self.assertEqual(len({(b["task_name"], b["task_instance_id"], b["episode_index"]) for b in selected}), 3)
        self.assertEqual(len(episodes["episode_indices"]), 3)
        for boundary in selected:
            self.assertNotEqual(boundary["left_subtask"], boundary["right_subtask"])
            self.assertGreaterEqual(boundary["boundary_frame"]-boundary["left_start"], 8)
        samples = result["sample_keys"]
        self.assertEqual(len({(s["root_id"], s["episode_index"], s["frame_index"]) for s in samples}), len(samples))
        for sample in samples:
            self.assertLess(sample["frame_index"], sample["segment_end"])
            self.assertGreaterEqual(sample["frame_index"], sample["segment_start"])
            self.assertLessEqual(sample["frame_index"]+4, sample["next_segment_end"])
            self.assertEqual(sample["valid_action_steps"], min(4, sample["distance_to_boundary_frames"]))
            self.assertEqual(sample["full_H_legal_before"], sample["distance_to_boundary_frames"]>=4)
            self.assertEqual(sample["executed_K_within_original_label"], sample["distance_to_boundary_frames"]>=2)

    def test_integrity_and_distinct_anchor_guard(self):
        document = fixture()
        document["intervals"][0]["subtask"] = "tampered"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            make(document)
        with self.assertRaisesRegex(ValueError, "distinct anchors"):
            probe.build_probe(fixture(), source_path="fixture", source_file_sha256="fixture",
                              horizon=4, execute=4)

    def test_explicit_runner_noise_contract_does_not_change_sample_selection(self):
        default, _ = make(fixture())
        changed, _ = make(fixture(), action_noise_seed=9)
        self.assertEqual(default["sample_keys"], changed["sample_keys"])
        self.assertEqual(default["pairing_contract"]["eval_openloop_action_noise_argument"],
                         "--action-noise-seed 0")
        self.assertEqual(changed["pairing_contract"]["eval_openloop_action_noise_argument"],
                         "--action-noise-seed 9")
        self.assertFalse(default["inference_executed"])
        self.assertTrue(default["pairing_contract"]["matching_action_noise_does_not_control_all_runtime_randomness"])

    def test_real_manifest_invariants_when_available(self):
        path = Path(__file__).with_name("probe_manifest.json")
        if not path.exists():
            self.skipTest("Generate the real probe first")
        result = json.loads(path.read_text())
        samples = result["sample_keys"]
        self.assertEqual(len(samples), result["statistics"]["selected_boundaries"]*4)
        self.assertEqual(len({(s["root_id"], s["episode_index"], s["frame_index"]) for s in samples}), len(samples))
        self.assertEqual(len(set(result["statistics"]["bucket_counts"].values())), 1)
        self.assertTrue(all(t["selected_boundaries"]<=3 for t in result["statistics"]["by_task"].values()))
        for sample in samples:
            self.assertIn(sample["distance_to_boundary_frames"], (64, 32, 16, 1))
            self.assertEqual(sample["valid_action_steps"], min(32, sample["distance_to_boundary_frames"]))
            self.assertLessEqual(sample["frame_index"]+32, sample["next_segment_end"])


if __name__ == "__main__":
    unittest.main()

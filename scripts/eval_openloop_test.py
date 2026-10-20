# ruff: noqa: PT009, PT027
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

_SPEC = importlib.util.spec_from_file_location("eval_openloop", Path(__file__).with_name("eval_openloop.py"))
assert _SPEC is not None
assert _SPEC.loader is not None
eval_openloop = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = eval_openloop
_SPEC.loader.exec_module(eval_openloop)


class FakeSegments:
    def __init__(self):
        self.segments = {
            7: (
                np.asarray([0, 2, 4]),
                np.asarray([2, 4, 6]),
                ["approach", "imputed", "grasp"],
            )
        }
        self._annotations = {7: np.asarray([True, False, True])}

    def annotation_flags(self, episode):
        return self._annotations[episode]

    def validate_coverage(self, lengths):
        assert lengths == {7: 6}

    def lookup(self, episode, frame, *, require_annotated=False):
        starts, ends, labels = self.segments[episode]
        index = int(np.searchsorted(starts, frame, side="right")) - 1
        if index < 0 or frame >= ends[index]:
            raise ValueError
        if require_annotated and not self._annotations[episode][index]:
            raise ValueError
        return labels[index], int(ends[index])


class EvalOpenLoopHelpersTest(unittest.TestCase):
    def test_input_hash_is_key_order_independent_but_checks_values_and_shape(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        first = {"image": image, "prompt": "pick up", "state": np.zeros(2, dtype=np.float32)}
        reordered = {"state": first["state"], "prompt": first["prompt"], "image": image.copy()}
        self.assertEqual(eval_openloop.input_content_sha256(first), eval_openloop.input_content_sha256(reordered))
        changed = {**first, "image": image.reshape(4, 3)}
        self.assertNotEqual(eval_openloop.input_content_sha256(first), eval_openloop.input_content_sha256(changed))
        changed = {**first, "prompt": "put down"}
        self.assertNotEqual(eval_openloop.input_content_sha256(first), eval_openloop.input_content_sha256(changed))

    def test_action_noise_is_keyed_to_frame_and_independent_of_iteration_order(self):
        keys = [(7, 100), (4, 2), (7, 99)]
        forward = {key: eval_openloop.paired_action_noise(0, *key, 32, 32) for key in keys}
        reverse = {key: eval_openloop.paired_action_noise(0, *key, 32, 32) for key in reversed(keys)}
        for key in keys:
            np.testing.assert_array_equal(forward[key], reverse[key])
            self.assertEqual(forward[key].shape, (32, 32))
            self.assertEqual(forward[key].dtype, np.float32)
        self.assertFalse(np.array_equal(forward[keys[0]], forward[keys[1]]))
        self.assertFalse(np.array_equal(forward[keys[0]], eval_openloop.paired_action_noise(1, 7, 100, 32, 32)))
        for args in [(-1, 0, 0, 32, 32), (0, 0, 0, 0, 32), (True, 0, 0, 32, 32)]:
            with self.assertRaises(ValueError):
                eval_openloop.paired_action_noise(*args)

    def test_filtered_episode_rows_use_relative_indices(self):
        metadata = [
            eval_openloop.EpisodeMeta(0, 0, 0, 3),
            eval_openloop.EpisodeMeta(1, 0, 3, 5),
            eval_openloop.EpisodeMeta(2, 1, 5, 9),
        ]

        filtered = eval_openloop.filter_episode_meta(metadata, [2, 1])
        self.assertEqual(
            [(episode.episode_index, episode.relative_from_index) for episode in filtered],
            [(1, 0), (2, 2)],
        )

        samples = eval_openloop.select_samples(filtered, per_group=10, seed=4)
        self.assertEqual(
            [(sample.global_frame_index, sample.dataset_index) for sample in samples],
            [(3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5)],
        )

    def test_subtask_selection_covers_canonical_annotated_segments_only(self):
        metadata = [eval_openloop.EpisodeMeta(7, 3, 100, 106, relative_from_index=0)]

        samples = eval_openloop.select_samples(metadata, per_group=10, seed=0, segments=FakeSegments())

        self.assertEqual(
            [(sample.subtask, sample.frame_index, sample.segment_end) for sample in samples],
            [
                ("approach", 0, 2),
                ("approach", 1, 2),
                ("grasp", 4, 6),
                ("grasp", 5, 6),
            ],
        )
        self.assertTrue(all(sample.subtask != "imputed" for sample in samples))

    def test_replay_samples_preserves_frames_and_valid_horizons(self):
        metadata = [eval_openloop.EpisodeMeta(7, 3, 100, 106, relative_from_index=8)]
        payload = {
            "sample_keys": [
                {
                    "task_index": 3,
                    "episode_index": 7,
                    "frame_index": 4,
                    "valid_action_steps": 2,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(payload))
            samples = eval_openloop.replay_samples(path, metadata, FakeSegments())

        self.assertEqual(
            samples,
            [
                eval_openloop.EvalSample(
                    task_index=3,
                    episode_index=7,
                    frame_index=4,
                    global_frame_index=104,
                    dataset_index=12,
                    subtask="grasp",
                    segment_end=6,
                )
            ],
        )

    def test_replay_samples_rejects_horizon_crossing_current_segment(self):
        metadata = [eval_openloop.EpisodeMeta(7, 3, 100, 106)]
        payload = {
            "sample_keys": [
                {
                    "task_index": 3,
                    "episode_index": 7,
                    "frame_index": 4,
                    "valid_action_steps": 3,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "ends before the replayed valid horizon"):
                eval_openloop.replay_samples(path, metadata, FakeSegments())

    def test_valid_action_mask_excludes_segment_boundary_and_padding(self):
        mask = eval_openloop.valid_action_mask(np.asarray([False, False, False, True]), frame_index=3, segment_end=5)
        self.assertEqual(mask.tolist(), [True, True, False, False])

        with self.assertRaisesRegex(ValueError, "outside segment"):
            eval_openloop.valid_action_mask(np.asarray([False]), frame_index=5, segment_end=5)

    def test_aggregation_reports_l1_rmse_and_ignores_invalid_steps(self):
        sample = eval_openloop.EvalSample(0, 1, 0, 0, 0)
        error = np.asarray([[1.0, -1.0], [2.0, 2.0], [100.0, 100.0]])
        record = eval_openloop.ErrorRecord(sample, {"model": error}, np.asarray([True, True, False]))

        overall = eval_openloop.aggregate_errors([record])["model"]
        self.assertAlmostEqual(overall["l1"], 1.5)
        self.assertAlmostEqual(overall["rmse"], np.sqrt(2.5))
        self.assertEqual(overall["valid_action_steps"], 2)
        self.assertEqual(overall["scalar_values"], 4)

        position = eval_openloop.aggregate_errors([record], position=1)["model"]
        self.assertAlmostEqual(position["l1"], 2.0)
        self.assertAlmostEqual(position["rmse"], 2.0)

    def test_sample_diagnostics_use_common_position_and_exclude_incomplete_prefix(self):
        sample = eval_openloop.EvalSample(0, 1, 0, 0, 0)
        error = np.arange(32, dtype=np.float64)[:, None] * np.ones((1, 23))
        valid = np.arange(32) < 1
        record = eval_openloop.ErrorRecord(sample, {"model": error}, valid)
        result = eval_openloop.sample_error_diagnostics(record)
        self.assertEqual(result["first_action"]["metrics"]["model"]["l1"], 0)
        self.assertEqual(result["complete_prefix_16"]["samples"], 0)
        self.assertEqual(result["complete_prefix_16"]["metrics"], {})
        self.assertEqual(result["per_chunk_position"][1]["metrics"], {})
        full = eval_openloop.ErrorRecord(sample, {"model": error}, np.ones(32, dtype=bool))
        combined = eval_openloop.comparable_error_metrics([record, full])
        self.assertEqual(combined["first_action"]["metrics"]["model"]["valid_action_steps"], 2)
        self.assertEqual(combined["complete_prefix_16"]["samples"], 1)
        self.assertEqual(combined["complete_prefix_16"]["metrics"]["model"]["valid_action_steps"], 16)
        self.assertEqual(combined["complete_prefix_16"]["metrics"]["model"]["l1"], 7.5)

    def test_boundary_replay_validates_identity_geometry_and_noise_contract(self):
        metadata = [eval_openloop.EpisodeMeta(7, 3, 100, 106)]
        row = {"episode_index": 7, "frame_index": 5, "valid_action_steps": 1,
               "subtask": "grasp", "root_id": "root", "source_kind": "replay", "repo_id": "repo",
               "task_name": "task", "task_instance_id": 10, "boundary_id": "boundary", "sample_id": "sample",
               "bucket": "last_1", "next_subtask": "place", "segment_start": 4, "segment_end": 6,
               "next_segment_end": 10, "distance_to_boundary_frames": 1, "full_H_legal_before": False,
               "executed_K_within_original_label": False, "recorded_full_H_future_available_by_index": True,
               "training_seen_source": True, "unexpected": "not copied"}
        payload = {"kind": "matched_training_boundary_probe", "schema_version": 1,
                   "model_horizon_H": 4, "executed_prefix_K": 2, "root_id": "root", "source_kind": "replay",
                   "repo_id": "repo", "pairing_contract": {"action_noise_seed": 0}, "sample_keys": [row]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload))
            kwargs = dict(names={3: "task"}, repo_id="repo", horizon=4, action_noise_seed=0)
            samples = eval_openloop.replay_samples(path, metadata, FakeSegments(), **kwargs)
            self.assertEqual(samples[0].probe_metadata["bucket"], "last_1")
            self.assertNotIn("unexpected", samples[0].probe_metadata)
            with self.assertRaisesRegex(ValueError, "noise-seed"):
                eval_openloop.replay_samples(path, metadata, FakeSegments(), **{**kwargs, "action_noise_seed": None})
            for field, value, error in [("subtask", "wrong", "subtask"), ("task_name", "wrong", "task name"),
                                        ("segment_end", 7, "interval"), ("bucket", "edge_H", "bucket"),
                                        ("root_id", "other", "root_id")]:
                changed = copy.deepcopy(payload)
                changed["sample_keys"][0][field] = value
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                    eval_openloop.replay_samples(path, metadata, FakeSegments(), **kwargs)


class EvalOpenLoopCliTest(unittest.TestCase):
    def run_evaluation(self, extra_args, *, sample_item=None, prediction=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.txt"
            episodes.write_text("7\n")
            segments = root / "segments.parquet"
            segments.write_text("fake segment artifact for provenance hashing\n")
            checkpoint = root / "checkpoint"
            checkpoint.write_text("fake checkpoint for provenance hashing\n")
            output = root / "result.json"
            argv = [
                "eval_openloop.py", "--dataset", str(root), "--episodes-file", str(episodes),
                "--pi05-ckpt", str(checkpoint), "--out", str(output), "--horizon", "2", "--limit", "1",
                *extra_args,
            ]
            if "provided" in extra_args:
                argv.extend(["--subtask-prompts", str(segments)])
            fake_module = SimpleNamespace(
                SubtaskFromSegments=SimpleNamespace(from_parquet=lambda *args, **kwargs: FakeSegments())
            )
            fake_item = sample_item if sample_item is not None else {
                "gt_action": np.ones((2, 23)),
                "valid": np.ones(2, dtype=np.bool_),
                "hold_action": np.zeros(23),
            }
            fake_prediction = np.zeros((2, 23)) if prediction is None else prediction
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(sys.modules, {"openpi.training.subtask_data": fake_module}),
                mock.patch.object(eval_openloop, "load_episode_meta", return_value=[
                    eval_openloop.EpisodeMeta(7, 3, 0, 6)
                ]),
                mock.patch.object(eval_openloop, "task_names", return_value={3: "task"}),
                mock.patch.object(eval_openloop, "build_pi05", return_value=(
                    lambda item: fake_prediction, lambda key: fake_item
                )) as build,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_openloop.main()
            return json.loads(output.read_text()), build.call_args.args[0]

    def test_raw_actions_are_opt_in_truncated_and_hash_verifiable(self):
        default, args = self.run_evaluation([])
        self.assertFalse(args.save_raw_actions)
        self.assertNotIn("raw_actions", default["sample_keys"][0])

        target = np.arange(4 * 23, dtype=np.float64).reshape(4, 23) / 10
        prediction = -np.arange(3 * 23, dtype=np.float32).reshape(3, 23) / 20
        valid = np.asarray([True, False, True, False])
        hold = np.arange(23, dtype=np.float32) / 100
        result, args = self.run_evaluation(
            ["--save-raw-actions", "--horizon", "4"],
            sample_item={"gt_action": target, "valid": valid, "hold_action": hold},
            prediction=prediction,
        )
        self.assertTrue(args.save_raw_actions)
        record = result["sample_keys"][0]
        raw = record["raw_actions"]
        self.assertEqual(raw["schema_version"], 1)
        self.assertEqual(set(raw), {
            "schema_version", "prediction", "ground_truth", "valid_mask", "hold_action", "dtypes", "sha256",
        })
        arrays = {name: np.asarray(raw[name], dtype=dtype) for name, dtype in raw["dtypes"].items()}
        np.testing.assert_array_equal(arrays["prediction"], prediction)
        np.testing.assert_array_equal(arrays["ground_truth"], target[:3])
        np.testing.assert_array_equal(arrays["valid_mask"], valid[:3])
        np.testing.assert_array_equal(arrays["hold_action"], np.tile(hold, (3, 1)))
        self.assertEqual(raw["sha256"], eval_openloop.input_content_sha256(arrays))
        signed_error = arrays["prediction"] - arrays["ground_truth"]
        expected_l1 = float(np.abs(signed_error[arrays["valid_mask"]]).mean())
        self.assertAlmostEqual(record["error_diagnostics"]["metrics"]["model"]["l1"], expected_l1)
        self.assertTrue((signed_error < 0).any())
        self.assertTrue(all(np.isfinite(value).all() for value in arrays.values()))

    def test_raw_actions_reject_nonfinite_hold_values(self):
        with self.assertRaisesRegex(RuntimeError, "Nonfinite raw action export"):
            self.run_evaluation(["--save-raw-actions"], sample_item={
                "gt_action": np.ones((2, 23)),
                "valid": np.ones(2, dtype=np.bool_),
                "hold_action": np.full(23, np.nan),
            })

    def test_joint_budget_default_and_explicit_override_reach_model_builder(self):
        for mode, override, expected in [("none", None, 200), ("provided", None, 512), ("provided", 768, 768)]:
            with self.subTest(mode=mode, override=override):
                extra = ["--subtask-conditioning", mode]
                if override is not None:
                    extra.extend(["--max-token-len", str(override)])
                result, args = self.run_evaluation(extra)
                self.assertEqual(args.max_token_len, expected)
                self.assertEqual(result["max_token_len"], expected)

    def test_episode_filter_does_not_imply_held_out_data(self):
        result, _ = self.run_evaluation([])
        self.assertTrue(result["sampling"]["episode_filter"])
        self.assertFalse(result["sampling"]["held_out_episode_filter"])
        self.assertEqual(result["sampling"]["split_label"], "unspecified")
        self.assertFalse(result["sampling"]["training_overlap_checked"])
        self.assertNotIn("held-out", result["note"])

    def test_action_noise_seed_is_explicit_and_separate_from_sample_seed(self):
        legacy, legacy_args = self.run_evaluation([])
        self.assertIsNone(legacy_args.action_noise_seed)
        self.assertIsNone(legacy["action_noise"]["seed"])
        paired, paired_args = self.run_evaluation(["--action-noise-seed", "17", "--seed", "2"])
        self.assertEqual(paired_args.action_noise_seed, 17)
        self.assertEqual(paired["action_noise"]["seed"], 17)
        self.assertEqual(paired["seed"], 2)

    def test_normalization_asset_is_separate_from_dataset_identity(self):
        result, args = self.run_evaluation(["--repo-id", "dataset", "--norm-asset-id", "saved-robot"])
        self.assertEqual(args.repo_id, "dataset")
        self.assertEqual(args.norm_asset_id, "saved-robot")
        self.assertEqual(result["repo_id"], "dataset")
        self.assertEqual(result["normalization"]["asset_id"], "saved-robot")

    def test_explicit_split_label_preserves_unverified_overlap_caveat(self):
        for label in ("in_sample", "held_out"):
            with self.subTest(label=label):
                result, _ = self.run_evaluation(["--split-label", label])
                self.assertEqual(result["sampling"]["split_label"], label)
                self.assertEqual(result["sampling"]["held_out_episode_filter"], label == "held_out")
                self.assertIn("overlap is not checked", result["note"])


if __name__ == "__main__":
    unittest.main()

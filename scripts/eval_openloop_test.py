# ruff: noqa: PT009, PT027
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

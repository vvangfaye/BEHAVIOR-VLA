import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.subtask_data import PureSubtaskExecutor
from openpi.training.subtask_data import StrictSubtaskActionIndex
from openpi.training.subtask_data import SubtaskActionIndex
from openpi.training.subtask_data import SubtaskFromSegments
from openpi.training.subtask_data import episode_lengths_in_dataset_order


def _write_table(tmp_path, rows):
    path = tmp_path / "segments.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(ep=42, start=0, end=5, text="move to radio"):
    return {"episode_index": ep, "start": start, "end": end, "subtask": text}


def test_half_open_intervals_preserve_whole_task_and_original_ids(tmp_path):
    transform = SubtaskFromSegments.from_parquet(
        _write_table(
            tmp_path,
            [
                _row(start=5, end=9, text="press radio"),
                _row(),
            ],
        )
    )
    transform.validate_coverage({42: 9})
    for frame, expected in [(0, "move to radio"), (4, "move to radio"), (5, "press radio"), (8, "press radio")]:
        data = {"episode_index": np.array([42]), "frame_index": frame, "prompt": "Turn on the radio."}
        result = transform(data)
        assert result["subtask"] == expected
        assert result["prompt"] == "Turn on the radio."
        assert "subtask" not in data
    for ep, frame in [(42, -1), (42, 9), (0, 0)]:
        with pytest.raises(ValueError, match="No subtask annotation"):
            transform({"episode_index": ep, "frame_index": frame, "prompt": "must not become the target"})


@pytest.mark.parametrize(
    "rows",
    [
        [_row(end=0)],
        [_row(start=-1)],
        [_row(text=" ")],
        [_row(text=None)],
        [_row(), _row(start=4, end=8)],
        [_row(), _row()],
        [_row(start=0.5)],
    ],
)
def test_reject_invalid_segment_tables(tmp_path, rows):
    with pytest.raises(ValueError, match=r"Invalid|Empty|Overlapping|integer|Null|empty"):
        SubtaskFromSegments.from_parquet(_write_table(tmp_path, rows))


def test_gap_and_tail_do_not_inherit_stale_labels(tmp_path):
    transform = SubtaskFromSegments.from_parquet(_write_table(tmp_path, [_row(), _row(start=7, end=10)]))
    with pytest.raises(ValueError, match="without gaps"):
        transform.validate_coverage({42: 10})
    with pytest.raises(ValueError, match="No subtask annotation"):
        transform({"episode_index": 42, "frame_index": 6})
    with pytest.raises(ValueError, match="Missing subtask annotations"):
        transform.validate_coverage({99: 10})


@pytest.mark.parametrize("value", [1.5, np.array([0, 1]), "0"])
def test_indices_cannot_be_silently_truncated(tmp_path, value):
    transform = SubtaskFromSegments.from_parquet(_write_table(tmp_path, [_row()]))
    with pytest.raises(ValueError, match="single integer"):
        transform({"episode_index": 42, "frame_index": value})


def test_selected_episode_filter_runs_before_label_validation(tmp_path):
    path = _write_table(tmp_path, [_row(ep=5, text=""), _row(), _row(ep=99, text=None)])
    transform = SubtaskFromSegments.from_parquet(path, episodes_index=[42])
    transform.validate_coverage({42: 5})
    assert set(transform.segments) == {42}
    assert transform.dropped_episodes == {}
    with pytest.raises(ValueError, match="Empty subtask label: episode=5"):
        SubtaskFromSegments.from_parquet(path)
    with pytest.raises(ValueError, match="No episodes selected"):
        SubtaskFromSegments.from_parquet(path, episodes_index=[])


@pytest.mark.parametrize("text", ["", " \n\t", None])
def test_drop_empty_label_removes_whole_episode_without_filling_targets(tmp_path, text):
    path = _write_table(tmp_path, [_row(), _row(start=5, end=10, text=text), _row(ep=99)])
    transform = SubtaskFromSegments.from_parquet(path, empty_label_policy="drop_episode")
    assert transform.dropped_episodes == {42: [(5, 10)]}
    assert set(transform.segments) == {99}
    transform.validate_coverage({99: 5})
    with pytest.raises(ValueError, match="No subtask annotation"):
        transform({"episode_index": 42, "frame_index": 0})


def test_drop_policy_still_rejects_malformed_intervals(tmp_path):
    path = _write_table(tmp_path, [_row(text=""), _row(start=-1)])
    with pytest.raises(ValueError, match="Invalid subtask interval"):
        SubtaskFromSegments.from_parquet(path, empty_label_policy="drop_episode")


def test_strict_action_index_stays_inside_each_segment_and_preserves_global_stride(tmp_path):
    labels = SubtaskFromSegments.from_parquet(
        _write_table(
            tmp_path,
            [
                _row(ep=42, start=0, end=5, text="move"),
                _row(ep=42, start=5, end=10, text="move"),
                _row(ep=99, start=0, end=6, text="place"),
            ],
        )
    )
    action_index = StrictSubtaskActionIndex.from_segments(labels, {42: 10, 99: 6}, action_horizon=3, frame_stride=2)

    assert action_index.raw_frames == 16
    assert action_index.eligible_anchors == 10
    assert action_index.boundary_rejected_anchors == 6
    assert [action_index.resolve(i) for i in range(len(action_index))] == [
        (0, 42, 0),
        (2, 42, 2),
        (6, 42, 6),
        (10, 99, 0),
        (12, 99, 2),
    ]
    assert action_index.resolve(-1) == (12, 99, 2)
    with pytest.raises(IndexError):
        action_index.resolve(5)


def test_strict_action_index_rejects_dataset_with_no_full_horizon(tmp_path):
    labels = SubtaskFromSegments.from_parquet(_write_table(tmp_path, [_row(end=3)]))
    with pytest.raises(ValueError, match="No action anchors"):
        StrictSubtaskActionIndex.from_segments(labels, {42: 3}, action_horizon=4)


@pytest.mark.parametrize("role_column", ["annotated", "is_annotated"])
def test_executor_index_is_role_aware_capped_and_deterministic(tmp_path, role_column):
    rows = [
        {**_row(start=0, end=7, text="move"), role_column: True},
        {**_row(start=7, end=10, text="imputed"), role_column: False},
        {**_row(start=10, end=14, text="press"), role_column: True},
    ]
    labels = SubtaskFromSegments.from_parquet(_write_table(tmp_path, rows))
    action_index = SubtaskActionIndex.from_segments(
        labels,
        {42: 14},
        action_horizon=4,
        frame_stride=2,
        max_samples_per_segment=3,
    )

    assert labels.annotation_column == role_column
    assert labels.role_counts() == {
        "annotated_segments": 2,
        "imputed_segments": 1,
        "annotated_frames": 11,
        "imputed_frames": 3,
    }
    assert action_index.eligible_anchors == 7
    assert action_index.capped_sample_count == 6
    assert [action_index.resolve(i) for i in range(len(action_index))] == [
        (0, 42, 0),
        (4, 42, 4),
        (6, 42, 6),
        (10, 42, 10),
        (12, 42, 12),
        (13, 42, 13),
    ]
    assert action_index.valid_action_step_distribution() == {1: 2, 2: 1, 3: 1, 4: 2}


def test_executor_cap_one_keeps_segment_final_frame(tmp_path):
    labels = SubtaskFromSegments.from_parquet(
        _write_table(tmp_path, [{**_row(start=0, end=7, text="move"), "is_annotated": True}])
    )
    action_index = SubtaskActionIndex.from_segments(
        labels,
        {42: 7},
        action_horizon=4,
        frame_stride=2,
        max_samples_per_segment=1,
    )

    assert action_index.eligible_anchors == 4
    assert action_index.resolve(0) == (6, 42, 6)
    assert action_index.valid_action_step_distribution() == {1: 1}


def test_role_aware_empty_imputed_interval_is_coverage_only(tmp_path):
    rows = [
        {
            **_row(start=0, end=2, text=""),
            "is_annotated": False,
            "provenance": "imputed_leading",
        },
        {
            **_row(start=2, end=5, text="move"),
            "is_annotated": True,
            "provenance": "annotated",
        },
    ]
    labels = SubtaskFromSegments.from_parquet(_write_table(tmp_path, rows), require_role_provenance=True)
    labels.validate_coverage({42: 5})
    assert labels.dropped_episodes == {}
    assert labels.provenance_counts() == {
        "segments": {"imputed_leading": 1, "annotated": 1},
        "frames": {"imputed_leading": 2, "annotated": 3},
    }
    action_index = SubtaskActionIndex.from_segments(labels, {42: 5}, action_horizon=2)
    assert [action_index.resolve(i) for i in range(len(action_index))] == [
        (2, 42, 2),
        (3, 42, 3),
        (4, 42, 4),
    ]


def test_executor_requires_role_aware_provenance(tmp_path):
    with pytest.raises(ValueError, match="requires role_aware_v1"):
        SubtaskFromSegments.from_parquet(_write_table(tmp_path, [_row()]), require_role_provenance=True)


def test_legacy_roles_default_true_and_executor_mask_combines_boundary_and_padding(tmp_path):
    labels = SubtaskFromSegments.from_parquet(
        _write_table(tmp_path, [_row(start=0, end=3, text="move"), _row(start=3, end=6, text="press")])
    )
    assert labels.annotation_column is None
    assert labels.annotation_flags(42).tolist() == [True, True]
    transform = PureSubtaskExecutor(labels, action_horizon=4, action_pad_key="action_is_pad")

    result = transform(
        {
            "episode_index": np.int64(42),
            "frame_index": np.int64(2),
            "prompt": "must be replaced",
            "action_is_pad": np.asarray([False, True, True, True]),
        }
    )
    assert result["prompt"] == "move"
    assert result["action_loss_mask"].tolist() == [True, False, False, False]

    terminal = transform(
        {
            "episode_index": np.int64(42),
            "frame_index": np.int64(5),
            "action_is_pad": np.asarray([False, True, True, True]),
        }
    )
    assert terminal["action_loss_mask"].tolist() == [True, False, False, False]


def test_executor_lookup_rejects_imputed_segment_without_fallback(tmp_path):
    labels = SubtaskFromSegments.from_parquet(
        _write_table(
            tmp_path,
            [
                {**_row(start=0, end=2, text="annotated"), "annotated": True},
                {**_row(start=2, end=4, text="imputed"), "annotated": False},
            ],
        )
    )
    transform = PureSubtaskExecutor(labels, action_horizon=2, action_pad_key="action_is_pad")
    with pytest.raises(ValueError, match="imputed"):
        transform(
            {
                "episode_index": 42,
                "frame_index": 2,
                "prompt": "whole task fallback is forbidden",
                "action_is_pad": np.zeros(2, dtype=bool),
            }
        )


def test_episode_lengths_follow_source_indices(tmp_path):
    episodes = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [99, 42],
                "length": [6, 10],
                "dataset_from_index": [10, 0],
            }
        ),
        episodes / "file-000.parquet",
    )

    assert episode_lengths_in_dataset_order(tmp_path, [99, 42]) == {42: 10, 99: 6}

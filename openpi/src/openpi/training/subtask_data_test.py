import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


@pytest.mark.parametrize("empty_text", [None, "", "   "])
def test_joint_rejects_empty_imputed_labels(tmp_path, empty_text):
    rows = [
        {**_row(start=0, end=2, text=empty_text), "is_annotated": False, "provenance": "imputed_leading"},
        {**_row(start=2, end=5, text="move"), "is_annotated": True, "provenance": "annotated"},
    ]
    segment_path = _write_table(tmp_path, rows)
    with pytest.raises(ValueError, match="Empty subtask label"):
        SubtaskFromSegments.from_parquet(segment_path)
    labels = SubtaskFromSegments.from_parquet(segment_path, empty_label_policy="drop_episode")
    assert labels.segments == {}
    assert labels.dropped_episodes == {42: [(0, 2)]}


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

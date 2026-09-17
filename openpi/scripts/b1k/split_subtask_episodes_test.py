import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.b1k import split_subtask_episodes as splitter


def _write_dataset(root: Path, rows: list[dict]) -> None:
    episode_dir = root / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [row["episode_index"] for row in rows],
                "task_index": [row["task_index"] for row in rows],
                "task_instance_id": [row["task_instance_id"] for row in rows],
            }
        ),
        episode_dir / "file-000.parquet",
    )


def _rows() -> list[dict]:
    return [
        {"episode_index": task * 100 + group * 10 + repeat, "task_index": task, "task_instance_id": group}
        for task in (14, 56)
        for group in range(6)
        for repeat in range(2)
    ]


def _write_retained(path: Path, rows: list[dict]) -> list[int]:
    episode_ids = [row["episode_index"] for row in reversed(rows)]
    path.write_text("".join(f"{episode_id}\n" for episode_id in episode_ids))
    return episode_ids


def test_split_is_deterministic_stratified_and_leakage_free(tmp_path):
    dataset = tmp_path / "dataset"
    rows = _rows()
    _write_dataset(dataset, rows)
    retained = tmp_path / "retained.txt"
    retained_ids = _write_retained(retained, rows)

    first = splitter.create_split(dataset, retained, [56, 14], tmp_path / "first", val_fraction=0.34, seed=7)
    second = splitter.create_split(dataset, retained, [56, 14], tmp_path / "second", val_fraction=0.34, seed=7)

    assert first == second
    assert (tmp_path / "first" / "train_episodes.txt").read_bytes() == (
        tmp_path / "second" / "train_episodes.txt"
    ).read_bytes()
    assert (tmp_path / "first" / "val_episodes.txt").read_bytes() == (
        tmp_path / "second" / "val_episodes.txt"
    ).read_bytes()
    assert json.loads((tmp_path / "first" / "split_manifest.json").read_text()) == first

    train_ids = {int(value) for value in (tmp_path / "first" / "train_episodes.txt").read_text().split()}
    val_ids = {int(value) for value in (tmp_path / "first" / "val_episodes.txt").read_text().split()}
    assert not train_ids & val_ids
    assert train_ids | val_ids == set(retained_ids)
    task_by_episode = {row["episode_index"]: row["task_index"] for row in rows}
    assert {task_by_episode[episode_id] for episode_id in train_ids} == {14, 56}
    assert {task_by_episode[episode_id] for episode_id in val_ids} == {14, 56}
    for task in (14, 56):
        stats = first["stats"]["per_task"][str(task)]
        assert stats["train_groups"] == 4
        assert stats["val_groups"] == 2

    side_by_group = {}
    for assignment in first["group_assignments"]:
        key = (assignment["task_index"], assignment["task_instance_id"])
        assert key not in side_by_group
        side_by_group[key] = assignment["split"]
        expected_side = train_ids if assignment["split"] == "train" else val_ids
        assert set(assignment["episode_indices"]) <= expected_side
    train_digest = hashlib.sha256((tmp_path / "first" / "train_episodes.txt").read_bytes()).hexdigest()
    val_digest = hashlib.sha256((tmp_path / "first" / "val_episodes.txt").read_bytes()).hexdigest()
    assert first["outputs"]["train_episodes"]["sha256"] == train_digest
    assert first["outputs"]["val_episodes"]["sha256"] == val_digest
    assert len(first["inputs"]["episode_metadata_sha256"]) == 64


def test_different_seeds_can_change_group_assignment(tmp_path):
    dataset = tmp_path / "dataset"
    rows = _rows()
    _write_dataset(dataset, rows)
    retained = tmp_path / "retained.txt"
    retained_ids = _write_retained(retained, rows)
    metadata, _ = splitter.read_episode_metadata(dataset)

    _, val_seed_1, _, _ = splitter.split_episode_groups(metadata, retained_ids, [14, 56], seed=1)
    _, val_seed_2, _, _ = splitter.split_episode_groups(metadata, retained_ids, [14, 56], seed=2)

    assert val_seed_1 != val_seed_2


@pytest.mark.parametrize(
    ("retained_ids", "task_indices", "match"),
    [
        ([1400, 999999], [14], "Unknown requested episode IDs"),
        ([1400, 1400], [14], "Duplicate requested episode IDs"),
        ([1400, 1410], [14, 56], "no retained episodes"),
    ],
)
def test_split_rejects_invalid_requested_ids_and_tasks(tmp_path, retained_ids, task_indices, match):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, _rows())
    metadata, _ = splitter.read_episode_metadata(dataset)

    with pytest.raises(ValueError, match=match):
        splitter.split_episode_groups(metadata, retained_ids, task_indices)


def test_split_rejects_missing_groups_empty_task_side_and_overlap():
    with pytest.raises(ValueError, match="Missing task_instance_id"):
        splitter.split_episode_groups(
            {
                1: {"episode_index": 1, "task_index": 14, "task_instance_id": None},
                2: {"episode_index": 2, "task_index": 14, "task_instance_id": 2},
            },
            [1, 2],
            [14],
        )
    with pytest.raises(ValueError, match="at least two task-instance groups"):
        splitter.split_episode_groups(
            {
                1: {"episode_index": 1, "task_index": 14, "task_instance_id": 1},
                2: {"episode_index": 2, "task_index": 14, "task_instance_id": 1},
            },
            [1, 2],
            [14],
        )
    with pytest.raises(ValueError, match="overlap"):
        splitter.validate_episode_split([1, 2], [2, 3], [1, 2, 3])


def test_create_split_refuses_nonempty_output_and_duplicate_file_ids(tmp_path):
    dataset = tmp_path / "dataset"
    rows = _rows()
    _write_dataset(dataset, rows)
    retained = tmp_path / "retained.txt"
    retained.write_text("1400\n1400\n")
    with pytest.raises(ValueError, match="Duplicate requested episode IDs"):
        splitter.create_split(dataset, retained, [14], tmp_path / "new-output")

    retained.write_text("".join(f"{row['episode_index']}\n" for row in rows if row["task_index"] == 14))
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="nonempty output directory"):
        splitter.create_split(dataset, retained, [14], output)
    assert (output / "keep.txt").read_text() == "do not overwrite"

"""Mixed export contract tests use real HDF5/parquet and a fake codec/reader."""
import copy
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_codex_training_data as data
from test_export_codex_collection import trajectory


class NumericRows:
    def __init__(self, rows, columns=None):
        self.rows, self.columns = rows, columns

    def select_columns(self, columns):
        return NumericRows(self.rows, columns)

    def __getitem__(self, item):
        rows = self.rows[item]
        return {key: [row[key] for row in rows] for key in self.columns}


class Dataset:
    roots = {}

    def __init__(self, *, root, episodes=None, delta_timestamps=None, **kwargs):
        assert kwargs["tolerance_s"] == 5e-3
        stored = self.roots[str(Path(root).resolve())]
        self.__dict__.update(stored.__dict__)
        self.rows = [r for r in stored.rows if episodes is None or r["episode_index"] in episodes]
        self.horizon = len(delta_timestamps["action"]) if delta_timestamps else None
        self.hf_dataset = NumericRows(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = copy.deepcopy(self.rows[index])
        if self.horizon:
            last = index
            while last + 1 < len(self.rows) and self.rows[last + 1]["episode_index"] == row["episode_index"]:
                last += 1
            row["action"] = np.stack([self.rows[min(index + i, last)]["action"] for i in range(self.horizon)])
        for key in data.CAMERAS:
            row[key] = row[key].transpose(2, 0, 1).astype(np.float32) / 255
        return row

    @classmethod
    def create(cls, *, root, fps, features, **kwargs):
        result = object.__new__(cls)
        result.root = Path(root).resolve(); result.root.mkdir(parents=True)
        result.meta = SimpleNamespace(fps=fps, features=features)
        result.rows, result.pending, result.episodes, result.tasks = [], [], [], []
        result.horizon = None
        cls.roots[str(result.root)] = result
        return result

    def add_frame(self, frame):
        self.pending.append(copy.deepcopy(frame))

    def save_episode(self):
        ep = len(self.episodes)
        for i, row in enumerate(self.pending):
            if row["task"] not in self.tasks:
                self.tasks.append(row["task"])
            row.update(episode_index=ep, frame_index=i, task_index=self.tasks.index(row["task"]))
            self.rows.append(row)
        self.episodes.append(dict(episode_index=ep, length=len(self.pending)))
        self.pending = []

    def finalize(self):
        target = self.root / "meta/episodes/chunk-000"
        target.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.episodes), target / "file-000.parquet")
        self.hf_dataset = NumericRows(self.rows)


@pytest.fixture
def inputs(tmp_path):
    teacher = trajectory(tmp_path / "teacher.hdf5")
    info = data.inspect_codex_collection_trajectory(teacher)
    root = tmp_path / "base"
    ds = Dataset.create(root=root, fps=30, features=data._features(info))
    prompt = data.canonical_task_prompts()["turning_on_radio"]
    for i in range(4):
        ds.add_frame({"action": np.full(23, 10 + i, np.float32), "observation.state": np.full(57, 10 + i, np.float32),
                      **{key: np.full((8, 8, 3), 10 + i, np.uint8) for key in data.CAMERAS}, "task": prompt})
    ds.save_episode(); ds.finalize()
    (root / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task_name="turning_on_radio", task=prompt)) + "\n")
    meta = root / "meta/episodes/chunk-000/file-000.parquet"
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0, length=4, task_index=0, task_instance_id=114,
                                            task_name="turning_on_radio", tasks=[prompt])]), meta)
    labels = root / "labels.parquet"
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0, start=0, end=2, subtask="move to radio"),
                                        dict(episode_index=0, start=2, end=4, subtask="turn on radio")]), labels)
    return dict(trajectories=[teacher], output=tmp_path / "mixed", repo_id="fixture/mixed", base_dataset=root,
                base_repo_id="fixture/base", base_segments=labels, base_episodes=[0], dataset_class=Dataset)


def test_mixing_preserves_dynamic_labels_and_numeric_alignment_without_claiming_real_codec(inputs):
    report = data.export_mixed(**inputs)
    assert report["new_frames"] == 3 and report["original_frames"] == 4
    assert report["new_frame_fraction"] == 3 / 7 and report["tasks"] == ["turning_on_radio"]
    result = data.verify_mixed(inputs["output"], dataset_class=Dataset)
    assert result["aligned_source_frames_verified"] == 3
    assert result["original_source_frames_verified"] == 4
    assert result["complete"] is False and result["real_lerobot_reader"] is False
    assert result["action_horizon"] == 32 and result["decoded_frame_samples"] == 7
    assert result["reader_tolerance_s"] == report["reader_tolerance_s"] == 5e-3


def test_export_and_both_verification_readers_use_training_tolerance(inputs):
    calls = []
    class RecordingDataset(Dataset):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            super().__init__(**kwargs)
    inputs["dataset_class"] = RecordingDataset
    data.export_mixed(**inputs)
    data.verify_mixed(inputs["output"], dataset_class=RecordingDataset)
    assert [Path(call["root"]) for call in calls] == [inputs["base_dataset"], inputs["output"], inputs["base_dataset"]]
    assert all(call["tolerance_s"] == 5e-3 and call["video_backend"] == "pyav" for call in calls)
    config_path = Path(data.__file__).resolve().parents[1] / "openpi/src/openpi/training/config.py"
    config = next(node for node in ast.walk(ast.parse(config_path.read_text()))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainConfig"
                  and any(key.arg == "name" and isinstance(key.value, ast.Constant) and key.value.value == "pi05_b1k"
                          for key in node.keywords))
    kwargs = next(key.value for node in ast.walk(config) if isinstance(node, ast.Call)
                  for key in node.keywords if key.arg == "dataset_kwargs")
    assert ast.literal_eval(kwargs)["tolerance_s"] == data.READER_TOLERANCE_S


def test_reader_tolerance_is_frozen_in_export_contract(inputs):
    data.export_mixed(**inputs)
    path = inputs["output"] / "export_report.json"
    report = data.read(path);report["reader_tolerance_s"] = 0.02
    report["fingerprint"] = data._digest({key: value for key, value in report.items() if key != "fingerprint"})
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reader tolerance"):
        data.verify_mixed(inputs["output"], dataset_class=Dataset)


@pytest.mark.parametrize("phase", ["export", "verify"])
def test_timestamp_decode_failure_is_never_skipped(inputs, phase):
    class FrameTimestampError(Exception):
        pass
    reads = []
    class FailingDataset(Dataset):
        def __getitem__(self, index):
            reads.append(index)
            raise FrameTimestampError("outside configured tolerance")
    if phase == "verify":data.export_mixed(**inputs)
    with pytest.raises(FrameTimestampError, match="outside configured tolerance"):
        if phase == "export":
            inputs["dataset_class"] = FailingDataset
            data.export_mixed(**inputs)
        else:data.verify_mixed(inputs["output"], dataset_class=FailingDataset)
    assert len(reads) == 1
    if phase == "export":assert not (inputs["output"] / "export_report.json").exists()


def test_released_snake_case_task_names_map_to_verified_full_prompt(inputs):
    path = inputs["base_dataset"] / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist(); rows[0]["tasks"] = ["turning_on_radio"]; del rows[0]["task_name"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    original = Dataset.roots[str(inputs["base_dataset"].resolve())]
    for row in original.rows:
        row["task"] = "turning_on_radio"
    data.export_mixed(**inputs)
    exported = Dataset.roots[str(inputs["output"].resolve())]
    assert {row["task"] for row in exported.rows} == {data.canonical_task_prompts()["turning_on_radio"]}
    assert data.verify_mixed(inputs["output"], dataset_class=Dataset)["frames"] == 7


@pytest.mark.parametrize("damage", ["canonical_prompt", "task_index", "decoded_name", "decoded_frame"])
def test_name_compatibility_does_not_weaken_task_and_frame_identity(inputs, damage):
    if damage == "canonical_prompt":
        path = inputs["base_dataset"] / "meta/tasks.jsonl"
        definition = json.loads(path.read_text()); definition["task"] = "Do another task"
        path.write_text(json.dumps(definition) + "\n")
    elif damage == "task_index":
        path = inputs["base_dataset"] / "meta/tasks.jsonl"
        definition = json.loads(path.read_text()); definition["task_name"] = "putting_shoes_on_rack"
        definition["task"] = data.canonical_task_prompts()["putting_shoes_on_rack"]
        path.write_text(json.dumps(definition) + "\n")
    else:
        row = Dataset.roots[str(inputs["base_dataset"].resolve())].rows[0]
        if damage == "decoded_name": row["task"] = "putting_shoes_on_rack"
        else: row["frame_index"] = 1
    with pytest.raises(ValueError): data.export_mixed(**inputs)
    assert not (inputs["output"] / "export_report.json").exists()


@pytest.mark.parametrize("damage", ["duplicate_teacher", "failed_teacher", "duplicate_original", "reserved_original", "wrong_task", "empty_label", "gap"])
def test_mixed_sources_reject_invalid_or_reserved_inputs_before_output(inputs, damage):
    if damage == "duplicate_teacher":
        inputs["trajectories"] *= 2
    elif damage == "failed_teacher":
        other = inputs["output"].parent / "failure.hdf5"
        inputs["trajectories"] = [trajectory(other, success=False)]
    elif damage == "duplicate_original":
        inputs["base_episodes"] *= 2
    elif damage in ("reserved_original", "wrong_task"):
        path = inputs["base_dataset"] / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        if damage == "reserved_original": rows[0]["task_instance_id"] = 301
        else: rows[0]["task_name"] = "putting_shoes_on_rack"
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        rows = pq.read_table(inputs["base_segments"]).to_pylist()
        if damage == "empty_label": rows[0]["subtask"] = ""
        else: rows[1]["start"] = 3
        pq.write_table(pa.Table.from_pylist(rows), inputs["base_segments"])
    with pytest.raises(ValueError): data.export_mixed(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("damage", ["action", "state", "image", "frame_identity", "original_source", "labels", "selection", "report"])
def test_reader_gate_rejects_shifted_or_changed_training_data(inputs, damage):
    data.export_mixed(**inputs)
    root = inputs["output"]
    ds = Dataset.roots[str(root.resolve())]
    if damage in ("action", "state", "image", "frame_identity"):
        key = {"action": "action", "state": "observation.state", "image": next(iter(data.CAMERAS)), "frame_identity": "frame_index"}[damage]
        ds.rows[0][key] = ds.rows[0][key] + (100 if damage == "image" else 1)
    elif damage == "original_source":
        Dataset.roots[str(inputs["base_dataset"].resolve())].rows[0]["action"] += 1
    elif damage == "labels":
        path = root / "meta/subtask_segments.parquet"
        rows = pq.read_table(path).to_pylist(); rows[0]["subtask"] = "changed"
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif damage == "selection":
        (root / "retained_episodes.txt").write_text("0\n")
    else:
        path = root / "export_report.json"; report = data.read(path); report["frames"] = 99
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError): data.verify_mixed(root, dataset_class=Dataset)

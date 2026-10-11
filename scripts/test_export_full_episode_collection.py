"""Real HDF5/Parquet contracts; injected readers never establish codec readiness."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_full_episode_collection as export
from full_episode_trajectory import NativeFullEpisodeTrajectory
from subtask_trajectory import CAMERAS


class MockWriter:
    created = None

    @classmethod
    def create(cls, **kwargs):
        writer = cls()
        writer.root = kwargs["root"]
        (writer.root / "meta").mkdir(parents=True)
        (writer.root / "meta/info.json").write_text(json.dumps({"fps": kwargs["fps"], "features": kwargs["features"]}))
        writer.frames, writer.episodes, writer.pending = [], [], []
        writer.finalized = False
        cls.created = writer
        return writer

    def add_frame(self, frame):
        self.pending.append(copy.deepcopy(frame))

    def save_episode(self):
        self.episodes.append({"episode_index": len(self.episodes), "length": len(self.pending)})
        self.frames.append(self.pending)
        self.pending = []

    def finalize(self):
        self.finalized = True
        folder = self.root / "meta/episodes/chunk-000"
        folder.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.episodes), folder / "file-000.parquet")


class MockHFDataset:
    def __init__(self, rows, columns=None):
        self.rows, self.columns = rows, columns

    def select_columns(self, columns):
        return MockHFDataset(self.rows, columns)

    def __getitem__(self, item):
        rows = self.rows[item]
        return {key: [row[key] for row in rows] for key in self.columns}


class MockReader:
    def __init__(self, **kwargs):
        writer = MockWriter.created
        assert Path(kwargs["root"]) == writer.root
        self.horizon = len(kwargs["delta_timestamps"]["action"])
        self.rows, self.episode_rows, self.positions = [], [], []
        tasks = {row["task"]: row["task_index"] for row in map(json.loads, (writer.root / "meta/tasks.jsonl").read_text().splitlines())}
        for episode, frames in enumerate(writer.frames):
            episode_rows = []
            for frame, values in enumerate(frames):
                row = {**values, "episode_index": episode, "frame_index": frame, "task_index": tasks[values["task"]]}
                self.positions.append((episode, frame))
                self.rows.append(row)
                episode_rows.append(row)
            self.episode_rows.append(episode_rows)
        self.hf_dataset = MockHFDataset(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = copy.deepcopy(self.rows[item])
        episode, frame = self.positions[item]
        rows = self.episode_rows[episode]
        row["action"] = np.stack([rows[min(frame + step, len(rows) - 1)]["action"] for step in range(self.horizon)])
        for key in CAMERAS:
            row[key] = row[key].transpose(2, 0, 1).astype(np.float32) / 255
        return row


def trajectory(path, *, instance=301, attempt="first", offset=0, success=True, truncated=False,
               fps=30, empty=False, same_label_source_changes=False):
    metadata = {"task": "turning_on_radio", "instance_id": instance, "rollout_id": 0,
                "run_id": "synthetic-native", "attempt_id": attempt, "checkpoint": {"step": 1375000},
                "source_manifest_sha256": "a" * 64, "fps": fps,
                "task_prompt": export.canonical_task_prompts()["turning_on_radio"]}
    with NativeFullEpisodeTrajectory(path, metadata) as writer:
        for step in range(80):
            frame = {"observation.state": np.full(57, offset + step, np.float32),
                     **{key: np.full((8, 8, 3), offset + step, np.uint8) for key in CAMERAS}}
            if empty:
                label, source = None, None
            elif same_label_source_changes:
                label, source = "turn on radio", "generated" if step % 16 == 0 else "cached"
            elif step < 40:
                label, source = "move to radio", "generated"
            elif step < 44:
                label, source = None, None
            else:
                label, source = "turn on radio", "cached"
            writer.record(frame, np.full(23, offset + step, np.float32), step == 79 and not truncated,
                          step == 79 and truncated, label, source)
        writer.finish({"task": metadata["task"], "instance_id": instance, "rollout_id": 0, "steps": 80,
                       "success": success, "terminated": not truncated, "truncated": truncated,
                       "final_goal_state": {"success": success},
                       "conditioning": {"mode": "whole_task", "oracle": False, "privileged": False}},
                      terminal_observation=frame)
    return path


def run_export(paths, output, **kwargs):
    return export.export_full_episode_collection(paths, output, "fixture/native", dataset_class=MockWriter, **kwargs)


def synthetic_verified_marker(root):
    """Synthetic proof shape for index validation, never a real-codec test result."""
    report, _, _ = export.load_export(root)
    proof = {"kind": export.VERIFICATION_KIND, "complete": True, "real_lerobot_reader": True,
             "real_codec_reader_verified": True, "aligned_source_frames_verified": report["frames"],
             "source_sha256": report["source_sha256"], "label_sha256": report["label_sha256"],
             "dataset_metadata_sha256": export.metadata_hashes(root), "test_fixture_only": True}
    export.write_json(root / "native_format_verification.json", proof)
    export.write_json(root / export.VERIFIED_MARKER,
                      {"source_sha256": report["source_sha256"], "label_sha256": report["label_sha256"],
                       "verification_sha256": export.file_sha256(root / "native_format_verification.json")})


def test_export_preserves_whole_episode_empty_intervals_and_raw_weak_source_trace(tmp_path):
    source = trajectory(tmp_path / "source.hdf5")
    root = tmp_path / "dataset"
    report = run_export([source], root)
    assert report["frames"] == 80 and report["episodes"] == 1 and report["unlabelled_frames"] == 4
    assert report["valid_labelled_anchors"] == 14
    assert report["training_instance_min"] == 301 and report["training_seen_instances"] == [["turning_on_radio", 301]]
    assert report["ready_for_training"] is False and report["real_lerobot_writer"] is False
    assert report["subtask_success_verified"] is False
    assert not (root / export.VERIFIED_MARKER).exists()
    labels = pq.read_table(root / "meta/subtask_segments.parquet").to_pylist()
    assert [(row["start"], row["end"], row["subtask"], row["subtask_source"]) for row in labels] == [
        (0, 40, "move to radio", "generated"), (40, 44, None, None), (44, 80, "turn on radio", "cached")]
    assert all(not row["is_human_annotated"] and not row["subtask_success_verified"] for row in labels)
    assert MockWriter.created.finalized and len(MockWriter.created.frames[0]) == 80
    for step, frame in enumerate(MockWriter.created.frames[0]):
        np.testing.assert_array_equal(frame["action"], np.full(23, step, np.float32))
        np.testing.assert_array_equal(frame["observation.state"], np.full(57, step, np.float32))
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    assert episode["training_seen"] is True and episode["independent_evaluation_eligible"] is False
    assert (root / "retained_episodes.txt").read_text() == "0\n"


@pytest.mark.parametrize("damage", ["failed", "truncated", "held_out", "preserved_validation", "wrong_fps", "duplicate", "changed_actions"])
def test_invalid_bc_sources_fail_before_creating_output(tmp_path, damage):
    path = trajectory(tmp_path / "source.hdf5", success=damage not in {"failed", "truncated"},
                      truncated=damage == "truncated", fps=20 if damage == "wrong_fps" else 30)
    paths, kwargs = [path], {}
    if damage == "held_out":
        kwargs["held_out_instances"] = [["turning_on_radio", 301]]
    if damage == "preserved_validation":
        kwargs["training_instance_min"] = 311
    if damage == "duplicate":
        paths.append(path)
    if damage == "changed_actions":
        with h5py.File(path, "r+") as stream:
            stream["action"][3] = np.ones(23, np.float32)
    root = tmp_path / "rejected"
    with pytest.raises(ValueError):
        run_export(paths, root, **kwargs)
    assert not root.exists()


def test_preflight_counts_failed_and_validation_attempts_without_relabelling(tmp_path):
    sources = [trajectory(tmp_path / "success.hdf5", instance=311),
               trajectory(tmp_path / "failed.hdf5", success=False, attempt="failed", offset=1),
               trajectory(tmp_path / "truncated.hdf5", success=False, truncated=True, attempt="cut", offset=2)]
    report = export.inspect_inputs(sources, training_instance_min=311)
    assert report["attempts"] == 3 and report["raw_frames"] == 240
    assert report["bc_eligible_episodes"] == 1 and report["bc_eligible_frames"] == 80
    assert report["valid_labelled_anchors"] == 14 and report["official_benchmark"] is False


def test_cached_source_transition_does_not_remove_all_training_windows(tmp_path):
    source = trajectory(tmp_path / "source.hdf5", same_label_source_changes=True)
    root = tmp_path / "dataset"
    report = run_export([source], root)
    assert report["segments"] == 10
    assert report["valid_labelled_anchors"] == 49
    rows = pq.read_table(root / "meta/subtask_segments.parquet").to_pylist()
    assert export.training_intervals(rows) == [dict(episode_index=0, start=0, end=80, subtask="turn on radio")]


def test_injected_reader_verifies_alignment_but_cannot_create_training_proof(tmp_path):
    root = tmp_path / "dataset"
    run_export([trajectory(tmp_path / "source.hdf5")], root)
    result = export.verify_export(root, dataset_class=MockReader)
    assert result["complete"] is True and result["aligned_source_frames_verified"] == 80
    assert result["real_lerobot_reader"] is False and result["ready_for_training"] is False
    assert not (root / export.VERIFIED_MARKER).exists()
    with pytest.raises(FileNotFoundError):
        export.prepare_training_index(root, root_id="native", held_out_instances=[])


def test_reader_alignment_failure_never_produces_verification_marker(tmp_path):
    root = tmp_path / "dataset"
    run_export([trajectory(tmp_path / "source.hdf5")], root)
    MockWriter.created.frames[0][3]["action"] = np.zeros(23, np.float32)
    with pytest.raises(ValueError, match="alignment"):
        export.verify_export(root, dataset_class=MockReader)
    assert not (root / export.VERIFIED_MARKER).exists()


def test_native_index_matches_existing_mixture_and_excludes_empty_and_tail_windows(tmp_path):
    root = tmp_path / "dataset"
    run_export([trajectory(tmp_path / "source.hdf5")], root)
    synthetic_verified_marker(root)
    held = [["turning_on_radio", 1]]
    index = export.prepare_training_index(root, root_id="native", held_out_instances=held)
    assert index["held_out_instances"] == held
    assert index["source_kind"] == "collected" and index["provenance"]["native_collection_kind"] == "native_full_episode_collection"
    assert index["intervals"] == [dict(episode_index=0, start=0, end=40, subtask="move to radio"),
                                   dict(episode_index=0, start=44, end=80, subtask="turn on radio")]
    assert index["provenance"]["valid_anchors"] == 14
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
    from openpi.training.subtask_mixture import MixturePlan
    path = tmp_path / "native_index.json"
    export.write_json(path, index)
    manifest = {"schema_version": 1, "kind": "b1k_three_source_mixture", "seed": 42,
                "global_batch_size": 128, "action_horizon": 32,
                "phases": [{"start_step": 0, "end_step": 5, "sources": [{"kind": "collected", "weight": 1,
                    "indexes": [{"path": str(path), "sha256": export.file_sha256(path)}]}]}]}
    plan = MixturePlan(manifest, base_dir=tmp_path)
    for number in range(plan.num_samples):
        ref = plan.reference(number)
        assert (ref["frame"] <= 8 and ref["subtask"] == "move to radio") or (
            44 <= ref["frame"] <= 48 and ref["subtask"] == "turn on radio")
    mixture_path = tmp_path / "mixture.json"
    export.write_json(mixture_path, manifest)
    from openpi.training.subtask_mixture_data import create_dataset
    config = SimpleNamespace(subtask_prediction=True, prompt_from_task=True, subtask_whole_task_prob=0,
                             frame_stride=1, episodes_index=None, memory_contract=None,
                             subtask_mixture_path=str(mixture_path), subtask_mixture_roots_path=None,
                             action_sequence_keys=("action",), data_cls=MockReader, dataset_kwargs={})
    dataset = create_dataset(config, 32, metadata_cls=lambda **kwargs: SimpleNamespace(fps=30))
    for number in (0, 17, 399):
        reference, sample = plan.reference(number), dataset[number]
        assert sample["subtask"] == reference["subtask"]
        assert sample["action"].shape == (32, 23) and sample["observation.state"].shape == (57,)
        np.testing.assert_array_equal(sample["action"][:, 0], np.arange(reference["frame"], reference["frame"] + 32))


@pytest.mark.parametrize("damage", ["metadata", "label_table", "source", "held_out"])
def test_index_rejects_post_readback_changes_and_validation_overlap(tmp_path, damage):
    root = tmp_path / "dataset"
    source = trajectory(tmp_path / "source.hdf5")
    run_export([source], root)
    synthetic_verified_marker(root)
    held = []
    if damage == "metadata":
        (root / "meta/info.json").write_text("{}")
    elif damage == "label_table":
        path = root / "meta/subtask_segments.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["subtask"] = "invented label"
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif damage == "source":
        with h5py.File(source, "r+") as stream:
            stream["action"][3] = np.zeros(23, np.float32)
    else:
        held = [["turning_on_radio", 301]]
    with pytest.raises(ValueError):
        export.prepare_training_index(root, root_id="native", held_out_instances=held)


def test_fully_unlabelled_episode_stays_exported_but_has_no_joint_anchors(tmp_path):
    root = tmp_path / "dataset"
    report = run_export([trajectory(tmp_path / "source.hdf5", empty=True)], root)
    assert report["frames"] == 80 and report["unlabelled_frames"] == 80 and report["valid_labelled_anchors"] == 0
    synthetic_verified_marker(root)
    with pytest.raises(ValueError, match="No nonempty"):
        export.prepare_training_index(root, root_id="native", held_out_instances=[])

"""Real HDF5/Parquet, mock LeRobot writer; no model, video codec or simulator."""
import copy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_collection_trajectory import CAMERAS, CodexCollectionTrajectory
from export_codex_collection import canonical_task_prompts, export_codex_collection

DEFAULT_PROMPT = object()


class MockDataset:
    created = None

    @classmethod
    def create(cls, **kwargs):
        result = cls()
        result.root = kwargs["root"]
        result.root.mkdir(parents=True)
        result.frames, result.episodes, result.pending = [], [], []
        result.finalized = False
        cls.created = result
        return result

    def add_frame(self, frame):
        self.pending.append(copy.deepcopy(frame))

    def save_episode(self):
        self.episodes.append(dict(episode_index=len(self.episodes), length=len(self.pending)))
        self.frames.append(self.pending)
        self.pending = []

    def finalize(self):
        self.finalized = True
        directory = self.root / "meta/episodes/chunk-000"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.episodes), directory / "file-000.parquet")


def trajectory(path, *, attempt="first", offset=0, success=True, instance=311, fps=30,
               task_prompt=DEFAULT_PROMPT):
    identity = dict(task="turning_on_radio", instance_id=instance, attempt_id=attempt,
                    checkpoint="synthetic/no-model", conditioning="task_and_subtask", fps=fps)
    if task_prompt is DEFAULT_PROMPT:
        task_prompt = canonical_task_prompts()[identity["task"]]
    if task_prompt is not None:
        identity["task_prompt"] = task_prompt
    with CodexCollectionTrajectory(path, identity) as writer:
        for step, label in enumerate(("move to radio", "move to radio", "turn on radio")):
            frame = {"observation.state": np.full(57, offset + step, np.float32),
                     **{key: np.full((8, 8, 3), offset + step, np.uint8) for key in CAMERAS}}
            writer.record(frame, np.full(23, offset + step, np.float32), step == 2, False, label)
        writer.finish(dict(steps=3, task_success=success, final_goal_state={"success": success},
                           terminated=True, truncated=False))
    return path


def export(paths, output, **kwargs):
    return export_codex_collection(paths, output, "fixture/codex", dataset_class=MockDataset, **kwargs)


def test_dynamic_labels_source_groups_and_values_survive_export(tmp_path):
    paths = [trajectory(tmp_path / "first.hdf5"),
             trajectory(tmp_path / "second.hdf5", attempt="second", offset=8, instance=312)]
    output = tmp_path / "dataset"
    report = export(paths, output)
    labels = pq.read_table(output / "meta/subtask_segments.parquet").to_pylist()
    assert labels == [dict(episode_index=episode, **segment) for episode in (0, 1) for segment in (
        dict(start=0, end=2, subtask="move to radio"), dict(start=2, end=3, subtask="turn on radio"))]
    assert report["segments"] == 4 and report["frames"] == 6 and report["source_groups"] == 2
    assert report["ready_for_training"] is False and report["real_codec_reader_verified"] is False
    assert report["real_lerobot_writer"] is False
    assert not (output / "_READY_FOR_TRAINING").exists()
    assert not (output / "_VERIFIED_EXPORT").exists()
    assert MockDataset.created.finalized
    for episode, offset in enumerate((0, 8)):
        for step, frame in enumerate(MockDataset.created.frames[episode]):
            np.testing.assert_array_equal(frame["action"], np.full(23, offset + step, np.float32))
            assert frame["task"] == canonical_task_prompts()["turning_on_radio"]
    episodes = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [row["task_instance_id"] for row in episodes] == [311, 312]
    provenance = json.loads((output / "meta/collection_provenance.json").read_text())
    assert [row["attempt_id"] for row in provenance] == ["first", "second"]
    assert provenance[0]["environment_result"]["success_source"] == "environment_task_and_final_goal"
    assert (output / "retained_episodes.txt").read_text() == "0\n1\n"


@pytest.mark.parametrize("kind", ["failed", "same_path", "same_identity", "same_content", "wrong_fps"])
def test_rejects_failed_duplicate_or_incompatible_sources_before_creating_output(tmp_path, kind):
    first = trajectory(tmp_path / "first.hdf5", success=kind != "failed")
    paths = [first]
    if kind == "same_path":
        paths.append(first)
    elif kind == "same_identity":
        paths.append(trajectory(tmp_path / "other.hdf5", offset=1))
    elif kind == "same_content":
        paths.append(trajectory(tmp_path / "other.hdf5", attempt="other"))
    elif kind == "wrong_fps":
        paths.append(trajectory(tmp_path / "other.hdf5", attempt="other", offset=1, fps=20))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        export(paths, output)
    assert not output.exists()


def test_reserved_eval_instances_are_rejected_even_for_tampered_success_metadata(tmp_path):
    path = trajectory(tmp_path / "reserved.hdf5")
    with h5py.File(path, "r+") as stream:
        identity = json.loads(stream.attrs["metadata"])
        identity["instance_id"] = 301
        stream.attrs["metadata"] = json.dumps(identity)
    with pytest.raises(ValueError, match="311..320"):
        export([path], tmp_path / "rejected")


def test_requires_explicit_inputs_fresh_root_and_full_prompt(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        export([], tmp_path / "empty")
    path = trajectory(tmp_path / "source.hdf5")
    with pytest.raises(ValueError, match="fresh"):
        export([path], tmp_path)
    with pytest.raises(ValueError, match="whole-task prompt"):
        export([path], tmp_path / "missing-prompt", task_prompts={})


def test_canonical_registry_loading_does_not_import_openpi():
    before = set(sys.modules)
    prompts = canonical_task_prompts()
    assert len(prompts) == 100 and "turning_on_radio" in prompts
    assert not any(name.startswith("openpi") for name in set(sys.modules) - before)


@pytest.mark.parametrize("source_prompt", [None, "", "  "])
def test_missing_source_task_prompt_never_becomes_training_label(tmp_path, source_prompt):
    path = trajectory(tmp_path / "missing-source-prompt.hdf5", task_prompt=source_prompt)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="source task_prompt"):
        export([path], output)
    assert not output.exists()


def test_export_prompt_must_exactly_match_collected_whole_task_label(tmp_path):
    path = trajectory(tmp_path / "source.hdf5")
    output = tmp_path / "rejected"
    prompt = canonical_task_prompts()["turning_on_radio"]
    with pytest.raises(ValueError, match="differs from collected task_prompt"):
        export([path], output, task_prompts={"turning_on_radio": prompt + " "})
    assert not output.exists()


def test_custom_source_prompt_is_preserved_only_with_matching_export_map(tmp_path):
    prompt = "Switch on the specified radio."
    path = trajectory(tmp_path / "custom-source.hdf5", task_prompt=prompt)
    with pytest.raises(ValueError, match="differs from collected task_prompt"):
        export([path], tmp_path / "wrong-canonical")
    export([path], tmp_path / "matching", task_prompts={"turning_on_radio": prompt})
    assert all(frame["task"] == prompt for frame in MockDataset.created.frames[0])

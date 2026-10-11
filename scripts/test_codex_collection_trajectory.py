"""CPU synthetic-array tests; these do not assert simulator or model success."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_collection_trajectory import (
    CAMERAS, CodexCollectionTrajectory, inspect_codex_collection_trajectory,
)


def metadata(**overrides):
    return dict(task="put_cups_away", instance_id=311, attempt_id="attempt-001",
                checkpoint="fixture/no-real-model", conditioning="task_and_subtask", fps=30, **overrides)


def frame(value=0):
    return {"observation.state": np.full(57, value, dtype=np.float32),
            **{key: np.full((8, 8, 3), value, dtype=np.uint8) for key in CAMERAS}}


def outcome(steps=3, success=True, terminated=True, truncated=False):
    return dict(steps=steps, task_success=success, final_goal_state={"success": success},
                terminated=terminated, truncated=truncated)


def write_attempt(path, success=True):
    with CodexCollectionTrajectory(path, metadata()) as writer:
        for index, label in enumerate(("grasp cup", "grasp cup", "place cup")):
            writer.record(frame(index), np.full(23, index, np.float32), index == 2, False, label)
        writer.finish(outcome(success=success))


def test_dynamic_segments_and_executed_action_roundtrip(tmp_path):
    path = tmp_path / "trajectory.hdf5"
    write_attempt(path)
    result = inspect_codex_collection_trajectory(path)
    assert result["segments"] == [dict(start=0, end=2, subtask="grasp cup"),
                                   dict(start=2, end=3, subtask="place cup")]
    assert result["result"]["success_source"] == "environment_task_and_final_goal"
    assert result["metadata"]["official_benchmark"] is False
    assert result["arrays_sha256"] == result["content_sha256"]
    with h5py.File(path) as stream:
        assert stream.attrs["schema_version"] == 2
        np.testing.assert_array_equal(stream["action"][:], np.repeat(np.arange(3)[:, None], 23, axis=1))
        np.testing.assert_array_equal(stream["observation.state"][:, 0], np.arange(3))


def test_failed_attempt_is_saved_but_never_accepted(tmp_path):
    path = tmp_path / "failed.hdf5"
    write_attempt(path, success=False)
    assert inspect_codex_collection_trajectory(path, require_success=False)["result"]["success"] is False
    with pytest.raises(ValueError, match="environment-confirmed"):
        inspect_codex_collection_trajectory(path)


@pytest.mark.parametrize("field,value", [("instance_id", 301), ("instance_id", 321),
                                        ("instance_id", True), ("conditioning", "subtask_only"),
                                        ("fps", 0), ("official_benchmark", True)])
def test_invalid_collection_identity_fails_before_file_creation(tmp_path, field, value):
    identity = metadata()
    identity[field] = value
    path = tmp_path / "invalid.hdf5"
    with pytest.raises(ValueError):
        CodexCollectionTrajectory(path, identity)
    assert not path.exists()


@pytest.mark.parametrize("label", [None, "", "  ", "grasp\ncup", "cup\x00"])
def test_missing_or_invalid_labels_never_append(tmp_path, label):
    with CodexCollectionTrajectory(tmp_path / "label.hdf5", metadata()) as writer:
        with pytest.raises(ValueError):
            writer.record(frame(), np.zeros(23), False, False, label)
        assert writer.count == 0


@pytest.mark.parametrize("terminal", [(True, False), (False, True)])
def test_cannot_append_after_terminal_or_join_reset_attempt(tmp_path, terminal):
    with CodexCollectionTrajectory(tmp_path / "terminal.hdf5", metadata()) as writer:
        writer.record(frame(), np.zeros(23), *terminal, "grasp cup")
        with pytest.raises(ValueError, match="terminal"):
            writer.record(frame(), np.zeros(23), False, False, "place cup")


def test_preinference_candidates_cannot_be_saved_as_executed_action(tmp_path):
    with CodexCollectionTrajectory(tmp_path / "chunk.hdf5", metadata()) as writer:
        with pytest.raises(ValueError, match="23-dimensional"):
            writer.record(frame(), np.zeros((16, 23)), False, False, "grasp cup")
        assert writer.count == 0


@pytest.mark.parametrize("change", [dict(success=True), dict(task_success=True),
                                     dict(final_goal_state={"success": True}), dict(steps=2),
                                     dict(terminated=False), dict(task_success=1)])
def test_finish_rejects_teacher_success_and_alignment_errors(tmp_path, change):
    with CodexCollectionTrajectory(tmp_path / "bad-result.hdf5", metadata()) as writer:
        writer.record(frame(), np.zeros(23), True, False, "grasp cup")
        result = outcome(steps=1, success=False)
        result.update(change)
        with pytest.raises(ValueError):
            writer.finish(result)


def test_success_cannot_be_claimed_before_environment_terminal(tmp_path):
    with CodexCollectionTrajectory(tmp_path / "premature.hdf5", metadata()) as writer:
        writer.record(frame(), np.zeros(23), False, False, "grasp cup")
        with pytest.raises(ValueError, match="environment-terminated"):
            writer.finish(outcome(steps=1, terminated=False))


def test_incomplete_file_cannot_be_accepted(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    with CodexCollectionTrajectory(path, metadata()) as writer:
        writer.record(frame(), np.zeros(23), False, False, "grasp cup")
    with pytest.raises(ValueError, match="Unfinished"):
        inspect_codex_collection_trajectory(path)


def test_zero_step_abort_is_auditable_but_not_training_data(tmp_path):
    path = tmp_path / "early-abort.hdf5"
    with CodexCollectionTrajectory(path, metadata()) as writer:
        writer.finish(outcome(steps=0, success=False, terminated=False))
    result = inspect_codex_collection_trajectory(path, require_success=False)
    assert result["steps"] == 0 and result["segments"] == []


@pytest.mark.parametrize("mutation", ["action", "labels", "segments", "indices", "result", "metadata"])
def test_detects_array_label_boundary_and_result_tampering(tmp_path, mutation):
    path = tmp_path / "tampered.hdf5"
    write_attempt(path)
    with h5py.File(path, "r+") as stream:
        if mutation == "action":
            stream["action"][0, 0] = 10
        elif mutation == "labels":
            stream["subtask"][0] = "open cupboard"
        elif mutation == "segments":
            stream.attrs["segments"] = json.dumps([dict(start=0, end=3, subtask="place cup")])
        elif mutation == "indices":
            stream["frame_index"][1] = 7
        elif mutation == "result":
            result = json.loads(stream.attrs["result"])
            result["success"] = False
            stream.attrs["result"] = json.dumps(result)
        else:
            identity = json.loads(stream.attrs["metadata"])
            identity["checkpoint"] = "different-model"
            stream.attrs["metadata"] = json.dumps(identity)
    with pytest.raises(ValueError):
        inspect_codex_collection_trajectory(path)


def test_native_mixed_metadata_roundtrip_and_legacy_metadata_stays_canonical(tmp_path):
    old = tmp_path / "legacy.hdf5";write_attempt(old)
    legacy = inspect_codex_collection_trajectory(old)["metadata"]
    assert "collection_mode" not in legacy and legacy["subtask_source"] == "provided"
    native = tmp_path / "native.hdf5"
    with CodexCollectionTrajectory(native, metadata(collection_mode="native_with_corrections")) as writer:
        writer.record(frame(),np.zeros(23),False,False,"native move to radio")
        writer.record(frame(1),np.ones(23),True,False,"teacher adjust grip")
        writer.finish(outcome(steps=2))
    info = inspect_codex_collection_trajectory(native)
    assert info["metadata"]["collection_mode"] == "native_with_corrections"
    assert info["metadata"]["subtask_source"] == "mixed_native_and_teacher"
    assert info["steps"] == 2


@pytest.mark.parametrize("mode,source", [("unknown","provided"),("teacher_conditioned","mixed_native_and_teacher"),
                                       ("native_with_corrections","provided")])
def test_trajectory_rejects_mode_source_mismatch_before_file(tmp_path, mode, source):
    path=tmp_path / "mismatch.hdf5"
    with pytest.raises(ValueError):CodexCollectionTrajectory(path,metadata(collection_mode=mode,subtask_source=source))
    assert not path.exists()

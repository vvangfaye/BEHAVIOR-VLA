import json
from pathlib import Path
import sys
from types import ModuleType

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import codex_collection_demonstrations as demos
from codex_collection_trajectory import CodexCollectionTrajectory
from subtask_trajectory import CAMERAS


def test_predeclared_local_target_reserves_examples_without_changing_default():
    labels = [{"subtask": "target" if i in (3, 8) else f"other-{i}"} for i in range(28)]
    assert demos._example_segment_indices(labels, 6) == [0, 5, 10, 16, 21, 27]
    selected = demos._example_segment_indices(labels, 6, "target")
    assert len(selected) == 6 and 3 in selected and 8 in selected
    with pytest.raises(ValueError, match="absent"):
        demos._example_segment_indices(labels, 6, "not present")
    with pytest.raises(ValueError, match="budget"):
        demos._example_segment_indices(labels, 1, "target")


def frame(step):
    return {"observation.state": np.full(57, step, np.float32),
            "action": np.full(23, step / 10, np.float32),
            **{key: np.full((16, 16, 3), step, np.uint8) for key in CAMERAS}}


def test_close_phase_reference_exposes_pre_close_pose_without_claiming_grasp(tmp_path):
    actions = np.zeros((90, 23), np.float32)
    actions[:, [14, 22]] = 1
    actions[45:, 14] = -1
    def reader(step):
        return {**frame(step), 'action': actions[step]}
    result = demos._prepare(tmp_path/'phase',
        dict(source_kind='original_train', instance_id=4, task='camera'),
        [dict(start=0, end=90, subtask='pick up camera')], 90, reader,
        gripper_phase_actions=actions)
    example = result['examples'][0]
    assert [row['frame'] for row in example['observations']] == [30, 45, 60]
    assert [row['state'][0] for row in example['observations']] == [30, 45, 60]
    assert example['executed_actions_start'] == 45
    assert example['executed_actions'][0][14] == -1
    assert example['reference_selection']['arms'] == ['left']
    assert example['reference_selection']['grasp_success_inferred'] is False
    context, images = demos.demonstration_context(tmp_path/'phase/manifest.json', 'camera')
    assert context['examples'][0]['reference_selection'] == example['reference_selection']
    from PIL import Image
    with Image.open(images[0]) as image:
        assert image.size == (672, 732)


def test_close_phase_does_not_mislabel_an_already_closed_gripper_as_new_close():
    actions = np.zeros((60, 23), np.float32)
    actions[:, 14] = -1
    frames, action_start, evidence = demos._gripper_phase(actions, 20, 60)
    assert frames == [20, 59] and action_start == 20
    assert evidence['kind'] == 'segment_endpoints_no_close_onset'


def test_close_phase_reference_rejects_misaligned_released_actions(tmp_path):
    actions = np.zeros((6, 23), np.float32)
    actions[:, [14, 22]] = 1
    actions[3:, 14] = -1
    with pytest.raises(ValueError, match='differ from the raw control event'):
        demos._prepare(tmp_path/'phase',
            dict(source_kind='original_train', instance_id=4),
            [dict(start=0, end=6, subtask='pick up camera')], 6, frame,
            gripper_phase_actions=actions)


def test_close_phase_selects_first_command_and_clips_frames_to_segment():
    actions = np.zeros((60, 23), np.float32)
    actions[:, [14, 22]] = 1
    actions[21:24, 22] = -1
    actions[40:, 14] = -1
    frames, action_start, evidence = demos._gripper_phase(actions, 20, 60)
    assert frames == [20, 21, 36] and action_start == 21
    assert evidence['arms'] == ['right']


def collected(tmp_path, *, success=True):
    path = tmp_path / "trajectory.hdf5"
    trajectory = CodexCollectionTrajectory(path, dict(task="turning_on_radio", instance_id=311,
        attempt_id="attempt-1", checkpoint="1688977", conditioning="task_and_subtask", fps=30,
        task_prompt="Turn on the radio"))
    for step in range(6):
        item = frame(step)
        trajectory.record({key: value for key, value in item.items() if key != "action"},
                          item["action"], step == 5, False,
                          "move to radio" if step < 3 else "turn on radio")
    trajectory.finish(dict(steps=6, task_success=success, final_goal_state={"success": success},
                           terminated=True, truncated=False))
    return path


def original(tmp_path, *, instance=114, success=True):
    root = tmp_path / "original"
    meta = root / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    info = dict(codebase_version="v3.0", data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                features={key: {"dtype": "video"} for key in CAMERAS})
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "tasks.jsonl").write_text(json.dumps(dict(task_index=0, task_name="turning_on_radio", task="Turn on the radio")))
    row = dict(episode_index=102, task_index=0, task_instance_id=instance,
               tasks=["Turn on the radio"], length=6, **{"data/chunk_index": 0, "data/file_index": 0})
    for key in CAMERAS:
        row.update({f"videos/{key}/from_timestamp": 0., f"videos/{key}/to_timestamp": .2,
                    f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0})
    pq.write_table(pa.Table.from_pylist([row]), meta / "episodes/chunk-000/file-000.parquet")
    labels = tmp_path / "segments.parquet"
    pq.write_table(pa.Table.from_pylist([dict(episode_index=102, start=0, end=3, subtask="move to radio"),
                                       dict(episode_index=102, start=3, end=6, subtask="turn on radio")]), labels)
    retained = tmp_path / "episodes.txt"
    retained.write_text("102\n")
    for name in ["data/chunk-000/file-000.parquet", *[f"videos/{key}/chunk-000/file-000.mp4" for key in CAMERAS]]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-backed-by-fake-reader")
    raw = tmp_path / "raw.hdf5"
    with h5py.File(raw, "w") as stream:
        group = stream.create_group("data/demo_0")
        group.attrs.update(task_name="turning_on_radio", instance_id=instance,
                           completion_reason="task_success" if success else "timeout")
        group.create_dataset("action", data=np.stack([frame(step)["action"] for step in range(6)]))
    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs["episodes"] == [102] and kwargs["download_videos"] is False
            assert kwargs["tolerance_s"] == 5e-3
        def __len__(self): return 6
        def __getitem__(self, index): return frame(index)
    return dict(dataset_root=root, segments_path=labels, raw_path=raw, episode_index=102,
                retained_episodes_path=retained, dataset_class=Dataset, output_dir=tmp_path / "package")


def test_original_phase_reference_uses_verified_raw_group_and_matching_reader(tmp_path):
    args = original(tmp_path)
    actions = np.stack([frame(step)['action'] for step in range(6)])
    actions[:, [14, 22]] = 1
    actions[4:, 14] = -1
    with h5py.File(args['raw_path'], 'r+') as stream:
        stream['data/demo_0/action'][:] = actions
    Base = args['dataset_class']
    class Dataset(Base):
        def __getitem__(self, index):
            return {**frame(index), 'action': actions[index]}
    args['dataset_class'] = Dataset
    result = demos.prepare_original_demonstrations(**args, gripper_phase_reference=True)
    target = result['examples'][1]
    assert target['reference_selection']['frame'] == 4
    assert target['executed_actions_start'] == 4
    assert [row['frame'] for row in target['observations']] == [3, 4, 5]
    assert result['metadata']['source_raw_group'] == '/data/demo_0'


def test_collection_success_creates_bounded_compact_context(tmp_path):
    source = collected(tmp_path)
    result = demos.prepare_collection_demonstrations(source, tmp_path / "package")
    context, images = demos.demonstration_context(tmp_path / "package/manifest.json", "turning_on_radio", "turn on radio")
    assert result["metadata"]["source_kind"] == "verified_collection"
    assert len(images) == 1 and images[0].is_file()
    assert context["examples"][0]["subtask"] == "turn on radio"
    assert context["examples"][0]["observations"][0]["frame"] == 3
    assert len(context["examples"][0]["executed_actions"]) == 3
    assert context["examples"][0]["recovery_verified"] is False
    assert context["subtask_sequence"] == ["move to radio", "turn on radio"]


def test_collection_failed_cannot_seed_examples(tmp_path):
    with pytest.raises(ValueError, match="whole-task successes"):
        demos.prepare_collection_demonstrations(collected(tmp_path, success=False), tmp_path / "package")
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("instance", [301, 305, 310, 311, 320, 321, -1])
def test_public_and_invalid_instances_cannot_impersonate_original_train(tmp_path, instance):
    with pytest.raises(ValueError, match="original train"):
        demos.prepare_original_demonstrations(**original(tmp_path, instance=instance))


def test_original_provenance_and_success_are_retained(tmp_path):
    arguments = original(tmp_path)
    result = demos.prepare_original_demonstrations(**arguments)
    assert result["metadata"]["episode_index"] == 102
    assert result["metadata"]["instance_id"] == 114
    assert result["metadata"]["source_raw_group"] == "/data/demo_0"
    assert result["metadata"]["reader_tolerance_s"] == 5e-3
    assert len(result["metadata"]["source_files"]) == 4
    assert result["examples"][0]["executed_actions"][1] == [0.1] * 23
    assert result["examples"][0]["observations"][0]["state"] == [0.] * 57


def test_production_reader_receives_existing_training_timestamp_tolerance(tmp_path, monkeypatch):
    args = original(tmp_path)
    base = args.pop("dataset_class")
    received = []
    class Dataset(base):
        def __init__(self, **kwargs):
            received.append(kwargs)
            super().__init__(**kwargs)
    module = ModuleType("lerobot.datasets")
    module.LeRobotDataset = Dataset
    monkeypatch.setitem(sys.modules, "lerobot.datasets", module)
    result = demos.prepare_original_demonstrations(**args)
    assert len(received) == 1
    assert received[0]["tolerance_s"] == result["metadata"]["reader_tolerance_s"] == 5e-3
    assert received[0]["video_backend"] == "pyav"


def test_reader_timestamp_failure_is_not_skipped_or_retried(tmp_path):
    args = original(tmp_path)
    base = args["dataset_class"]
    reads = []
    class FrameTimestampError(Exception):
        pass
    class Dataset(base):
        def __getitem__(self, index):
            reads.append(index)
            raise FrameTimestampError("outside configured tolerance")
    args["dataset_class"] = Dataset
    with pytest.raises(FrameTimestampError, match="outside configured tolerance"):
        demos.prepare_original_demonstrations(**args)
    assert len(reads) == 1
    assert not (args["output_dir"] / "manifest.json").exists()


def standard_original(tmp_path):
    args = original(tmp_path)
    metadata = args["dataset_root"] / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(metadata).to_pylist()
    rows[0].update(raw_episode_id=1140, annotation_path="annotations/task-0000/episode_00001140.json")
    pq.write_table(pa.Table.from_pylist(rows), metadata)
    canonical = tmp_path / "task-0000/episode_00001140.hdf5"
    canonical.parent.mkdir()
    args["raw_path"].rename(canonical)
    args["raw_path"] = canonical
    with h5py.File(args["raw_path"], "a") as stream:
        data, group = stream["data"], stream["data/demo_0"]
        for key in ("task_name", "instance_id", "completion_reason"):
            del group.attrs[key]
        data.attrs["config"] = json.dumps({"task": {"type": "BehaviorTask",
            "activity_name": "turning_on_radio", "activity_instance_id": 114,
            "activity_definition_id": 0, "termination_config": {"max_steps": 10000}}})
        data.attrs["scene_file"] = json.dumps({"versions": {"omnigibson": {
            "version": "3.7.0-alpha", "git_hash": "a" * 40}}, "state": {"private_fixture": "never_teacher_input"}})
        group.create_dataset("terminated", data=np.array([False, False, False, True, True, True]))
        group.create_dataset("truncated", data=np.zeros(6, dtype=bool))
        # Dereferencing simulator state would fail; success and teacher examples
        # must use only termination signals and the real-observation reader.
        group["state"] = h5py.ExternalLink("/nonexistent/raw-state-fixture", "/state")
    args["expected_raw_sha256"] = demos._sha(args["raw_path"])
    return args


@pytest.mark.parametrize("recorded_instance", [0, 114])
def test_standard_original_schema_uses_recorded_success_and_preserves_version_gap(tmp_path, recorded_instance):
    args = standard_original(tmp_path)
    with h5py.File(args["raw_path"], "a") as stream:
        config = json.loads(stream["data"].attrs["config"])
        config["task"]["activity_instance_id"] = recorded_instance
        stream["data"].attrs["config"] = json.dumps(config)
    args["expected_raw_sha256"] = demos._sha(args["raw_path"])
    result = demos.prepare_original_demonstrations(**args)
    metadata = result["metadata"]
    assert metadata["source_raw_group"] == "/data/demo_0"
    assert metadata["success_source"] == "original_behavior_task_terminated_without_truncation"
    assert metadata["termination_evidence"]["terminated_count"] == 3
    assert metadata["termination_evidence"]["truncated_count"] == 0
    assert metadata["recording_omnigibson_version"] == "3.7.0-alpha"
    assert metadata["recording_omnigibson_git_hash"] == "a" * 40
    assert metadata["success_semantics_audit"]["exact_recording_source_semantics_verified"] is False
    assert metadata["identity_source"] == "published_metadata_raw_id_and_sha"
    assert metadata["recorded_config_instance"] == recorded_instance
    assert metadata["instance_id"] == 114
    assert metadata["published_raw_relative_path"] == "task-0000/episode_00001140.hdf5"
    assert metadata["expected_raw_sha256"] == metadata["source_raw_sha256"]
    assert "never_teacher_input" not in json.dumps(result)


@pytest.mark.parametrize("mutation", [
    "wrong_task", "wrong_instance", "instance_alias_conflict", "unsupported_type",
    "unsupported_termination", "invalid_max_steps", "no_success", "final_not_success",
    "truncated", "integer_terminated", "signal_length", "signal_matrix",
    "missing_signal", "action_width", "action_length", "duplicate", "explicit_failure",
    "conflicting_group_task", "conflicting_group_instance",
])
def test_standard_original_schema_rejects_unproved_success(tmp_path, mutation):
    args = standard_original(tmp_path)
    with h5py.File(args["raw_path"], "a") as stream:
        data, group = stream["data"], stream["data/demo_0"]
        config = json.loads(data.attrs["config"])
        task = config["task"]
        if mutation == "wrong_task": task["activity_name"] = "putting_shoes_on_rack"
        elif mutation == "wrong_instance": task["activity_instance_id"] = 115
        elif mutation == "instance_alias_conflict": task["instance_id"] = 115
        elif mutation == "unsupported_type": task["type"] = "PointNavigationTask"
        elif mutation == "unsupported_termination": task["termination_config"]["collision"] = True
        elif mutation == "invalid_max_steps": task["termination_config"]["max_steps"] = True
        elif mutation == "no_success": group["terminated"][:] = False
        elif mutation == "final_not_success": group["terminated"][-1] = False
        elif mutation == "truncated": group["truncated"][-1] = True
        elif mutation in {"integer_terminated", "signal_length", "signal_matrix", "missing_signal"}:
            del group["terminated"]
            if mutation != "missing_signal":
                value = (np.ones(6, dtype=np.int64) if mutation == "integer_terminated" else
                         np.ones(5, dtype=bool) if mutation == "signal_length" else np.ones((6, 1), dtype=bool))
                group.create_dataset("terminated", data=value)
        elif mutation in {"action_width", "action_length"}:
            del group["action"]
            group.create_dataset("action", data=np.zeros((6, 22) if mutation == "action_width" else (5, 23)))
        elif mutation == "duplicate": stream.copy(group, data, name="demo_1")
        elif mutation == "explicit_failure": group.attrs["completion_reason"] = "timeout"
        elif mutation == "conflicting_group_task": group.attrs["task_name"] = "putting_shoes_on_rack"
        elif mutation == "conflicting_group_instance": group.attrs["instance_id"] = 115
        data.attrs["config"] = json.dumps(config)
    args["expected_raw_sha256"] = demos._sha(args["raw_path"])
    with pytest.raises(ValueError): demos.prepare_original_demonstrations(**args)
    assert not Path(args["output_dir"]).exists()


@pytest.mark.parametrize("mutation", ["missing_sha", "wrong_sha", "filename", "parent", "raw_id", "task_index", "encoded_task", "annotation"])
def test_standard_schema_requires_frozen_sha_and_exact_published_raw_path(tmp_path, mutation):
    args = standard_original(tmp_path)
    if mutation == "missing_sha": args.pop("expected_raw_sha256")
    elif mutation == "wrong_sha": args["expected_raw_sha256"] = "0" * 64
    elif mutation in {"filename", "parent"}:
        path = (args["raw_path"].with_name("episode_00001150.hdf5") if mutation == "filename"
                else tmp_path / "task-0001" / args["raw_path"].name)
        path.parent.mkdir(exist_ok=True)
        args["raw_path"].rename(path); args["raw_path"] = path
    else:
        metadata = args["dataset_root"] / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(metadata).to_pylist()
        if mutation == "raw_id": rows[0]["raw_episode_id"] = 1150
        elif mutation == "task_index": rows[0]["task_index"] = 1
        elif mutation == "encoded_task":
            rows[0]["raw_episode_id"] = 10000
            rows[0]["annotation_path"] = "annotations/task-0000/episode_00010000.json"
            path = args["raw_path"].with_name("episode_00010000.hdf5")
            args["raw_path"].rename(path); args["raw_path"] = path
        else: rows[0]["annotation_path"] = "annotations/task-0000/episode_00001150.json"
        pq.write_table(pa.Table.from_pylist(rows), metadata)
    with pytest.raises(ValueError): demos.prepare_original_demonstrations(**args)
    assert not Path(args["output_dir"]).exists()


@pytest.mark.parametrize("mutation", [None, "task", "instance", "type"])
def test_extended_success_cannot_override_conflicting_config_identity(tmp_path, mutation):
    args = original(tmp_path)
    task = {"type": "BehaviorTask", "activity_name": "turning_on_radio", "activity_instance_id": 114}
    if mutation == "task": task["activity_name"] = "putting_shoes_on_rack"
    if mutation == "instance": task["activity_instance_id"] = 113
    if mutation == "type": task["type"] = "OtherTask"
    with h5py.File(args["raw_path"], "a") as stream:
        stream["data"].attrs["config"] = json.dumps({"task": task})
    if mutation is None:
        result = demos.prepare_original_demonstrations(**args)
        assert result["metadata"]["success_source"] == "original_raw_completion_reason_task_success"
    else:
        with pytest.raises(ValueError): demos.prepare_original_demonstrations(**args)


@pytest.mark.parametrize("mutation", ["not_retained", "not_successful", "label_gap", "wrong_instance", "wrong_task"])
def test_original_admission_checks(tmp_path, mutation):
    args = original(tmp_path, success=mutation != "not_successful")
    if mutation == "not_retained": args["retained_episodes_path"].write_text("999\n")
    if mutation == "label_gap":
        pq.write_table(pa.Table.from_pylist([dict(episode_index=102, start=1, end=6, subtask="move")]), args["segments_path"])
    if mutation == "wrong_instance":
        with h5py.File(args["raw_path"], "a") as stream: stream["data/demo_0"].attrs["instance_id"] = 113
    if mutation == "wrong_task":
        metadata = args["dataset_root"] / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(metadata).to_pylist()
        rows[0]["tasks"] = ["putting_shoes_on_rack"]
        pq.write_table(pa.Table.from_pylist(rows), metadata)
    with pytest.raises(ValueError): demos.prepare_original_demonstrations(**args)


@pytest.mark.parametrize("mutation", ["manifest", "image", "task", "path_escape", "holdout"])
def test_context_rejects_tampered_or_other_task_package(tmp_path, mutation):
    package = tmp_path / "package"
    demos.prepare_collection_demonstrations(collected(tmp_path), package)
    path = package / "manifest.json"
    data = json.loads(path.read_text())
    if mutation == "manifest": data["steps"] += 1
    if mutation == "image": (package / data["examples"][0]["image"]).write_bytes(b"changed")
    if mutation == "path_escape": data["examples"][0]["image"] = "../outside.png"
    if mutation == "holdout": data["metadata"]["instance_id"] = 301
    if mutation in {"path_escape", "holdout"}:
        data["fingerprint"] = demos._digest({key: value for key, value in data.items() if key != "fingerprint"})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        demos.demonstration_context(path, "other" if mutation == "task" else "turning_on_radio")


def test_package_requires_fresh_output(tmp_path):
    source = collected(tmp_path)
    demos.prepare_collection_demonstrations(source, tmp_path / "package")
    with pytest.raises(FileExistsError): demos.prepare_collection_demonstrations(source, tmp_path / "package")

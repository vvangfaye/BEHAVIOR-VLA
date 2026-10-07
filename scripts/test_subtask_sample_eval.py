import argparse
import copy
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import eval_b1k_subtask_dynamic as sample_dynamic
import eval_b1k_subtask_samples as runner
import subtask_sample_panel as panel_module
from subtask_sample_panel import (
    build_panel,
    infer_predicate,
    load_panel,
    preflight_panel,
    raw_hdf5_path,
    validate_raw_hdf5,
)
from summarize_subtask_closed_loop import summarize


def _write_panel_inputs(tmp_path: Path, *, extra_supported_segments: bool = False):
    dataset = tmp_path / "dataset"
    metadata_dir = dataset / "meta" / "episodes" / "chunk-000"
    metadata_dir.mkdir(parents=True)
    metadata = []
    segments = []
    episode = 0
    for task_index, task in (
        (14, "carrying_in_groceries"),
        (56, "make_rose_centerpieces"),
    ):
        for index in range(12):
            raw_episode_id = task_index * 10000 + index
            metadata.append(
                {
                    "episode_index": episode,
                    "task_index": task_index,
                    "task_instance_id": index,
                    "raw_episode_id": raw_episode_id,
                    "annotation_path": f"annotations/task-{task_index:04d}/episode_{raw_episode_id:08d}.json",
                    "length": 64,
                    "tasks": [task],
                }
            )
            if task_index == 14:
                anchor_subtask = "move to paper bag"
                short_subtask = "pick up tomato"
                anchor_objects = ["paper_bag_1"]
                short_objects = ["tomato_1"]
            else:
                anchor_subtask = "move to rose"
                short_subtask = "pick up rose"
                anchor_objects = ["rose_1"]
                short_objects = ["rose_1"]
            common = {
                "episode_index": episode,
                "annotation_path": metadata[-1]["annotation_path"],
                "source_interval_index": 0,
                "manipulating_object_ids": [],
                "reference_object_ids": [],
                "memory_prefix": "",
                "spatial_prefix": "",
                "is_annotated": True,
                "provenance": "annotated",
            }
            segments.extend(
                [
                    {
                        **common,
                        "start": 0,
                        "end": 40,
                        "subtask": anchor_subtask,
                        "source_segment_index": index,
                        "source_skill_index": index,
                        "skill_description": "move to",
                        "object_ids": anchor_objects,
                    },
                    {
                        **common,
                        "start": 40,
                        "end": 52,
                        "subtask": short_subtask,
                        "source_segment_index": 100 + index,
                        "source_skill_index": 100 + index,
                        "skill_description": "pick up from",
                        "object_ids": short_objects,
                    },
                    {
                        **common,
                        "start": 52,
                        "end": 64,
                        "subtask": "hand over object",
                        "source_segment_index": 200 + index,
                        "source_skill_index": 200 + index,
                        "skill_description": "hand over",
                        "object_ids": short_objects,
                    },
                ]
            )
            if extra_supported_segments and task_index == 14:
                segments.append(
                    {
                        **common,
                        "start": 8,
                        "end": 40,
                        "subtask": anchor_subtask,
                        "source_segment_index": 300 + index,
                        "source_skill_index": 300 + index,
                        "skill_description": "move to",
                        "object_ids": anchor_objects,
                    }
                )
            episode += 1
    pq.write_table(pa.Table.from_pylist(metadata), metadata_dir / "file-000.parquet")
    segments_path = tmp_path / "segments.parquet"
    pq.write_table(pa.Table.from_pylist(segments), segments_path)
    train_path = tmp_path / "train.txt"
    train_path.write_text("".join(f"{index}\n" for index in range(24)))
    split_path = tmp_path / "split.json"
    split_path.write_text("{}\n")
    return dataset, segments_path, train_path, split_path


def _write_refingerprinted_panel(path: Path, panel: dict) -> None:
    panel["fingerprint"] = panel_module._canonical_hash(
        {name: value for name, value in panel.items() if name != "fingerprint"}
    )
    path.write_text(json.dumps(panel))


def test_panel_selection_expands_anchored_episodes_deterministically(tmp_path):
    inputs = _write_panel_inputs(tmp_path)
    first = build_panel(*inputs, seed=1, episodes_per_task=2)
    second = build_panel(*inputs, seed=1, episodes_per_task=2)

    assert first == second
    assert first["schema_version"] == 2
    assert "reserves" not in first
    assert first["selection"]["episode_anchors"] == {
        "version": "subtask_closed_loop_panel_v1",
        "seed": 1,
        "episodes_per_task": 2,
        "minimum_segment_frames": 32,
        "uniqueness": [
            "episode_index",
            "task_instance_id",
            "source_segment_index",
        ],
    }
    assert first["selection"]["scheduled_segments"]["scope"] == (
        "all_annotated_segments_in_selected_episodes"
    )
    assert len(first["episode_anchors"]) == 4
    assert len(first["episodes"]) == 4
    assert len(first["samples"]) == 8
    assert len(first["exclusions"]) == 4
    assert len({anchor["episode_index"] for anchor in first["episode_anchors"]}) == 4
    assert first["counts"] == {
        "episode_anchors": 4,
        "episodes": 4,
        "annotated_segments": 12,
        "samples": 8,
        "exclusions": 4,
        "exclusions_by_reason": {"unsupported_predicate": 4},
        "by_task": {
            "carrying_in_groceries": {
                "episode_anchors": 2,
                "episodes": 2,
                "annotated_segments": 6,
                "samples": 4,
                "exclusions": 2,
            },
            "make_rose_centerpieces": {
                "episode_anchors": 2,
                "episodes": 2,
                "annotated_segments": 6,
                "samples": 4,
                "exclusions": 2,
            },
        },
    }
    assert {item["reason"] for item in first["exclusions"]} == {
        "unsupported_predicate"
    }
    for episode in first["episodes"]:
        episode_samples = [
            sample
            for sample in first["samples"]
            if sample["episode_index"] == episode["episode_index"]
        ]
        assert [sample["rollout_id"] for sample in episode_samples] == [0, 1]
        assert [sample["start"] for sample in episode_samples] == [0, 40]
        assert episode["sample_ids"] == [
            sample["sample_id"] for sample in episode_samples
        ]
        assert episode["anchor_sample_id"] == episode_samples[0]["sample_id"]
        assert episode["counts"] == {
            "annotated_segments": 3,
            "samples": 2,
            "exclusions": 1,
        }

    path = tmp_path / "panel.json"
    path.write_text(json.dumps(first))
    assert load_panel(path) == first
    corrupted = json.loads(path.read_text())
    corrupted["samples"][0]["start"] = 1
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_panel(path)


def test_production_panel_guard_requires_frozen_v1_anchors_and_counts():
    panel = {
        "episode_anchors": [
            {"sample_id": sample_id, "selection_rank": rank}
            for sample_id, rank in panel_module.FROZEN_PRODUCTION_ANCHORS
        ],
        "counts": {
            **panel_module.PRODUCTION_COUNTS,
            "exclusions_by_reason": {"unsupported_predicate": 4},
        },
        "exclusions": [{"skill_description": "hand over"}] * 4,
    }

    assert panel_module.validate_frozen_production_panel(panel) is panel

    corrupted = copy.deepcopy(panel)
    corrupted["episode_anchors"][0]["selection_rank"] = "0" * 64
    with pytest.raises(ValueError, match="frozen v1 panel"):
        panel_module.validate_frozen_production_panel(corrupted)


def test_load_panel_uses_declared_episode_count_and_checks_rollouts(tmp_path):
    panel = build_panel(*_write_panel_inputs(tmp_path), seed=1, episodes_per_task=2)
    path = tmp_path / "panel.json"

    corrupted = copy.deepcopy(panel)
    same_episode = [
        sample
        for sample in corrupted["samples"]
        if sample["episode_index"] == corrupted["samples"][0]["episode_index"]
    ]
    same_episode[1]["rollout_id"] = 7
    _write_refingerprinted_panel(path, corrupted)
    with pytest.raises(ValueError, match="dense and ordered"):
        load_panel(path)

    corrupted = copy.deepcopy(panel)
    corrupted["episodes"][0]["sample_ids"].reverse()
    _write_refingerprinted_panel(path, corrupted)
    with pytest.raises(ValueError, match="sample IDs are inconsistent"):
        load_panel(path)

    corrupted = copy.deepcopy(panel)
    corrupted["counts"]["exclusions"] = 0
    _write_refingerprinted_panel(path, corrupted)
    with pytest.raises(ValueError, match="counts are inconsistent"):
        load_panel(path)


@pytest.mark.parametrize(
    ("task", "row", "expected"),
    [
        (
            "carrying_in_groceries",
            {
                "skill_description": "place in",
                "subtask": "place tomato in fridge",
                "object_ids": [],
            },
            {"kind": "inside", "target": "tomato", "reference": "refrigerator"},
        ),
        (
            "carrying_in_groceries",
            {
                "skill_description": "hand over",
                "subtask": "hand over tomato",
                "object_ids": [],
            },
            None,
        ),
        (
            "make_rose_centerpieces",
            {
                "skill_description": "place in",
                "subtask": "place rose in vase",
                "object_ids": [],
            },
            {"kind": "rose_resolved_increment", "value": 1},
        ),
    ],
)
def test_predicate_inference(task, row, expected):
    row = {
        "manipulating_object_ids": [],
        "reference_object_ids": [],
        **row,
    }
    assert infer_predicate(task, row) == expected


def _sample() -> dict:
    return {
        "task": "carrying_in_groceries",
        "task_index": 14,
        "episode_index": 1,
        "raw_episode_id": 140010,
        "task_instance_id": 7,
        "episode_length": 40,
        "source_segment_index": 3,
        "source_interval_index": 0,
        "start": 5,
        "end": 37,
        "segment_frames": 32,
        "skill_description": "move to",
        "subtask": "move to paper bag",
        "predicate": {"kind": "near", "target": "sack"},
        "sample_id": "sample-1",
        "rollout_id": 0,
    }


def _write_hdf5(
    path: Path,
    *,
    samples: int = 40,
    git_hash: str | None = "test-revision",
    trailing_state: bool = False,
) -> None:
    path.parent.mkdir(parents=True)
    state_samples = samples + int(trailing_state)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["config"] = json.dumps({"robots": [{"model": "R1Pro"}]})
        data.attrs["scene_file"] = json.dumps({})
        if git_hash is not None:
            data.attrs["omnigibson_git_hash"] = git_hash
        group = data.create_group("demo_0")
        group.attrs["num_samples"] = state_samples
        group.attrs["transitions"] = json.dumps({"1": {}, "5": {}, "9": {}})
        group.create_dataset(
            "state", data=np.ones((state_samples, 6), dtype=np.float32)
        )
        group.create_dataset(
            "state_size", data=np.full(state_samples, 6, dtype=np.int64)
        )
        group.create_dataset("action", data=np.zeros((samples, 2), dtype=np.float32))
        group.create_group("init_metadata")


def test_complete_recorded_scene_backfills_only_recorded_objects(tmp_path):
    scene_dir = (
        tmp_path
        / "2026-challenge-task-instances"
        / "scenes"
        / "house"
        / "json"
    )
    scene_dir.mkdir(parents=True)

    def object_info(name):
        return {
            "class_module": "objects",
            "class_name": "Object",
            "args": {"name": name, "category": name, "model": "model"},
        }

    full_scene = {
        "objects_info": {
            "init_info": {
                "counter": object_info("counter"),
                "extra": object_info("extra"),
            }
        },
        "state": {
            "registry": {
                "object_registry": {
                    "counter": {"root_link": {"pos": [1, 2, 3]}},
                    "extra": {"root_link": {"pos": [4, 5, 6]}},
                }
            }
        },
    }
    (scene_dir / "instance.json").write_text(json.dumps(full_scene))
    recorded_scene = {
        "objects_info": {"init_info": {"counter": object_info("counter")}},
        "state": {"registry": {"object_registry": {}}},
    }

    completed, full_path, backfilled = runner.complete_recorded_scene_file(
        {"scene": {"scene_model": "house", "scene_instance": "instance"}},
        recorded_scene,
        tmp_path,
    )

    assert full_path == scene_dir / "instance.json"
    assert backfilled == 1
    assert set(completed["objects_info"]["init_info"]) == {"counter"}
    assert set(completed["state"]["registry"]["object_registry"]) == {"counter"}
    assert recorded_scene["state"]["registry"]["object_registry"] == {}


def test_complete_recorded_scene_upgrades_legacy_robot_controller_state(tmp_path):
    scene_dir = (
        tmp_path
        / "2026-challenge-task-instances"
        / "scenes"
        / "house"
        / "json"
    )
    scene_dir.mkdir(parents=True)
    (scene_dir / "instance.json").write_text(
        json.dumps(
            {
                "objects_info": {"init_info": {}},
                "state": {"registry": {"object_registry": {}}},
            }
        )
    )
    legacy_controller = {
        "goal_is_valid": True,
        "goal": {"target": [1.0, 2.0]},
    }
    recorded_scene = {
        "objects_info": {"init_info": {"robot_r1": {}}},
        "state": {
            "registry": {
                "object_registry": {
                    "robot_r1": {"controllers": {"base": legacy_controller}}
                }
            }
        },
    }

    completed, _, backfilled = runner.complete_recorded_scene_file(
        {
            "scene": {"scene_model": "house", "scene_instance": "instance"},
            "robots": [{"name": "robot_r1"}],
        },
        recorded_scene,
        tmp_path,
    )

    robot_state = completed["state"]["registry"]["object_registry"]["robot_r1"]
    assert backfilled == 0
    assert robot_state["controller_groups"]["base"] == {
        "goal_set": True,
        "goals": {"target": [1.0, 2.0]},
    }
    assert "controllers" in recorded_scene["state"]["registry"]["object_registry"]["robot_r1"]


def test_alias_robot_observation_namespace():
    proprio = object()
    rgb = object()
    obs = {
        "robot::proprio": proprio,
        "robot::robot:zed_link:Camera:0::rgb": rgb,
        "task_id": object(),
    }

    aliased = runner.alias_robot_observation_namespace(obs, "robot", "robot_r1")

    assert aliased["robot_r1::proprio"] is proprio
    assert aliased["robot_r1::robot_r1:zed_link:Camera:0::rgb"] is rgb
    assert aliased["robot::proprio"] is proprio


def test_configure_recorded_robot_preserves_identity_and_rewrites_cameras():
    recorded = {"name": "robot", "model": "r1pro", "controller_config": {"base": {}}}
    evaluator = {
        "name": "robot_r1",
        "model": "r1pro",
        "obs_modalities": ["proprio", "rgb"],
        "proprio_obs": ["base_qvel"],
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_height": 1080}}},
    }
    configured, eval_config = runner._configure_recorded_robot(
        recorded,
        evaluator,
        {"camera_sensor_names": {"head": "robot_r1:zed_link:Camera:0"}},
    )

    assert configured["name"] == "robot"
    assert configured["controller_config"] == {"base": {}}
    assert configured["obs_modalities"] == ["proprio", "rgb"]
    assert eval_config["camera_sensor_names"]["head"] == "robot:zed_link:Camera:0"


def test_raw_hdf5_contract_and_path(tmp_path):
    sample = _sample()
    path = raw_hdf5_path(tmp_path, sample)
    _write_hdf5(path)

    value = validate_raw_hdf5(path, sample, include_sha256=False)

    assert value["demo_group"] == "demo_0"
    assert value["selected_frame"] == 5
    assert value["transition_count_before_frame"] == 1
    assert value["omnigibson_git_hash"] == "test-revision"
    assert runner.transition_steps_before({"9": {}, "1": {}, "5": {}}, 5) == [1]


def test_preflight_groups_episode_hash_and_open_with_per_sample_evidence(
    tmp_path, monkeypatch
):
    first = {**_sample(), "rollout_id": 0}
    second = {
        **_sample(),
        "sample_id": "sample-2",
        "source_segment_index": 4,
        "start": 9,
        "end": 40,
        "segment_frames": 31,
        "rollout_id": 1,
    }
    path = raw_hdf5_path(tmp_path, first)
    _write_hdf5(path, trailing_state=True)

    open_calls = []
    hash_calls = []
    real_h5py_file = h5py.File
    real_file_sha256 = panel_module.file_sha256

    def counted_open(*args, **kwargs):
        open_calls.append(Path(args[0]))
        return real_h5py_file(*args, **kwargs)

    def counted_hash(value):
        hash_calls.append(Path(value))
        return real_file_sha256(value)

    monkeypatch.setattr(h5py, "File", counted_open)
    monkeypatch.setattr(panel_module, "file_sha256", counted_hash)
    report = preflight_panel(
        {"fingerprint": "panel", "samples": [first, second]}, tmp_path
    )

    assert report["valid"]
    assert report["counts"] == {
        "expected_episodes": 1,
        "validated_episodes": 1,
        "expected_samples": 2,
        "validated_samples": 2,
    }
    assert open_calls == [path]
    assert hash_calls == [path]
    assert len(report["episodes"]) == 1
    assert report["episodes"][0]["sample_ids"] == ["sample-1", "sample-2"]
    assert report["episodes"][0]["state_layout"] == "trailing_state"
    assert len(report["episodes"][0]["sha256"]) == 64
    row_hash = __import__('hashlib').sha256(np.ones(6, dtype=np.float32).tobytes()).hexdigest()
    assert all(item['selected_state_sha256'] == row_hash for item in report['samples'])
    assert [{k:v for k,v in item.items() if k != 'selected_state_sha256'} for item in report["samples"]] == [
        {
            "sample_id": "sample-1",
            "episode_index": 1,
            "rollout_id": 0,
            "selected_frame": 5,
            "selected_state_size": 6,
            "transition_count_before_frame": 1,
        },
        {
            "sample_id": "sample-2",
            "episode_index": 1,
            "rollout_id": 1,
            "selected_frame": 9,
            "selected_state_size": 6,
            "transition_count_before_frame": 2,
        },
    ]


def test_runner_caches_raw_episode_once_for_all_sample_starts(tmp_path, monkeypatch):
    first = _sample()
    second = {
        **_sample(),
        "sample_id": "sample-2",
        "source_segment_index": 4,
        "start": 9,
        "end": 40,
        "segment_frames": 31,
        "rollout_id": 1,
    }
    path = raw_hdf5_path(tmp_path, first)
    _write_hdf5(path)
    real_h5py_file = h5py.File
    open_calls = []
    hash_calls = []

    def counted_open(*args, **kwargs):
        open_calls.append(Path(args[0]))
        return real_h5py_file(*args, **kwargs)

    def counted_hash(value):
        hash_calls.append(Path(value))
        return "a" * 64

    monkeypatch.setattr(h5py, "File", counted_open)
    monkeypatch.setattr(runner, "file_sha256", counted_hash)

    cache = runner.load_episode_cache(path, [first, second])

    assert open_calls == [path]
    assert hash_calls == [path]
    assert len(cache["states"]) == 10
    assert cache["transition_by_step"] == {1: {}, 5: {}, 9: {}}
    assert cache["sample_evidence"]["sample-1"]["selected_frame"] == 5
    assert cache["sample_evidence"]["sample-2"]["selected_frame"] == 9
    assert cache["sample_evidence"]["sample-2"]["sha256"] == "a" * 64


def test_preflight_reports_one_missing_file_error_per_episode(tmp_path):
    first = {**_sample(), "rollout_id": 0}
    second = {
        **_sample(),
        "sample_id": "sample-2",
        "source_segment_index": 4,
        "rollout_id": 1,
    }

    report = preflight_panel(
        {"fingerprint": "panel", "samples": [first, second]},
        tmp_path,
        include_sha256=False,
    )

    assert not report["valid"]
    assert report["counts"]["validated_episodes"] == 0
    assert report["counts"]["validated_samples"] == 0
    assert len(report["errors"]) == 1
    assert report["errors"][0]["scope"] == "episode"
    assert report["errors"][0]["sample_ids"] == ["sample-1", "sample-2"]
    assert report["errors"][0]["error"].startswith("FileNotFoundError:")


def test_raw_hdf5_accepts_missing_git_hash(tmp_path):
    sample = _sample()
    path = raw_hdf5_path(tmp_path, sample)
    _write_hdf5(path, git_hash=None)

    value = validate_raw_hdf5(path, sample, include_sha256=False)

    assert value["omnigibson_git_hash"] is None


def test_raw_hdf5_accepts_legacy_trailing_state(tmp_path):
    sample = _sample()
    path = raw_hdf5_path(tmp_path, sample)
    _write_hdf5(path, trailing_state=True)

    value = validate_raw_hdf5(path, sample, include_sha256=False)

    assert value["action_samples"] == sample["episode_length"]
    assert value["state_samples"] == sample["episode_length"] + 1
    assert value["state_layout"] == "trailing_state"
    assert value["selected_frame"] == sample["start"]


def test_raw_hdf5_rejects_frame_count_mismatch(tmp_path):
    sample = _sample()
    path = raw_hdf5_path(tmp_path, sample)
    _write_hdf5(path, samples=39)

    with pytest.raises(ValueError, match="expected 40"):
        validate_raw_hdf5(path, sample, include_sha256=False)


class _FakeSignals:
    def __init__(self, *, distances=None, inside=None, grasped=()):
        self.distances = {} if distances is None else distances
        self.grasped = frozenset(grasped)
        self.open = {}
        self.inside = {} if inside is None else inside
        self.on_top = {}
        self.tilts = {}


class _FakeAdapter:
    task_name = "carrying_in_groceries"
    rose_ids = ()
    target_ids = {"sack-id": "sack-id"}
    resolved_object_names = {}

    def __init__(self, signals):
        self.signals = signals

    def observe(self):
        return self.signals


class _FakeSelector:
    def __init__(self, signals, *, phase="close_car"):
        self.adapter = _FakeAdapter(signals)
        self.phase = phase
        self.recoveries = 0

    def refresh(self, step):
        return {"step": step, "phase": self.phase, "locked_object": None}

    @property
    def summary(self):
        return {"recoveries": self.recoveries}


def test_completion_monitor_uses_predicate_only_and_debounces_recovery(monkeypatch):
    monkeypatch.setattr(runner, "_symbol_ids", lambda: {"sack": "sack-id"})
    signals = _FakeSignals(distances={"sack-id": 2.0})
    selector = _FakeSelector(signals, phase="close_car")
    monitor = runner.CompletionMonitor(
        object(),
        {"kind": "near", "target": "sack"},
        selector=selector,
        near_distance=1.0,
        tilt_threshold=1.0,
    )

    assert monitor.metric_eligible
    assert monitor.setup_evidence["phase_compatible"] is False
    signals.distances["sack-id"] = 0.5
    selector.recoveries = 1
    assert not monitor.refresh(1)
    assert not monitor.refresh(2)
    assert monitor.refresh(3)
    assert monitor.result()["diagnostic_recoveries"] == 1


@pytest.mark.parametrize(
    ("distances", "expected_reason"),
    [
        ({"sack-id": 0.5}, "predicate_already_true"),
        ({}, "predicate_unobservable"),
        ({"sack-id": float("nan")}, "predicate_unobservable"),
    ],
)
def test_completion_monitor_setup_excludes_invalid_initial_predicate(
    monkeypatch, distances, expected_reason
):
    monkeypatch.setattr(runner, "_symbol_ids", lambda: {"sack": "sack-id"})
    monitor = runner.CompletionMonitor(
        object(),
        {"kind": "near", "target": "sack"},
        selector=_FakeSelector(_FakeSignals(distances=distances)),
        near_distance=1.0,
        tilt_threshold=1.0,
    )

    assert not monitor.metric_eligible
    assert monitor.setup_exclusion_reason == expected_reason
    assert not monitor.refresh(1)
    assert monitor.result()["setup_evidence"]


def test_completion_monitor_scores_rose_increment_over_any_rose(monkeypatch):
    rose_1 = "rose.n.01_1"
    rose_2 = "rose.n.01_2"
    vase = "vase.n.01_1"
    monkeypatch.setattr(runner, "_symbol_ids", lambda: {"vase": vase})
    signals = _FakeSignals(
        inside={(rose_1, vase): False, (rose_2, vase): False}
    )
    selector = _FakeSelector(signals, phase="move_to_rose")
    selector.adapter.task_name = "make_rose_centerpieces"
    selector.adapter.rose_ids = (rose_1, rose_2)
    selector.adapter.target_ids = {
        rose_1: "scene:rose-one",
        rose_2: "scene:rose-two",
    }
    selector.adapter.resolved_object_names = {
        rose_1: "rose_1",
        rose_2: "rose_2",
    }

    monitor = runner.CompletionMonitor(
        object(),
        {"kind": "rose_resolved_increment", "value": 1},
        {"manipulating_object_ids": ["scene:rose-two"], "object_ids": ["vase"]},
        selector=selector,
        near_distance=1.0,
        tilt_threshold=1.0,
    )

    assert monitor.metric_eligible
    assert monitor.active_rose == rose_2  # provenance only
    assert monitor.initial_resolved_count == 0
    # A rose other than the annotated one still completes "place rose in vase".
    signals.inside[(rose_1, vase)] = True
    assert not monitor.refresh(1)
    assert not monitor.refresh(2)
    assert monitor.refresh(3)
    assert monitor.satisfied


def test_completion_monitor_scores_pick_rose_over_any_rose(monkeypatch):
    rose_1 = "rose.n.01_1"
    rose_2 = "rose.n.01_2"
    monkeypatch.setattr(runner, "_symbol_ids", lambda: {})
    signals = _FakeSignals(distances={rose_1: 2.0, rose_2: 3.0})
    selector = _FakeSelector(signals, phase="pick_rose")
    selector.adapter.task_name = "make_rose_centerpieces"
    selector.adapter.rose_ids = (rose_1, rose_2)
    selector.adapter.target_ids = {rose_1: rose_1, rose_2: rose_2}
    selector.adapter.resolved_object_names = {rose_1: "rose_80", rose_2: "rose_79"}

    monitor = runner.CompletionMonitor(
        object(),
        {"kind": "grasped", "target": "active_rose"},
        {"manipulating_object_ids": ["rose"], "object_ids": ["rose"]},
        selector=selector,
        near_distance=1.0,
        tilt_threshold=1.0,
    )

    assert monitor.metric_eligible
    # Grasping any runtime rose satisfies the generic "pick up rose" instruction.
    signals.grasped = frozenset({rose_1})
    assert not monitor.refresh(1)
    assert not monitor.refresh(2)
    assert monitor.refresh(3)
    assert monitor.satisfied


def test_completion_monitor_uses_locked_rose_for_generic_annotation(monkeypatch):
    rose_1 = "rose.n.01_1"
    rose_2 = "rose.n.01_2"
    monkeypatch.setattr(runner, "_symbol_ids", lambda: {})
    signals = _FakeSignals(distances={rose_1: 2.0, rose_2: 3.0})
    selector = _FakeSelector(signals, phase="move_to_rose")
    selector.adapter.task_name = "make_rose_centerpieces"
    selector.adapter.rose_ids = (rose_1, rose_2)
    selector.adapter.target_ids = {rose_1: rose_1, rose_2: rose_2}
    selector.adapter.resolved_object_names = {rose_1: "rose_80", rose_2: "rose_79"}
    selector.refresh = lambda step: {
        "step": step,
        "phase": "move_to_rose",
        "locked_object": rose_2,
    }

    monitor = runner.CompletionMonitor(
        object(),
        {"kind": "near", "target": "active_rose"},
        {"manipulating_object_ids": ["rose"], "object_ids": ["rose"]},
        selector=selector,
        near_distance=1.0,
        tilt_threshold=1.0,
    )

    assert monitor.metric_eligible
    assert monitor.active_rose == rose_2


def test_sparse_materialization_plan_captures_before_same_frame_transition():
    assert runner.sparse_materialization_plan(
        [2, 5], {"0": {}, "2": {}, "4": {}, "5": {}}
    ) == [
        ("load", 0),
        ("transition", 0),
        ("load", 1),
        ("load", 2),
        ("capture", 2),
        ("transition", 2),
        ("load", 3),
        ("load", 4),
        ("transition", 4),
        ("load", 5),
        ("capture", 5),
    ]


def test_runner_processes_and_publishes_complete_episode_bundle(tmp_path, monkeypatch):
    first = _sample()
    second = {
        **_sample(),
        "sample_id": "sample-2",
        "rollout_id": 1,
        "source_segment_index": 4,
        "start": 10,
        "end": 40,
        "segment_frames": 30,
        "subtask": "pick up tomato",
        "predicate": {"kind": "grasped", "target": "tomato"},
    }
    panel = {
        "fingerprint": "panel-fingerprint",
        "episodes": [
            {
                "task": first["task"],
                "task_instance_id": first["task_instance_id"],
                "episode_index": first["episode_index"],
                "raw_episode_id": first["raw_episode_id"],
                "sample_ids": [first["sample_id"], second["sample_id"]],
            }
        ],
        "samples": [first, second],
    }
    attempt_dir = tmp_path / "attempt"
    lease = SimpleNamespace(
        task=first["task"],
        instance_id=first["task_instance_id"],
        attempt_dir=attempt_dir,
        lease_token="lease",
    )

    class FakeQueue:
        def __init__(self, _):
            self.published = None

        def metadata(self, *, readonly):
            assert readonly
            return {
                "run_id": "run",
                "mode": "train",
                "require_videos": False,
                "panel_fingerprint": panel["fingerprint"],
                "profile": runner.QUEUE_PROFILE,
                "evaluation_kind": runner.EVALUATION_KIND,
                "expected_episode_jobs": 1,
                "expected_segment_rollouts": 2,
                "num_rollouts": 1,
            }

        def claim(self, task, worker, attempt_root):
            assert task == first["task"]
            assert worker == "run:worker:generation-0"
            assert attempt_root == str(tmp_path / "attempts")
            return lease

        def publish(self, claimed, output_dir, **kwargs):
            self.published = (claimed, output_dir, kwargs)

        def reconcile(self, *args, **kwargs):
            return False

        def fail(self, *args, **kwargs):
            raise AssertionError("episode should not fail")

    queue = FakeQueue(None)
    monkeypatch.setattr(runner, "EvaluationQueue", lambda _: queue)
    monkeypatch.setattr(runner, "load_panel", lambda _: panel)
    monkeypatch.setattr(runner, "load_episode_cache", lambda path, samples, **kwargs: {"cache": True, "grasp_serialization_audit": {"legacy_missing_assisted_grasp_state": False}})
    monkeypatch.setattr(runner, "_build_evaluator_cfg", lambda *args: object())

    evaluator = object()

    class EvaluatorContext:
        def __enter__(self):
            return evaluator

        def __exit__(self, *args):
            return False

    create_calls = []

    def create_evaluator(*args):
        create_calls.append(args)
        return EvaluatorContext()

    monkeypatch.setattr(runner, "create_sample_evaluator", create_evaluator)
    monkeypatch.setattr(
        runner,
        "materialize_episode_start_states",
        lambda evaluator, cache, samples: {
            sample["sample_id"]: {"sample_id": sample["sample_id"]}
            for sample in samples
        },
    )
    executed = []

    def run_sample(base, evaluator, args, sample, cache, materialized, output, fingerprint, **kwargs):
        executed.append((sample["rollout_id"], materialized["sample_id"]))
        return {"rollout_id": sample["rollout_id"], "metric_eligible": False}

    monkeypatch.setattr(runner, "run_sample", run_sample)

    class FakeGM:
        HEADLESS = False
        ENABLE_TRANSITION_RULES = False

        @staticmethod
        def unlocked():
            return nullcontext()

    base = SimpleNamespace(
        gm=FakeGM(),
        DEFAULT_EVAL_SEED=1,
        seed_everything=lambda seed: seed,
        logger=SimpleNamespace(exception=lambda *args: None),
    )
    args = SimpleNamespace(
        subtask_panel=tmp_path / "panel.json",
        task_name=first["task"],
        queue_db="queue.db",
        queue_run_id="run",
        queue_worker="worker",
        queue_generation=0,
        queue_attempt_root=str(tmp_path / "attempts"),
        mode="train",
        write_video=False,
        raw_data_root=tmp_path / "raw",
        headless=True,
        sample_max_steps=32,
        output_dir=tmp_path / "results" / first["task"],
    )

    results = runner.run(args, base_module=base)

    assert [r["rollout_id"] for r in results] == [0, 1]
    assert len(create_calls) == 1
    assert executed == [(0, "sample-1"), (1, "sample-2")]
    assert runner._artifact_stem(first).endswith("_0")
    assert runner._artifact_stem(second).endswith("_1")
    assert queue.published[0] is lease
    assert queue.published[2]["rollout_specs"] == [
        {"rollout_id": 0, "sample_id": "sample-1", "video": "required_if_eligible"},
        {"rollout_id": 1, "sample_id": "sample-2", "video": "required_if_eligible"},
    ]
    assert "num_rollouts" not in queue.published[2]


def test_setup_excluded_rollout_writes_json_without_video(tmp_path, monkeypatch):
    sample = {**_sample(), "rollout_id": 4}
    restoration = {"selected_frame": sample["start"], "sha256": "a" * 64}
    monkeypatch.setattr(
        runner,
        "restore_materialized_sample_state",
        lambda evaluator, cache, materialized: restoration,
    )

    class ExcludedMonitor:
        metric_eligible = False
        setup_exclusion_reason = "predicate_already_true"
        setup_evidence = {"initial_predicate_value": True}

        def __init__(self, evaluator, predicate, sample):
            self.predicate = predicate

        def result(self):
            return {
                "predicate": self.predicate,
                "dwell": 0,
                "debounce_steps": 3,
                "setup_evidence": self.setup_evidence,
            }

    monkeypatch.setattr(runner, "CompletionMonitor", ExcludedMonitor)

    class FakeGM:
        ENABLE_TRANSITION_RULES = False

        @staticmethod
        def unlocked():
            return nullcontext()

    class FakeEvaluator:
        external_subtask = None
        env = SimpleNamespace(task=SimpleNamespace(success=False))
        conditioning_metadata = {"subtask_mode": "external_fixed", "subtask": sample["subtask"]}
        subtask_events = []
        observed_subtasks = []
        observed_subtask_sources = []
        metrics = []

        def start_recording(self, *args, **kwargs):
            raise AssertionError("excluded rollout must not record")

        def stop_recording(self):
            raise AssertionError("excluded rollout must not stop an unstarted recording")

    base = SimpleNamespace(
        gm=FakeGM(),
        verify_external_subtask_rollout=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("excluded rollout must not call the policy verifier")
        ),
    )
    args = SimpleNamespace(write_video=True, video_fps=20, sample_max_steps=8)

    result = runner.run_sample(
        base,
        FakeEvaluator(),
        args,
        sample,
        {},
        {},
        tmp_path,
        "panel",
    )

    assert result["rollout_id"] == 4
    assert result["metric_eligible"] is False
    assert result["result_state"] == "setup_excluded"
    assert result["steps"] == 0
    assert result["success"] is False
    assert result["failure_reason"] == "predicate_already_true"
    assert (tmp_path / "json" / f"{runner._artifact_stem(sample)}.json").is_file()
    assert not (tmp_path / "videos").exists()


def _valid_result(sample: dict, fingerprint: str) -> dict:
    setup_evidence = {
        "initial_predicate_value": False,
        "initial_phase": "diagnostic_phase",
        "compatible_phases": [],
        "phase_compatible": False,
    }
    return {
        "task": sample["task"],
        "instance_id": sample["task_instance_id"],
        "rollout_id": sample["rollout_id"],
        "sample_id": sample["sample_id"],
        "episode_index": sample["episode_index"],
        "raw_episode_id": sample["raw_episode_id"],
        "source_segment_index": sample["source_segment_index"],
        "source_interval_index": sample["source_interval_index"],
        "segment_start": sample["start"],
        "segment_end": sample["end"],
        "segment_frames": sample["segment_frames"],
        "subtask": sample["subtask"],
        "predicate": sample["predicate"],
        "panel_fingerprint": fingerprint,
        "metric_eligible": True,
        "result_state": "evaluated",
        "setup_exclusion_reason": None,
        "setup_evidence": setup_evidence,
        "steps": 3,
        "success": True,
        "terminal_credit": False,
        "failure_reason": None,
        "terminated": False,
        "truncated": False,
        "task_success": False,
        "conditioning": {
            "subtask_mode": "external_fixed",
            "subtask": sample["subtask"],
            "source": "provided",
            "subtask_source": "provided",
            "training_set_diagnostic": True,
            "official_benchmark": False,
        },
        "subtask_events": [],
        "observed_subtasks": [sample["subtask"]] * 3,
        "observed_subtask_sources": ["provided"] * 3,
        "monitor": {
            "predicate": sample["predicate"],
            "dwell": 3,
            "satisfied": True,
            "debounce_steps": 3,
            "metric_eligible": True,
            "setup_exclusion_reason": None,
            "setup_evidence": setup_evidence,
            "initial_snapshot": {"phase": "diagnostic_phase"},
            "final_snapshot": {"phase": "diagnostic_phase"},
            "diagnostic_recoveries": 1,
            "selector_summary": {"recoveries": 1},
        },
        "restoration": {
            "selected_frame": sample["start"],
            "selected_state_size": 6,
            "state_prefix_samples": sample["start"] + 1,
            "topology_transition_steps": [1],
            "full_snapshot": True,
            "sha256": "a" * 64,
        },
        "q_score": {"final": 0.25},
    }


def _summary_panel(samples, exclusions=None):
    exclusions = exclusions or []
    first = samples[0]
    return {
        "fingerprint": "panel",
        "episodes": [
            {
                "task": first["task"],
                "task_instance_id": first["task_instance_id"],
                "sample_ids": [sample["sample_id"] for sample in samples],
            }
        ],
        "samples": samples,
        "exclusions": exclusions,
    }


def _summary_manifest(panel):
    samples = panel["samples"]
    episode = panel["episodes"][0]
    return {
        "profile": "training_subtask_closed_loop_episode_segments",
        "evaluation_kind": "training_subtask_closed_loop",
        "panel_fingerprint": panel["fingerprint"],
        "checkpoint": "/checkpoint",
        "expected_episode_jobs": 1,
        "expected_segment_rollouts": len(samples),
        "expected_annotated_segments": len(samples) + len(panel["exclusions"]),
        "expected_exclusions": len(panel["exclusions"]),
        "sample_max_steps": 2048,
        "jobs": [
            {
                "task": episode["task"],
                "instance_id": episode["task_instance_id"],
                "sample_ids": episode["sample_ids"],
            }
        ],
        "raw_preflight": {
            "schema_version": 2,
            "valid": True,
            "episodes": [
                {
                    "sample_ids": episode["sample_ids"],
                    "sha256": "a" * 64,
                }
            ],
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "episode_index": sample["episode_index"],
                    "rollout_id": sample["rollout_id"],
                    "selected_frame": sample["start"],
                    "selected_state_size": 6,
                    "transition_count_before_frame": 1,
                }
                for sample in samples
            ],
        },
    }


def _write_summary_result(results, sample, result):
    path = (
        results
        / "pi05"
        / sample["task"]
        / "json"
        / f"{sample['task']}_{sample['task_instance_id']}_{sample['rollout_id']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


def test_strict_summary_accepts_exact_sample_and_rejects_missing_video(tmp_path):
    sample = _sample()
    panel = _summary_panel([sample])
    results = tmp_path / "results"
    _write_summary_result(results, sample, _valid_result(sample, panel["fingerprint"]))

    incomplete = summarize(
        panel, results, checkpoint="/checkpoint", require_videos=True
    )
    assert not incomplete["complete"]
    assert len(incomplete["missing_videos"]) == 1

    video_path = (
        results
        / "pi05"
        / sample["task"]
        / "videos"
        / f"{sample['task']}_{sample['task_instance_id']}_0.mp4"
    )
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    manifest = _summary_manifest(panel)
    complete = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=manifest,
    )
    assert complete["complete"]
    assert complete["successes"] == 1
    assert complete["overall"]["metric_eligible"] == 1
    assert (
        complete["per_prompt_label"][sample["task"]][sample["subtask"]][
            "completion_rate"
        ]
        == 1.0
    )

    manifest["raw_preflight"]["episodes"][0]["sha256"] = "b" * 64
    mismatched = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=manifest,
    )
    assert not mismatched["complete"]
    assert any(
        "restoration SHA256" in error
        for error in mismatched["invalid_results"][0]["errors"]
    )


def test_strict_summary_excludes_setup_and_unsupported_from_denominator(tmp_path):
    first = _sample()
    second = {
        **_sample(),
        "rollout_id": 1,
        "sample_id": "sample-2",
        "source_segment_index": 4,
        "start": 8,
        "end": 40,
        "segment_frames": 32,
        "skill_description": "pour",
        "subtask": "pour paper bag",
        "predicate": {"kind": "not_inside", "target": "milk", "reference": "sack"},
    }
    unsupported = {
        **second,
        "sample_id": "sample-3",
        "source_segment_index": 5,
        "skill_description": "hand over",
        "subtask": "hand over milk",
        "reason": "unsupported_predicate",
    }
    unsupported.pop("rollout_id")
    unsupported.pop("predicate")
    panel = _summary_panel([first, second], [unsupported])
    results = tmp_path / "results"
    first_result = _valid_result(first, panel["fingerprint"])
    first_result["truncated"] = True
    second_result = _valid_result(second, panel["fingerprint"])
    setup_evidence = {"initial_predicate_value": True}
    second_result.update(
        {
            "metric_eligible": False,
            "result_state": "setup_excluded",
            "setup_exclusion_reason": "predicate_already_true",
            "setup_evidence": setup_evidence,
            "steps": 0,
            "success": False,
            "failure_reason": "predicate_already_true",
            "observed_subtasks": [],
            "observed_subtask_sources": [],
        }
    )
    second_result["monitor"].update(
        {
            "dwell": 0,
            "metric_eligible": False,
            "setup_exclusion_reason": "predicate_already_true",
            "setup_evidence": setup_evidence,
        }
    )
    _write_summary_result(results, first, first_result)
    second_path = _write_summary_result(results, second, second_result)
    video = (
        results
        / "pi05"
        / first["task"]
        / "videos"
        / f"{first['task']}_{first['task_instance_id']}_0.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    summary = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=_summary_manifest(panel),
    )

    assert summary["complete"]
    assert summary["expected_annotated_segments"] == 3
    assert summary["accounted_annotations"] == 3
    assert summary["coverage"] == 1.0
    assert summary["overall"]["metric_eligible"] == 1
    assert summary["overall"]["setup_excluded"] == 1
    assert summary["overall"]["unsupported"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["video_completeness"] == {
        "required": 1,
        "completed": 1,
        "permitted": 2,
    }
    assert summary["per_skill_description"]["hand over"]["unsupported"] == 1

    duplicate = results / "pi05" / "duplicate" / "json" / "duplicate.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(json.dumps(first_result))
    duplicated = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=_summary_manifest(panel),
    )
    assert not duplicated["complete"]
    assert len(duplicated["duplicate_results"]) == 1
    assert len(duplicated["unexpected_results"]) == 1

    duplicate.unlink()
    second_path.unlink()
    missing = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=_summary_manifest(panel),
    )
    assert not missing["complete"]
    assert len(missing["missing_results"]) == 1


def test_strict_summary_accepts_terminal_completion_credit(tmp_path):
    sample = _sample()
    panel = _summary_panel([sample])
    results = tmp_path / "results"
    result = _valid_result(sample, panel["fingerprint"])
    # A whole-task terminal condition ended the episode on the same step the
    # subtask predicate first held, so the debounce window could not accrue.
    result.update({"steps": 5, "terminated": True, "terminal_credit": True})
    result["monitor"].update({"dwell": 1, "satisfied": True})
    result["observed_subtasks"] = [sample["subtask"]] * 5
    result["observed_subtask_sources"] = ["provided"] * 5
    _write_summary_result(results, sample, result)
    video = (
        results
        / "pi05"
        / sample["task"]
        / "videos"
        / f"{sample['task']}_{sample['task_instance_id']}_0.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    summary = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=_summary_manifest(panel),
    )
    assert summary["complete"]
    assert summary["successes"] == 1

    # Credit without a satisfied predicate at termination is rejected.
    result["monitor"]["satisfied"] = False
    _write_summary_result(results, sample, result)
    rejected = summarize(
        panel,
        results,
        checkpoint="/checkpoint",
        require_videos=True,
        run_manifest=_summary_manifest(panel),
    )
    assert not rejected["complete"]
    assert any(
        "terminal completion credit" in error
        for error in rejected["invalid_results"][0]["errors"]
    )


def test_panel_shell_mode_requires_queue(tmp_path):
    script = SCRIPTS / "eval_b1k.sh"
    env = os.environ.copy()
    env.update(
        {
            "PI05_SUBTASK_SAMPLE_PANEL": str(tmp_path / "panel.json"),
            "PI05_SUBTASK_RAW_DATA_ROOT": str(tmp_path),
            "PROVIDED_SUBTASK_AS_PROMPT": "1",
            "NUM_ROLLOUTS": "1",
        }
    )
    completed = subprocess.run(
        ["bash", str(script), "carrying_in_groceries"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "requires PI05_EVAL_QUEUE_DB" in completed.stderr


def _dynamic_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=tmp_path / "run",
        runtime_root=tmp_path / "runtime",
        checkpoint=tmp_path / "checkpoint",
        repo_id="repo",
        gpus=list(range(8)),
        base_port=9000,
        max_attempts=2,
        max_restarts_per_slot=3,
        sample_max_steps=2048,
        write_video=True,
        raw_data_root=tmp_path / "raw",
        eval_env=tmp_path / "eval-env",
        pi05_env=tmp_path / "pi05-env",
        omnigibson_data_path=tmp_path / "data",
        run_id=None,
    )


def _dynamic_panel(tmp_path: Path) -> dict:
    return build_panel(
        *_write_panel_inputs(tmp_path, extra_supported_segments=True),
        seed=1,
        episodes_per_task=2,
    )


def test_dynamic_manifest_and_child_environment_are_panel_specific(tmp_path):
    panel = _dynamic_panel(tmp_path)
    args = _dynamic_args(tmp_path)
    manifest = sample_dynamic.create_manifest(
        args,
        panel,
        {"valid": True, "samples": []},
    )

    assert sample_dynamic.validate_manifest(manifest, panel) == manifest
    assert manifest["schema_version"] == 2
    assert manifest["profile"] == sample_dynamic.PROFILE
    assert manifest["evaluation_kind"] == sample_dynamic.EVALUATION_KIND
    assert manifest["episodes_per_task"] == 2
    assert manifest["expected_episode_jobs"] == 4
    assert manifest["expected_segment_rollouts"] == 10
    assert manifest["expected_annotated_segments"] == 14
    assert manifest["expected_exclusions"] == 4
    metadata = sample_dynamic._queue_metadata(manifest)
    assert metadata["profile"] == sample_dynamic.PROFILE
    assert metadata["evaluation_kind"] == sample_dynamic.EVALUATION_KIND
    assert metadata["expected_episode_jobs"] == 4
    assert metadata["expected_segment_rollouts"] == 10
    assert [job["rollout_count"] for job in manifest["jobs"]] == [3, 3, 2, 2]
    for episode, job in zip(panel["episodes"], manifest["jobs"], strict=True):
        assert job == {
            "task": episode["task"],
            "instance_id": episode["task_instance_id"],
            "episode_index": episode["episode_index"],
            "raw_episode_id": episode["raw_episode_id"],
            "rollout_count": len(episode["sample_ids"]),
            "sample_ids": episode["sample_ids"],
            "rollout_specs": [
                {
                    "rollout_id": rollout_id,
                    "sample_id": sample_id,
                    "video": "required_if_eligible",
                }
                for rollout_id, sample_id in enumerate(episode["sample_ids"])
            ],
        }

    corrupted = copy.deepcopy(manifest)
    corrupted["jobs"][0]["rollout_specs"][1]["rollout_id"] = 7
    payload = {key: value for key, value in corrupted.items() if key != "fingerprint"}
    corrupted["fingerprint"] = sample_dynamic.dynamic._fingerprint(payload)
    with pytest.raises(ValueError, match="rollout_specs"):
        sample_dynamic.validate_manifest(corrupted, panel)

    corrupted = copy.deepcopy(manifest)
    corrupted["require_videos"] = False
    payload = {key: value for key, value in corrupted.items() if key != "fingerprint"}
    corrupted["fingerprint"] = sample_dynamic.dynamic._fingerprint(payload)
    with pytest.raises(ValueError, match="requires videos"):
        sample_dynamic.validate_manifest(corrupted, panel)

    specs = sample_dynamic.worker_specs(list(range(4)), 9000)
    assert [spec.port for spec in specs] == [9000, 9001, 9002, 9003]
    assert specs[0].tasks == sample_dynamic.dynamic.TASKS
    assert specs[-1].tasks == tuple(reversed(sample_dynamic.dynamic.TASKS))
    spec = specs[0]
    environment = sample_dynamic.child_environment(manifest, spec, 2)
    assert environment["PI05_SUBTASK_SAMPLE_PANEL"] == str(
        tmp_path / "run" / "panel.json"
    )
    assert environment["PI05_SUBTASK_RAW_DATA_ROOT"] == str(tmp_path / "raw")
    assert environment["PROVIDED_SUBTASK_AS_PROMPT"] == "1"
    assert environment["NUM_ROLLOUTS"] == "1"
    assert environment["MODE"] == "train"


def test_dynamic_reconcile_uses_episode_rollout_specs(tmp_path):
    panel = _dynamic_panel(tmp_path)
    manifest = sample_dynamic.create_manifest(
        _dynamic_args(tmp_path), panel, {"valid": True, "samples": []}
    )
    queue = sample_dynamic.EvaluationQueue(manifest["queue_db"])
    queue.initialize(
        ((job["task"], job["instance_id"]) for job in manifest["jobs"]),
        max_attempts=2,
    )
    lease = queue.claim(
        "carrying_in_groceries", "run:slot-0:generation-0", manifest["attempt_root"]
    )
    job = next(
        job
        for job in manifest["jobs"]
        if (job["task"], job["instance_id"])
        == (lease.task, lease.instance_id)
    )
    lease.json_dir.mkdir(parents=True)
    for spec in job["rollout_specs"]:
        lease.result_path(spec["rollout_id"]).write_text(
            json.dumps(
                {
                    "task": lease.task,
                    "instance_id": lease.instance_id,
                    "rollout_id": spec["rollout_id"],
                    "sample_id": spec["sample_id"],
                    "steps": 0,
                    "success": False,
                    "metric_eligible": False,
                    "result_state": "setup_excluded",
                }
            )
        )

    assert sample_dynamic.reconcile_lease(queue, manifest, lease)
    assert queue.counts(readonly=True)["counts"]["succeeded"] == 1
    assert len(list(Path(manifest["results_root"]).rglob("*.json"))) == 3


def test_dynamic_status_counts_segment_rollouts_and_episode_jobs(tmp_path, capsys):
    panel = _dynamic_panel(tmp_path)
    args = _dynamic_args(tmp_path)
    manifest = sample_dynamic.create_manifest(
        args, panel, {"valid": True, "samples": []}
    )
    args.run_root.mkdir()
    (args.run_root / "panel.json").write_text(json.dumps(panel))
    (args.run_root / "manifest.json").write_text(json.dumps(manifest))
    queue = sample_dynamic.EvaluationQueue(manifest["queue_db"])
    queue.initialize(
        ((job["task"], job["instance_id"]) for job in manifest["jobs"]),
        max_attempts=2,
    )

    first_job = manifest["jobs"][0]
    sample_by_id = {sample["sample_id"]: sample for sample in panel["samples"]}
    json_dir = (
        Path(manifest["results_root"])
        / "pi05"
        / first_job["task"]
        / "json"
    )
    json_dir.mkdir(parents=True)
    for index, spec in enumerate(first_job["rollout_specs"][:2]):
        sample = sample_by_id[spec["sample_id"]]
        result = _valid_result(sample, panel["fingerprint"])
        result["rollout_id"] = spec["rollout_id"]
        if index == 0:
            result["metric_eligible"] = True
        else:
            result.update(
                {
                    "success": False,
                    "metric_eligible": False,
                    "result_state": "setup_excluded",
                }
            )
        (json_dir / f"{first_job['task']}_{first_job['instance_id']}_{index}.json").write_text(
            json.dumps(result)
        )

    sample_dynamic._run_status(argparse.Namespace(run_root=args.run_root))
    status = json.loads(capsys.readouterr().out)
    assert status["expected_episode_jobs"] == 4
    assert status["completed_episode_jobs"] == 0
    assert status["episode_queue_counts"] == {
        "pending": 4,
        "running": 0,
        "succeeded": 0,
        "exhausted": 0,
    }
    assert status["expected_segment_rollouts"] == 10
    assert status["completed_segment_rollouts"] == 2
    assert status["remaining_segment_rollouts"] == 8
    assert status["eligible_segment_rollouts"] == 1
    assert status["successful_segment_rollouts"] == 1
    assert status["setup_excluded_segment_rollouts"] == 1
    assert status["success_rate"] == 1.0
    assert status["invalid_results"] == []

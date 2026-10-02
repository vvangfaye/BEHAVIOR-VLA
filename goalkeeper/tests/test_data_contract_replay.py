import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import h5py
import numpy as np
import pytest

from goalkeeper.data import check_dataset, load_manifest, load_record, merge_manifests, write_record_archive
from goalkeeper.synthetic import make_dataset


def _real_fixture(tmp_path):
    manifest = Path(make_dataset(tmp_path / "source", groups=40, windows=1))
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["source"] = "training_demo"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def test_formal_dataset_contract_is_sealed_and_detects_tampering(tmp_path):
    source = _real_fixture(tmp_path)
    manifest = Path(merge_manifests([source], tmp_path / "formal"))
    checked = check_dataset(manifest.parent)
    assert checked["row_count"] == 40
    assert all(checked["splits"][name]["groups"] for name in ("train", "dev", "calibration"))
    assert (manifest.parent / "_READY").is_file()
    with pytest.raises(FileExistsError, match="sealed"):
        merge_manifests([source], manifest.parent)
    record = next((manifest.parent / "records").glob("*.npz"))
    with record.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        check_dataset(manifest.parent)


def test_formal_merge_rejects_synthetic_rows(tmp_path):
    manifest = make_dataset(tmp_path / "synthetic", groups=20, windows=1)
    with pytest.raises(ValueError, match="synthetic"):
        merge_manifests([manifest], tmp_path / "formal")


def test_formal_merge_copies_an_indexed_episode_archive_once(tmp_path):
    manifest = _real_fixture(tmp_path)
    raw_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    records, queries, atom_keys = [], None, None
    for row in raw_rows:
        with np.load(manifest.parent / row["record"], allow_pickle=False) as data:
            records.append({key: data[key].copy() for key in
                            ("visual", "proprio", "actions", "truth", "visibility", "future_truth")})
            if queries is None:
                queries, atom_keys = data["queries"].copy(), data["atom_keys"].copy()
    archive = manifest.parent / "episode.npz"
    write_record_archive(archive, records, queries, atom_keys)
    for index, row in enumerate(raw_rows):
        row["record"] = archive.name
        row["record_index"] = index
    manifest.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))

    merged = Path(merge_manifests([manifest], tmp_path / "formal"))
    checked = check_dataset(merged.parent)
    rows = load_manifest(merged)
    assert checked["record_file_count"] == 1
    assert len(list((merged.parent / "records").glob("*.npz"))) == 1
    assert [row["record_index"] for row in rows] == list(range(40))
    assert load_record(rows[-1])["actions"].shape == load_record(rows[0])["actions"].shape


def _load_replay_module():
    path = Path(__file__).parents[1] / "scripts" / "replay_hdf5.py"
    spec = importlib.util.spec_from_file_location("goalkeeper_replay_hdf5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mocked_hdf5_replay_emits_t_plus_one_and_exact_predicates(tmp_path, monkeypatch):
    replay_module = _load_replay_module()
    bddl = tmp_path / "task.bddl"
    bddl.write_text("(define (problem p) (:objects a - switch) (:goal (on a)))")
    source = tmp_path / "demo.hdf5"
    with h5py.File(source, "w") as data:
        root = data.create_group("data")
        root.attrs["config"] = json.dumps({
            "task": {"activity_name": "carrying_in_groceries"},
            "scene": {"scene_model": "training_scene", "scene_instance": "scene_instance_3"},
        })
        demo = root.create_group("demo_0")
        demo.create_dataset("action", data=np.zeros((2, 2), np.float32))
        demo.create_dataset("state", data=np.arange(12, dtype=np.float32).reshape(3, 4))
        demo.create_dataset("state_size", data=np.array([4, 4, 4]))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "approved": True, "source": "training_demo", "task": "carrying_in_groceries", "task_id": 14,
        "episode_id": 0, "scene_model": "training_scene", "scene_instance": "scene_instance_3",
        "instance_id": 3,
        "demo_sha256": sha(source), "bddl_sha256": sha(bddl),
    }))

    predicate_class = type("OnPredicate", (), {})
    from bddl import predicates
    monkeypatch.setitem(predicates.TOKEN_TO_PREDICATE, "on", predicate_class)

    class Task:
        value = False
        calls = []
        def _evaluate_predicate(self, predicate, *entities):
            self.calls.append((predicate, entities))
            self.value = not self.value
            return self.value

    class FakeBase:
        @classmethod
        def create_from_hdf5(cls, **kwargs):
            return cls()
        def __init__(self, *args, **kwargs):
            self.env = types.SimpleNamespace(task=Task(), close=lambda: None)
            self.input_hdf5 = types.SimpleNamespace(close=lambda: None)
        def _process_obs(self, obs, info): return obs
        def playback_episode(self, **kwargs):
            # DataPlaybackWrapper reset produces one replaceable pre-restore frame.
            for index in range(4):
                obs = {"head": np.full((2, 2, 3), index, np.uint8),
                       "left": np.full((2, 2, 3), index, np.uint8),
                       "right": np.full((2, 2, 3), index, np.uint8),
                       "state": np.array([index, index + 1], np.float32)}
                self._process_obs(obs, {})
        def close_dataset(self): pass

    package = types.ModuleType("omnigibson")
    envs = types.ModuleType("omnigibson.envs")
    hdf = types.ModuleType("omnigibson.envs.hdf5_data_wrapper")
    hdf.HDF5PlaybackWrapper = FakeBase
    monkeypatch.setitem(sys.modules, "omnigibson", package)
    monkeypatch.setitem(sys.modules, "omnigibson.envs", envs)
    monkeypatch.setitem(sys.modules, "omnigibson.envs.hdf5_data_wrapper", hdf)

    def fake_open(input_path, output_path, task, program, keys, *, scene_model, scene_instance,
                  capture_stride, features=None, render_iterations=1):
        assert (scene_model, scene_instance) == ("training_scene", "scene_instance_3")
        wrapper_type = replay_module._make_wrapper(
            program, keys, capture_stride=capture_stride, features=features
        )
        return wrapper_type(), "training_scene", types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(replay_module, "_open_wrapper", fake_open)
    output = tmp_path / "episode.npz"
    result = replay_module.replay(source, output, bddl, approval, task="carrying_in_groceries", episode_id=0,
                                  rgb_head_key="head", rgb_left_key="left", rgb_right_key="right",
                                  proprio_key="state")
    with np.load(output, allow_pickle=False) as episode:
        assert episode["actions"].shape[0] == 2
        assert episode["proprio"].shape[0] == 3
        assert episode["truth"].shape == episode["visibility"].shape == (3, 1)
        assert np.all(episode["visibility"] == -1)
        assert episode["group_id"].item() == result["group_id"]
    assert len(Task.calls) == 3
    assert all(call == (predicate_class, ("a",)) for call in Task.calls)

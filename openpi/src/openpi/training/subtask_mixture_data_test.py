"""Reader integration checks without a simulator, JAX or a video decoder."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.recovery_index import build_index
from openpi.training.subtask_mixture import MixturePlan
from openpi.training.subtask_mixture_data import create_dataset
from openpi.training.subtask_mixture_data import execution_contract
from openpi.training.subtask_mixture_data import resolve_roots
from openpi.training.subtask_mixture_data import RGB_KEYS


CAMERAS = sorted(RGB_KEYS)


def make_root(path, *, collected=False):
    meta = path / "meta"
    episodes_dir = meta / "episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    features = {"observation.state": {"shape": [57 if collected else 61], "dtype": "float32"},
                **{key: {"shape": [2, 2, 3], "dtype": "video"} for key in CAMERAS}}
    (meta / "info.json").write_text(json.dumps({"features": features, "fps": 30}))
    task_name = "collect_task" if collected else "expert_task"
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task_name": task_name,
                                               "task": "Natural " + task_name}) + "\n")
    rows = [{"episode_index": 42, "length": 90, "dataset_from_index": 1000, "task_index": 0,
             "task_instance_id": 2}]
    if not collected:
        rows.insert(0, {"episode_index": 7, "length": 64, "dataset_from_index": 0, "task_index": 0,
                        "task_instance_id": 1})
    pq.write_table(pa.Table.from_pylist(rows), episodes_dir / "file-000.parquet")
    files = [{"path": f.relative_to(path).as_posix(), "role": "metadata",
              "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(meta.rglob("*")) if f.is_file()]
    return [{**row, "task_name": task_name} for row in rows], files


class FakeLeRobot:
    """Models episode filtering in original parquet order, not source row offsets."""
    corrupt_identity = False
    corrupt_action = False
    corrupt_image = False
    padded_action = False

    def __init__(self, *, root, episodes, delta_timestamps, **kwargs):
        from pathlib import Path
        root = Path(root)
        assert len(delta_timestamps["action"]) == 32
        assert delta_timestamps["action"][-1] == pytest.approx(31 / 30)
        rows = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        self.selected = [(row["episode_index"], frame) for row in rows if row["episode_index"] in episodes
                         for frame in range(row["length"])]
        self.dim = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["shape"][0]

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index):
        episode, frame = self.selected[index]
        actions = np.full((32, 23), frame, np.float32)
        if self.corrupt_action:
            actions[0, 0] = np.nan
        return {"episode_index": episode, "frame_index": frame + int(self.corrupt_identity), "task_index": 0,
                "observation.state": np.zeros(self.dim), "action": actions,
                "action_is_pad": np.full(32, self.padded_action, dtype=bool),
                **{key: np.zeros((3, 2, 2)) if not self.corrupt_image else np.zeros((1, 2, 2)) for key in CAMERAS}}


@pytest.fixture
def mixture(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    metadata, files = make_root(old)
    fresh_meta, fresh_files = make_root(new, collected=True)
    segments = [{"episode_index": 7, "start": 0, "end": 64, "subtask": "replay skill"},
                {"episode_index": 42, "start": 0, "end": 10, "subtask": ""},
                {"episode_index": 42, "start": 10, "end": 90, "subtask": "recovered skill"}]
    indexes = []
    for kind, ids in (("recovered", [42]), ("replay", [7])):
        indexes.append(build_index(metadata, segments, dataset_root=old, root_id="expert", repo_id="original",
                                   source_kind=kind, episode_indices=ids, provenance={"source_files": files}))
    proof = {"episode_index": 42, "length": 90, "task_name": "collect_task", "task_instance_id": 2,
             "source_kind": "collected", "content_sha256": "a" * 64, "trial_id": "trial-1"}
    indexes.append(build_index(fresh_meta, [{"episode_index": 42, "start": 0, "end": 90, "subtask": "fresh skill"}],
                               dataset_root=new, root_id="fresh", repo_id="fresh", source_kind="collected",
                               episode_indices=[42], provenance={"source_files": fresh_files, "collected_episodes": [proof]}))
    sources = []
    for index in indexes:
        path = tmp_path / (index["source_kind"] + ".json")
        path.write_text(json.dumps(index))
        sources.append({"kind": index["source_kind"], "weight": 1 / 3,
                        "indexes": [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]})
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "b1k_three_source_mixture", "seed": 42,
                               "global_batch_size": 16, "action_horizon": 32,
                               "phases": [{"start_step": 0, "end_step": 50, "sources": sources}]}))
    return SimpleNamespace(subtask_prediction=True, prompt_from_task=True, subtask_whole_task_prob=0,
                           frame_stride=1, episodes_index=None, memory_contract=None,
                           subtask_mixture_path=str(path), subtask_mixture_roots_path=None,
                           action_sequence_keys=("action",), data_cls=FakeLeRobot, dataset_kwargs={})


def reader(config):
    return create_dataset(config, 32, metadata_cls=lambda **kwargs: SimpleNamespace(fps=30))


def test_real_index_to_reader_preserves_raw_state_and_independent_ids(mixture):
    dataset = reader(mixture)
    seen = set()
    for i in range(120):
        ref = dataset.mixture_plan.reference(i)
        sample = dataset[i]
        seen.add(ref["source_kind"])
        assert sample["subtask"] == ref["subtask"]
        assert sample["prompt"] == "Natural " + ref["task_name"]
        assert sample["episode_index"] == ref["episode_index"]
        assert sample["frame_index"] == ref["frame"]
        assert sample["action"][0, 0] == ref["frame"]
        assert ref["frame"] + 32 <= ref["segment_end"]
        assert sample["observation.state"].shape == ((57,) if ref["root_id"] == "fresh" else (61,))
    assert seen == {"recovered", "replay", "collected"}
    # The selected expert subset is [7, 42]; source offset 1000 is intentionally wrong for row access.
    assert dataset.readers["expert"].offsets == {7: 0, 42: 64}


def test_relocation_uses_identical_index_and_samples(mixture, tmp_path):
    first = reader(mixture)
    relocated = tmp_path / "staged"
    shutil.copytree(tmp_path / "old", relocated)
    mapping = tmp_path / "roots.json"
    mapping.write_text(json.dumps({"expert": str(relocated)}))
    mixture.subtask_mixture_roots_path = str(mapping)
    second = reader(mixture)
    assert first.mixture_plan.fingerprint == second.mixture_plan.fingerprint
    for i in (0, 17, 433):
        assert first[i]["subtask"] == second[i]["subtask"]
        np.testing.assert_array_equal(first[i]["action"], second[i]["action"])


def test_reader_opens_only_sampled_episode_and_bounds_cache(mixture, monkeypatch):
    calls = []
    class TrackingReader(FakeLeRobot):
        def __init__(self, **kwargs):
            calls.append(kwargs["episodes"])
            super().__init__(**kwargs)
    mixture.data_cls = TrackingReader
    dataset = reader(mixture)
    assert calls == []
    for i in range(50):
        ref = dataset.mixture_plan.reference(i)
        sample = dataset[i]
        assert sample["episode_index"] == ref["episode_index"]
    assert calls and all(len(selected) == 1 for selected in calls)
    assert all(len(r._datasets) <= 2 for r in dataset.readers.values())


def test_metadata_mutation_rejected_before_any_reader(mixture, tmp_path):
    with (tmp_path / "old/meta/tasks.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="metadata changed"):
        reader(mixture)


@pytest.mark.parametrize("option", ["corrupt_identity", "corrupt_action", "corrupt_image", "padded_action"])
def test_runtime_sample_alignment_and_shapes_fail_closed(mixture, monkeypatch, option):
    monkeypatch.setattr(FakeLeRobot, option, True)
    with pytest.raises(ValueError):
        reader(mixture)[0]


def test_incompatible_timebase_is_rejected(mixture):
    with pytest.raises(ValueError, match="30 Hz"):
        create_dataset(mixture, 32, metadata_cls=lambda **kwargs: SimpleNamespace(fps=15))


@pytest.mark.parametrize("field,value", [("frame_stride", 8), ("episodes_index", [42]),
                                         ("subtask_whole_task_prob", .1), ("memory_contract", {})])
def test_legacy_selection_cannot_override_frozen_plan(mixture, field, value):
    setattr(mixture, field, value)
    with pytest.raises(ValueError):
        reader(mixture)


def test_duplicate_physical_roots_rejected(mixture, tmp_path):
    mapping = tmp_path / "roots.json"
    mapping.write_text(json.dumps({"expert": str(tmp_path / "old"), "fresh": str(tmp_path / "old")}))
    with pytest.raises(ValueError, match="alias"):
        resolve_roots(MixturePlan.load(mixture.subtask_mixture_path), mapping)


def test_execution_contract_binds_normalization_and_topology():
    norm = {"state": SimpleNamespace(mean=np.zeros(23), std=np.ones(23))}
    original = execution_contract(128, 2, 42, norm)
    assert original != execution_contract(128, 1, 42, norm)
    norm["state"].std[0] = 2
    assert original != execution_contract(128, 2, 42, norm)


def test_four_source_reader_emits_supervision_and_source_identity(mixture, tmp_path):
    path=tmp_path/'mixture.json';m=json.loads(path.read_text())
    old=json.loads((tmp_path/'collected.json').read_text())
    full=tmp_path/'full';shutil.copytree(tmp_path/'new',full)
    old.update(root_id='full',repo_id='full',dataset_root=str(full),source_kind='full_episode')
    old['episodes'][0]['content_sha256']='b'*64
    old['provenance'].update(label_semantics='self_generated_weak_subtask',language_loss_enabled=False)
    old['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in old.items() if k!='fingerprint'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    f=tmp_path/'full.json';f.write_text(json.dumps(old))
    m.update(schema_version=2,kind='b1k_four_source_mixture')
    m['phases'][0]['sources'].append(dict(kind='full_episode',weight=1/3,indexes=[dict(path=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest())]))
    path.write_text(json.dumps(m));dataset=reader(mixture);seen=set()
    for i in range(200):
        ref=dataset.mixture_plan.reference(i);sample=dataset[i];seen.add(ref['source_kind'])
        assert bool(sample['has_subtask_target'])==(ref['source_kind']!='full_episode')
        assert int(sample['source_kind_id'])=={'replay':0,'recovered':1,'collected':2,'full_episode':3}[ref['source_kind']]
        assert sample['subtask']==ref['subtask']
    assert len(seen)==4

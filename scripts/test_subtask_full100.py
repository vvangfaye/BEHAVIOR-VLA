"""CPU coverage of full100 panels, role predicates and Nebula input integrity."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "nebula"))
import subtask_sample_panel as panels
import subtask_completion as completion
import eval_b1k_subtask_dynamic as dynamic
import subtask_eval


def row(skill, targets=("apple_1",), references=()):
    return {"skill_description": skill, "manipulating_object_ids": list(targets),
            "reference_object_ids": list(references), "object_ids": [*targets, *references],
            "completion_contract": completion.CONTRACT}


@pytest.mark.parametrize("skill,state", [("place on", "OnTop"), ("place in", "Inside"), ("place under", "Under")])
def test_relational_completion_preserves_exact_role_ids(skill, state):
    predicate = completion.infer_role_predicate(row(skill, references=("table_abcd_0",)))
    assert predicate["targets"] == ["apple_1"]
    assert predicate["references"] == ["table_abcd_0"] and predicate["state"] == state


def test_next_to_requires_both_reference_objects():
    assert completion.infer_role_predicate(row("place on next to", references=("table",))) is None
    predicate = completion.infer_role_predicate(row("place on next to", references=("table", "vase")))
    assert predicate["references"] == ["table", "vase"] and predicate["secondary_state"] == "NextTo"


@pytest.mark.parametrize("skill", ["pour", "hand over", "chop", "press", "ignite", "spray", "attach"])
def test_unknown_skill_is_excluded(skill):
    assert completion.infer_role_predicate(row(skill)) is None


def test_unambiguous_binding_and_ambiguous_binding_rejection():
    obj = SimpleNamespace(name="apple_1")
    env = SimpleNamespace(task=SimpleNamespace(object_scope={"apple.n.01_1": obj}), scene=SimpleNamespace(objects=[obj]))
    assert completion.resolve_object(env, "apple_1") is obj
    assert completion.resolve_object(env, "apple.n.01_1") is obj
    duplicate = SimpleNamespace(name="apple1")
    env.scene.objects.append(duplicate)
    with pytest.raises(ValueError, match="Ambiguous"):
        completion.resolve_object(env, "APPLE-1")
    with pytest.raises(ValueError, match="absent"):
        completion.resolve_object(env, "apple_2")


class Observer:
    def __init__(self, values):
        self.values = iter(values)

    def observe(self):
        value = next(self.values)
        return value, {"value": value}


def test_completion_requires_three_consecutive_control_steps():
    monitor = completion.RoleCompletionMonitor(None, {"kind": "role_bound"}, observer=Observer([False, True, False, True, True, True]))
    assert monitor.metric_eligible
    assert [monitor.refresh(step) for step in range(1, 6)] == [False, False, False, False, True]
    result = monitor.result()
    assert result["dwell"] == 3 and result["satisfied"] is True and result["diagnostic_recoveries"] == 0


def test_already_completed_and_unobservable_samples_are_excluded():
    initial_true = completion.RoleCompletionMonitor(None, {}, observer=Observer([True]))
    assert not initial_true.metric_eligible and initial_true.setup_exclusion_reason == "predicate_already_true"
    class Missing:
        def observe(self):
            raise ValueError("missing object")
    missing = completion.RoleCompletionMonitor(None, {}, observer=Missing())
    assert not missing.metric_eligible and missing.setup_exclusion_reason == "predicate_unobservable"
    assert missing.setup_evidence["error"] == "missing object"


@pytest.fixture
def full100(tmp_path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta/episodes/chunk-000"
    meta.mkdir(parents=True)
    metadata, segments = [], []
    for index, task in enumerate(sorted(panels.challenge_tasks())):
        for instance in range(4):
            episode = index * 4 + instance
            metadata.append({"episode_index": episode, "tasks": [task], "task_index": index,
                             "task_instance_id": instance, "raw_episode_id": episode,
                             "annotation_path": f"annotations/{episode}.json", "length": 64})
            common = {"episode_index": episode, "annotation_path": f"annotations/{episode}.json",
                      "source_interval_index": 0, "source_skill_index": 0,
                      "memory_prefix": "", "spatial_prefix": "", "is_annotated": True,
                      "provenance": "annotated"}
            segments.append({**common, **row("pick up from"), "start": 0, "end": 40,
                             "source_segment_index": 0, "subtask": "pick up from apple"})
            segments.append({**common, **row("pour"), "start": 40, "end": 64,
                             "source_segment_index": 1, "subtask": "pour apple"})
    pq.write_table(pa.Table.from_pylist(metadata), meta / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(segments), tmp_path / "segments.parquet")
    (tmp_path / "episodes.txt").write_text("\n".join(map(str, range(400))))
    (tmp_path / "split.json").write_text("{}")
    panel = panels.build_panel(dataset, tmp_path / "segments.parquet", tmp_path / "episodes.txt", tmp_path / "split.json",
                               episodes_per_task=4, task_scope="full100")
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel))
    return path, panel


def test_full100_balanced_selection_and_explicit_unsupported_denominator(full100):
    path, panel = full100
    assert len(panel["selection"]["tasks"]) == 100 and len(panel["episodes"]) == 400
    assert panel["counts"]["samples"] == 400 and panel["counts"]["exclusions"] == 400
    assert panels.load_panel(path) == panel
    specs = dynamic.worker_specs([0, 1, 2, 3], 8140, tuple(panel["selection"]["tasks"]))
    assert all(set(spec.tasks) == panels.challenge_tasks() for spec in specs)


def test_task_mapping_and_predicate_tampering_are_rejected(full100):
    path, panel = full100
    broken = copy.deepcopy(panel)
    broken["selection"]["task_index_to_name"]["99"] = "unknown"
    broken["fingerprint"] = panels._canonical_hash({k: v for k, v in broken.items() if k != "fingerprint"})
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="cover challenge tasks"):
        panels.load_panel(path)
    broken = copy.deepcopy(panel)
    broken["samples"][0]["predicate"]["targets"] = ["another_object"]
    broken["fingerprint"] = panels._canonical_hash({k: v for k, v in broken.items() if k != "fingerprint"})
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="invalid predicate"):
        panels.load_panel(path)


def test_joint_manifest_preserves_training_inference_semantics(full100, tmp_path):
    path, panel = full100
    args = argparse.Namespace(run_root=tmp_path / "run", runtime_root=None, run_id="joint100",
                              checkpoint=tmp_path / "checkpoint", repo_id="repo", gpus=[0, 1, 2, 3], base_port=8140,
                              max_attempts=2, max_restarts_per_slot=8, sample_max_steps=2048, write_video=True,
                              raw_data_root=tmp_path / "raw", eval_env=tmp_path / "eval", pi05_env=tmp_path / "pi05",
                              omnigibson_data_path=tmp_path / "assets", subtask_prediction=True)
    manifest = dynamic.create_manifest(args, panel, {"valid": True})
    dynamic.validate_manifest(manifest, panel)
    assert manifest["expected_episode_jobs"] == 400
    settings = manifest["child_environment"]
    assert settings["SUBTASK_PREDICTION"] == "1" and settings["PROVIDED_SUBTASK_AS_PROMPT"] == "0"
    assert settings["MAX_TOKEN_LEN"] == "512"


def inventory_for(panel):
    source = "oss://test-bucket/raw"
    return {"schema_version": 1, "raw_source": source, "panel_fingerprint": panel["fingerprint"], "episodes": [
        {"relative_path": f"task-{e['task_index']:04d}/episode_{e['raw_episode_id']:08d}.hdf5",
         "key": f"raw/task-{e['task_index']:04d}/episode_{e['raw_episode_id']:08d}.hdf5", "bytes": 100,
         "etag": "frozen", "crc64": "42"} for e in panel["episodes"]]}


def test_nebula_rejects_changed_raw_objects_before_submission(full100, tmp_path):
    path, panel = full100
    inventory = inventory_for(panel)
    inv_path = tmp_path / "inventory.json"
    inv_path.write_text(json.dumps(inventory))
    subtask_eval.load_inputs(path, inv_path)
    client = SimpleNamespace(key=lambda s: s.replace("oss://test-bucket/", ""), bucket=SimpleNamespace(
        head_object=lambda _: SimpleNamespace(content_length=100, etag="changed", headers={"x-oss-hash-crc64ecma": "42"})))
    with pytest.raises(ValueError, match="changed since panel"):
        subtask_eval.preflight(path, inv_path, inventory["raw_source"], client)
    inventory["episodes"][-1] = inventory["episodes"][0]
    inv_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="duplicate"):
        subtask_eval.load_inputs(path, inv_path)


@pytest.mark.parametrize("grasped,relation,secondary,expected", [
    (True, True, True, False), (False, True, True, True),
    (False, False, True, False), (False, True, False, False)])
def test_placement_requires_release_and_every_relation(grasped, relation, secondary, expected):
    target = SimpleNamespace(name="apple")
    refs = [SimpleNamespace(name="table"), SimpleNamespace(name="vase")]
    observer = completion.RoleStateObserver.__new__(completion.RoleStateObserver)
    observer.targets, observer.references = [target], refs
    observer.predicate = {"operation": "placed", "state": "OnTop", "secondary_state": "NextTo"}
    observer._grasped = lambda obj: grasped
    observer._state = lambda obj, name, reference=None: relation if name == "OnTop" else secondary
    result, evidence = observer.observe()
    assert result is expected
    assert evidence["targets"][0]["grasped"] is grasped


def test_stationary_restoration_does_not_keep_stale_contact_predicate(monkeypatch):
    import sys
    state_class = type("OnTop", (), {})
    monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(
        object_states=SimpleNamespace(OnTop=state_class)))
    class CachedState:
        actual = False
        cached = False
        def clear_cache(self):
            self.cached = None
        def get_value(self, other):
            if self.cached is None:
                self.cached = self.actual
            return self.cached
    state = CachedState()
    target = SimpleNamespace(name="container", states={state_class: state})
    reference = object()
    assert completion.RoleStateObserver._state(target, "OnTop", reference) is False
    state.actual = True
    assert completion.RoleStateObserver._state(target, "OnTop", reference) is True


def test_unsupported_prompt_has_explicit_counts_and_empty_csv_rate(full100, tmp_path):
    import csv
    import summarize_subtask_closed_loop as summaries
    _, panel = full100
    summary = summaries.summarize(panel, tmp_path / "missing_results", checkpoint="checkpoint", require_videos=False)
    task = sorted(panel["selection"]["tasks"])[0]
    excluded = summary["per_prompt_label"][task]["pour apple"]
    assert excluded["unsupported"] == 4 and excluded["completion_rate"] is None
    subtask_eval.write_reports(summary, tmp_path)
    with (tmp_path / "subtask_success.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 200
    selected = next(item for item in records if item["task"] == task and item["subtask"] == "pour apple")
    assert selected["unsupported"] == "4" and selected["completion_rate"] == ""

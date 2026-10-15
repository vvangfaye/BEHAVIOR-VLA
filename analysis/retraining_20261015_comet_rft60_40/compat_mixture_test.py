"""CPU checks for distributed sample identity and safe curriculum continuation."""

import collections
import copy
import hashlib
import json
from pathlib import Path

import pytest

from openpi.training.subtask_mixture import MixturePlan
from openpi.training.subtask_mixture import check_consumed_contract
from openpi.training.subtask_mixture import compare_consumed_contract
from openpi.training.subtask_mixture import consumed_contract
from openpi.training.subtask_mixture import save_consumed_contract


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def fingerprint(index):
    index["fingerprint"] = hashlib.sha256(canonical({k: v for k, v in index.items() if k != "fingerprint"})).hexdigest()
    return index


def make_index(root, kind, *, episode=0, instance=1, length=40, task="task", subtask="pick", held=None):
    record = {"episode_index": episode, "length": length, "task_name": task, "task_instance_id": instance}
    if kind == "collected":
        record["content_sha256"] = hashlib.sha256(root.encode()).hexdigest()
    return fingerprint({"schema_version": 1, "kind": "b1k_valid_intervals", "source_kind": kind,
                        "root_id": root, "repo_id": "repo", "dataset_root": f"/data/{root}", "action_horizon": 4,
                        "held_out_instances": held or [], "episodes": [record],
                        "intervals": [{"episode_index": episode, "start": 5, "end": length, "subtask": subtask}],
                        "provenance": {"source": "fixture"}})


def write_index(tmp_path, index, name=None):
    path = tmp_path / (name or (index["root_id"] + ".json"))
    path.write_text(json.dumps(index, ensure_ascii=False))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source(tmp_path, kind, indexes, weight):
    return {"kind": kind, "weight": weight, "indexes": [write_index(tmp_path, i) for i in indexes]}


def manifest(sources, *, steps=1000, batch=16):
    return {"schema_version": 1, "kind": "b1k_three_source_mixture", "seed": 42,
            "global_batch_size": batch, "action_horizon": 4,
            "phases": [{"start_step": 0, "end_step": steps, "sources": sources}]}


def load(tmp_path, data, name="mixture.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return MixturePlan.load(path)


def two_sources(tmp_path):
    return [source(tmp_path, "recovered", [make_index("old", "recovered")], 0.5),
            source(tmp_path, "collected", [make_index("new", "collected", length=4000)], 0.5)]


def test_mixture_kind_weight_ignores_pool_size_and_number_of_roots(tmp_path):
    sources = two_sources(tmp_path)
    sources[1]["indexes"] += [write_index(tmp_path, make_index(f"new{i}", "collected", length=5000, task=f"task{i}"))
                              for i in range(4)]
    plan = load(tmp_path, manifest(sources))
    counts = collections.Counter(plan.reference(i)["source_kind"] for i in range(plan.num_samples))
    assert 0.48 < counts["recovered"] / plan.num_samples < 0.52
    assert plan.num_samples == 16000
    assert len(plan.indexes) == 6


def test_group_and_instance_balance_ignores_segment_duration(tmp_path):
    index = make_index("old", "recovered")
    index["episodes"].extend([
        {"episode_index": 1, "length": 10000, "task_name": "task", "task_instance_id": 2},
        {"episode_index": 2, "length": 12, "task_name": "other", "task_instance_id": 3},
    ])
    index["intervals"].extend([
        {"episode_index": 1, "start": 0, "end": 10000, "subtask": "pick"},
        {"episode_index": 2, "start": 0, "end": 12, "subtask": "place"},
    ])
    plan = load(tmp_path, manifest([source(tmp_path, "recovered", [fingerprint(index)], 1)]))
    counts = collections.Counter(plan.reference(i)["task_instance_id"] for i in range(plan.num_samples))
    assert 0.23 < counts[1] / plan.num_samples < 0.27
    assert 0.23 < counts[2] / plan.num_samples < 0.27
    assert 0.48 < counts[3] / plan.num_samples < 0.52


def test_action_windows_stay_inside_labelled_segment(tmp_path):
    index = make_index("old", "recovered", length=9)
    plan = load(tmp_path, manifest([source(tmp_path, "recovered", [index], 1)]))
    assert {plan.reference(i)["frame"] for i in range(100)} == {5}
    assert plan.episodes_by_root == {"old": [0]}
    assert plan.episode_records_by_root["old"][0]["length"] == 9


def test_disjoint_rank_indices_reconstruct_global_samples_and_resume(tmp_path):
    plan = load(tmp_path, manifest(two_sources(tmp_path), steps=30))
    ranks = [list(range(rank, plan.num_samples, 2)) for rank in range(2)]
    assert set(ranks[0]).isdisjoint(ranks[1])
    assert sorted(ranks[0] + ranks[1]) == list(range(plan.num_samples))
    recovered = {i: plan.reference(i) for rank in ranks for i in reversed(rank)}
    assert [recovered[i] for i in range(plan.num_samples)] == [plan.reference(i) for i in range(plan.num_samples)]
    reloaded = load(tmp_path, plan.manifest, "resumed.json")
    assert [reloaded.reference(i) for i in range(16 * 12, plan.num_samples)] == [
        plan.reference(i) for i in range(16 * 12, plan.num_samples)]


def test_same_episode_number_in_distinct_roots_is_separate(tmp_path):
    plan = load(tmp_path, manifest(two_sources(tmp_path)))
    refs = {plan.reference(i)["root_id"] for i in range(100)}
    assert refs == {"old", "new"}
    assert plan.episodes_by_root == {"old": [0], "new": [0]}


def test_phase_boundaries(tmp_path):
    sources = two_sources(tmp_path)
    first = copy.deepcopy(sources)
    first[0]["weight"], first[1]["weight"] = 1, 0
    last = copy.deepcopy(sources)
    last[0]["weight"], last[1]["weight"] = 0, 1
    data = manifest(first, steps=2)
    data["phases"].append({"start_step": 2, "end_step": 4, "sources": last})
    plan = load(tmp_path, data)
    assert all(plan.reference(i)["source_kind"] == "recovered" for i in range(32))
    assert all(plan.reference(i)["source_kind"] == "collected" for i in range(32, 64))
    with pytest.raises(IndexError):
        plan.reference(64)


def test_resume_accepts_changed_unconsumed_suffix_and_extended_budget(tmp_path):
    data = manifest(two_sources(tmp_path), steps=20)
    original = load(tmp_path, data)
    contract = consumed_contract(original, 10)
    changed = copy.deepcopy(data)
    changed["phases"][0]["end_step"] = 10
    future = copy.deepcopy(changed["phases"][0])
    future.update(start_step=10, end_step=40)
    future["sources"][0]["weight"] = 0.8
    future["sources"][1]["weight"] = 0.2
    changed["phases"].append(future)
    continuation = load(tmp_path, changed)
    compare_consumed_contract(contract, continuation, 10)
    assert [original.reference(i) for i in range(160)] == [continuation.reference(i) for i in range(160)]
    assert [original.reference(i) for i in range(160, 320)] != [continuation.reference(i) for i in range(160, 320)]
    with pytest.raises(ValueError, match="consumed mixture"):
        compare_consumed_contract(consumed_contract(original, 11), continuation)


@pytest.mark.parametrize("field,value", [("seed", 43), ("global_batch_size", 32)])
def test_resume_rejects_sampling_identity_change_even_at_step_zero(tmp_path, field, value):
    data = manifest(two_sources(tmp_path))
    contract = consumed_contract(load(tmp_path, data), 0)
    data[field] = value
    with pytest.raises(ValueError, match="consumed mixture"):
        compare_consumed_contract(contract, load(tmp_path, data))


def test_resume_rejects_changed_consumed_source_and_wrong_step(tmp_path):
    data = manifest(two_sources(tmp_path))
    old = load(tmp_path, data)
    contract = consumed_contract(old, 10)
    data["phases"][0]["sources"][0]["weight"] = 0.8
    with pytest.raises(ValueError, match="consumed mixture"):
        compare_consumed_contract(contract, load(tmp_path, data))
    with pytest.raises(ValueError, match="restored optimizer step"):
        compare_consumed_contract(contract, old, 11)
    with pytest.raises(ValueError, match="exceeds"):
        consumed_contract(old, old.num_steps + 1)


def test_index_location_and_root_location_do_not_change_sampling_identity(tmp_path):
    data = manifest(two_sources(tmp_path))
    original = load(tmp_path, data)
    moved = tmp_path / "moved"
    moved.mkdir()
    for src in data["phases"][0]["sources"]:
        for ref in src["indexes"]:
            index = json.loads(Path(ref["path"]).read_text())
            index["dataset_root"] = "/different/mount/" + index["root_id"]
            ref.update(write_index(moved, fingerprint(index)))
    relocated = load(tmp_path, data)
    compare_consumed_contract(consumed_contract(original, 10), relocated)
    assert original.fingerprint == relocated.fingerprint
    assert [original.reference(i) for i in range(100)] == [relocated.reference(i) for i in range(100)]


def test_checkpoint_contract_round_trip_and_tampering(tmp_path):
    plan = load(tmp_path, manifest(two_sources(tmp_path)))
    directory = tmp_path / "assets"
    saved = save_consumed_contract(directory, plan, 7)
    assert saved["manifest"] == plan.manifest
    check_consumed_contract(directory, plan, 7)
    saved["step"] = 8
    with pytest.raises(ValueError, match="fingerprint"):
        compare_consumed_contract(saved, plan)
    with pytest.raises(ValueError, match="requires a saved"):
        check_consumed_contract(tmp_path / "missing", plan)


def test_resume_ignores_redundant_identical_phase_split(tmp_path):
    data = manifest(two_sources(tmp_path), steps=20)
    original = load(tmp_path, data)
    second = copy.deepcopy(data["phases"][0])
    data["phases"][0]["end_step"] = 5
    second["start_step"] = 5
    data["phases"].append(second)
    compare_consumed_contract(consumed_contract(original, 10), load(tmp_path, data))


def test_file_sha_and_embedded_fingerprint_are_both_checked(tmp_path):
    data = manifest(two_sources(tmp_path))
    ref = data["phases"][0]["sources"][0]["indexes"][0]
    path = Path(ref["path"])
    index = json.loads(path.read_text())
    index["intervals"][0]["start"] = 6
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="file SHA256"):
        load(tmp_path, data)
    ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Index fingerprint"):
        load(tmp_path, data)


def test_all_indexes_agree_on_holdout_and_exclude_it(tmp_path):
    data = manifest(two_sources(tmp_path))
    index = make_index("new", "collected", held=[["unseen", 9]])
    data["phases"][0]["sources"][1]["indexes"] = [write_index(tmp_path, index)]
    with pytest.raises(ValueError, match="disagree on held-out"):
        load(tmp_path, data)
    bad = make_index("bad", "recovered", held=[["task", 1]])
    with pytest.raises(ValueError, match="held-out instance"):
        load(tmp_path, manifest([source(tmp_path, "recovered", [bad], 1)]))


def test_recovered_replay_overlap_is_rejected_across_different_roots(tmp_path):
    sources = [source(tmp_path, "recovered", [make_index("old", "recovered")], 0.8),
               source(tmp_path, "replay", [make_index("copy", "replay")], 0.2)]
    with pytest.raises(ValueError, match="overlaps recovered/replay"):
        load(tmp_path, manifest(sources))


def test_same_root_episode_conflict_is_rejected(tmp_path):
    a = make_index("original", "recovered")
    b = make_index("original", "replay", instance=2)
    sources = [{"kind": "recovered", "weight": 0.8, "indexes": [write_index(tmp_path, a, "a.json")]},
               {"kind": "replay", "weight": 0.2, "indexes": [write_index(tmp_path, b, "b.json")]}]
    with pytest.raises(ValueError, match="within one root"):
        load(tmp_path, manifest(sources))


def test_duplicate_collected_content_across_roots_is_rejected(tmp_path):
    a = make_index("new", "collected")
    b = make_index("copy", "collected")
    b["episodes"][0]["content_sha256"] = a["episodes"][0]["content_sha256"]
    fingerprint(b)
    with pytest.raises(ValueError, match="Duplicate collected"):
        load(tmp_path, manifest([source(tmp_path, "collected", [a, b], 1)]))


def test_collected_identity_missing_fails_closed(tmp_path):
    index = make_index("new", "collected")
    del index["episodes"][0]["content_sha256"]
    with pytest.raises(ValueError, match="content_sha256"):
        load(tmp_path, manifest([source(tmp_path, "collected", [fingerprint(index)], 1)]))


@pytest.mark.parametrize("changes,match", [
    ({"start": 8, "end": 9}, "no complete action window"),
    ({"start": 2, "end": 100}, "exceeds episode"),
    ({"subtask": ""}, "nonempty text"),
    ({"episode_index": 99}, "absent episode"),
])
def test_invalid_intervals_fail_closed(tmp_path, changes, match):
    index = make_index("old", "recovered")
    index["intervals"][0].update(changes)
    with pytest.raises(ValueError, match=match):
        load(tmp_path, manifest([source(tmp_path, "recovered", [fingerprint(index)], 1)]))


def test_duplicate_kind_and_bad_phase_schedule_are_rejected(tmp_path):
    sources = two_sources(tmp_path)
    with pytest.raises(ValueError, match="repeated source kind"):
        load(tmp_path, manifest(sources + [copy.deepcopy(sources[0])]))
    data = manifest(sources)
    data["phases"][0]["start_step"] = 1
    with pytest.raises(ValueError, match="contiguous"):
        load(tmp_path, data)


def two_groups(tmp_path):
    return source(tmp_path, "recovered", [make_index("weak", "recovered", task="weak"),
                                           make_index("strong", "recovered", task="strong")], 1)


def test_weak_group_upweight_retains_all_groups_and_instances(tmp_path):
    data_source = two_groups(tmp_path)
    data_source["group_weights"] = [{"task_name": "weak", "subtask": "pick", "weight": 3}]
    plan = load(tmp_path, manifest([data_source]))
    counts = collections.Counter(plan.reference(i)["task_name"] for i in range(plan.num_samples))
    assert 0.73 < counts["weak"] / plan.num_samples < 0.77
    assert counts["strong"] > 0
    assert set(plan.episodes_by_root) == {"weak", "strong"}


def test_consumed_group_weights_cannot_change_but_future_weights_can(tmp_path):
    data = manifest([two_groups(tmp_path)], steps=20)
    original = load(tmp_path, data)
    contract = consumed_contract(original, 10)
    data["phases"][0]["sources"][0]["group_weights"] = [
        {"task_name": "weak", "subtask": "pick", "weight": 3}]
    with pytest.raises(ValueError, match="consumed mixture"):
        compare_consumed_contract(contract, load(tmp_path, data))
    future = copy.deepcopy(data["phases"][0])
    future["start_step"] = 10
    data["phases"][0]["end_step"] = 10
    del data["phases"][0]["sources"][0]["group_weights"]
    data["phases"].append(future)
    compare_consumed_contract(contract, load(tmp_path, data))


@pytest.mark.parametrize("weight", [0, -1, float("inf"), True])
def test_invalid_group_weight_rejected(tmp_path, weight):
    data_source = two_groups(tmp_path)
    data_source["group_weights"] = [{"task_name": "weak", "subtask": "pick", "weight": weight}]
    with pytest.raises(ValueError, match="finite and positive"):
        load(tmp_path, manifest([data_source]))


def test_unknown_group_weight_is_rejected(tmp_path):
    data_source = two_groups(tmp_path)
    data_source["group_weights"] = [{"task_name": "absent", "subtask": "pick", "weight": 3}]
    with pytest.raises(ValueError, match="Unknown or repeated"):
        load(tmp_path, manifest([data_source]))


def test_four_sources_weights_and_resume_are_independent_of_archive_counts(tmp_path):
    sources = []
    for kind, weight in [('replay', .5), ('recovered', .25), ('collected', .2), ('full_episode', .05)]:
        index = make_index(kind, kind, instance=len(sources)+1)
        if kind == 'full_episode':
            index['episodes'][0]['content_sha256'] = 'f'*64
            index['provenance'].update(label_semantics='self_generated_weak_subtask', language_loss_enabled=False)
            fingerprint(index)
        sources.append(source(tmp_path,kind,[index],weight))
    data=manifest(sources,steps=10000);data.update(schema_version=2,kind='b1k_four_source_mixture')
    plan=load(tmp_path,data)
    counts=collections.Counter(plan.reference(i)['source_kind'] for i in range(plan.num_samples))
    for s in sources:assert abs(counts[s['kind']]/plan.num_samples-s['weight'])<.005
    saved=consumed_contract(plan,321)
    compare_consumed_contract(saved,load(tmp_path,data),321)
    assert plan.reference(321*plan.global_batch_size)==load(tmp_path,data).reference(321*plan.global_batch_size)
    data['phases'][0]['sources'][0]['weight']=.4
    with pytest.raises(ValueError,match='sampling changed'):
        compare_consumed_contract(saved,load(tmp_path,data),321)


def test_weak_text_cannot_be_mislabeled_as_supervised_collected(tmp_path):
    index=make_index('weak','collected')
    index['provenance']['label_semantics']='self_generated_weak_subtask'
    with pytest.raises(ValueError,match='Weak full-episode text'):
        load(tmp_path,manifest([source(tmp_path,'collected',[fingerprint(index)],1)]))


def test_task_balancing_not_proportional_to_number_of_subtasks(tmp_path):
    index=make_index('old','recovered',task='a',length=100)
    index['episodes'].append(dict(episode_index=1,length=100,task_name='b',task_instance_id=2))
    index['intervals']=[dict(episode_index=0,start=i*10,end=(i+1)*10,subtask=str(i)) for i in range(8)]
    index['intervals'].append(dict(episode_index=1,start=0,end=100,subtask='only'))
    s=source(tmp_path,'recovered',[fingerprint(index)],1);s['group_sampling']='task_balanced'
    plan=load(tmp_path,manifest([s]));counts=collections.Counter(plan.reference(i)['task_name'] for i in range(plan.num_samples))
    assert .48 < counts['a']/plan.num_samples < .52

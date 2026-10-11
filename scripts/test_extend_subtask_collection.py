"""Bounded extra collection waves preserve original evidence and use fresh seeds."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extend_subtask_collection as extension
from full_subtask_collection import HISTORY, NATIVE, freeze, validate
from subtask_improvement import digest


@pytest.fixture(scope="module")
def original():
    panel = dict(fingerprint="synthetic-panel", samples=[])
    proofs, groups = {}, []
    identity = dict(checkpoint_step=1375000, checkpoint_manifest_sha256="e" * 64, checkpoint_source="oss://fixture/model")
    inputs = {"metrics": "f" * 64}
    for index in range(299):
        successes = 0 if index < 43 else 10 if index < 47 else 25 if index < 75 else 75
        group = dict(task=f"task_{index}", subtask="pick up", assignment_index=index,
                     eligible=100, successes=successes, rate=successes / 100, underpowered=False)
        samples = []
        for position in range(2 if index >= 272 else 3):
            sid = f"sample-{index}-{position}"
            sample = dict(sample_id=sid, task=group["task"], subtask=group["subtask"], task_index=index,
                          task_instance_id=position, raw_episode_id=index * 3 + position,
                          episode_index=index * 3 + position, start=0)
            proof = dict(sample_id=sid, baseline_metric_eligible=True, raw_sha256=digest([sid, "raw"]),
                         selected_state_sha256=digest([sid, "state"]), baseline_result_sha256=digest([sid, "baseline"]),
                         state_protocol=HISTORY if index % 2 else NATIVE,
                         prefix_actions_sha256=digest([sid, "prefix"]),
                         raw_sample_evidence=dict(selected_state_size=400, transition_count_before_frame=0))
            samples.append(sample); panel["samples"].append(sample); proofs[sid] = proof
        plan = freeze(panel, group, samples, proofs, identity, [["holdout_task", 99]],
                      accepted_input_sha256=inputs, evidence_manifest_sha256="b" * 64, seed=1011 + index)
        groups.append(dict(index=index, state="complete", group=group, plan=plan, code_root="/frozen/code",
            source_manifest_sha256="a" * 64, raw_data_root="/frozen/raw", output_root=f"/frozen/campaign/groups/{index:03d}"))
    campaign = dict(state="complete", groups=groups, source_manifest_sha256="a" * 64,
                    source_evidence_sha256="b" * 64, code_root="/frozen/code", expected_trials=8700)
    evidence = dict(complete=True, samples=list(proofs.values()), source_manifest_sha256="a" * 64,
                    accepted_input_sha256=inputs)
    return campaign, {panel["fingerprint"]: panel}, evidence


def build(original, **kwargs):
    return extension.build_extension(*original, parent_campaign_sha256="c" * 64,
        source_evidence_sha256="b" * 64, output_root="/mnt/afs/new_extension", deadline_epoch=1792174505.218891, **kwargs)


def test_exact_authorized_budget_and_no_zero_or_half_rate_additions(original):
    result = build(original)
    assert result["requested_group_count"] == 32 and result["shard_count"] == 36
    assert result["expected_trials"] == 1080 and result["cumulative_planned_trials"] == 9780
    assert {entry["index"] for entry in result["groups"]} == set(range(299, 335))
    assert {entry["original_group_index"] for entry in result["groups"]} == set(range(43, 75))
    assert len(result["deferred_zero_groups"]) == 43
    assert all(entry["plan"]["expected_trials"] == 30 for entry in result["groups"])
    assert [(entry["original_group_index"], entry["wave"]) for entry in result["groups"][:4]] == [(43, 1), (43, 2), (44, 1), (44, 2)]
    assert {entry["planned_cumulative_trials"] for entry in result["groups"] if entry["original_group_index"] < 47} == {90}
    assert {entry["planned_cumulative_trials"] for entry in result["groups"] if entry["original_group_index"] >= 47} == {60}


def test_new_trial_and_policy_seed_sets_never_overlap_original_or_other_waves(original):
    result = build(original)
    old = [trial for entry in original[0]["groups"] for trial in entry["plan"]["trials"]]
    new = [trial for entry in result["groups"] for trial in entry["plan"]["trials"]]
    assert len({trial["trial_id"] for trial in new}) == len({trial["policy_seed"] for trial in new}) == 1080
    assert not {trial["trial_id"] for trial in new} & {trial["trial_id"] for trial in old}
    assert not {trial["policy_seed"] for trial in new} & {trial["policy_seed"] for trial in old}
    assert result == build(original)


def test_original_native_history_baseline_sources_holdouts_and_starts_are_unchanged(original):
    before = copy.deepcopy(original)
    result = build(original)
    samples = {sample["sample_id"]: sample for sample in original[1]["synthetic-panel"]["samples"]}
    for entry in result["groups"]:
        old = original[0]["groups"][entry["original_group_index"]]
        validate(entry["plan"], samples)
        assert entry["group"] == old["group"]
        assert entry["code_root"] == old["code_root"] and entry["raw_data_root"] == old["raw_data_root"]
        assert {trial["sample_id"] for trial in entry["plan"]["trials"]} == {trial["sample_id"] for trial in old["plan"]["trials"]}
        for key, value in old["plan"].items():
            if key not in {"seed", "trials", "fingerprint"}: assert entry["plan"][key] == value
    assert original == before


@pytest.mark.parametrize("damage", ["parent_hash", "source_hash", "code_identity", "rates", "missing_group", "short_original", "source_proof", "plan_hash"])
def test_changed_original_evidence_or_scope_is_rejected(original, damage):
    campaign, panels, evidence = copy.deepcopy(original)
    if damage == "parent_hash":
        with pytest.raises(ValueError):
            extension.build_extension(campaign, panels, evidence, parent_campaign_sha256="bad",
                source_evidence_sha256="b" * 64, output_root="/new", deadline_epoch=1)
        return
    if damage == "source_hash": campaign["source_evidence_sha256"] = "d" * 64
    elif damage == "code_identity": campaign["groups"][43]["code_root"] = "/other/code"
    elif damage == "rates": campaign["groups"][43]["group"]["rate"] = .2
    elif damage == "missing_group": campaign["groups"].pop()
    elif damage == "short_original":
        plan = campaign["groups"][43]["plan"]; plan["trials"].pop(); plan["expected_trials"] -= 1
        plan["fingerprint"] = digest({key: value for key, value in plan.items() if key != "fingerprint"})
    elif damage == "source_proof":
        next(row for row in evidence["samples"] if row["sample_id"] == "sample-43-0")["selected_state_sha256"] = "9" * 64
    else: campaign["groups"][43]["plan"]["fingerprint"] = "d" * 64
    with pytest.raises((ValueError, KeyError)): build((campaign, panels, evidence))


def test_existing_actual_frozen_campaign_stages_independently_reconstructible_plans(tmp_path):
    root = Path(__file__).resolve().parents[1] / "eval_results/sco_subtask_collection_20261011/frozen_full299_g4"
    if not root.is_dir(): pytest.skip("Actual frozen full299 source is not present on this checkout")
    options = dict(output_root="/mnt/afs/fixture_extension", deadline_epoch=1792174505.218891)
    first = extension.prepare(root, tmp_path / "local", **options)
    second = extension.prepare(root, tmp_path / "remote", expected_fingerprint=first["fingerprint"], **options)
    assert first == second
    for entry in first["groups"]:
        directory = tmp_path / "local/groups" / f"{entry['index']:03d}"
        assert (directory / "parent_panel.json").is_symlink()
        assert json.loads((directory / "prepared.json").read_text()) == entry
        assert extension.sha(directory / "source_evidence.json") == first["source_evidence_sha256"]
    with pytest.raises(ValueError, match="already exists"): extension.prepare(root, tmp_path / "local", **options)
    with pytest.raises(ValueError, match="reconstruction"):
        extension.prepare(root, tmp_path / "bad", expected_fingerprint="0" * 64, **options)
    assert not (tmp_path / "bad").exists()

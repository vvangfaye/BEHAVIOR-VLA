#!/usr/bin/env python3
"""Freeze bounded extra waves from the existing, independently audited campaign.

This tool does no cloud operations or simulation. Run the same command on an
AFS worker with --expected-fingerprint to stage a byte-identical campaign from
the original inputs; only new plans/receipts are written, panels/evidence are
referenced through symlinks to the already frozen source files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from full_subtask_collection import freeze, validate
from subtask_improvement import digest
from subtask_sample_panel import load_panel


KIND = "low_subtask_collection_extension"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing SHA256 identity: " + name)


def build_extension(campaign, panels, evidence, *, parent_campaign_sha256, source_evidence_sha256,
                    output_root, deadline_epoch, seed_base=20261011):
    """Pure deterministic planning; preserve every original baseline/source identity."""
    _check_hash(parent_campaign_sha256, "parent campaign")
    _check_hash(source_evidence_sha256, "source evidence")
    if not Path(output_root).is_absolute() or not isinstance(deadline_epoch, (int, float)) or isinstance(deadline_epoch, bool) or not math.isfinite(deadline_epoch):
        raise ValueError("Extension requires an absolute staging root and a finite original deadline")
    if type(seed_base) is not int or not 0 <= seed_base < 2**31:
        raise ValueError("New wave seed must be a bounded integer")
    if campaign.get("state") != "complete" or campaign.get("source_evidence_sha256") != source_evidence_sha256:
        raise ValueError("Original campaign/source audit is incomplete or changed")
    groups = campaign.get("groups", [])
    if len(groups) != 299 or {g["index"] for g in groups} != set(range(299)):
        raise ValueError("Extension requires exactly the existing 299 unique group identities")
    proofs = {}
    for proof in evidence.get("samples", []):
        sid = proof["sample_id"]
        if sid in proofs: raise ValueError("Duplicate original source proof")
        proofs[sid] = proof
    if evidence.get("complete") is not True or not proofs:
        raise ValueError("Original independent source audit is incomplete")
    source_identity = campaign["source_manifest_sha256"]
    audit_parent = campaign.get("source_audit_parent_source_manifest_sha256", source_identity)
    if evidence.get("source_manifest_sha256") not in {source_identity, audit_parent}:
        raise ValueError("Source evidence belongs to another code/audit identity")
    old_trials, used_seeds = set(), set()
    moderate, hard, zeros = [], [], []
    for entry in sorted(groups, key=lambda row: row["index"]):
        plan, group = entry["plan"], entry["group"]
        if plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
            raise ValueError("Original collection plan fingerprint changed")
        if entry["source_manifest_sha256"] != source_identity or entry["code_root"] != campaign["code_root"]:
            raise ValueError("Original group code identity differs from the campaign")
        if plan["selected_groups"] != [group] or plan["source_evidence_manifest_sha256"] != source_evidence_sha256:
            raise ValueError("Original group baseline/source identity differs")
        eligible, successes = group["eligible"], group["successes"]
        if type(eligible) is not int or type(successes) is not int or eligible < 1 or not 0 <= successes <= eligible:
            raise ValueError("Invalid original baseline denominator")
        rate = successes / eligible
        if not math.isclose(rate, group["rate"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Original success rate denominator changed")
        if plan["expected_trials"] != len(plan["trials"]): raise ValueError("Original trial budget changed")
        for trial in plan["trials"]:
            if trial["trial_id"] in old_trials: raise ValueError("Duplicate original trial identity")
            old_trials.add(trial["trial_id"]); used_seeds.add(trial["policy_seed"])
        if rate == 0: zeros.append(entry)
        elif rate < .2: hard.append(entry)
        elif rate < .5: moderate.append(entry)
    if len(old_trials) != 8700 or (len(moderate), len(hard), len(zeros)) != (28, 4, 43):
        raise ValueError("Frozen campaign differs from authorized 28 moderate / 4 hard / 43 zero groups")
    new_trials, entries = set(), []
    for original in sorted(moderate + hard, key=lambda row: row["index"]):
        old, group = original["plan"], original["group"]
        panel = panels[old["panel_fingerprint"]]
        samples_by_id = {row["sample_id"]: row for row in panel["samples"]}
        ids = sorted({trial["sample_id"] for trial in old["trials"]})
        validate(old, samples_by_id)
        if len(ids) != 3 or old["expected_trials"] != 30:
            raise ValueError("Authorized extension expects unchanged three starts / 30 initial trials")
        if old["accepted_input_sha256"] != evidence["accepted_input_sha256"]:
            raise ValueError("Original accepted-baseline evidence differs")
        selected = [samples_by_id[sid] for sid in ids]
        selected_proofs = {sid: proofs[sid] for sid in ids}
        waves = 2 if group["rate"] < .2 else 1
        for wave in range(1, waves + 1):
            candidate_seed = seed_base + original["index"] * 100 + wave
            for _ in range(1000):
                plan = freeze(panel, group, selected, selected_proofs, old["checkpoint_identity"], old["held_out_instances"],
                    accepted_input_sha256=old["accepted_input_sha256"], evidence_manifest_sha256=source_evidence_sha256,
                    repeats=10, max_steps=old["max_steps"], seed=candidate_seed)
                trial_ids = {trial["trial_id"] for trial in plan["trials"]}
                policy_seeds = {trial["policy_seed"] for trial in plan["trials"]}
                if (len(trial_ids) == 30 and len(policy_seeds) == 30 and not trial_ids & (old_trials | new_trials)
                        and not policy_seeds & used_seeds): break
                candidate_seed += 1
            else: raise ValueError("Could not freeze fully independent extension seeds")
            validate(plan, samples_by_id)
            for key, value in old.items():
                if key not in {"seed", "trials", "fingerprint"} and plan.get(key) != value:
                    raise ValueError("Extension changed original source/baseline contract: " + key)
            new_trials.update(trial_ids); used_seeds.update(policy_seeds)
            index = 299 + len(entries)
            entry = dict(index=index, original_group_index=original["index"], wave=wave, state="complete",
                output_root=str(Path(output_root) / "groups" / f"{index:03d}"), original_output_root=original["output_root"],
                code_root=original["code_root"], source_manifest_sha256=source_identity, raw_data_root=original["raw_data_root"],
                plan=plan, group=copy.deepcopy(group), parent_plan_fingerprint=old["fingerprint"],
                planned_cumulative_trials=30 + waves * 30)
            if "source_audit_parent_source_manifest_sha256" in original:
                entry["source_audit_parent_source_manifest_sha256"] = original["source_audit_parent_source_manifest_sha256"]
            entries.append(entry)
    payload = dict(schema_version=1, kind=KIND, state="complete", requested_group_count=32, shard_count=36,
        expected_trials=1080, source_manifest_sha256=source_identity, source_evidence_sha256=source_evidence_sha256,
        parent_campaign_sha256=parent_campaign_sha256, code_root=campaign["code_root"], output_root=str(output_root),
        deadline_epoch=deadline_epoch, groups=entries,
        selection=dict(moderate_rate_interval=[.2, .5], moderate_upper_exclusive=True, moderate_groups=28,
                       moderate_total_trials=60, positive_below_point_two_groups=4, hard_total_trials=90,
                       baseline_rates_unchanged=True, starts_unchanged=True, maximum_trials_per_shard=30),
        deferred_zero_groups=[dict(original_group_index=row["index"], group=copy.deepcopy(row["group"]),
            parent_plan_fingerprint=row["plan"]["fingerprint"], reason="await_first_round_and_restoration_evidence") for row in zeros],
        original_expected_trials=8700, cumulative_planned_trials=9780, trial_ids_disjoint_from_original=True,
        policy_seeds_disjoint_from_original=True)
    if len(entries) != 36 or len(new_trials) != 1080: raise ValueError("Extension budget differs from authorization")
    return {**payload, "fingerprint": digest(payload)}


def prepare(frozen_root, output, *, output_root, deadline_epoch, expected_fingerprint=None, seed_base=20261011):
    source, output = Path(frozen_root).resolve(), Path(output)
    if output.exists() or output.is_symlink(): raise ValueError("Extension staging output already exists")
    if source == output.resolve() or source in output.resolve().parents or output.resolve() in source.parents:
        raise ValueError("Extension output must be separate from original frozen inputs")
    campaign_path, evidence_path = source / "campaign.json", source / "source_evidence.json"
    campaign, evidence = json.loads(campaign_path.read_text()), json.loads(evidence_path.read_text())
    panel_paths = [p for p in (source / "parent_panel.json", source / "additional_panel.json") if p.is_file()]
    panel_map, files = {}, {"campaign.json": sha(campaign_path), "source_evidence.json": sha(evidence_path)}
    for path in panel_paths:
        panel = load_panel(path)
        if panel["fingerprint"] in panel_map: raise ValueError("Duplicate frozen panel identity")
        panel_map[panel["fingerprint"]] = (panel, path)
        files[path.name] = sha(path)
    extension = build_extension(campaign, {key: value[0] for key, value in panel_map.items()}, evidence,
        parent_campaign_sha256=files["campaign.json"], source_evidence_sha256=files["source_evidence.json"],
        output_root=output_root, deadline_epoch=deadline_epoch, seed_base=seed_base)
    extension["parent_input_files_sha256"] = files
    extension["fingerprint"] = digest({k: v for k, v in extension.items() if k != "fingerprint"})
    if expected_fingerprint is not None and extension["fingerprint"] != expected_fingerprint:
        raise ValueError("Remote extension reconstruction differs from the locally frozen manifest")
    output.parent.mkdir(parents=True, exist_ok=True); output.mkdir()
    try:
        for entry in extension["groups"]:
            directory = output / "groups" / f"{entry['index']:03d}"; directory.mkdir(parents=True)
            (directory / "parent_panel.json").symlink_to(panel_map[entry["plan"]["panel_fingerprint"]][1])
            (directory / "source_evidence.json").symlink_to(evidence_path)
            for name, value in (("collection_plan.json", entry["plan"]), ("prepared.json", entry)):
                with (directory / name).open("x") as stream: stream.write(json.dumps(value, indent=2) + "\n")
        if any(sha(source / name) != value for name, value in files.items()):
            raise ValueError("Original frozen inputs changed during extension preparation")
        with (output / "campaign.json").open("x") as stream:
            stream.write(json.dumps(extension, indent=2) + "\n")
        return extension
    except BaseException:
        shutil.rmtree(output); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", required=True, help="Absolute final AFS root written into worker entries")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--expected-fingerprint", help="Required when independently reconstructing remote staged inputs")
    parser.add_argument("--seed-base", type=int, default=20261011)
    args = parser.parse_args()
    result = prepare(args.frozen_root, args.output, output_root=args.output_root, deadline_epoch=args.deadline_epoch,
                     expected_fingerprint=args.expected_fingerprint, seed_base=args.seed_base)
    print(json.dumps({key: result[key] for key in ("fingerprint", "requested_group_count", "shard_count", "expected_trials", "deadline_epoch")}))


if __name__ == "__main__": main()

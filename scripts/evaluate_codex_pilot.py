#!/usr/bin/env python3
"""Verify the fixed 20-instance no-Agent panel and report its paired outcomes.

This is an offline artifact verifier: it imports no teacher, policy, simulator,
cloud SDK, or model. It neither launches evaluations nor selects favorable runs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

TASKS = ("turning_on_radio", "putting_shoes_on_rack")
BUDGETS = {"turning_on_radio": 3225, "putting_shoes_on_rack": 11590}
PANEL_KEYS = {(task, instance, 0) for task in TASKS for instance in range(301, 311)}
HEX = re.compile(r"^[a-f0-9]{64}$")
SHARDS = [
    {"id": "radio-pilot", "task": TASKS[0], "instance_ids": [301], "write_video": True},
    {"id": "a000", "task": TASKS[0], "instance_ids": list(range(302, 306)), "write_video": False},
    {"id": "a001", "task": TASKS[0], "instance_ids": list(range(306, 311)), "write_video": False},
    {"id": "a088", "task": TASKS[1], "instance_ids": list(range(301, 306)), "write_video": False},
    {"id": "a089", "task": TASKS[1], "instance_ids": list(range(306, 311)), "write_video": False},
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "size": path.stat().st_size}


def verify_file(ref):
    if (not isinstance(ref, dict) or not {"path", "sha256", "size"} <= ref.keys()
            or not HEX.fullmatch(str(ref["sha256"])) or type(ref["size"]) is not int or ref["size"] < 1):
        raise ValueError("Missing bounded artifact hash/size receipt")
    path = Path(ref["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size != ref["size"] or digest(path) != ref["sha256"]:
        raise ValueError("Artifact bytes differ from their immutable receipt")
    return path


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_contract(source_manifest_sha256):
    if not HEX.fullmatch(source_manifest_sha256):
        raise ValueError("Source snapshot requires an immutable manifest hash")
    return {"source_manifest_sha256": source_manifest_sha256,
            "mode": "public_test", "prompt_mode": "whole_task", "oracle": False,
            "privileged": False, "external_agent": False, "external_subtask_selector": None,
            "subtask_prediction": "model_native", "action_conditioning": "task_and_subtask",
            "evaluation_seed": 0, "seed_scope": "once_per_ordered_shard_process",
            "fresh_policy_server_per_shard": True, "num_rollouts": 1,
            "max_steps_argument": None, "step_budget_rule": "int(1.5 * mean_human_demo_length)",
            "resolved_step_budgets": BUDGETS, "execute_horizon": 16,
            "max_token_len": 512, "max_subtask_len": 64,
            "server_gpu": 0, "eval_gpu": 0, "gpu_count": 1, "pi05_mem_fraction": 0.35,
            "ordered_shards": SHARDS}


def validate_result(value, key):
    if tuple(value.get(name) for name in ("task", "instance_id", "rollout_id")) != key or key not in PANEL_KEYS:
        raise ValueError("Result identity is outside the fixed paired panel")
    if type(value.get("steps")) is not int or not 1 <= value["steps"] <= BUDGETS[key[0]]:
        raise ValueError("Result has no real steps or exceeds the official task budget")
    if any(type(value.get(k)) is not bool for k in ("success", "terminated", "truncated")):
        raise ValueError("Result terminal fields must be booleans")
    if not (value["terminated"] or value["truncated"]):
        raise ValueError("Incomplete rollouts cannot enter the paired report")
    if value["success"] and not value["terminated"]:
        raise ValueError("Successful rollouts require environment termination")
    if not value["success"] and value["truncated"] and value["steps"] != BUDGETS[key[0]]:
        raise ValueError("Unsuccessful truncated result did not reach the official step budget")
    conditioning = value.get("conditioning", {})
    expected = {"mode": "whole_task", "oracle": False, "privileged": False,
                "source": "policy_server_whole_task_prompt", "comparable_to_whole_task": True}
    if any(conditioning.get(k) != v for k, v in expected.items()):
        raise ValueError("Evaluation must use unprivileged model-native whole-task conditioning")
    if any(value.get(k) not in (None, [], {}) for k in ("prompt_events", "selector_events", "selector_summary",
                                                       "subtask_vote_events", "subtask_vote_summary")):
        raise ValueError("Oracle, selector, or voting overlays are outside this evaluation contract")
    events = value.get("subtask_events")
    if not isinstance(events, list) or not events or not any(e.get("source") == "generated" for e in events):
        raise ValueError("Actual native-generated subtask evidence is required")
    previous = -1
    for event in events:
        if (event.get("source") not in ("generated", "cached") or type(event.get("step")) is not int
                or not previous <= event["step"] < value["steps"]
                or not isinstance(event.get("subtask"), str) or not event["subtask"].strip()):
            raise ValueError("Teacher/provided/oracle or malformed subtask events are forbidden")
        previous = event["step"]
    if events[0]["step"] != 0:
        raise ValueError("Native subtask evidence must begin before the first action")
    final = value.get("final_goal_state", {})
    if type(final.get("success")) is not bool or final["success"] != value["success"]:
        raise ValueError("Task success and final goal state disagree")
    q = value.get("q_score", {}).get("final")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Final Q must be a finite goal-completion score in [0,1]")
    return {"success": value["success"], "q": float(q), "steps": value["steps"]}


def _checked_result(path, descriptor, key, shard_id, receipts):
    ref = file_ref(path)
    if ref["sha256"] != descriptor["sha256"] or ref["size"] != descriptor.get("size", descriptor.get("bytes")):
        raise ValueError("Baseline original JSON does not match its historical hash receipt")
    validate_result(load(path), key)
    return {"task": key[0], "instance_id": key[1], "rollout_id": key[2],
            "shard_id": shard_id, "json": ref, "provenance_receipts": receipts, "video": None}


def prepare_baseline(root):
    """Resolve the predeclared canonical sources; never choose by success/Q."""
    root = Path(root)
    ready = load(root / "preparation_ready.json")
    plan = load(root / "production_plan.json")
    if ready["checkpoint_step"] != 1688977 or plan["reuse_pilot_regardless_of_model_score"] is not True:
        raise ValueError("Baseline checkpoint or canonical pilot reuse rule changed")
    scope = ready["scope"]
    if (scope["prompt_mode"] != "whole_task" or scope["oracle_prompts"] is not False
            or scope["external_subtask_selector"] is not None
            or scope["max_steps"] != "official default 1.5x mean human demo length"):
        raise ValueError("Historical baseline is not the requested official native protocol")
    source = ready["source_manifest_sha256"]
    hashes = load(root / "source_overlay_sha256.json")
    # Verify archived source bytes, not the evolving working tree.
    with tarfile.open(root / "source_overlay.tar.gz") as archive:
        for name, expected in hashes.items():
            raw = archive.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Archived evaluation source hash changed")
        raw = archive.extractfile("BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/eval_utils.py").read()
        assignments = {node.targets[0].id: ast.literal_eval(node.value)
                       for node in ast.parse(raw).body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "DEFAULT_EVAL_SEED"}
        if assignments.get("DEFAULT_EVAL_SEED") != 0:
            raise ValueError("Archived seed differs from paired protocol")
    driver_hash = digest(root / "pilot_driver_source.py")
    rows = []
    accepted = load(root / "pilot_accepted.json")
    if (accepted["state"] != "accepted" or accepted["checkpoint_step"] != 1688977
            or accepted["source_manifest_sha256"] != source or accepted["driver_sha256"] != driver_hash
            or not accepted["full_artifact_sha256_verified"]
            or not accepted["canonical_result_kept_regardless_of_score"]):
        raise ValueError("Original pilot provenance is incomplete")
    row = _checked_result(root / "pilot_artifacts/turning_on_radio_301_0.json", accepted["checked_artifacts"]["json"],
                          (TASKS[0], 301, 0), "radio-pilot", [file_ref(root / "pilot_accepted.json")])
    # The original video was decoded on AFS and its audit receipt is retained;
    # local video bytes need not be downloaded again for the paired score table.
    row["video"] = {"historical_verified": True, **accepted["checked_artifacts"]["video"],
                    "receipt": file_ref(root / "pilot_accepted.json")}
    rows.append(row)
    levels = {"radio-pilot": "accepted_full_artifact_audit"}
    for shard in SHARDS[1:]:
        ident = shard["id"]
        if ident in ("a000", "a088"):
            accepted_path = root / f"production_audits/{ident}_accepted.json"
            report_path = root / f"production_audits/{ident}_report.json"
            accepted, report = load(accepted_path), load(report_path)
            if (accepted["report_sha256"] != digest(report_path) or accepted["state"] != "accepted"
                    or report["checkpoint_step"] != 1688977 or report["source_manifest_sha256"] != source
                    or report["driver_sha256"] != driver_hash or not report["full_artifact_sha256_verified"]
                    or not report["canonical_results_kept_regardless_of_score"]):
                raise ValueError("Full shard receipt changed or lacks canonical provenance")
            descriptors = {r["instance_id"]: r["json"] for r in report["results"]}
            refs = [file_ref(accepted_path), file_ref(report_path)]
            levels[ident] = "accepted_full_artifact_audit"
        else:
            receipt_path = root / f"production_partial_results_{ident}_current.json"
            original_path = root / f"production_{ident}_afs_originals_current.json"
            receipt, originals = load(receipt_path), load(original_path)
            manifest = originals["RESULT_MANIFEST.json"]
            status = originals["_ASSIGNED_SUCCESS"]
            if (receipt["source_manifest_sha256"] != source or status["state"] != "complete"
                    or status["checkpoint_step"] != 1688977
                    or manifest["metadata"]["source_manifest_sha256"] != source
                    or manifest["metadata"]["checkpoint_step"] != 1688977
                    or manifest["metadata"]["require_videos"] is not False):
                raise ValueError("Original-byte receipt lacks completed native checkpoint provenance")
            descriptors = {r["instance_id"]: r["json"] for r in manifest["rollouts"]}
            for result in receipt["results"]:
                desc = descriptors[result["instance_id"]]
                if desc["sha256"] != result["sha256"] or desc["size"] != result["bytes"]:
                    raise ValueError("Partial readback and completed result manifest disagree")
            refs = [file_ref(receipt_path), file_ref(original_path)]
            levels[ident] = "original_bytes_and_completed_manifest_verified_full_shard_audit_pending"
        if set(descriptors) != set(shard["instance_ids"]):
            raise ValueError("Historical ordered shard has missing or extra instances")
        for instance in shard["instance_ids"]:
            path = root / f"production_artifacts/{ident}/{shard['task']}_{instance}_0.json"
            rows.append(_checked_result(path, descriptors[instance], (shard["task"], instance, 0), ident, refs))
    return {"kind": "codex_pilot_native_eval_panel_v1", "role": "baseline",
            "checkpoint": {"id": "1688977", "kind": "baseline", "path": ready["checkpoint"]},
            "runtime_contract": runtime_contract(source), "results": rows,
            "source_receipts": [file_ref(root/name) for name in ("preparation_ready.json", "production_plan.json",
                                "source_overlay_sha256.json", "source_overlay.tar.gz", "pilot_driver_source.py")],
            "verification_levels": levels, "selection_rule": "fixed tasks, all IDs301–310, canonical rollout0 regardless of outcome",
            "comparison_complete": False, "baseline_reuse": "only if candidate matches the entire runtime contract"}


def _panel_rows(manifest):
    if manifest.get("kind") != "codex_pilot_native_eval_panel_v1":
        raise ValueError("Unknown native evaluation panel format")
    rows = {}
    for row in manifest.get("results", []):
        key = tuple(row.get(name) for name in ("task", "instance_id", "rollout_id"))
        if key in rows or key not in PANEL_KEYS:
            raise ValueError("Duplicate or out-of-scope evaluation result")
        expected_shard = next(s for s in SHARDS if s["task"] == key[0] and key[1] in s["instance_ids"])
        if row.get("shard_id") != expected_shard["id"]:
            raise ValueError("Result is assigned to a different process-seed shard")
        path = verify_file(row["json"])
        rows[key] = validate_result(load(path), key)
        for receipt in row.get("provenance_receipts", []):
            verify_file(receipt)
        video = row.get("video")
        if expected_shard["write_video"] and not video:
            raise ValueError("Required pilot video evidence is missing")
        if video:
            if video.get("historical_verified") is True:
                if manifest["role"] != "baseline":
                    raise ValueError("New candidate videos require actual artifact bytes")
                verify_file(video["receipt"])
            else:
                verify_file(video)
    if set(rows) != PANEL_KEYS:
        raise ValueError("Exactly twenty original result JSONs are required per checkpoint")
    return rows


def _artifact_identity(ref):
    return None if ref is None else {"sha256": ref["sha256"], "size": ref["size"]}


def _result_identity(row):
    return {**{key: row[key] for key in ("task", "instance_id", "rollout_id", "shard_id")},
            "json": _artifact_identity(row["json"]), "video": _artifact_identity(row.get("video"))}


def _validate_candidate(manifest, contract):
    checkpoint = manifest["checkpoint"]
    if manifest.get("role") != "finetuned" or checkpoint.get("kind") != "trained" or checkpoint.get("id") in (None, "", "1688977"):
        raise ValueError("Candidate must identify a distinct actually trained checkpoint")
    train_path = verify_file(checkpoint["training_receipt"])
    train = load(train_path)
    if (train.get("kind") != "codex_pilot_trained_checkpoint_v1" or train.get("checkpoint_id") != checkpoint["id"]
            or train.get("parent_checkpoint_step") != 1688977
            or type(train.get("optimizer_steps")) is not int or train["optimizer_steps"] < 1
            or train.get("ready_for_evaluation") is not True or train.get("real_lerobot_readback_verified") is not True
            or train.get("base_normalization_preserved") is not True):
        raise ValueError("Training receipt does not prove a ready trained checkpoint")
    # The immutable cloud receipt can retain its AFS path. A downloaded copy
    # supplied by the panel must match its original byte identity exactly.
    checkpoint_manifest = checkpoint.get("checkpoint_manifest", train["checkpoint_manifest"])
    if _artifact_identity(checkpoint_manifest) != _artifact_identity(train["checkpoint_manifest"]):
        raise ValueError("Downloaded checkpoint manifest differs from the training receipt")
    verify_file(checkpoint_manifest)
    received = {}
    for ref in manifest.get("run_receipts", []):
        receipt = load(verify_file(ref))
        ident = receipt.get("shard_id")
        if ident in received or ident not in {s["id"] for s in SHARDS}:
            raise ValueError("Missing/duplicate/unexpected candidate process receipt")
        if (receipt.get("kind") != "codex_pilot_native_eval_run_v1" or receipt.get("state") != "complete"
                or receipt.get("checkpoint_id") != checkpoint["id"]
                or receipt.get("checkpoint_manifest_sha256") != train["checkpoint_manifest"]["sha256"]
                or receipt.get("runtime_contract") != contract
                or receipt.get("external_agent") is not False or receipt.get("teacher_imported") is not False):
            raise ValueError("Candidate runtime/checkpoint/Agent provenance does not match")
        metadata = receipt.get("policy_metadata", {})
        if metadata.get("subtask_generation") is not True or metadata.get("subtask_action_conditioning") != "task_and_subtask":
            raise ValueError("Candidate server did not verify its joint native policy contract")
        keys = receipt.get("environment_keys")
        if not isinstance(keys, list) or not keys:
            raise ValueError("Evaluation child environment inventory is required")
        for key in keys:
            if (not isinstance(key, str) or key.startswith(("CODEX_", "TEACHER_", "EXTERNAL_SUBTASK", "PROVIDED_SUBTASK"))
                    or key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PI05_SUBTASK_COLLECTION_PLAN", "PI05_SUBTASK_PROBE_PLAN",
                               "PI05_PAIRED_RETEST_SEED", "PI05_SUBTASK_SAMPLE_PANEL")):
                raise ValueError("Teacher/collection configuration leaked into evaluation")
        expected = [row for row in manifest["results"] if row["shard_id"] == ident]
        actual = receipt.get("results")
        if not isinstance(actual, list) or [_result_identity(r) for r in actual] != [_result_identity(r) for r in expected]:
            raise ValueError("Run receipt does not bind the exact ordered result artifacts")
        shard = next(s for s in SHARDS if s["id"] == ident)
        if [row["instance_id"] for row in expected] != shard["instance_ids"]:
            raise ValueError("Candidate instance execution order differs from the baseline seed protocol")
        if shard["write_video"]:
            video = expected[0]["video"]
            checked = receipt.get("video_validation", {})
            count, fps = checked.get("frames"), checked.get("fps")
            if (checked.get("video_sha256") != video["sha256"] or type(count) is not int or count < 2
                    or isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
                    or checked.get("decoded_frames") != [0, count//2, count-1]):
                raise ValueError("Candidate pilot video must have first/middle/last decoded-frame evidence")
        received[ident] = receipt
    if set(received) != {s["id"] for s in SHARDS}:
        raise ValueError("All five independently seeded process receipts are required")


def compare_panels(baseline_path, candidate_path):
    baseline, candidate = load(baseline_path), load(candidate_path)
    if baseline.get("role") != "baseline" or baseline.get("checkpoint", {}).get("id") != "1688977":
        raise ValueError("Baseline must be the predeclared 1688977 checkpoint")
    contract = baseline["runtime_contract"]
    if contract != runtime_contract(contract["source_manifest_sha256"]) or candidate.get("runtime_contract") != contract:
        raise ValueError("Paired evaluation config, source, seed schedule or budget differs")
    for ref in baseline.get("source_receipts", []):
        verify_file(ref)
    if not baseline.get("source_receipts"):
        raise ValueError("Baseline reuse needs original source/hash receipts")
    # Rebuild from the historical chain. Merely editing a panel's scores or
    # swapping an arbitrary file_ref must not turn it into a canonical baseline.
    source_root = Path(baseline["source_receipts"][0]["path"]).parent
    if baseline != prepare_baseline(source_root):
        raise ValueError("Baseline panel no longer matches its original canonical receipt chain")
    before, after = _panel_rows(baseline), _panel_rows(candidate)
    _validate_candidate(candidate, contract)
    rows = [{"task": key[0], "instance_id": key[1], "rollout_id": key[2],
             "baseline": before[key], "finetuned": after[key],
             "q_delta": after[key]["q"]-before[key]["q"]} for key in sorted(PANEL_KEYS)]
    def aggregate(selected):
        count = len(selected)
        first = sum(row["baseline"]["success"] for row in selected)
        second = sum(row["finetuned"]["success"] for row in selected)
        return {"instances": count, "baseline_successes": first, "finetuned_successes": second,
                "baseline_success_rate": first/count, "finetuned_success_rate": second/count,
                "paired_wins": sum(not row["baseline"]["success"] and row["finetuned"]["success"] for row in selected),
                "paired_losses": sum(row["baseline"]["success"] and not row["finetuned"]["success"] for row in selected),
                "both_success": sum(row["baseline"]["success"] and row["finetuned"]["success"] for row in selected),
                "both_failure": sum(not row["baseline"]["success"] and not row["finetuned"]["success"] for row in selected),
                "baseline_mean_q": sum(row["baseline"]["q"] for row in selected)/count,
                "finetuned_mean_q": sum(row["finetuned"]["q"] for row in selected)/count,
                "mean_q_delta": sum(row["q_delta"] for row in selected)/count}
    return {"kind": "codex_pilot_paired_report_v1", "comparison_complete": True, "external_agent": False,
            "baseline_panel": file_ref(baseline_path), "candidate_panel": file_ref(candidate_path),
            "runtime_contract": contract, "baseline_verification_levels": baseline.get("verification_levels"),
            "overall": aggregate(rows), "by_task": {task: aggregate([r for r in rows if r["task"] == task]) for task in TASKS},
            "pairs": rows, "interpretation": "Q is final goal completion in [0,1]; this fixed 20-instance pilot is not a full benchmark."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("prepare-baseline")
    baseline.add_argument("--baseline-root", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    for command in (baseline, comparison):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_baseline(args.baseline_root) if args.command == "prepare-baseline" else compare_panels(args.baseline, args.candidate)
    write_new(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "comparison_complete": result["comparison_complete"]}))


if __name__ == "__main__":
    main()

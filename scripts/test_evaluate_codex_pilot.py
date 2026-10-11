"""Offline paired-report tests. Generated artifacts are fixtures, never real evaluations."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from evaluate_codex_pilot import BUDGETS, SHARDS, TASKS, compare_panels, digest, file_ref, load, prepare_baseline, validate_result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def result(task, instance, success=False):
    return {"task": task, "instance_id": instance, "rollout_id": 0,
        "steps": 100 if success else BUDGETS[task], "success": success,
        "terminated": success, "truncated": not success,
        "conditioning": {"mode": "whole_task", "oracle": False, "privileged": False,
            "source": "policy_server_whole_task_prompt", "comparable_to_whole_task": True},
        "prompt_events": [], "selector_events": [], "selector_summary": None,
        "subtask_vote_events": [], "subtask_vote_summary": None,
        "subtask_events": [{"step": 0, "subtask": "fixture subtask", "source": "generated"},
                           {"step": 16, "subtask": "fixture next subtask", "source": "cached"}],
        "final_goal_state": {"success": success}, "q_score": {"final": 1. if success else .2}}


def historical_baseline(root):
    source = "a"*64
    save(root / "preparation_ready.json", {"checkpoint_step": 1688977, "checkpoint": "/fixture/1688977",
        "source_manifest_sha256": source, "scope": {"prompt_mode": "whole_task", "oracle_prompts": False,
        "external_subtask_selector": None, "max_steps": "official default 1.5x mean human demo length"}})
    save(root / "production_plan.json", {"reuse_pilot_regardless_of_model_score": True})
    driver = root / "pilot_driver_source.py"
    driver.write_text("# fixture-only immutable driver\n")
    name = "BEHAVIOR-1K/OmniGibson/omnigibson/eval/utils/eval_utils.py"
    raw = b"DEFAULT_EVAL_SEED = 0\n"
    with tarfile.open(root / "source_overlay.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    save(root / "source_overlay_sha256.json", {name: hashlib.sha256(raw).hexdigest()})
    for shard in SHARDS:
        reports = []
        for instance in shard["instance_ids"]:
            success = (shard["task"], instance) in ((TASKS[0], 301), (TASKS[1], 302))
            relative = "pilot_artifacts" if shard["id"] == "radio-pilot" else f"production_artifacts/{shard['id']}"
            path = save(root / relative / f"{shard['task']}_{instance}_0.json", result(shard["task"], instance, success))
            reports.append({"instance_id": instance, "json": file_ref(path)})
        common = {"checkpoint_step": 1688977, "source_manifest_sha256": source, "driver_sha256": digest(driver),
            "full_artifact_sha256_verified": True, "canonical_results_kept_regardless_of_score": True}
        if shard["id"] == "radio-pilot":
            save(root / "pilot_accepted.json", {**common, "state": "accepted", "canonical_result_kept_regardless_of_score": True,
                "checked_artifacts": {"json": reports[0]["json"], "video": {"path": "fixture.mp4", "size": 123, "sha256": "b"*64}}})
        elif shard["id"] in ("a000", "a088"):
            report = save(root / f"production_audits/{shard['id']}_report.json", {**common, "results": reports})
            save(root / f"production_audits/{shard['id']}_accepted.json", {"state": "accepted", "report_sha256": digest(report)})
        else:
            save(root / f"production_partial_results_{shard['id']}_current.json", {
                "source_manifest_sha256": source, "results": [{"instance_id": r["instance_id"],
                    "sha256": r["json"]["sha256"], "bytes": r["json"]["size"]} for r in reports]})
            save(root / f"production_{shard['id']}_afs_originals_current.json", {
                "_ASSIGNED_SUCCESS": {"state": "complete", "checkpoint_step": 1688977},
                "RESULT_MANIFEST.json": {"metadata": {"source_manifest_sha256": source, "checkpoint_step": 1688977,
                    "require_videos": False}, "rollouts": reports}})
    return prepare_baseline(root)


def refresh_runs(candidate, output):
    refs = []
    training = load(candidate["checkpoint"]["training_receipt"]["path"])
    for shard in SHARDS:
        rows = [r for r in candidate["results"] if r["shard_id"] == shard["id"]]
        receipt = {"kind": "codex_pilot_native_eval_run_v1", "state": "complete", "shard_id": shard["id"],
            "checkpoint_id": candidate["checkpoint"]["id"], "checkpoint_manifest_sha256": training["checkpoint_manifest"]["sha256"],
            "runtime_contract": candidate["runtime_contract"], "external_agent": False, "teacher_imported": False,
            "policy_metadata": {"subtask_generation": True, "subtask_action_conditioning": "task_and_subtask"},
            "environment_keys": ["PATH", "PI05_CKPT", "SUBTASK_PREDICTION", "OMNIGIBSON_HEADLESS"], "results": rows}
        if shard["write_video"] and rows and rows[0]["video"]:
            receipt["video_validation"] = {"video_sha256": rows[0]["video"]["sha256"], "frames": 100,
                "fps": 30., "decoded_frames": [0, 50, 99]}
        refs.append(file_ref(save(output / f"run-{shard['id']}.json", receipt)))
    candidate["run_receipts"] = refs


@pytest.fixture
def panels(tmp_path):
    baseline = historical_baseline(tmp_path / "original")
    before = save(tmp_path / "baseline.json", baseline)
    trained = file_ref(save(tmp_path / "checkpoint_manifest.json", {"fixture_only": True, "weights": "fake-test-weights"}))
    training = file_ref(save(tmp_path / "training.json", {"kind": "codex_pilot_trained_checkpoint_v1",
        "checkpoint_id": "fixture-trained-256", "parent_checkpoint_step": 1688977, "optimizer_steps": 256,
        "ready_for_evaluation": True, "real_lerobot_readback_verified": True,
        "base_normalization_preserved": True, "checkpoint_manifest": trained}))
    candidate = {"kind": baseline["kind"], "role": "finetuned",
        "checkpoint": {"kind": "trained", "id": "fixture-trained-256", "training_receipt": training},
        "runtime_contract": copy.deepcopy(baseline["runtime_contract"]), "results": []}
    for row in baseline["results"]:
        key = row["task"], row["instance_id"]
        success = key in ((TASKS[0], 302), (TASKS[1], 302), (TASKS[1], 303))
        path = save(tmp_path / "new" / f"{key[0]}_{key[1]}_0.json", result(*key, success=success))
        video = None
        if key == (TASKS[0], 301):
            video_path = tmp_path / "new/fixture.mp4"
            video_path.write_bytes(b"fixture-only video; not a real rollout")
            video = file_ref(video_path)
        candidate["results"].append({"task": key[0], "instance_id": key[1], "rollout_id": 0,
            "shard_id": row["shard_id"], "json": file_ref(path), "video": video})
    refresh_runs(candidate, tmp_path)
    after = save(tmp_path / "candidate.json", candidate)
    return before, after, candidate, tmp_path


def test_full_paired_counts_include_all_outcomes_and_goal_completion(panels):
    before, after, _, _ = panels
    report = compare_panels(before, after)
    assert report["comparison_complete"] and report["external_agent"] is False
    assert len(report["pairs"]) == 20
    score = report["overall"]
    assert (score["baseline_successes"], score["finetuned_successes"], score["paired_wins"], score["paired_losses"]) == (2, 3, 2, 1)
    assert score["both_success"] == 1 and score["both_failure"] == 16
    assert score["mean_q_delta"] == pytest.approx(.04)
    assert report["by_task"][TASKS[0]]["instances"] == 10
    assert report["baseline_verification_levels"]["a001"].endswith("full_shard_audit_pending")


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_instance", "budget", "seed", "source", "order"])
def test_incomplete_or_unpaired_panels_rejected(panels, change):
    before, after, candidate, output = panels
    if change == "missing": candidate["results"].pop()
    elif change == "duplicate": candidate["results"][-1] = candidate["results"][0]
    elif change == "wrong_instance": candidate["results"][0]["instance_id"] = 311
    elif change == "order": candidate["results"][1:5] = list(reversed(candidate["results"][1:5]))
    else:
        key, value = {"budget": ("max_steps_argument", 4500), "seed": ("evaluation_seed", 1),
            "source": ("source_manifest_sha256", "d"*64)}[change]
        candidate["runtime_contract"][key] = value
    refresh_runs(candidate, output)
    save(after, candidate)
    with pytest.raises(ValueError): compare_panels(before, after)


@pytest.mark.parametrize("source", ["provided", "oracle", "teacher_cartesian_correction", "external_selector"])
def test_external_subtask_events_disallowed(source):
    value = result(TASKS[0], 301)
    value["subtask_events"][1]["source"] = source
    with pytest.raises(ValueError, match="forbidden"): validate_result(value, (TASKS[0], 301, 0))


@pytest.mark.parametrize("change", ["unfinished", "early_truncation", "success_mismatch", "nan_q", "no_generated", "privileged"])
def test_invalid_results_never_count_as_failure_or_success(change):
    value = result(TASKS[0], 301)
    if change == "unfinished": value["truncated"] = False
    elif change == "early_truncation": value["steps"] = 2048
    elif change == "success_mismatch": value["final_goal_state"]["success"] = True
    elif change == "nan_q": value["q_score"]["final"] = float("nan")
    elif change == "no_generated": value["subtask_events"][0]["source"] = "cached"
    else: value["conditioning"]["privileged"] = True
    with pytest.raises(ValueError): validate_result(value, (TASKS[0], 301, 0))


def test_modified_result_bytes_fail_hash_receipt(panels):
    before, after, candidate, _ = panels
    path = Path(candidate["results"][0]["json"]["path"])
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="immutable receipt"): compare_panels(before, after)


def test_baseline_cannot_swap_canonical_outcome(panels):
    before, after, candidate, _ = panels
    baseline = load(before)
    baseline["results"][1]["json"] = candidate["results"][1]["json"]
    save(before, baseline)
    with pytest.raises(ValueError, match="canonical receipt chain"): compare_panels(before, after)


@pytest.mark.parametrize("change", ["teacher_env", "teacher_import", "checkpoint_mismatch", "video_undecoded", "missing_run"])
def test_runtime_must_prove_no_agent_and_trained_checkpoint(panels, change):
    before, after, candidate, _ = panels
    if change == "missing_run": candidate["run_receipts"].pop()
    else:
        ref = candidate["run_receipts"][0]
        receipt = load(ref["path"])
        if change == "teacher_env": receipt["environment_keys"].append("CODEX_TEACHER_URL")
        elif change == "teacher_import": receipt["teacher_imported"] = True
        elif change == "checkpoint_mismatch": receipt["checkpoint_manifest_sha256"] = "0"*64
        else: receipt["video_validation"]["decoded_frames"] = [0]
        save(ref["path"], receipt)
        candidate["run_receipts"][0] = file_ref(ref["path"])
    save(after, candidate)
    with pytest.raises(ValueError): compare_panels(before, after)


def test_untrained_candidate_rejected(panels):
    before, after, candidate, _ = panels
    ref = candidate["checkpoint"]["training_receipt"]
    training = load(ref["path"])
    training["optimizer_steps"] = 0
    save(ref["path"], training)
    candidate["checkpoint"]["training_receipt"] = file_ref(ref["path"])
    save(after, candidate)
    with pytest.raises(ValueError, match="ready trained checkpoint"): compare_panels(before, after)


def test_downloaded_artifact_paths_may_differ_but_cloud_hashes_must_match(panels):
    before, after, candidate, _ = panels
    training_ref = candidate["checkpoint"]["training_receipt"]
    training = load(training_ref["path"])
    candidate["checkpoint"]["checkpoint_manifest"] = dict(training["checkpoint_manifest"])
    training["checkpoint_manifest"]["path"] = "/remote/afs/checkpoint/ready.json"
    save(training_ref["path"], training)
    candidate["checkpoint"]["training_receipt"] = file_ref(training_ref["path"])
    for index, ref in enumerate(candidate["run_receipts"]):
        receipt = load(ref["path"])
        for row in receipt["results"]:
            row["json"]["path"] = "/remote/afs/results/"+Path(row["json"]["path"]).name
            if row["video"]:
                row["video"]["path"] = "/remote/afs/results/fixture.mp4"
        save(ref["path"], receipt)
        candidate["run_receipts"][index] = file_ref(ref["path"])
    save(after, candidate)
    assert compare_panels(before, after)["overall"]["instances"] == 20

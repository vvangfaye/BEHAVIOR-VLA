"""Freeze the selected Huber trial only after the current evaluation is complete.

This prepares a new bundle; it never submits a cloud job or changes running work.
The original candidate remains immutable for comparison and audit.
"""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "expert_huber_candidate"
DEST = CANDIDATE / "selected_stage"
VARIANTS = {
    "expert_huber_candidate": ("b1k-expert-huber5-1688977-20261017", 5.0),
    "expert_huber50_candidate": ("b1k-expert-huber50-1688977-20261017", 50.0),
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def validate_evaluation():
    path = ROOT / "heldout20_baseline_vs_expert1k/build_report.py"
    spec = importlib.util.spec_from_file_location("completed_huber_reference", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    plan, pairs, models, proofs, missing = report.collect()
    if missing:
        raise ValueError(f"Current reference evaluation is incomplete: {len(missing)} episodes missing")
    native = report.verify_native_completion(plan)
    exclusion = report.verify_exclusion(pairs)
    return {"summaries": {role: report.summary(rows, pairs) for role, rows in models.items()},
            "paired": report.paired(models["baseline"], models["candidate"], pairs),
            "episode_proofs": proofs, "native_completion": native, "exclusion": exclusion}


def validate_ablation_evidence():
    """Use the completed original panel and training audit, not partial new outcomes."""
    report_path = ROOT / "final_comparison_proof.json"
    report = read(report_path)
    if not report.get("complete") or not report.get("protocol_and_seed_checks_passed"):
        raise ValueError("Original comparison is incomplete")
    for model in ("baseline1688977", "mixed1000", "mixed5000", "expert1000"):
        if report["summaries"][model]["episodes"] != 20:
            raise ValueError("Original comparison lacks a full model panel")
    for proof in report["input_proofs"]:
        if digest(ROOT / proof["file"]) != proof["sha256"]:
            raise ValueError("Original comparison input changed")
    accepted_path = ROOT / "replay_cpu_audit/accepted_result.json"
    result_path = ROOT / "replay_cpu_audit/result.json"
    accepted, result = read(accepted_path), read(result_path)
    if (not accepted.get("accepted") or not result.get("complete")
            or accepted["result_sha256"] != digest(result_path)
            or result["windows"] != 1829 or result.get("training") is not False
            or result.get("data_modified") is not False):
        raise ValueError("Training-data audit is incomplete or changed")
    sensitivity_path = ROOT / "expert_huber_candidate/target_threshold_sensitivity.json"
    sensitivity = read(sensitivity_path)
    if sensitivity["source_sha256"] != digest(result_path):
        raise ValueError("Threshold evidence source changed")
    counts = {row["threshold"]: row["windows_with_target_above_threshold"]
              for row in sensitivity["thresholds"]}
    if counts[5] != 389 or counts[50] != 1:
        raise ValueError("Threshold decision evidence differs")
    return {"basis": "completed_original_panel_and_training_data_ablation",
            "original_panel_summaries": report["summaries"],
            "proofs": [{"file": str(p.relative_to(ROOT)), "sha256": digest(p)}
                       for p in (report_path, accepted_path, result_path, sensitivity_path)],
            "original_input_proofs_checked": len(report["input_proofs"]),
            "ongoing_holdout_results_used_for_threshold_selection": False}


def validate_admission(policy):
    if policy == "complete_current_panel":
        return validate_evaluation()
    if policy == "preregistered_training_ablation":
        return validate_ablation_evidence()
    raise ValueError("Unknown admission policy")


def main(candidate_name="expert_huber_candidate", admission="complete_current_panel"):
    global CANDIDATE, DEST
    expected_run, expected_delta = VARIANTS[candidate_name]
    CANDIDATE = ROOT / candidate_name
    DEST = CANDIDATE / "selected_stage"
    if DEST.exists():
        raise FileExistsError("Selected stage already exists; do not overwrite or resubmit")
    if admission == "preregistered_training_ablation" and expected_delta != 50.0:
        raise ValueError("The independent training-data ablation selects only Huber 50")
    evaluation = validate_admission(admission)
    selection_path = CANDIDATE / "selection.json"
    selection = read(selection_path)
    if (selection.get("selected_run_id") != expected_run
            or not selection.get("rationale")
            or selection.get("admission_policy", "complete_current_panel") != admission
            or (admission == "complete_current_panel" and selection.get("current_evaluation_reviewed") is not True)):
        raise ValueError("A concrete selection and rationale are required")
    source = CANDIDATE / "bundle"
    preparation = read(CANDIDATE / "preparation.json")
    manifest_path = source / "bundle_manifest.json"
    if digest(manifest_path) != preparation["candidate_manifest_sha256"]:
        raise ValueError("Prepared candidate identity changed")
    manifest = read(manifest_path)
    for name, expected in manifest.items():
        if digest(source / name) != expected:
            raise ValueError("Candidate source changed: " + name)
    validation = read(CANDIDATE / "validation.json")
    loss_path = source / "openpi/src/openpi/shared/action_loss.py"
    if (not validation.get("passed") or validation.get("delta") != expected_delta
            or validation["loss_source_sha256"] != digest(loss_path)):
        raise ValueError("CPU loss validation is absent or stale")
    shutil.copytree(source, DEST / "bundle", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    bundle = DEST / "bundle"
    job_path = bundle / "nebula/worker_job.json"
    job = read(job_path)
    job["formal_training_authorized"] = True
    save(job_path, job)
    auth_path = bundle / "inputs/formal_authorization.json"
    auth = read(auth_path)
    auth.update(formal_training_allowed=True, kind="selected_huber_trial_under_active_improvement_goal",
                rationale=selection["rationale"])
    save(auth_path, auth)
    argv_path = bundle / "inputs/compiled/training_argv.json"
    argv = read(argv_path)
    argv["launch_authorized"] = True
    save(argv_path, argv)
    new_manifest = {name: digest(bundle / name) for name in manifest}
    changed = sorted(name for name in manifest if manifest[name] != new_manifest[name])
    if changed != sorted(["nebula/worker_job.json", "inputs/formal_authorization.json",
                          "inputs/compiled/training_argv.json"]):
        raise ValueError("Unexpected selected-bundle change")
    save(bundle / "bundle_manifest.json", new_manifest)
    save(DEST / "selection_proof.json", {
        "selection_sha256": digest(selection_path), "source_manifest_sha256": digest(manifest_path),
        "selected_manifest_sha256": digest(bundle / "bundle_manifest.json"),
        "evaluation": evaluation, "admission_policy": admission, "changed_files": changed,
        "updates": 1000, "gpus": 16, "maximum_runtime_seconds": 21600,
        "maximum_gpu_hours": 96, "automatic_extension": False,
        "cloud_jobs_created": 0, "submitted": False,
    })
    print(json.dumps({"prepared": True, "submitted": False, "changed_files": changed}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(VARIANTS), default="expert_huber_candidate")
    parser.add_argument("--admission", choices=["complete_current_panel", "preregistered_training_ablation"],
                        default="complete_current_panel")
    args = parser.parse_args()
    main(args.candidate, args.admission)

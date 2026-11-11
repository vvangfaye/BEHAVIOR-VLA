"""Prepare, but do not submit, a fresh expert-only learning-rate comparison."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "expert_control/formal_stage/bundle"
DEST = ROOT / "expert_lowerlr_candidate/bundle"
RUN = "b1k-expert-lr03-1688977-20261017"
FACTOR = 0.3


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("Inspect the existing candidate; never overwrite a prepared run")
    original = read(SOURCE / "bundle_manifest.json")
    for name, expected in original.items():
        if sha(SOURCE / name) != expected:
            raise ValueError("Source bundle changed: " + name)

    # This module uses only the standard library. Runtime Optax validation remains
    # mandatory in the unchanged trainer if this candidate is later selected.
    module_path = SOURCE / "openpi/src/openpi/shared/learning_rate_contract.py"
    spec = importlib.util.spec_from_file_location("lowerlr_contract", module_path)
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    old_lr = read(SOURCE / "inputs/compiled/learning_rate_contract.json")
    fields = dict(old_lr["schedule"])
    fields.update(peak_lr=3e-7, decay_lr=6e-8)
    schedule_type = dataclasses.make_dataclass("CosineDecaySchedule", [
        ("warmup_steps", int), ("peak_lr", float), ("decay_steps", int), ("decay_lr", float)
    ])
    new_lr = contracts.make_contract(schedule_type(**fields))
    if any(abs(contracts.expected_rate(new_lr, step) /
               contracts.expected_rate(old_lr, step) - FACTOR) > 1e-12
           for step in range(1001)):
        raise ValueError("The proposed schedule is not a uniform 0.3 scaling")

    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    save(DEST / "inputs/compiled/learning_rate_contract.json", new_lr)
    argv_file = DEST / "inputs/compiled/training_argv.json"
    argv_doc = read(argv_file)
    argv = argv_doc["argv"]
    for option, value in {"--exp-name": RUN, "--num-train-steps": "1000",
                          "--lr-schedule.peak-lr": "3e-07",
                          "--lr-schedule.decay-lr": "6e-08"}.items():
        if argv.count(option) != 1:
            raise ValueError("Expected one argument: " + option)
        argv[argv.index(option) + 1] = value
    argv_doc["launch_authorized"] = False
    save(argv_file, argv_doc)

    job_file = DEST / "nebula/worker_job.json"
    job = read(job_file)
    job.update(run_id=RUN, attempt_id="stage1000-g0", resume=False,
               experiment="expert_only_learning_rate_comparison",
               output="/data/oss_bucket_0/wangfaye/result/behavior-challenge/expert_lowerlr/" + RUN,
               formal_training_authorized=False)
    save(job_file, job)
    auth_file = DEST / "inputs/formal_authorization.json"
    auth = read(auth_file)
    auth.update(formal_training_allowed=False, peak_learning_rate=3e-7,
                kind="prepared_expert_lowerlr_candidate_pending_selection",
                rationale="Test retention versus adaptation by scaling only the LR trajectory to 0.3.")
    save(auth_file, auth)
    contract_file = DEST / "inputs/formal_run_contract.json"
    contract = read(contract_file)
    contract.update(run_id=RUN, learning_rate_contract_sha256=sha(
        DEST / "inputs/compiled/learning_rate_contract.json"))
    save(contract_file, contract)

    manifest = {name: sha(DEST / name) for name in original}
    changed = sorted(name for name in original if manifest[name] != original[name])
    expected_changes = sorted([
        "inputs/compiled/learning_rate_contract.json", "inputs/compiled/training_argv.json",
        "inputs/formal_authorization.json", "inputs/formal_run_contract.json", "nebula/worker_job.json"
    ])
    if changed != expected_changes:
        raise ValueError("Unexpected changes in the controlled candidate")
    save(DEST / "bundle_manifest.json", manifest)
    unchanged_contract = dict(contract)
    source_contract = read(SOURCE / "inputs/formal_run_contract.json")
    for name in ("run_id", "learning_rate_contract_sha256"):
        unchanged_contract.pop(name)
        source_contract.pop(name)
    if unchanged_contract != source_contract:
        raise ValueError("Non-LR training identity changed")
    save(DEST.parent / "preparation.json", {
        "run_id": RUN, "cloud_jobs_created": 0, "execution_enabled": False,
        "selection_pending": "Complete the current baseline/expert holdout comparison before selecting another trial.",
        "hypothesis": "Smaller updates may retain baseline skills while learning expert behaviors; no improvement claimed.",
        "factor": FACTOR, "updates": 1000, "fresh_from_checkpoint": 1688977,
        "resume": False, "same_data_and_sampling_schedule": True,
        "same_optimizer_batch_seed_normalization_and_source": True,
        "schedule_ratio_verified_updates": [0, 1000],
        "runtime_optax_check": "Deferred to unchanged trainer; not exercised locally.",
        "source_manifest_sha256": sha(SOURCE / "bundle_manifest.json"),
        "candidate_manifest_sha256": sha(DEST / "bundle_manifest.json"),
        "changed_files": changed,
        "evaluation": {
            "compare_to": "expert1000 and original1688977 on identical instances, seeds and budgets",
            "selection_panel": "Existing 20 fixed instances and 20 fine-tuning-held-out instances",
            "confirmation": "Once used to select a configuration, these panels are tuning evidence; independent confirmation is needed.",
            "missing_or_failed_episodes": "Never scored as zero or silently excluded"
        }
    })
    print(json.dumps({"prepared": True, "execution_enabled": False,
                      "changed_files": changed, "cloud_jobs_created": 0}))


if __name__ == "__main__":
    main()

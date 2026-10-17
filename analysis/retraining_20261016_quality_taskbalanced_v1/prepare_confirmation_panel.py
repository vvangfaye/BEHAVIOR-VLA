"""Freeze a separate, unsubmitted panel before choosing the next checkpoint."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SALT = "b1k-confirmation-v1-20261017"
OUTPUT = ROOT / "heldout20_confirmation_panel.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("Never silently replace a predeclared confirmation panel")
    source = ROOT / "heldout20_candidate_panel.json"
    declared = json.loads(source.read_text())
    tuning_pairs = set(map(tuple, declared["pairs"]))
    tuning_tasks = {task for task, _ in tuning_pairs}
    heldout, training = None, set()
    for proof in declared["source_proofs"]:
        path = ROOT / proof["path"]
        if digest(path) != proof["sha256"]:
            raise ValueError("Training index changed")
        index = json.loads(path.read_text())
        here = set(map(tuple, index["held_out_instances"]))
        heldout = here if heldout is None else heldout & here
        training.update((row["task_name"], row["task_instance_id"]) for row in index["episodes"])

    # Historical snapshots are read only to exclude already evaluated identities;
    # success, Q and other outcomes do not enter candidate ranking.
    evaluated, history_proofs = set(), []
    for path in sorted((ROOT / "evaluation_history").glob("*.json")):
        data = json.loads(path.read_text())
        records = data.get("records", [])
        if not records:
            continue
        evaluated.update((row["task"], row["instance_id"]) for row in records)
        history_proofs.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})

    by_task = {}
    for task, instance in sorted(heldout - training - evaluated - tuning_pairs):
        if task not in tuning_tasks:
            by_task.setdefault(task, instance)
    ranked = sorted(by_task.items(), key=lambda pair: hashlib.sha256(
        f"{SALT}|{pair[0]}|{pair[1]}".encode()).hexdigest())
    pairs = sorted(ranked[:20])
    if len(pairs) != 20 or len({task for task, _ in pairs}) != 20:
        raise ValueError("Insufficient independent task identities")
    if set(pairs) & (training | evaluated | tuning_pairs):
        raise ValueError("Confirmation panel overlap")
    value = {
        "kind": "predeclared_confirmation_panel_candidate",
        "execution_enabled": False, "cloud_jobs_created": 0,
        "pairs": pairs, "eligible_tasks": len(by_task),
        "selection_rule": "For each task outside the current tuning panel, take its smallest eligible heldout instance; rank tasks by SHA256(salt|task|instance), retain 20, then sort for display.",
        "salt": SALT,
        "selection_uses_evaluation_outcomes": False,
        "training_indexes_checked": len(declared["source_proofs"]),
        "indexed_training_pairs_checked": len(training),
        "declared_heldout_source": {"path": str(source.relative_to(ROOT)), "sha256": digest(source)},
        "historical_identity_sources": history_proofs,
        "same_task_overlap_with_current_tuning_panel": 0,
        "training_pair_overlap": 0, "checked_historical_evaluation_pair_overlap": 0,
        "baseline_pretraining_exclusion_proven": False,
        "runtime_scene_state_files_verified": False,
        "candidate_checkpoint_selected": False,
        "required_before_submission": [
            "Freeze the selected candidate checkpoint identity before obtaining confirmation outcomes.",
            "Recheck training index hashes and exclude these pairs from any new training data.",
            "Verify each scene state file and native task budget; freeze the actual protocol hash.",
            "Compare original and selected candidate with identical pairs, policy/environment seed 0 and native frame budgets."
        ],
        "limitations": [
            "The initial model's pretraining exposure is unknown.",
            "This panel changes task composition and must not be merged with or directly compared to the current panel's aggregate score.",
            "If its outcomes guide further selection, it becomes tuning evidence and needs a fresh confirmation panel."
        ]
    }
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"confirmation_pairs": len(pairs), "eligible_tasks": len(by_task),
                      "training_overlap": 0, "current_task_overlap": 0, "execution_enabled": False}))


if __name__ == "__main__":
    main()

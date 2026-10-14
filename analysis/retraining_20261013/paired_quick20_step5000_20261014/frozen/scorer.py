"""Paired heldout development decisions from already artifact-accepted receipts.

This scorer does not submit jobs or certify result files. The caller must first
validate native result/seed/checkpoint evidence against its frozen evaluation
plan. Missing or infrastructure-failed receipts cannot become training gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate(bundle, expected_pairs, policy_seed):
    if bundle.get("kind") != "artifact_accepted_finetune_development_results":
        raise ValueError("Artifact acceptance is required before scoring")
    for field in ("checkpoint_manifest_sha256", "protocol_sha256", "acceptance_manifest_sha256"):
        if not _sha(bundle.get(field)):
            raise ValueError("Missing immutable evaluation identity: " + field)
    if bundle.get("policy_seed") != policy_seed:
        raise ValueError("Evaluation policy seeds differ")
    expected = {tuple(x) for x in expected_pairs}
    if len(expected) != len(expected_pairs) or not expected:
        raise ValueError("Expected development pairs must be unique and nonempty")
    rows = {}
    for row in bundle.get("records", []):
        key = (row.get("task"), row.get("instance_id"))
        if key not in expected or key in rows:
            raise ValueError("Unexpected or duplicated development receipt")
        if row.get("artifact_accepted") is not True or row.get("policy_seed_acknowledged") is not True:
            raise ValueError("Unverified result/seed evidence cannot become a development score")
        if row.get("policy_seed") != policy_seed or type(row.get("steps")) is not int or row["steps"] <= 0:
            raise ValueError("Receipt lacks matching seed and actual simulator steps")
        if not _sha(row.get("result_sha256")):
            raise ValueError("Native result identity missing")
        success, q = row.get("model_success"), row.get("q_score")
        if type(success) is not bool or type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("Invalid actual task-success/Q metrics")
        if success != (q == 1):
            raise ValueError("Task success differs from final environment Q")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("Incomplete development results; do not score only the completed subset")
    return rows


def compare(baseline, candidate, expected_pairs, *, policy_seed=42, completed_updates, full_development=True):
    if type(completed_updates) is not int or not 0 <= completed_updates <= 20000:
        raise ValueError("Checkpoint progress exceeds the original finite training budget")
    if baseline.get("protocol_sha256") != candidate.get("protocol_sha256"):
        raise ValueError("Evaluation robot/wrapper/limits protocols differ")
    old = validate(baseline, expected_pairs, policy_seed)
    new = validate(candidate, expected_pairs, policy_seed)
    if baseline["checkpoint_manifest_sha256"] == candidate["checkpoint_manifest_sha256"]:
        raise ValueError("Baseline and fine-tuned candidate must have distinct weight identities")
    tasks = sorted({key[0] for key in old})
    per_task = []
    improved = regressed = 0
    for task in tasks:
        pairs = [key for key in old if key[0] == task]
        per_task.append(dict(task=task, instances=len(pairs),
            baseline_success=mean(old[key]["model_success"] for key in pairs),
            candidate_success=mean(new[key]["model_success"] for key in pairs),
            baseline_q=mean(old[key]["q_score"] for key in pairs),
            candidate_q=mean(new[key]["q_score"] for key in pairs)))
        improved += sum(not old[key]["model_success"] and new[key]["model_success"] for key in pairs)
        regressed += sum(old[key]["model_success"] and not new[key]["model_success"] for key in pairs)
    old_macro = mean(x["baseline_success"] for x in per_task)
    new_macro = mean(x["candidate_success"] for x in per_task)
    old_q = mean(x["q_score"] for x in old.values())
    new_q = mean(x["q_score"] for x in new.values())
    net_additional_failures = regressed - improved
    guard = net_additional_failures >= 4 and old_q - new_q >= .03 - 1e-12
    better = new_macro > old_macro or (math.isclose(new_macro, old_macro, abs_tol=1e-12) and new_q > old_q)
    # Quick20 is a technical check. It cannot grant continuation beyond5000.
    eligible = full_development and len(old) == 102 and policy_seed == 42 and completed_updates >= 5000
    decision = ("second_seed_replication_only" if full_development and len(old) == 102 and policy_seed == 43 else
                "technical_check_only" if not eligible else
                "budget_complete_select_best" if completed_updates == 20000 else
                "review_required_regression" if guard else
                "continue_within_original_budget" if better else "review_required_no_improvement")
    return dict(kind="paired_finetune_development_comparison", completed_updates=completed_updates,
        pairs=len(old), tasks=len(tasks), policy_seed=policy_seed,
        baseline_macro_task_success=old_macro, candidate_macro_task_success=new_macro,
        macro_task_success_change=new_macro-old_macro,
        baseline_mean_q=old_q, candidate_mean_q=new_q,
        improved_pairs=improved, regressed_pairs=regressed,
        net_additional_failed_pairs=net_additional_failures,
        regression_review_triggered=guard, candidate_better_by_selection_rule=better,
        decision=decision, may_continue_past5000=eligible and completed_updates < 20000 and better and not guard,
        max_original_authorized_updates=20000,
        artifact_validation_performed_by_this_scorer=False,
        evidence="single-seed development comparison, not proof of unseen generalization or causality",
        per_task=per_task)


def main():
    parser = argparse.ArgumentParser()
    for field in ("baseline", "candidate", "pairs", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--completed-updates", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--quick-subset", action="store_true")
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())
    if isinstance(pairs, dict):
        pairs = pairs["pairs"]
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), pairs,
        policy_seed=args.policy_seed, completed_updates=args.completed_updates, full_development=not args.quick_subset)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()

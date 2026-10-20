"""Compare predictions of identical demonstrated actions from two observations.

Uses only saved CPU inference outputs. This is not an intervention on the
environment and cannot identify a policy's physical feedback response.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "parent": ROOT / "cpu_raw_feasibility_20261020/baseline36_raw.json",
    "finetuned": ROOT / "calibration_cpu_followup_20261020/expert1000_36_raw.json",
}
GROUPS = {
    "base": list(range(3)), "torso": list(range(3, 7)),
    "left_arm": list(range(7, 14)), "left_gripper": [14],
    "right_arm": list(range(15, 22)), "right_gripper": [22],
}


def average(values):
    return sum(values) / len(values)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    norm_path = Path(data["finetuned"]["artifacts"]["norm_stats"]["path"])
    std = json.loads(norm_path.read_text())["norm_stats"]["actions"]["std"]
    for model in data.values():
        assert model["normalization"]["use_quantile_norm"] is False
        assert model["artifacts"]["norm_stats"]["sha256"] == digest(norm_path)
    by_model = {}
    for name, model in data.items():
        rows = defaultdict(dict)
        for sample in model["sample_keys"]:
            meta = sample["probe_metadata"]
            rows[meta["boundary_id"]][meta["bucket"]] = sample
        by_model[name] = rows
    assert set(by_model["parent"]) == set(by_model["finetuned"])
    per_boundary = []
    for boundary in sorted(by_model["parent"]):
        parent_rows, tuned_rows = by_model["parent"][boundary], by_model["finetuned"][boundary]
        for bucket in ["edge_H", "inside_K"]:
            a, b = parent_rows[bucket], tuned_rows[bucket]
            for key in ["episode_index", "frame_index", "subtask", "action_noise_sha256", "policy_input_sha256", "ground_truth_sha256"]:
                assert a[key] == b[key], (boundary, bucket, key)
        old, fresh = parent_rows["edge_H"], parent_rows["inside_K"]
        offset = fresh["frame_index"] - old["frame_index"]
        assert offset == 16 and old["subtask"] == fresh["subtask"]
        steps = min(old["valid_action_steps"]-offset, fresh["valid_action_steps"])
        assert steps == 16
        groups = {}
        for group, dims in GROUPS.items():
            stats = {}
            for name in FILES:
                old_sample = by_model[name][boundary]["edge_H"]["raw_actions"]
                fresh_sample = by_model[name][boundary]["inside_K"]["raw_actions"]
                errors_old, errors_fresh, revisions = [], [], []
                for t in range(steps):
                    assert old_sample["valid_mask"][offset+t] and fresh_sample["valid_mask"][t]
                    for d in dims:
                        gt_old = old_sample["ground_truth"][offset+t][d]
                        gt_fresh = fresh_sample["ground_truth"][t][d]
                        assert abs(gt_old-gt_fresh) < 1e-10
                        scale = std[d] + 1e-6
                        pred_old = old_sample["prediction"][offset+t][d]
                        pred_fresh = fresh_sample["prediction"][t][d]
                        errors_old.append(((pred_old-gt_old)/scale)**2)
                        errors_fresh.append(((pred_fresh-gt_fresh)/scale)**2)
                        revisions.append(((pred_old-pred_fresh)/scale)**2)
                stats[name] = {"old_observation_mse": average(errors_old),
                               "fresh_observation_mse": average(errors_fresh),
                               "prediction_revision_mse": average(revisions)}
            stats["finetune_delta"] = {
                key: stats["finetuned"][key]-stats["parent"][key]
                for key in stats["parent"]
            }
            groups[group] = stats
        per_boundary.append({"boundary_id": boundary, "task": old["task"],
                             "episode_index": old["episode_index"],
                             "old_frame": old["frame_index"], "fresh_frame": fresh["frame_index"],
                             "identical_target_steps": steps, "groups": groups})
    by_task = {}
    for task in sorted({r["task"] for r in per_boundary}):
        records = [r for r in per_boundary if r["task"] == task]
        by_task[task] = {
            group: {
                model: {key: average([r["groups"][group][model][key] for r in records])
                        for key in records[0]["groups"][group][model]}
                for model in ["parent", "finetuned", "finetune_delta"]
            } for group in GROUPS
        }
    blocks = [group for task in by_task.values() for group in task.values()]
    summary = {
        "task_group_blocks": len(blocks),
        "old_fit_improves_fresh_fit_worsens": sum(
            b["finetune_delta"]["old_observation_mse"] < 0
            and b["finetune_delta"]["fresh_observation_mse"] > 0 for b in blocks),
        "old_fit_improves": sum(b["finetune_delta"]["old_observation_mse"] < 0 for b in blocks),
        "fresh_fit_improves": sum(b["finetune_delta"]["fresh_observation_mse"] < 0 for b in blocks),
        "revision_mse_increases": sum(b["finetune_delta"]["prediction_revision_mse"] > 0 for b in blocks),
    }
    report = {
        "status": "complete", "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in FILES.items()},
        "normalization": {"path": str(norm_path), "sha256": digest(norm_path), "formula": "squared raw difference / (std + 1e-6)^2"},
        "protocol": {"same_absolute_target_actions": True, "target_match_tolerance": 1e-10,
                     "same_label": True, "observation_gap_frames": 16,
                     "new_inference": False, "new_simulation": False,
                     "saved_action_noise_hashes_match_between_checkpoints": True},
        "summary": summary, "by_task": by_task, "per_boundary": per_boundary,
        "limitations": [
            "Three tasks and nine training-seen episodes selected near annotation boundaries.",
            "One noise draw per observation; noise differs between the old and fresh observations.",
            "Old predictions are chunk positions 16..31, which the current K=16 deployment does not execute.",
            "Fresh observations follow the expert trajectory, not a physically perturbed or model-generated state.",
            "Smaller revision discrepancy is not automatically better; correct reactivity can require changing a prediction.",
            "This measures prediction revision along demonstrations, not corrective control or a causal effect of observation freshness."
        ], "goal_complete": False,
    }
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

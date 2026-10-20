"""Strictly pair the completed free-label screens; annotation is a reference."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    names = ("baseline1688977", "expert1000")
    paths = {name: root / f"{name}.json" for name in names}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for value in data.values():
        assert value["status"] == "complete" and len(value["samples"]) == 36
        assert value["generation_calls"] == 37 and value["action_sampling_calls"] == 0
        assert all(value["same_model_repeat"].values())
        assert value["backend"] == "cpu" and value["parameter_dtypes"] == ["bfloat16"]
    base, candidate = (data[n] for n in names)
    assert base["model_config"] == candidate["model_config"]
    assert base["normalization_sha256"] == candidate["normalization_sha256"]
    assert base["cache_manifest_sha256"] == candidate["cache_manifest_sha256"]
    indexes = {}
    for name, value in data.items():
        index = {(r["episode_index"], r["frame_index"]): r for r in value["samples"]}
        assert len(index) == 36
        indexes[name] = index
    assert indexes[names[0]].keys() == indexes[names[1]].keys()
    rows = []
    groups = defaultdict(lambda: {"samples": 0, "changed_labels": 0,
                                 "baseline_annotation_agreements": 0, "candidate_annotation_agreements": 0,
                                 "invalid_baseline": 0, "invalid_candidate": 0})
    for key in sorted(indexes[names[0]]):
        b, c = (indexes[name][key] for name in names)
        for field in ("task", "annotation_label", "frames_to_annotation_end", "provided_policy_input_sha256",
                      "free_generation_input_sha256", "processed_prefix_sha256"):
            assert b[field] == c[field], (key, field)
        row = {"task": b["task"], "episode_index": key[0], "frame_index": key[1],
               "frames_to_annotation_end": b["frames_to_annotation_end"], "annotation": b["annotation_label"],
               "baseline_label": b["generated_label"], "candidate_label": c["generated_label"],
               "changed": b["generated_label"] != c["generated_label"],
               "baseline_annotation_agreement": b["valid_generation"] and b["generated_label"] == b["annotation_label"],
               "candidate_annotation_agreement": c["valid_generation"] and c["generated_label"] == c["annotation_label"]}
        rows.append(row)
        for group_key in ("all", b["task"], f"boundary_distance_{b['frames_to_annotation_end']}"):
            stats = groups[group_key]
            stats["samples"] += 1
            stats["changed_labels"] += int(row["changed"])
            stats["baseline_annotation_agreements"] += int(row["baseline_annotation_agreement"])
            stats["candidate_annotation_agreements"] += int(row["candidate_annotation_agreement"])
            stats["invalid_baseline"] += int(not b["valid_generation"])
            stats["invalid_candidate"] += int(not c["valid_generation"])
    with (root / "paired_labels.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "complete", "kind": "paired_free_subtask_label_screen",
              "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for name, path in paths.items()},
              "strict_input_and_config_pairing": True, "summary": dict(groups),
              "changed_rows": [r for r in rows if r["changed"]],
              "limits": ["Three selected tasks, nine training-seen episodes and 36 correlated anchors.",
                         "Exact string agreement with annotation is not a semantic or optimal-control correctness score.",
                         "Boundary anchors can legitimately admit a different next skill.",
                         "No task-completion outcomes or causal mechanism are measured."]}
    with (root / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"summary": report["summary"], "changed_rows": report["changed_rows"]}, indent=2))


if __name__ == "__main__":
    main()

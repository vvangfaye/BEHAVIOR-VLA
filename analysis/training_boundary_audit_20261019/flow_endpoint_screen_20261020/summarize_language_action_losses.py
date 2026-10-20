"""Separate existing joint-loss measurements; no new inference or training."""
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parent
    paths = {name: root / f"{name}.json" for name in ("baseline1688977", "expert1000")}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    index = {name: {(s["sample_id"], s["draw_index"]): s for s in d["samples"]}
             for name, d in data.items()}
    assert index["baseline1688977"].keys() == index["expert1000"].keys()
    groups = {}
    for key, base in index["baseline1688977"].items():
        candidate = index["expert1000"][key]
        for field in ("policy_input_sha256", "transformed_training_sample_sha256", "subtask"):
            assert base[field] == candidate[field]
        assert base["rng"] == candidate["rng"]
        group = groups.setdefault(key[0], {"episode": base["episode_index"],
                                          "frame": base["frame_index"],
                                          "subtask": base["subtask"], "values": {}})
        for name, sample in (("baseline1688977", base), ("expert1000", candidate)):
            metrics = sample["cfm"]["metrics"]
            values = {k: float(metrics[k]["values"]) for k in
                      ("action_loss", "subtask_loss", "subtask_token_accuracy", "subtask_tokens")}
            group["values"].setdefault(name, []).append(values)
    rows = []
    for group in groups.values():
        row = {k: v for k, v in group.items() if k != "values"}
        for name, values in group["values"].items():
            row[name] = {
                "draws": len(values),
                "mean_action_loss": mean(v["action_loss"] for v in values),
                "mean_subtask_token_ce": mean(v["subtask_loss"] for v in values),
                "subtask_token_ce_range": [min(v["subtask_loss"] for v in values),
                                           max(v["subtask_loss"] for v in values)],
                "token_accuracy_range": [min(v["subtask_token_accuracy"] for v in values),
                                         max(v["subtask_token_accuracy"] for v in values)],
                "supervised_tokens": sorted(set(v["subtask_tokens"] for v in values)),
            }
        b, c = row["baseline1688977"], row["expert1000"]
        row["action_loss_delta"] = c["mean_action_loss"] - b["mean_action_loss"]
        row["language_ce_delta"] = c["mean_subtask_token_ce"] - b["mean_subtask_token_ce"]
        rows.append(row)
    result = {
        "status": "complete", "kind": "existing_joint_loss_decomposition",
        "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in paths.items()},
        "rows": sorted(rows, key=lambda row: row["episode"]),
        "scope": "Three training-seen expert anchors; eight flow-noise/time draws each. Language labels are supplied for teacher-forced token scoring, not generated and not annotated autonomous-stage truth.",
        "findings": [
            "Both checkpoints have perfect teacher-forced token argmax accuracy on these three short labels.",
            "Per-token language CE is already near numerical saturation on these anchors.",
            "Language and action-loss changes have mixed alignment across the anchors.",
            "These anchors cannot identify whether autonomous stage selection explains closed-loop changes.",
        ],
        "limits": ["No new model inference or simulator run.",
                   "Eight flow draws are not independent task or language examples.",
                   "Near-zero finite-precision loss is not proof of probability exactly one.",
                   "No inference about all tasks, failure states, or ability retention."],
    }
    (root / "language_action_loss_decomposition.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()

"""Leave-one-task-out PCA of saved numerical action changes, not task effects."""
import json

import numpy as np

from audit_temporal_probe_basis import GROUPS, HERE, load, source_function


def main():
    checksum = source_function("input_content_sha256")
    base_data, base, base_source = load("baseline1688977", checksum)
    result = {"kind": "posthoc_leave_one_task_out_action_subspace", "sources": {"baseline1688977": base_source},
              "rank_grid": [1, 2, 4, 8, 16], "folds": []}
    for name in ("mixed1000", "mixed5000", "expert1000"):
        data, rows, result["sources"][name] = load(name, checksum)
        assert rows.keys() == base.keys()
        assert data["normalization"] == base_data["normalization"]
        assert data["action_noise"] == base_data["action_noise"]
        samples = []
        for key in rows:
            b, ba = base[key]
            r, ra = rows[key]
            for field in ("task", "policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
                assert r[field] == b[field]
            assert np.array_equal(ba["valid_mask"], ra["valid_mask"])
            if ba["valid_mask"][:16].all():
                samples.append((r["task"], ra["prediction"][:16].astype(float) - ba["prediction"][:16].astype(float)))
        for task in sorted({s[0] for s in samples}):
            train = np.stack([s[1] for s in samples if s[0] != task])
            test = np.stack([s[1] for s in samples if s[0] == task])
            for group, (lo, hi) in GROUPS.items():
                tr, te = train[:, :, lo:hi], test[:, :, lo:hi]
                # Fit scaling only on training tasks, one scale per control coordinate.
                scale = np.maximum(np.sqrt(np.mean(tr**2, axis=(0, 1))), 1e-6)
                x = (tr / scale).reshape(len(tr), -1)
                y = (te / scale).reshape(len(te), -1)
                # Uncentered SVD includes any common change in the probe span.
                _, singular, vectors = np.linalg.svd(x, full_matrices=False)
                identifiable = int(np.sum(singular > max(singular[0] * 1e-12, 1e-12)))
                energy = float(np.sum(y*y))
                scores = {}
                for rank in result["rank_grid"]:
                    actual = min(rank, identifiable)
                    q = vectors[:actual]
                    retained = float(np.sum((y @ q.T)**2))
                    score = None if energy <= 1e-12 else retained / energy
                    assert score is None or -1e-12 <= score <= 1 + 1e-12
                    scores[str(rank)] = score
                result["folds"].append({"checkpoint": name, "heldout_task": task, "group": group,
                                        "training_anchors": len(tr), "test_anchors": len(te),
                                        "numerical_training_rank": identifiable,
                                        "heldout_scaled_energy_fraction": scores})
    summary = {}
    for rank in result["rank_grid"]:
        values = [f["heldout_scaled_energy_fraction"][str(rank)] for f in result["folds"]]
        values = [v for v in values if v is not None]
        summary[str(rank)] = {"macro_checkpoint_task_group_fraction": float(np.mean(values)),
                              "minimum_fraction": float(min(values)),
                              "blocks_at_least_95_percent": sum(v >= .95 for v in values),
                              "blocks": len(values), "maximum_total_action_basis_dimension": 6 * rank}
    result["summary"] = summary
    non_gripper = [f for f in result["folds"] if "gripper" not in f["group"]]
    result["non_gripper_summary"] = {}
    for rank in result["rank_grid"]:
        values = [f["heldout_scaled_energy_fraction"][str(rank)] for f in non_gripper]
        values = [v for v in values if v is not None]
        result["non_gripper_summary"][str(rank)] = {
            "non_gripper_macro_fraction": float(np.mean(values)),
            "blocks_at_least_95_percent": sum(v >= .95 for v in values),
            "blocks": len(values),
        }
    result["status"] = "complete"
    result["limits"] = ["All three checkpoints share an initialization and selected training-seen observations.",
                        "Each fold fits a checkpoint-specific basis on two tasks; this is not a universal or future-update basis.",
                        "Per-coordinate scale is fitted on training tasks only; metric differs from the DCT coordinate-wise summary.",
                        "Numerical action-change coverage does not establish physical-effect or task-gradient coverage.",
                        "This diagnostic does not establish that low-rank physical probing is impossible or sufficient."]
    result["new_model_calls"] = result["new_simulator_calls"] = 0
    (HERE / "learned_probe_basis_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

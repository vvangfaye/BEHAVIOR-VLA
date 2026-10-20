"""Post-hoc compression audit of saved outputs; no model or simulator calls."""
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = HERE.parent / "calibration_screen_20261020" / "retrieved"
GROUPS = {"base": (0, 3), "torso": (3, 7), "left_arm": (7, 14),
          "left_gripper": (14, 15), "right_arm": (15, 22),
          "right_gripper": (22, 23)}
RANKS = (1, 2, 4, 8, 16)


def source_function(name):
    path = ROOT / "BEHAVIOR-VLAs/scripts/eval_openloop.py"
    module = ast.parse(path.read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Any": Any, "hashlib": hashlib, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load(name, checksum):
    path = INPUT / f"{name}.json"
    data = json.loads(path.read_text())
    rows = {}
    for row in data["sample_keys"]:
        raw = row["raw_actions"]
        arrays = {key: np.asarray(raw[key], dtype=raw["dtypes"][key])
                  for key in ("prediction", "ground_truth", "hold_action", "valid_mask")}
        assert checksum(arrays) == raw["sha256"]
        assert checksum(arrays["ground_truth"]) == row["ground_truth_sha256"]
        assert arrays["prediction"].shape == (32, 23)
        assert np.isfinite(arrays["prediction"]).all()
        key = (row["episode_index"], row["frame_index"])
        assert key not in rows
        rows[key] = (row, arrays)
    return data, rows, {"path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    checksum = source_function("input_content_sha256")
    base_data, base_rows, base_source = load("baseline1688977", checksum)
    # Fixed DCT-II basis. No fitting or rank selection uses the saved outputs.
    t = np.arange(16, dtype=float)[:, None]
    k = np.arange(16, dtype=float)[None, :]
    basis = np.cos(np.pi * (t + 0.5) * k / 16) * np.sqrt(2 / 16)
    basis[:, 0] /= np.sqrt(2)
    assert np.max(np.abs(basis.T @ basis - np.eye(16))) < 1e-13
    sources, results = {"baseline1688977": base_source}, {}
    for name in ("mixed1000", "mixed5000", "expert1000"):
        data, rows, sources[name] = load(name, checksum)
        assert rows.keys() == base_rows.keys()
        assert data["normalization"] == base_data["normalization"]
        assert data["action_noise"] == base_data["action_noise"]
        tasks = {}
        included, excluded = [], []
        for key in rows:
            base_row, base = base_rows[key]
            row, candidate = rows[key]
            for field in ("task", "policy_input_sha256", "ground_truth_sha256", "action_noise_sha256"):
                assert row[field] == base_row[field]
            assert np.array_equal(base["valid_mask"], candidate["valid_mask"])
            if not np.all(base["valid_mask"][:16]):
                excluded.append(key)
                continue
            delta = candidate["prediction"][:16].astype(float) - base["prediction"][:16].astype(float)
            tasks.setdefault(row["task"], []).append(delta)
            included.append(key)
        blocks = []
        for task, values in sorted(tasks.items()):
            delta = np.stack(values)
            coefficients = np.einsum("tk,ntd->nkd", basis, delta)
            total = np.sum(delta * delta, axis=(0, 1))
            meaningful = np.sqrt(total / (16 * len(delta))) > 1e-6
            for group, (lo, hi) in GROUPS.items():
                keep = meaningful[lo:hi]
                shares = {}
                for rank in RANKS:
                    retained = np.sum(coefficients[:, :rank] ** 2, axis=(0, 1))
                    ratios = retained[lo:hi][keep] / total[lo:hi][keep]
                    assert np.all(ratios >= -1e-12) and np.all(ratios <= 1 + 1e-12)
                    if rank == 16:
                        assert np.max(np.abs(ratios - 1), initial=0) < 1e-12
                    shares[str(rank)] = None if len(ratios) == 0 else float(np.mean(ratios))
                blocks.append({"task": task, "group": group, "anchors": len(values),
                               "coordinates_retained": int(sum(keep)),
                               "coordinate_mean_energy_fraction": shares})
        summary = {}
        for rank in RANKS:
            shares = [b["coordinate_mean_energy_fraction"][str(rank)] for b in blocks
                      if b["coordinate_mean_energy_fraction"][str(rank)] is not None]
            summary[str(rank)] = {"macro_task_group_fraction": float(np.mean(shares)),
                                  "minimum_task_group_fraction": float(np.min(shares)),
                                  "blocks_at_least_95_percent": sum(x >= .95 for x in shares),
                                  "blocks": len(shares), "action_basis_dimension": rank * 23}
        results[name] = {"included_anchors": len(included), "excluded_incomplete_prefix": len(excluded),
                         "summary": summary, "blocks": blocks}
    report = {"status": "complete", "kind": "posthoc_saved_action_geometry_audit",
              "sources": sources, "basis": "Fixed orthonormal DCT-II over the deployed first 16 actions",
              "ranks": list(RANKS), "coordinate_rms_floor": 1e-6,
              "aggregation": "Sum energy across anchors within each task/coordinate, compute unit-free retained fractions, average coordinates within group, then macro-average task/groups.",
              "results": results,
              "limits": ["Three selected tasks and training-seen provided-subtask observations only.",
                         "Describes common-noise numerical checkpoint deltas, not a sampler-invariant policy difference.",
                         "Action-space energy is not physical transition sensitivity or task utility.",
                         "A basis covering action deltas need not cover the physical gradient or future training updates.",
                         "Low rank does not establish a new algorithm or suffice for training-gradient identifiability."],
              "new_model_calls": 0, "new_simulator_calls": 0}
    (HERE / "temporal_probe_basis_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"anchors": r["included_anchors"], "summary": r["summary"]}
                      for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()

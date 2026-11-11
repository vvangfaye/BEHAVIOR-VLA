"""Post-hoc descriptive analysis of saved predictions; no new efficacy gate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "pilot_ada01"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read(PILOT / "plan.json")
    data_path = Path(plan["data_directory"]) / "data_plan.json"
    data = read(data_path)
    results = {}
    sources = {str(data_path): sha(data_path)}
    reference = None
    for method in plan["methods"]:
        path = PILOT / f"formal_{method}" / "output/test.npz"
        sources[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            samples = z["samples"].astype(np.float64)
            target = z["target"].astype(np.float64)
            index, episode = z["index"].copy(), z["episode"].copy()
            noise_hashes = z["noise_sha256"].copy()
        identifiers = (target, index, episode, noise_hashes)
        if reference is None:
            reference = identifiers
        else:
            for expected, actual in zip(reference, identifiers):
                np.testing.assert_array_equal(expected, actual)
        tasks = np.asarray([data["samples"][int(i)]["task_name"] for i in index])
        prediction = samples.mean(axis=1)
        error = prediction - target
        by_task = {}
        for task in sorted(set(tasks)):
            mask = tasks == task
            e = error[mask]
            total = float(np.square(e).mean())
            bias = float(np.square(e.mean(axis=0)).mean())
            centered = float(np.square(e - e.mean(axis=0)).mean())
            assert abs(total - bias - centered) < 1e-12
            by_task[task] = dict(
                observations=int(mask.sum()),
                episodes=int(len(np.unique(episode[mask]))),
                mean_prediction_mse=total,
                task_common_bias_mse=bias,
                task_centered_error_mse=centered,
                common_bias_fraction=bias / total,
                within_observation_draw_variance=float(
                    np.square(samples[mask] - prediction[mask, None]).mean()
                ),
            )
        results[method] = by_task
    comparisons = {}
    for task in results["extra_fm"]:
        control = results["extra_fm"][task]
        candidate = results["relative_difference"][task]
        comparisons[task] = {
            key + "_delta": candidate[key] - control[key]
            for key in ["mean_prediction_mse", "task_common_bias_mse",
                        "task_centered_error_mse", "within_observation_draw_variance"]
        }
    report = dict(
        observed_at=datetime.now(timezone.utc).isoformat(),
        analysis_status="Post-hoc descriptive decomposition; original failed screen unchanged.",
        definitions={
            "error": "Mean of four generated chunks minus one observed expert chunk.",
            "identity": "E||e||^2 = ||E e||^2 + E||e-E e||^2, grouped by task.",
            "units": "Mean square per normalized physical action coordinate and time slot.",
        },
        by_method=results,
        candidate_minus_extra_fm=comparisons,
        source_sha256=sources,
        conclusions=[
            "Centered error decreases on polishing_shoes but increases on both other tasks.",
            "Common task-wide additive bias explains less than 3.5 percent of control MSE on each task.",
            "A pure task-wide additive offset is therefore not the dominant observed offline error component.",
        ],
        limitations=[
            "Four draws and one expert target per observation do not identify the true conditional mean or valid action set.",
            "Task centering is a descriptive oracle operation, not a deployable correction or a general calibration test.",
            "Centered action error does not identify physical feedback quality or off-support corrections.",
            "No independent significance test, new training, inference, or simulator episode was added.",
        ],
        original_gate_passed=False,
        goal_complete=False,
    )
    assert all(x["common_bias_fraction"] < .035 for x in results["extra_fm"].values())
    (ROOT / "error_components.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(candidate_minus_extra_fm=comparisons,
                          control_common_bias_fraction={k: v["common_bias_fraction"]
                                                        for k, v in results["extra_fm"].items()}), indent=2))


if __name__ == "__main__":
    main()

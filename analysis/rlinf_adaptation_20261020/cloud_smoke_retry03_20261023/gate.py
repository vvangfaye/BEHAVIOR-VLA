"""Frozen admission checks; a failed comparison never starts PPO."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import env_observation, label_text, torch_model, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(plan["output"])
    model = torch_model(plan)
    probes = []
    for path in sorted((root / "jax").glob("*/probe_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            raw = {
                "state": data["state"],
                "images": list(data["images"]),
                "prompt": str(data["prompt"]),
            }
            actions, result = model.predict_action_batch(
                env_observation(raw),
                mode="eval",
                noise=torch.from_numpy(data["noise"]).cuda(),
            )
            text = label_text(model, result)
            error = np.abs(actions[0].cpu().numpy() - data["actions"])
            # In units of original action normalization, across the executed 23-D actions.
            error = error / (model.checkpoint_io.stats["actions"]["std"][None] + 1e-6)
            probes.append(
                {
                    "path": str(path),
                    "subtask_equal": text == str(data["subtask"]),
                    "jax_subtask": str(data["subtask"]),
                    "torch_subtask": text,
                    "mean_error": float(error.mean()),
                    "max_error": float(error.max()),
                }
            )
    pairs = []
    for instance in plan["eval_instances"]:

        def read(backend, name):
            return json.loads((root / backend / str(instance) / name).read_text())

        a, b = read("jax", "result.json"), read("torch", "result.json")
        ia, ib = read("jax", "initial.json"), read("torch", "initial.json")
        assert ia["physical_state"].keys() == ib["physical_state"].keys()
        initial_error = max(
            float(
                np.abs(
                    np.asarray(ia["physical_state"][k][part])
                    - np.asarray(ib["physical_state"][k][part])
                ).max()
            )
            for k in ia["physical_state"]
            for part in (0, 1)
        )
        pairs.append(
            {
                "instance": instance,
                "jax": a,
                "torch": b,
                "initial_max_error": initial_error,
                "score_gap": abs(a["score"] - b["score"]),
                "success_equal": a["success"] == b["success"],
            }
        )
    thresholds = plan["gate"]
    checks = {
        "enough_probes": len(probes) >= thresholds["minimum_probes"],
        "same_greedy_subtasks": bool(probes)
        and all(x["subtask_equal"] for x in probes),
        "same_input_mean_error": bool(probes)
        and max(x["mean_error"] for x in probes) <= thresholds["mean_action_error"],
        "same_input_max_error": bool(probes)
        and max(x["max_error"] for x in probes) <= thresholds["max_action_error"],
        "initial_state": all(
            x["initial_max_error"] <= thresholds["initial_error"] for x in pairs
        ),
        "closed_loop_score": all(
            x["score_gap"] <= thresholds["score_gap"] for x in pairs
        ),
        "closed_loop_success": all(x["success_equal"] for x in pairs),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "probes": probes,
        "pairs": pairs,
        "scope": "Two instances and sampled states; does not prove all-task equivalence.",
    }
    write(root / "gate.json", report)
    print(json.dumps({"passed": report["passed"], "checks": checks}), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

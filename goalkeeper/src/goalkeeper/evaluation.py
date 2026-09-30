"""Read official output files without modifying them; reject cherry-picked runs."""
from pathlib import Path
import json
import math
import numpy as np


def load_results(directory, expected):
    # expected is an explicit list from the chosen official task/instance split.
    keys = [tuple(x) for x in expected]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError("Expected keys must be unique and nonempty")
    expected_set, found = set(keys), {}
    for path in Path(directory).rglob("*.json"):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or not {"task", "instance_id", "rollout_id"} <= row.keys():
            continue
        key = (row["task"], row["instance_id"], row["rollout_id"])
        if key not in expected_set: raise ValueError(f"Unexpected rollout {key}; isolate the prescribed run")
        if key in found: raise ValueError(f"Duplicate rollout {key}; do not select the best repetition")
        q, success = row["q_score"]["final"], row["success"]
        if type(success) is not bool or isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError(f"Invalid official metric in {path}")
        found[key] = (q, float(success))
    # The submission rules count missing rollout instances as zero.
    values = np.array([found.get(k, (0., 0.)) for k in keys])
    return keys, values, len(found)


def summarize(directory, expected, baseline=None, bootstraps=2000, seed=7):
    keys, values, observed = load_results(directory, expected)
    tasks = sorted({k[0] for k in keys})
    out = {"expected": len(keys), "observed": observed, "missing_counted_as_zero": len(keys) - observed,
           "q_score": float(values[:, 0].mean()), "success_rate": float(values[:, 1].mean()),
           "scope": "full_100x10" if len(tasks) == 100 and all(sum(k[0] == t for k in keys) == 10 for t in tasks) else "subset",
           "per_task": {t: dict(zip(["q_score", "success_rate"], values[[k[0] == t for k in keys]].mean(0).tolist())) for t in tasks}}
    if baseline:
        _, reference, baseline_observed = load_results(baseline, expected)
        delta = values - reference
        out["paired_difference"] = dict(zip(["q_score", "success_rate"], delta.mean(0).tolist()))
        out["baseline_observed"] = baseline_observed
        # Resample matched instances within each fixed task, preserving pairing.
        if bootstraps < 1: raise ValueError("Positive bootstrap count required")
        rng = np.random.default_rng(seed)
        groups = [np.array([i for i, k in enumerate(keys) if k[0] == t]) for t in tasks]
        samples = np.array([delta[np.concatenate([rng.choice(g, len(g)) for g in groups])].mean(0)
                            for _ in range(bootstraps)])
        ci = np.quantile(samples, [.025, .975], axis=0).T
        out["paired_difference_95_percent_ci"] = dict(zip(["q_score", "success_rate"], ci.tolist()))
        out["ci_scope"] = "paired instance bootstrap conditional on this fixed task set; not unseen-task uncertainty"
    return out

"""Analytic feasibility probe for action-space zeroth-order gradients.

No robot, VLA, simulator, or policy training is involved. The rank-r projection
is a chosen synthetic policy Jacobian, not an estimate of our VLA's Jacobian.
"""
import json
from pathlib import Path

import numpy as np


def main():
    dimension, repetitions, block_size = 2 * 16 * 23, 512, 32
    rng = np.random.default_rng(20261020)
    ranks = (1, 32, dimension)
    records = []
    for directions in (1, 8, 32, 128):
        sums = {rank: {"squared_error": 0.0, "cosine": 0.0, "first": 0.0} for rank in ranks}
        for begin in range(0, repetitions, block_size):
            count = min(block_size, repetitions - begin)
            u = rng.standard_normal((count, directions, dimension))
            # For L(a)=.5*||a+e_1||^2 at a=0, the central two-point loss
            # difference is exactly 2*h*u_1 for every h>0.
            estimate = np.mean(u[..., :1] * u, axis=1)
            for rank in ranks:
                projected = estimate[:, :rank]
                squared = np.sum(projected**2, axis=1)
                sums[rank]["squared_error"] += float(np.sum(squared - 2 * projected[:, 0] + 1))
                sums[rank]["cosine"] += float(np.sum(projected[:, 0] / np.sqrt(squared)))
                sums[rank]["first"] += float(np.sum(projected[:, 0]))
        for rank in ranks:
            records.append({
                "action_dimension": dimension,
                "synthetic_policy_jacobian_rank": rank,
                "directions_per_estimate": directions,
                "repetitions": repetitions,
                "expected_relative_squared_error": (rank + 1) / directions,
                "measured_relative_squared_error": sums[rank]["squared_error"] / repetitions,
                "measured_mean_cosine": sums[rank]["cosine"] / repetitions,
                "measured_mean_first_component": sums[rank]["first"] / repetitions,
                "physical_branch_count_if_two_branches_per_loss": 4 * directions,
                "control_steps_if_each_branch_executes_16": 4 * directions * 16,
            })
    report = {
        "status": "complete",
        "kind": "analytic_quadratic_gradient_estimator_probe",
        "loss": "L(a)=0.5*||a+e_1||^2, evaluated at a=0",
        "estimator": "mean_m[(L(a+h*u)-L(a-h*u))/(2*h) * u], u~N(0,I)",
        "general_identity": "For Gaussian u and a sufficiently integrable loss, the expectation is the gradient of the Gaussian-smoothed loss, not necessarily of the original nonsmooth loss.",
        "projection_identity": "For quadratic loss with gradient g and linearized policy Jacobian J, E||J^T(g_hat-g)||^2=(||g||^2||J||_F^2+||J^T*g||^2)/m.",
        "chosen_projection": "J selects the first r action coordinates; g=e_1 lies inside this range. Thus relative MSE=(r+1)/m in this synthetic example.",
        "results": records,
        "conclusions": [
            "Reconstructing a full dynamics Jacobian is unnecessary for a stochastic gradient step.",
            "Two-point query count alone does not imply an accurate cheap update; variance can be large.",
            "Raw action-space variance does not alone determine propagated parameter-gradient variance.",
            "The relevant VLA Jacobian geometry and physical-loss noise remain unmeasured.",
        ],
        "limits": [
            "No physical response loss was evaluated.",
            "No real VLA parameter Jacobian was measured.",
            "Chosen low-rank examples do not establish low effective variance in this project.",
            "The branch/control counts exclude rendering, settling, restoration, model calls and teacher target construction.",
            "No novelty claim for the zeroth-order estimator: it is a known optimization method.",
        ],
        "new_model_calls": 0, "new_simulator_calls": 0, "training": False,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([r for r in records if r["directions_per_estimate"] == 32], indent=2))


if __name__ == "__main__":
    main()

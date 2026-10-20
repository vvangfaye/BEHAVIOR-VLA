"""Finite mathematical checks only; no policy, simulator or GPU calls."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def distances(points, metric):
    delta = points[:, None] - points[None, :]
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", delta, metric, delta), 0))


def expected_score(predicted, target, d):
    return predicted @ d @ target - 0.5 * predicted @ d @ predicted


def score_gradient(residual, metric):
    return metric @ residual / np.sqrt(residual @ metric @ residual)


def physical_cost(residual, metric):
    return float(residual @ metric @ residual)


def update(residual, learned_metric, step):
    gradient = score_gradient(residual, learned_metric)
    return residual - step * gradient / np.linalg.norm(gradient)


def main():
    points = np.array([[-1., 0.], [0., 1.], [1., -1.]])
    target = np.array([0.2, 0.3, 0.5])
    candidates = np.array([[i / 100, j / 100, (100 - i - j) / 100]
                           for i in range(101) for j in range(101 - i)])
    metrics = [np.eye(2), np.array([[1., .9], [.9, 1.]]),
               np.array([[1.6, -.3], [-.3, .4]])]
    proper_checks = []
    for metric in metrics:
        d = distances(points, metric)
        losses = np.array([expected_score(q, target, d) for q in candidates])
        winner = candidates[np.argmin(losses)]
        truth_loss = expected_score(target, target, d)
        assert np.allclose(winner, target)
        assert np.min(losses - truth_loss) > -1e-12
        # Exact expectation of the proposed two-independent-sample estimator.
        q = np.array([.45, .2, .35])
        enumerated = sum(q[i] * q[j] * target[k] *
                         (d[i, k] + d[j, k] - d[i, j]) / 2
                         for i in range(3) for j in range(3) for k in range(3))
        direct = expected_score(q, target, d)
        assert abs(enumerated - direct) < 1e-12
        proper_checks.append({"metric": metric.tolist(), "grid_minimizer": winner.tolist(),
                              "two_sample_expectation_error": abs(enumerated - direct)})

    physical_metric = np.array([[1., .9], [.9, 1.]])
    residual = np.array([1., -2.])
    grad = score_gradient(residual, physical_metric)
    fd = []
    for i in range(2):
        eps = np.eye(2)[i] * 1e-6
        fplus = np.sqrt((residual + eps) @ physical_metric @ (residual + eps))
        fminus = np.sqrt((residual - eps) @ physical_metric @ (residual - eps))
        fd.append((fplus - fminus) / 2e-6)
    assert np.max(np.abs(np.array(fd) - grad)) < 1e-8
    # Nonnegative diagonal losses can only move (-,+) at this residual.
    angles = np.linspace(0, np.pi / 2, 20001)
    feasible = np.stack([-np.cos(angles), np.sin(angles)], axis=1)
    step = .1
    diag_costs = np.array([physical_cost(residual + step * direction, physical_metric)
                           for direction in feasible])
    full_next = update(residual, physical_metric, step)
    assert full_next[0] > residual[0]
    assert physical_cost(full_next, physical_metric) < diag_costs.min()

    # Toy outer optimization: optimize the correlation of a trace-normalized metric.
    # Physical cost is supplied as an oracle in this mathematical construction only.
    train = [np.array([1., -2.]), np.array([1.2, -2.1])]
    heldout = [np.array([.8, -1.7]), np.array([-1., 2.]), np.array([1.4, -2.4])]
    correlations = np.linspace(-.9, .9, 1801)
    def loss(c, examples):
        m = np.array([[1., c], [c, 1.]])
        return float(np.mean([physical_cost(update(e, m, step), physical_metric)
                              for e in examples]))
    best_c = float(correlations[np.argmin([loss(c, train) for c in correlations])])
    assert loss(best_c, heldout) < loss(0, heldout)

    result = {
        "kind": "finite_analytic_sanity_checks_not_VLA_evidence",
        "proper_score_checks": proper_checks,
        "equal_update_norm_control": {
            "metric_trace": float(np.trace(physical_metric)),
            "metric_eigenvalues": np.linalg.eigvalsh(physical_metric).tolist(),
            "initial_residual": residual.tolist(), "step_norm": step,
            "energy_gradient": grad.tolist(),
            "max_gradient_finite_difference_error": float(np.max(np.abs(np.array(fd) - grad))),
            "initial_physical_cost": physical_cost(residual, physical_metric),
            "euclidean_next_cost": physical_cost(update(residual, np.eye(2), step), physical_metric),
            "best_nonnegative_diagonal_direction_cost": float(diag_costs.min()),
            "full_metric_next_cost": physical_cost(full_next, physical_metric)
        },
        "toy_outer_optimization": {
            "correlation": best_c, "train_cost": loss(best_c, train),
            "heldout_cost": loss(best_c, heldout), "euclidean_heldout_cost": loss(0, heldout),
            "limits": "Two training residuals and three held-out residuals share the same known quadratic dynamics cost. This does not demonstrate task-family or VLA transfer."
        },
        "non_claims": [
            "A finite grid is not a general proof of strict propriety.",
            "The example does not beat arbitrary full-matrix preconditioning or generic meta-learning.",
            "No simulator throughput, native policy backward pass, or practical meta-learning sample complexity was validated.",
            "No novelty or closed-loop VLA efficacy is established."
        ],
        "new_gpu_jobs": 0,
        "new_human_data": 0
    }
    output = ROOT / 'learned_proper_metric_math_check_20261020.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({"output": str(output),
                      "equal_update_norm_control": result["equal_update_norm_control"],
                      "toy_outer_optimization": result["toy_outer_optimization"]}, indent=2))


if __name__ == '__main__':
    main()

"""Population counterexample, not a robot experiment or novelty claim.

The parameter tying and threshold were chosen to construct a counterexample,
not sampled as a representative model of VLA behavior.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_SCALE_SLOPE = -1.5
PHI = math.exp(-0.5) / math.sqrt(2 * math.pi)
SUCCESS0 = 0.5 * math.erfc(1 / math.sqrt(2))
TARGET_MEAN = PHI / SUCCESS0
TARGET_SECOND_MOMENT = 1 + TARGET_MEAN


def success(theta):
    sigma = math.exp(LOG_SCALE_SLOPE * theta)
    return 0.5 * math.erfc((1 - theta) / (sigma * math.sqrt(2)))


def nll(theta):
    log_sigma = LOG_SCALE_SLOPE * theta
    residual_second = TARGET_SECOND_MOMENT - 2 * theta * TARGET_MEAN + theta**2
    return log_sigma + 0.5 * residual_second * math.exp(-2 * log_sigma)


def cfm(theta, grid_size):
    t = np.linspace(0, 1, grid_size)
    variance = math.exp(2 * LOG_SCALE_SLOPE * theta)
    denominator = (1 - t)**2 + t**2 * variance
    c = (t * variance - (1 - t)) / denominator
    d = theta * (1 - c * t)
    # v_theta(x_t,t) - (a-z), x_t=t*a+(1-t)*z.
    a_coefficient = c * t - 1
    z_coefficient = c * (1 - t) + 1
    moment = (a_coefficient**2 * TARGET_SECOND_MOMENT + z_coefficient**2
              + d**2 + 2 * a_coefficient * d * TARGET_MEAN)
    return float(np.trapz(moment, t))


def main():
    # These analytic derivatives use an exactly uniform t in [0,1].
    cfm_gradient = -math.pi / 4 * TARGET_MEAN * (2 + LOG_SCALE_SLOPE)
    nll_gradient = -TARGET_MEAN * (1 + LOG_SCALE_SLOPE)
    success_gradient = PHI * (1 + LOG_SCALE_SLOPE)
    derivative_checks = []
    for grid in [10001, 20001, 40001]:
        eps = 1e-5
        estimate = (cfm(eps, grid) - cfm(-eps, grid)) / (2 * eps)
        derivative_checks.append(dict(grid=grid, derivative=estimate,
                                      analytic_error=abs(estimate - cfm_gradient)))
        assert abs(estimate - cfm_gradient) < 1e-6
    rows = []
    for learning_rate in [0.001, 0.01]:
        for method, gradient in [("success_CFM", cfm_gradient), ("success_MLE", nll_gradient)]:
            theta = -learning_rate * gradient
            row = dict(method=method, learning_rate=learning_rate, theta=theta,
                       cfm_loss=cfm(theta, 40001), nll=nll(theta), success=success(theta))
            if method == "success_CFM":
                assert row["cfm_loss"] < cfm(0, 40001)
                assert row["success"] < SUCCESS0
            else:
                assert row["nll"] < nll(0)
                assert row["success"] > SUCCESS0
            rows.append(row)
    result = dict(
        observed_at=datetime.now(timezone.utc).isoformat(),
        setup={
            "policy": "a ~ Normal(mu=theta, sigma=exp(-1.5*theta))",
            "success": "a > 1; one-step fully observed environment",
            "initial_parameter": 0,
            "fixed_training_distribution": "Normal(0,1) conditioned on a>1",
            "flow_path": "x_t = (1-t)*z+t*a, z~Normal(0,1), t~Uniform[0,1]",
            "model_velocity": "Exact marginal Gaussian velocity of the parameterized policy family.",
            "loss": "Population E[(v_theta(x_t,t)-(a-z))^2]",
            "source_of_restriction": "One shared parameter ties mean and log scale; explicitly chosen to show existence.",
        },
        baseline=dict(cfm_loss=cfm(0, 40001), nll=nll(0), success=SUCCESS0),
        analytic_gradients=dict(cfm=cfm_gradient, nll=nll_gradient, success=success_gradient),
        cfm_descent_success_direction=-cfm_gradient * success_gradient,
        mle_descent_success_direction=-nll_gradient * success_gradient,
        quadrature_derivative_checks=derivative_checks,
        one_step_results=rows,
        conclusion="Population success-only CFM descent can oppose success improvement under shared parameter constraints. MLE behaves differently in this example.",
        limitations=[
            "Not a new general theorem: different surrogate objectives need not share finite-update improvement behavior.",
            "Not evidence that this mechanism caused any of the observed VLA results.",
            "No claim that all flow-matching updates hurt success or that unconstrained exact fitting of success data fails.",
            "MLE uses the tractable Gaussian endpoint density; the native VLA executed-action likelihood remains nontrivial.",
            "No VLA training method is validated by this counterexample.",
        ],
        new_gpu_jobs=0,
        new_simulator_episodes=0,
        goal_complete=False,
    )
    (ROOT / "success_cfm_direction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["baseline", "analytic_gradients", "one_step_results"]}, indent=2))


if __name__ == "__main__":
    main()

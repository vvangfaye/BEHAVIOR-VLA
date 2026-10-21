"""Exact-noise first-update check for a CFM-ratio policy surrogate.

This is a constructed restricted Gaussian family, not a reproduction of
FPO/FPO++ experiments or a statement about their general empirical behavior.
"""
from datetime import datetime, timezone
import json
import math

import numpy as np

from check_success_cfm_direction import LOG_SCALE_SLOPE, PHI, ROOT, SUCCESS0, success


TIME = np.linspace(0, 1, 40001)


def coefficients(theta):
    t = TIME
    variance = math.exp(2 * LOG_SCALE_SLOPE * theta)
    c = (t * variance - (1 - t)) / ((1 - t)**2 + t**2 * variance)
    d = theta * (1 - c * t)
    a, z = c * t - 1, c * (1 - t) + 1
    return np.asarray([np.trapz(a*a, t), np.trapz(2*a*d, t),
                       np.trapz(z*z + d*d, t)])


BASE = coefficients(0)


def surrogate(theta):
    # Exact integral over the original N(0,1) actions after completing square.
    da, db, dc = coefficients(theta) - BASE
    precision = 1 + 2 * da
    assert precision > 0
    tilted_mean = -db / precision
    tilted_std = 1 / math.sqrt(precision)
    normalizer = math.exp(-dc + db*db / (2*precision)) / math.sqrt(precision)
    tail = 0.5 * math.erfc((1 - tilted_mean) / (tilted_std * math.sqrt(2)))
    return normalizer * (tail - SUCCESS0)


def main():
    true_gradient = PHI * (1 + LOG_SCALE_SLOPE)
    surrogate_gradient = PHI * math.pi / 4 * (2 + LOG_SCALE_SLOPE)
    eps = 1e-5
    finite_difference = (surrogate(eps) - surrogate(-eps)) / (2 * eps)
    assert abs(finite_difference - surrogate_gradient) < 1e-7
    assert true_gradient * surrogate_gradient < 0
    theta_new = 0.01 * surrogate_gradient
    assert surrogate(theta_new) > surrogate(0)
    assert success(theta_new) < SUCCESS0
    report = dict(
        observed_at=datetime.now(timezone.utc).isoformat(),
        setup="Same tied Gaussian family and binary reward as success_cfm_direction.json.",
        advantage="1[a>1] - J(0); exact baseline, no critic error.",
        ratio="exp(L_0(a)-L_theta(a)); L is the exact expected uniform-time CFM loss.",
        objective="E_{a~N(0,1)}[(1[a>1]-J(0))*ratio]",
        true_reward_gradient=true_gradient,
        cfm_ratio_surrogate_gradient=surrogate_gradient,
        independently_integrated_surrogate_gradient=finite_difference,
        gradient_product=true_gradient * surrogate_gradient,
        step=dict(theta=theta_new, surrogate_before=surrogate(0),
                  surrogate_after=surrogate(theta_new), success_before=SUCCESS0,
                  success_after=success(theta_new)),
        inference=[
            "Exact advantage centering does not remove this constructed direction mismatch.",
            "At theta=0 all ratios are one; positive-width clipping and the FPO++ asymmetric penalty do not change the first derivative.",
            "Additional Monte Carlo samples cannot remove a discrepancy that remains for the exact expected CFM loss.",
        ],
        scope_limits=[
            "Specific uniform-time CFM schedule and tied Gaussian parameter family.",
            "The analytic unclipped first-update surrogate is studied, not all finite clipped updates.",
            "FPO's reported estimator unbiasedness concerns its ELBO/CFM gradient, not automatically the true endpoint reward gradient.",
            "Neither a VLA result nor proof that every CFM-based RL method fails.",
            "Does not establish a novel repair or override the relative-flow pilot's failed screen.",
        ],
        new_gpu_jobs=0, new_simulator_episodes=0, goal_complete=False,
    )
    (ROOT / "centered_flow_surrogate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["true_reward_gradient",
                      "cfm_ratio_surrogate_gradient", "gradient_product", "step"]}, indent=2))


if __name__ == "__main__":
    main()

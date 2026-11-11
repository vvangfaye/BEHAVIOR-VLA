"""Check a proposed reward-to-transport regression identity; not a robot experiment."""
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss


def main():
    # Bounded advantage, including a component invisible to a Gaussian's
    # first-order mean/variance updates. No arbitrary reward table is fitted.
    alpha, beta = 1.0, 2.0
    coefficient = alpha**2 * np.exp(-alpha**2 / 2) / (beta**2 * np.exp(-beta**2 / 2))
    offset = np.exp(-alpha**2 / 2) - coefficient * np.exp(-beta**2 / 2)
    scale = 0.5 / (1 + coefficient + abs(offset))

    def advantage(z):
        return scale * (np.cos(alpha * z) - coefficient * np.cos(beta * z) - offset)

    nodes, weights = hermgauss(160)
    nodes, weights = nodes * np.sqrt(2), weights / np.sqrt(np.pi)
    angles, angle_weights = leggauss(160)
    angles = (angles + 1) * np.pi / 4
    angle_weights *= np.pi / 4
    sine, cosine = np.sin(angles), np.cos(angles)

    def velocity_and_derivative(z):
        z = np.asarray(z)
        v, dv = np.zeros_like(z), np.zeros_like(z)
        for frequency, amplitude in [(alpha, scale), (beta, -scale * coefficient)]:
            attenuation = np.exp(-0.5 * frequency**2 * cosine**2)
            phase = frequency * z[..., None] * sine
            v += np.sum(angle_weights * (-amplitude * frequency * cosine * attenuation) * np.sin(phase), axis=-1)
            dv += np.sum(angle_weights * (-amplitude * frequency**2 * cosine * sine * attenuation) * np.cos(phase), axis=-1)
        return v, dv

    grid = np.linspace(-5, 5, 101)
    v, dv = velocity_and_derivative(grid)
    stein_error = np.max(np.abs(grid * v - dv - advantage(grid)))

    # Conditional mean of the actual regression target, integrating out its
    # independent Gaussian variable and angle. This checks the reversed
    # Gaussian coupling used to train from already observed (z, return) pairs.
    numerical_v = []
    for z in grid[::10]:
        endpoint = sine[:, None] * z + cosine[:, None] * nodes[None, :]
        integrand = np.sum(weights[None, :] * advantage(endpoint) * nodes[None, :], axis=1)
        numerical_v.append(np.sum(angle_weights * integrand))
    target_error = np.max(np.abs(np.asarray(numerical_v) - velocity_and_derivative(grid[::10])[0]))

    a = advantage(nodes)
    variance = float(weights @ (a * a))
    value = lambda z: 0.5 + advantage(z)
    v_nodes = velocity_and_derivative(nodes)[0]
    baseline_value = float(weights @ value(nodes))
    steps = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    improvements = [
        {"step": step, "value": float(weights @ value(nodes + step * v_nodes)),
         "finite_difference_slope": float(weights @ (value(nodes + step * v_nodes) - value(nodes)) / step)}
        for step in steps
    ]
    # Values can be uniformly close while their derivatives are arbitrarily
    # poor. This is a conceptual contrast to differentiating a fitted critic,
    # not evidence that any particular existing critic has this error.
    noise_amplitude, frequency = 0.005, 1000.0
    result = {
        "scope": "Deterministic one-dimensional population quadrature; no learned VLA, simulator, or performance result.",
        "regression_identity_max_abs_error": float(target_error),
        "gaussian_stein_equation_max_abs_error": float(stein_error),
        "centered_advantage_mean": float(weights @ a),
        "gaussian_mean_score_gradient": float(weights @ (a * nodes)),
        "gaussian_log_std_score_gradient": float(weights @ (a * (nodes**2 - 1))),
        "exact_distribution_improvement_derivative": variance,
        "baseline_value": baseline_value,
        "finite_step_results": improvements,
        "illustrative_critic_error": {"max_value_error": noise_amplitude, "max_gradient_error": noise_amplitude * frequency,
                                      "note": "e(z)=epsilon*sin(omega*z); not measured on project critics."},
        "limitations": ["Gaussian-policy comparison is local and does not cover RWFM, NPO or other expressive policies.",
                        "No finite-sample, high-dimensional field-learning or distillation guarantee is established.",
                        "No claim of mathematical novelty for the OU/Stein representation."]
    }
    assert stein_error < 1e-10 and target_error < 1e-10
    assert abs(result["gaussian_mean_score_gradient"]) < 1e-12
    assert abs(result["gaussian_log_std_score_gradient"]) < 1e-12
    assert abs(improvements[0]["finite_difference_slope"] - variance) < 1e-6
    path = Path(__file__).with_name("reward_transport_regression_math_check_20261020.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

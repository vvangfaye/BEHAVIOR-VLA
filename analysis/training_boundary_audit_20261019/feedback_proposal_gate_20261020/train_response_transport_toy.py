"""Small deterministic optimization experiment; no robot or neural-model claims.

Plant x_next = x + a + a^2. The policy tracks a moving nominal reference
r_next = r + b + b^2, with a = b + k * (x-r).
The demonstrated nominal command changes from b=0 to b=2. Observations at
x=r identify b, but leave k unidentifiable from imitation alone.
"""
import json
import math
from pathlib import Path

EPS = 0.01
TARGET_B = 2.0
PARENT_K = -0.5


def objective(b, k, method):
    fit = (b - TARGET_B) ** 2
    if method == "bc":
        return fit
    if method == "action_derivative_matching":
        return fit + (k - PARENT_K) ** 2
    # Mean squared difference of centered physical responses / EPS, over
    # delta = +/- EPS. Parent linear gain is 0.5, quadratic coefficient 0.25.
    linear_residual = 0.5 + (1 + 2 * b) * k
    curvature_residual = k * k - 0.25
    return fit + linear_residual ** 2 + EPS ** 2 * curvature_residual ** 2


def gradient(b, k, method):
    if method == "bc":
        return 2 * (b - TARGET_B), 0.0
    if method == "action_derivative_matching":
        return 2 * (b - TARGET_B), 2 * (k - PARENT_K)
    z = 0.5 + (1 + 2 * b) * k
    return (2 * (b - TARGET_B) + 4 * k * z,
            2 * (1 + 2 * b) * z + 4 * EPS ** 2 * k * (k * k - 0.25))


def error_trajectory(b, k, initial_error, horizon=10):
    # Derived by subtracting the moving nominal reference after each step.
    error = initial_error
    trace = [error]
    for _ in range(horizon):
        error = (1 + (1 + 2 * b) * k) * error + k * k * error * error
        trace.append(error)
    return trace


def main():
    methods = ("bc", "action_derivative_matching", "physical_response_matching")
    numerical_errors = []
    for method in methods:
        for b, k in ((0.0, -0.5), (0.9, -0.2), (2.0, -0.1)):
            analytic = gradient(b, k, method)
            h = 1e-6
            numerical = ((objective(b+h, k, method)-objective(b-h, k, method))/(2*h),
                         (objective(b, k+h, method)-objective(b, k-h, method))/(2*h))
            numerical_errors.extend(abs(x-y) for x,y in zip(analytic, numerical))
    assert max(numerical_errors) < 1e-7
    results = []
    # Three step sizes check that the positive example is not one tuned run.
    for lr in (0.005, 0.01, 0.02):
        for method in methods:
            b, k = 0.0, PARENT_K
            for _ in range(20000):
                db, dk = gradient(b, k, method)
                b, k = b-lr*db, k-lr*dk
                assert math.isfinite(b) and math.isfinite(k)
            trace = error_trajectory(b, k, 0.01)
            result = {"method": method, "learning_rate": lr, "steps": 20000,
                      "nominal_command": b, "feedback_gain": k,
                      "nominal_fit_squared_error": (b-TARGET_B)**2,
                      "linearized_error_multiplier": 1+(1+2*b)*k,
                      "error_after_10_steps": trace[-1], "error_trace": trace,
                      "final_objective": objective(b, k, method)}
            results.append(result)
            assert result["nominal_fit_squared_error"] < 1e-10
            if method == "physical_response_matching":
                assert abs(result["linearized_error_multiplier"] - 0.5) < 1e-5
                assert abs(trace[-1]) < 1e-5
            else:
                assert abs(result["linearized_error_multiplier"] + 1.5) < 1e-8
                assert abs(trace[-1]) > 0.1
    report = {
        "kind": "deterministic_scalar_training_prototype",
        "plant": "x_next = x + a + a^2",
        "policy": "a = b + k*(x-r); r_next = r+b+b^2",
        "training_data": "Nominal observations x=r, target action b=2; no perturbed expert actions.",
        "feedback_signal": "Known analytic physical transition; parent perturbation response at b=0,k=-0.5.",
        "maximum_gradient_check_error": max(numerical_errors),
        "results": results,
        "what_it_establishes": "The proposed physical-response loss has a simple trainable positive example where nominal adaptation and preserved action derivatives conflict, without requiring corrective action labels.",
        "not_established": [
            "Any new algorithmic novelty: physical stability losses and derivative matching have established prior art.",
            "A benefit over Stable-BC, optimal control or ordinary RL with a suitable dense objective.",
            "A differentiable or sufficiently accurate learned dynamics model for actual OmniGibson contact states.",
            "Useful parent response labels, computational efficiency, or closed-loop VLA gains.",
            "Suitability of a uniform response-retention penalty when a task requires different feedback."
        ],
        "new_gpu_jobs": 0,
        "new_training_of_vla": False,
    }
    path = Path(__file__).with_suffix(".json")
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"gradient_check_error": max(numerical_errors),
                      "results": [{k:v for k,v in x.items() if k != "error_trace"} for x in results]}, indent=2))


if __name__ == "__main__":
    main()

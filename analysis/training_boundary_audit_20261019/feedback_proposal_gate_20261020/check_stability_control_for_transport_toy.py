"""Strong simple control for the scalar transport toy, not a Stable-BC replica."""
import json
from pathlib import Path
import train_response_transport_toy as toy


def objective(b, k):
    rho = 1 + (1 + 2 * b) * k
    excess = max(0.0, abs(rho) - 0.5)
    return (b - toy.TARGET_B) ** 2 + excess ** 2


def gradient(b, k):
    rho = 1 + (1 + 2 * b) * k
    excess = max(0.0, abs(rho) - 0.5)
    sign = 1 if rho >= 0 else -1
    return (2 * (b - toy.TARGET_B) + 4 * excess * sign * k,
            2 * excess * sign * (1 + 2 * b))


def main():
    errors = []
    for b, k in ((0.1, -0.5), (0.9, -0.2), (2, -0.15), (1.5, -0.8)):
        h = 1e-6
        numeric = ((objective(b+h,k)-objective(b-h,k))/(2*h),
                   (objective(b,k+h)-objective(b,k-h))/(2*h))
        errors.extend(abs(x-y) for x,y in zip(numeric,gradient(b,k)))
    assert max(errors) < 1e-7
    rows = []
    for lr in (0.005, 0.01, 0.02):
        b, k = 0.0, toy.PARENT_K
        for _ in range(20000):
            db, dk = gradient(b, k)
            b, k = b-lr*db, k-lr*dk
        trace = toy.error_trajectory(b, k, 0.01)
        row = {"learning_rate": lr, "nominal_command": b, "feedback_gain": k,
               "nominal_fit_squared_error": (b-toy.TARGET_B)**2,
               "linearized_error_multiplier": 1+(1+2*b)*k,
               "error_after_10_steps": trace[-1]}
        rows.append(row)
        assert row["nominal_fit_squared_error"] < 1e-10
        assert abs(row["linearized_error_multiplier"]) <= 0.500001
        assert abs(trace[-1]) < 2e-5
    result = {"kind": "analytic_toy_prior_art_control", "gradient_check_error": max(errors),
              "method": "Imitation plus squared hinge on absolute discrete-time error multiplier above 0.5.",
              "results": rows,
              "interpretation": "A generic stability penalty also solves this constructed problem without inheriting the parent response. The transport toy distinguishes physical response matching from action derivative matching, but does not establish novelty or necessity relative to stability-based imitation.",
              "limitation": "This is a simple scalar discrete-time control, not a reproduction of the full published Stable-BC method and not an empirical VLA result."}
    with Path(__file__).with_suffix('.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

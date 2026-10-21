#!/usr/bin/env python3
"""Check a proposed response-training objective on the EXISTING hybrid toy.

This is a standard-library mathematical check, not a VLA/physics experiment.
The plant and event-sensitivity phenomenon come from the Oct 20 audit. The
new check fits a state-feedback coefficient using cached plant secants and
checks the result with the original event-resolved endpoint function.
"""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "training_boundary_audit_20261019/hybrid_contact_probe_20261020/hybrid_contact_counterexample.py"
spec = importlib.util.spec_from_file_location("existing_hybrid_toy", SOURCE)
plant = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(plant)

X0, Y0, U0, T = Q(-1, 100), Q(-1), Q(1, 100), Q(2)


def endpoint(x, u):
    state, hit_time, hit = plant.constant_endpoint(x, Y0, u, T)
    assert hit and 0 < hit_time < T
    return state[1]


def true_risk(k):
    rows = []
    for dx in (Q(-1, 1000), Q(-1, 10000), Q(1, 10000), Q(1, 1000)):
        x = X0 + dx
        u = U0 + Q(str(k)) * dx
        error = endpoint(x, u)
        rows.append({"dx": float(dx), "u": float(u), "endpoint_error": float(error),
                     "absolute_gain": abs(float(error / dx))})
    return {"rows": rows, "max_absolute_gain": max(r["absolute_gain"] for r in rows)}


def check():
    state_step, action_step = Q(1, 100000), Q(1, 1000000)
    a = (endpoint(X0 + state_step, U0) - endpoint(X0 - state_step, U0)) / (2 * state_step)
    b = (endpoint(X0, U0 + action_step) - endpoint(X0, U0 - action_step)) / (2 * action_step)
    assert a == -1 / U0
    exact_b = X0 / (U0 * U0)
    assert abs(float(b - exact_b)) < 2e-6

    # u_k(x)=u0+k(x-x0). BC at x0 cannot distinguish these controllers.
    # Cached local endpoint difference is (A+B*k)*dx.
    # U0 normalizes the toy's sensitivity units; it is fixed during fitting.
    af, bf, scale = float(a), float(b), float(U0)

    def loss(k):
        return (scale * (af + bf * k)) ** 2

    def gradient(k):
        return 2 * scale * scale * bf * (af + bf * k)

    eps = 1e-6
    for k in (0.0, -0.4, -1.3):
        numeric = (loss(k + eps) - loss(k - eps)) / (2 * eps)
        assert abs(numeric - gradient(k)) < 1e-8

    k = 0.0
    for _ in range(64):
        k -= 0.2 * gradient(k)
    expected = -af / bf
    assert abs(k - expected) < 1e-12
    assert endpoint(X0, U0) == 0
    baseline, trained = true_risk(0.0), true_risk(k)
    assert baseline["max_absolute_gain"] == 100.0
    assert trained["max_absolute_gain"] < 2e-6

    # Independently verify the exact limiting controller, using rational
    # arithmetic through the ORIGINAL event-resolved plant.
    for dx in (Q(-1, 1000), Q(-1, 10000), Q(1, 10000), Q(1, 1000)):
        assert endpoint(X0 + dx, U0 - dx) == 0

    return {
        "scope": "Mathematical objective/gradient check on an existing idealized hybrid system only",
        "plant_source": str(SOURCE),
        "plant_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "new_vla_training": False, "new_simulator_episodes": 0, "new_gpu_jobs": 0,
        "cached_A": af, "cached_B": bf, "analytic_B": float(exact_b),
        "objective": "[u0*(A+B*k)]^2; nominal BC error is identically zero for every k",
        "fitted_k": k, "secant_optimal_k": expected, "analytic_optimal_k": -1.0,
        "baseline": baseline, "fitted": trained,
        "analytic_controller_endpoint_errors_exactly_zero": True,
        "nominal_action_error_all_controllers": 0.0,
        "fixed_event_time_proxy": {
            "description": "If the event time is artificially frozen, this toy's terminal y has A=B=0.",
            "response_gradient": 0.0,
            "caution": "This is a specific incorrect proxy, not a claim that every Stable-BC implementation does this."
        },
        "limits": [
            "Event-time sensitivity and stabilizing feedback are established control concepts.",
            "The check does not establish novelty, native VLA trainability, data coverage or closed-loop VLA gains.",
            "The idealized absorbing-contact model is not validated as an OmniGibson contact model.",
            "A/B linearization is local; changing event order, missing an event or large action changes require separate handling."
        ]
    }


if __name__ == "__main__":
    result = check()
    target = HERE / "response_objective_check.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": str(target), "fitted_k": result["fitted_k"],
                      "baseline_gain": result["baseline"]["max_absolute_gain"],
                      "fitted_gain": result["fitted"]["max_absolute_gain"]}, indent=2))

#!/usr/bin/env python3
"""Exact checks of the response objective's interpretation and cache limits.

Standard-library arithmetic only. No VLA, GPU, simulator or training run.
"""
from fractions import Fraction as Q
import json
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, x):
    return [dot(row, x) for row in matrix]


def norm2(x):
    return dot(x, x)


def check_equivalence():
    # Rank-deficient response: the third action coordinate is unobservable.
    c = [[Q(1), Q(2), Q(0)], [Q(2), Q(4), Q(0)]]
    b = [Q(3), Q(7)]
    target = [Q(-17, 25), Q(-34, 25), Q(0)]
    residual = [x + y for x, y in zip(matvec(c, target), b)]
    assert residual == [Q(-2, 5), Q(1, 5)]
    assert norm2(residual) == Q(1, 5)
    for x in ([Q(0)] * 3, [Q(1), Q(-2), Q(100)], target,
              [Q(-17, 25), Q(-34, 25), Q(-89)]):
        direct = norm2([z + y for z, y in zip(matvec(c, x), b)])
        weighted_label = norm2(matvec(c, [z - y for z, y in zip(x, target)]))
        assert direct == weighted_label + norm2(residual)
    return {
        "identity": "||C x+b||^2 = ||C(x-x*)||^2 + ||(I-C C^+)b||^2, x*=-C^+ b",
        "C": [[float(z) for z in row] for row in c],
        "b": list(map(float, b)), "minimum_norm_target": list(map(float, target)),
        "irreducible_residual_squared": float(norm2(residual)),
        "meaning": "A weighted corrective action-difference target plus a constant; nullspace actions remain unconstrained.",
        "caution": "This is not equality to every published CCIL algorithm or to full pointwise action labels."
    }


def check_stale_cache():
    # Smooth plant, no event ambiguity: F(s,a)=s*(1+kappa*a)+a.
    # At anchor (s*,a*)=(0,0), A=B=1 for every kappa.
    # Policy a(s)=m-s has cached paired response identically zero, but its
    # true paired response at +/-epsilon is 2*epsilon*kappa*m.
    epsilon, kappa = Q(1, 10000), Q(1000)
    rows = []
    for m in (Q(0), Q(1, 1000), Q(1, 500)):
        def policy(s):
            return m - s

        def plant(s, a):
            return s * (1 + kappa * a) + a

        ap, am = policy(epsilon), policy(-epsilon)
        cached_difference = 2 * epsilon + ap - am
        true_difference = plant(epsilon, ap) - plant(-epsilon, am)
        true_gain = true_difference / (2 * epsilon)
        assert cached_difference == 0
        assert true_gain == kappa * m
        rows.append({"mean_action_shift": float(m), "nominal_action_squared_error": float(m*m),
                     "cached_gain": 0.0, "true_paired_gain": float(true_gain)})
    return {
        "plant": "F(s,a)=s*(1+1000*a)+a", "anchor": [0, 0],
        "policy": "a(s)=m-s", "rows": rows,
        "meaning": "Small nominal action error and small input perturbations do not alone validate a stale A/B cache.",
        "necessary_condition": "A bound on response variation with action/state, or refreshed/independently validated local responses, is needed.",
        "caution": "Analytic counterexample, not a measured OmniGibson or VLA failure."
    }


if __name__ == "__main__":
    result = {"scope": "Exact mathematical interpretation and counterexample checks",
              "new_model_or_simulator_runs": 0,
              "equivalence": check_equivalence(), "cache_limit": check_stale_cache()}
    target = Path(__file__).with_suffix(".json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

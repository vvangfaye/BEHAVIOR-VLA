"""Exact finite-state check of an SMC unnormalized path-score estimator.

This checks a candidate estimator, not a VLA, simulator, or novelty claim.
Run with the project's numpy-enabled research Python.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def bernoulli_probability(bits, probabilities):
    return float(np.prod([p if b else 1 - p for b, p in zip(bits, probabilities)]))


def enumerate_particles(p, q, potential, particles=2):
    """Two decisions: first action u, second v; success iff v == u.

    Resample after u with an arbitrary positive potential H(u). Retain the
    ancestor's score and cancel H in the terminal importance weight.
    Parameter coordinates are the three policy logits (p, q[0], q[1]).
    """
    mass = 0.0
    mean = np.zeros(4)
    second = np.zeros(4)
    naive = np.zeros(3)
    for first in itertools.product((0, 1), repeat=particles):
        probability_first = bernoulli_probability(first, [p] * particles)
        h = np.array([potential[u] for u in first])
        z1 = h.mean()
        alpha = h / h.sum()
        for ancestors in itertools.product(range(particles), repeat=particles):
            probability_ancestors = float(np.prod(alpha[list(ancestors)]))
            states = [first[i] for i in ancestors]
            for last in itertools.product((0, 1), repeat=particles):
                probability = (probability_first * probability_ancestors
                               * bernoulli_probability(last, [q[u] for u in states]))
                weights, scores = [], []
                for u, v in zip(states, last):
                    weights.append(float(u == v) / potential[u])
                    score = np.array([u - p, 0.0, 0.0])
                    score[1 + u] = v - q[u]
                    scores.append(score)
                weights = np.array(weights)
                scores = np.array(scores)
                estimate = np.r_[z1 * weights.mean(),
                                 z1 * (weights[:, None] * scores).mean(axis=0)]
                mass += probability
                mean += probability * estimate
                second += probability * estimate ** 2
                if weights.sum() > 0:
                    naive += probability * (weights[:, None] * scores).sum(axis=0) / weights.sum()
    truth = np.array([(1-p)*(1-q[0]) + p*q[1],
                      p*(1-p)*(q[1]+q[0]-1),
                      -(1-p)*q[0]*(1-q[0]), p*q[1]*(1-q[1])])
    assert abs(mass - 1) < 1e-12
    assert np.max(np.abs(mean-truth)) < 1e-12
    return {"particles": particles, "potential": list(potential),
            "reference_value_and_logit_gradient": truth.tolist(),
            "exact_expected_estimator": mean.tolist(),
            "maximum_absolute_bias": float(np.max(np.abs(mean-truth))),
            "estimator_variance": (second-mean**2).tolist(),
            "normalizer_dropped_expected_gradient": naive.tolist(),
            "normalizer_dropped_gradient_bias": (naive-truth[1:]).tolist()}


def absorbing_chain(p=0.7, horizon=40, particles=32):
    """Exact moments for a special chain with known absorbing failure.

    Each stage succeeds independently with probability p. Splitting retains
    successful particles at each level. This oracle level structure does not
    establish a variance improvement for general manipulation problems.
    """
    success = p ** horizon
    coefficient = horizon * (1-p)
    splitting_second = (p*p+p*(1-p)/particles) ** horizon
    splitting_var = coefficient**2 * (splitting_second-success**2)
    mc_var = coefficient**2 * success * (1-success) / particles
    # Independent trajectories can stop after first failure. This is only
    # an expected-work adjustment, not an exact hard-budget experiment.
    expected_mc_steps = (1-p**horizon)/(1-p)
    equivalent_mc_episodes = particles*horizon/expected_mc_steps
    work_adjusted_mc_var = coefficient**2*success*(1-success)/equivalent_mc_episodes
    return {"p": p, "horizon": horizon, "particles": particles,
            "true_success_probability": success,
            "true_logit_gradient": coefficient*success,
            "splitting_gradient_variance": splitting_var,
            "mc_gradient_variance_equal_trajectory_slots": mc_var,
            "variance_ratio_equal_trajectory_slots": mc_var/splitting_var,
            "expected_work_adjusted_mc_gradient_variance": work_adjusted_mc_var,
            "variance_ratio_expected_control_steps": work_adjusted_mc_var/splitting_var,
            "scope": "Special known absorbing chain; no empirical VLA claim."}


if __name__ == "__main__":
    result = {"classification": "exact_algorithm_check_only",
              "two_stage_checks": [enumerate_particles(.4, (.3, .8), h, n)
                                   for h in ((1., 1.), (.1, 3.), (2., .2))
                                   for n in (2, 3)],
              "absorbing_chain": absorbing_chain(),
              "vla_training_or_simulator_calls": 0}
    path = Path(__file__).with_name("particle_credit_check.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path),
                      "maximum_bias": max(x["maximum_absolute_bias"] for x in result["two_stage_checks"]),
                      "chain": result["absorbing_chain"]}, indent=2))

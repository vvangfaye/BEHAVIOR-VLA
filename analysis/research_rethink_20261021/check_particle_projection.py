"""Exact check of a likelihood-free, weighted endpoint-projection target.

Two-decision finite MDP; no VLA training or simulator interaction. The binary
energy score is evaluated analytically, which allows exact enumeration rather
than a favourable random seed to validate the particle-weight construction.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def path_payload(u, v, learner):
    # Expected energy score of Bernoulli(r) against observed binary y is
    # r^2 - 2*r*y + y, with its derivative in the learner's logit coordinate.
    loss = 0.0
    grad = np.zeros(3)
    for coordinate, target in ((0, u), (1+u, v)):
        r = learner[coordinate]
        loss += r*r - 2*r*target + target
        grad[coordinate] += 2*(r-target)*r*(1-r)
    return np.r_[loss, grad]


def probability(bits, probs):
    return float(np.prod([p if b else 1-p for b, p in zip(bits, probs)]))


def check(potential, count, old, learner):
    p, *q = old
    reference = sum(probability((u, v), (p, q[u])) * (u == v)
                    * path_payload(u, v, learner)
                    for u, v in itertools.product((0, 1), repeat=2))
    expected = np.zeros(4)
    mass = 0.0
    for first in itertools.product((0, 1), repeat=count):
        prob_first = probability(first, [p]*count)
        h = np.array([potential[u] for u in first])
        z1 = h.mean()
        alpha = h / h.sum()
        for ancestor in itertools.product(range(count), repeat=count):
            prob_ancestor = float(np.prod(alpha[list(ancestor)]))
            states = [first[i] for i in ancestor]
            for last in itertools.product((0, 1), repeat=count):
                prob = prob_first*prob_ancestor*probability(last, [q[u] for u in states])
                estimate = z1/count*sum((u == v)/potential[u]*path_payload(u, v, learner)
                                         for u, v in zip(states, last))
                expected += prob*estimate
                mass += prob
    assert abs(mass-1) < 1e-12
    assert np.max(np.abs(reference-expected)) < 1e-12
    return dict(potential=list(potential), particles=count,
                reference_loss_and_gradient=reference.tolist(),
                expected_particle_loss_and_gradient=expected.tolist(),
                maximum_absolute_bias=float(np.max(np.abs(reference-expected))))


def shared_parameter_counterexample():
    # Two contexts share one logit shift theta. R=action at context 0 and
    # R=1-action at context 1. This finite-capacity policy cannot realize the
    # unconstrained posterior; a proper projection loss is not a PG estimator.
    state_probability = np.array([.8, .2])
    p = np.array([.1, .5])
    direction = np.array([1., -1.])
    true_gradient = float(np.sum(state_probability * direction*p*(1-p)))
    negative_half_projection_gradient = float(np.sum(state_probability*direction*(p*(1-p))**2))
    assert true_gradient > 0 and negative_half_projection_gradient < 0
    return dict(true_return_gradient=true_gradient,
                negative_half_projection_loss_gradient=negative_half_projection_gradient,
                inference="Finite-step projection can oppose return improvement in a shared, misspecified policy class.")


if __name__ == "__main__":
    old = (.4, .3, .8)
    learner = (.2, .5, .7)
    old_success = (1-old[0])*(1-old[1])+old[0]*old[2]
    posterior = (old[0]*old[2]/old_success, 0., 1.)
    result = dict(
        classification="exact_surrogate_target_check_not_a_vla_result",
        checks=[check(h, n, old, learner) for h in ((1., 1.), (.1, 3.), (2., .2)) for n in (2, 3)],
        old_success=old_success,
        unconstrained_success_conditioned_action_probabilities=posterior,
        unconstrained_projected_success=(1-posterior[0])*(1-posterior[1])+posterior[0]*posterior[2],
        finite_parameter_limit=shared_parameter_counterexample(),
        new_model_calls=0, new_simulator_calls=0,
    )
    output = Path(__file__).with_name("particle_projection_check.json")
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(output=str(output),
                         maximum_bias=max(c["maximum_absolute_bias"] for c in result["checks"]),
                         finite_parameter_limit=result["finite_parameter_limit"]), indent=2))

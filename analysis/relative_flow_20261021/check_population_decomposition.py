"""Check the exact population meaning of the paired objective in one dimension."""
import json
from pathlib import Path

import numpy as np


def normal(x, mean, variance):
    return np.exp(-0.5 * np.square(x - mean) / variance) / np.sqrt(2 * np.pi * variance)


def main():
    # Unequal context probabilities and multimodal, unequal-variance endpoints.
    contexts = [(0.3, np.array([-1.0, 2.0]), np.array([0.8, 0.2])),
                (0.7, np.array([-2.0, 0.5, 3.0]), np.array([0.2, 0.5, 0.3]))]
    rows = []
    for t in [0.4, 0.9, 0.95, 0.99]:
        worst = 0.0
        shared_shift_error = 0.0
        positive_excess = 0.0
        for x in np.linspace(-3, 3, 41):
            densities, conditional_mean, conditional_variance = [], [], []
            for prior, a, probability in contexts:
                joint = probability * normal(x, (1 - t) * a, t * t)
                posterior = joint / joint.sum()
                mu = posterior @ a
                densities.append(prior * joint.sum())
                conditional_mean.append(mu)
                conditional_variance.append(posterior @ np.square(a - mu))
            densities = np.array(densities)
            context_posterior = densities / densities.sum()
            truth = (x - np.array(conditional_mean)) / t
            error = np.array([0.23 + 0.11 * x, -0.17 + 0.03 * x])
            candidate = truth + error / t
            expected_error = context_posterior @ error
            error_variance = context_posterior @ np.square(error - expected_error)
            irreducible = context_posterior @ conditional_variance
            factor = densities.sum() ** 2 * np.sqrt(4 * np.pi * t * t)
            decomposition = factor * 2 * (error_variance + irreducible)

            def paired_loss(velocity):
                value = 0.0
                for ci, (pc, a_set, pa) in enumerate(contexts):
                    for di, (pd, b_set, pb) in enumerate(contexts):
                        for ai, a in enumerate(a_set):
                            for bi, b in enumerate(b_set):
                                w = np.exp(-(1-t)**2 * (a-b)**2 / (4*t*t))
                                q = normal(x, (1-t)*(a+b)/2, t*t/2)
                                residual = t*(velocity[ci]-velocity[di])-(b-a)
                                value += pc*pd*pa[ai]*pb[bi]*w*q*residual**2
                return value

            actual = paired_loss(candidate)
            bayes = paired_loss(truth)
            worst = max(worst, abs(actual-decomposition))
            shared_shift_error = max(shared_shift_error,
                                     abs(actual-paired_loss(candidate+0.4+0.2*x)))
            positive_excess = max(positive_excess, actual-bayes)
            assert abs(actual-bayes-factor*2*error_variance) < 1e-11
        assert worst < 1e-11 and shared_shift_error < 1e-11
        rows.append(dict(time=t, max_decomposition_error=worst,
                         max_common_field_shift_error=shared_shift_error,
                         maximum_context_error_excess=positive_excess))
    result = dict(passed=True, rows=rows,
                  conclusion="The population auxiliary excess is an overlap-density-weighted variance of conditional flow errors, with the conditional mean error unidentified.",
                  implications=[
                      "Ordinary flow matching must identify the missing common field.",
                      "The objective is not a new source of information absent from the original demonstrations.",
                      "A potential contribution must demonstrate finite-data or optimization benefits of centering conditional errors.",
                      "Pairing only within a task centers within that task, not across all tasks."],
                  limits="Discrete one-dimensional exact enumeration; no claim of VLA benefit or independent methodological novelty.",
                  new_training_runs=0, new_simulator_episodes=0)
    path = Path(__file__).with_name("population_decomposition.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

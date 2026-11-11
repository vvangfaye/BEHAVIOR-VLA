"""Counterexamples for deriving new feedback information from point labels.

This is a synthetic algebra audit, not a model or simulator experiment.
"""
import json
import math
from pathlib import Path


def main():
    # For unordered weighted pairs, relative-action matching is a graph
    # Laplacian norm of pointwise BC residuals, not new target information.
    errors = [[0.3, -0.2], [-0.4, 0.7], [0.1, 0.9], [0.5, -0.3]]
    edges = [(0, 1, 0.5), (0, 3, 1.2), (1, 2, 0.8), (2, 3, 2.1)]
    pair_loss = sum(w * sum((a-b)**2 for a, b in zip(errors[i], errors[j]))
                    for i, j, w in edges)
    laplacian = [[0.0]*len(errors) for _ in errors]
    for i, j, w in edges:
        laplacian[i][i] += w
        laplacian[j][j] += w
        laplacian[i][j] -= w
        laplacian[j][i] -= w
    trace = sum(laplacian[i][j] * sum(a*b for a, b in zip(errors[i], errors[j]))
                for i in range(len(errors)) for j in range(len(errors)))
    assert math.isclose(pair_loss, trace, rel_tol=1e-14, abs_tol=1e-14)

    # All demonstrations lie on y=0. Dynamics are xdot=u. Stable and unstable
    # policies have identical action values and every pairwise action difference.
    states = [(i/8, 0.0) for i in range(-8, 9)]
    def policy(s, gain):
        return (-s[0], gain*s[1])
    observed = [policy(s, -1.0) for s in states]
    invisible_gains = [-3.0, -1.0, 0.0, 1.0, 3.0]
    candidates = []
    for gain in invisible_gains:
        actions = [policy(s, gain) for s in states]
        point_loss = sum(sum((a-b)**2 for a, b in zip(p, q))
                         for p, q in zip(actions, observed))
        relative_loss = sum(sum(((actions[i][k]-actions[j][k])-
                                 (observed[i][k]-observed[j][k]))**2 for k in range(2))
                            for i in range(len(states)) for j in range(i))
        assert point_loss == relative_loss == 0.0
        candidates.append({'transverse_feedback_gain': gain,
                           'point_label_loss': point_loss,
                           'all_pair_difference_loss': relative_loss,
                           'continuous_time_transverse_error_multiplier_after_3s': math.exp(3*gain),
                           'locally_asymptotically_stable': gain < 0})
    result = {'kind': 'synthetic_algebra_counterexample',
              'pair_loss': pair_loss, 'laplacian_quadratic_form': trace,
              'absolute_identity_error': abs(pair_loss-trace),
              'policies': candidates,
              'conclusion': 'Re-pairing the same state/action point labels can change weighting or inductive bias but cannot distinguish these feedback laws. New information or an explicit prior is necessary.',
              'limits': 'A rank-deficient synthetic dataset only. Does not establish low-rank support or feedback damage in the actual VLA dataset. No novelty claim for this counterexample.'}
    out = Path(__file__).with_suffix('.json')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'identity_error': result['absolute_identity_error'],
                      'indistinguishable_policy_gains': invisible_gains,
                      'status': 'passed'}))


if __name__ == '__main__':
    main()

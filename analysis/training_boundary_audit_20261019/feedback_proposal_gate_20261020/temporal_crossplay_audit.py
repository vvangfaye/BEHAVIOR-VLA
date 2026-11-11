"""Analytic counterexamples for interpreting a 2x2 policy crossover.

All numbers are constructed probabilities, not robot results.
Rows select the prefix policy; columns select the continuation policy.
"""
import json
from pathlib import Path


def matrix(q, r):
    return [[qi * rj for rj in r] for qi in q]


def contrast(m):
    return m[1][1] - m[1][0] - m[0][1] + m[0][0]


def determinant(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def main():
    q, r = [.4, .8], [.5, .9]
    independent = matrix(q, r)
    assert abs(contrast(independent) - (q[1]-q[0])*(r[1]-r[0])) < 1e-12
    assert abs(contrast(independent)) > .1
    assert abs(determinant(independent)) < 1e-12

    case_a = matrix([.9, .6], [.9, .6])
    case_b = matrix([.2, .7], [.3, .8])
    pooled = [[(case_a[i][j]+case_b[i][j])/2 for j in [0, 1]] for i in [0, 1]]
    assert abs(determinant(case_a)) < 1e-12 and abs(determinant(case_b)) < 1e-12
    assert abs(determinant(pooled)) > .05

    # Both policies attain the maximum possible reward from the initial state.
    # A chooses left then action 0; B chooses right then action 1.
    # Left requires action 0 and right requires action 1. Each continuation
    # uses a fixed action; the state itself is fully observed.
    routes = ['left', 'right']
    final_actions = [0, 1]
    required = {'left': 0, 'right': 1}
    crossplay = [[int(final_actions[j] == required[routes[i]]) for j in [0, 1]] for i in [0, 1]]
    assert crossplay == [[1, 0], [0, 1]]

    result = {
        'status': 'complete', 'evidence_type': 'analytic examples only, not VLA measurements',
        'matrix_convention': 'rows=prefix A/B, columns=continuation A/B',
        'additive_interaction_under_independence': {
            'prefix_success': q, 'continuation_success': r, 'return_matrix': independent,
            'additive_interaction': contrast(independent), 'multiplicative_determinant': determinant(independent),
            'identity': 'I = E_x[(q_B(x)-q_A(x))*(r_B(x)-r_A(x))] under conditional product success',
            'consequence': 'A nonzero additive crossover interaction does not reject independent success-rate multiplication.'
        },
        'heterogeneity_can_break_pooled_product_form': {
            'conditional_matrices': [case_a, case_b], 'pooled_matrix': pooled,
            'conditional_determinants': [determinant(case_a), determinant(case_b)],
            'pooled_determinant': determinant(pooled),
            'consequence': 'A nonzero determinant of the pooled return table does not establish within-instance coupling.'
        },
        'crossplay_failure_without_self_play_degradation': {
            'fully_observed': True, 'return_matrix': crossplay,
            'diagonal_return': 1, 'cross_return': 0,
            'consequence': 'Two policies can both be optimal from the task start while failing when mixed. Crossplay compatibility is an additional objective, not automatically evidence of better task execution.'
        },
        'design_requirements': [
            'Keep paired initial-state identity when estimating interaction; do not interpret pooled task tables as conditional effects.',
            'Report additive causal interaction as a scale-specific effect, not a unique measure of motor skill or maladaptive coadaptation.',
            'Specify what conditional stage-independence model is being rejected and estimate its components with sufficient repeats.',
            'A compatibility regularizer must improve actual self-policy performance, not merely off-diagonal success.',
            'Separate state-distribution changes from language state, cache, controller and restoration mismatches.'
        ],
        'prior_overlap': [
            {'url': 'https://arxiv.org/html/2607.16506v1', 'method': 'Foresight residual RL',
             'overlap': 'Optimizes earlier terminal states using downstream success.'},
            {'url': 'https://arxiv.org/abs/1406.5979', 'method': 'AggreVaTe',
             'overlap': 'Uses downstream cost-to-go to improve decisions.'},
            {'url': 'https://proceedings.mlr.press/v37/changb15.html', 'method': 'LOLS',
             'overlap': 'Learner/reference rollout-based policy improvement.'}
        ],
        'decision': 'Keep crossover as a controlled diagnostic; do not present temporal crossplay regularization as an established novel training paradigm.',
        'goal_complete': False,
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({
        'independent_additive_interaction': contrast(independent),
        'pooled_determinant_despite_conditional_independence': determinant(pooled),
        'optimal_policy_crossplay_matrix': crossplay,
    }))


if __name__ == '__main__':
    main()

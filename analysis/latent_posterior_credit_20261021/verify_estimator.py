"""Exact finite-state verification of stationary posterior path averaging.

No robot, learned policy, environment rollout, or performance claim is involved.
The four internal states encode two bits; only the first bit at the final
internal step is observed/executed. The last bit is an unexecuted suffix.
"""
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(92741)
N, K, D = 4, 3, 2
INITIAL = np.array([0.1, 0.2, 0.3, 0.4])
BASE = RNG.normal(scale=0.7, size=(K, N, N))
FEATURE = RNG.normal(scale=0.9, size=(K, N, N, D))
THETA = np.array([0.23, -0.31])
PATHS = np.array(list(itertools.product(range(N), repeat=K + 1)))
ACTION = PATHS[:, -1] // 2


def joint(theta):
    logits = BASE + np.einsum('kijd,d->kij', FEATURE, theta)
    logits -= logits.max(axis=-1, keepdims=True)
    transition = np.exp(logits)
    transition /= transition.sum(axis=-1, keepdims=True)
    centered = FEATURE - np.einsum('kij,kijd->kid', transition, FEATURE)[:, :, None, :]
    p = INITIAL[PATHS[:, 0]].copy()
    s = np.zeros((len(PATHS), D))
    for k in range(K):
        src, dst = PATHS[:, k], PATHS[:, k + 1]
        p *= transition[k, src, dst]
        s += centered[k, src, dst]
    return p, s


def gibbs_kernel(paths, posterior):
    """Random-scan exact heat-bath updates, including hidden endpoint suffix."""
    size, width = paths.shape
    kernel = np.zeros((size, size))
    for site in range(width):
        other = [j for j in range(width) if j != site]
        groups = {}
        for i, path in enumerate(paths):
            groups.setdefault(tuple(path[other]), []).append(i)
        for group in groups.values():
            idx = np.array(group)
            weights = posterior[idx] / posterior[idx].sum()
            kernel[np.ix_(idx, idx)] += np.tile(weights, (len(idx), 1)) / width
    return kernel


def main():
    probability, score = joint(THETA)
    baseline = 0.3
    reward = ACTION.astype(float) - baseline
    pg = ((probability * reward)[:, None] * score).sum(axis=0)
    eps = 1e-6
    finite_difference = []
    for d in range(D):
        e = np.eye(D)[d] * eps
        finite_difference.append((joint(THETA + e)[0][ACTION == 1].sum()
                                  - joint(THETA - e)[0][ACTION == 1].sum()) / (2 * eps))
    finite_difference = np.array(finite_difference)
    conditional = []
    marginal_mean = np.zeros(D)
    marginal_second = 0.0
    conditional_trace = {m: 0.0 for m in (1, 2, 4, 8, 16, 32, 64)}
    total_second = 0.0
    bad_mean = {m: np.zeros(D) for m in (0, 1, 4, 16)}
    for action in (0, 1):
        indices = np.flatnonzero(ACTION == action)
        paths, s = PATHS[indices], score[indices]
        p_action = probability[indices].sum()
        posterior = probability[indices] / p_action
        mean = posterior @ s
        centered = s - mean
        transition = gibbs_kernel(paths, posterior)
        row_error = np.max(np.abs(transition.sum(axis=1) - 1))
        stationary_error = np.max(np.abs(posterior @ transition - posterior))
        detailed_error = np.max(np.abs(posterior[:, None] * transition
                                      - posterior[None, :] * transition.T))
        assert row_error < 1e-13 and stationary_error < 1e-13 and detailed_error < 1e-13
        r = action - baseline
        marginal_mean += p_action * r * mean
        marginal_second += p_action * r**2 * np.dot(mean, mean)
        var0 = np.einsum('i,id,id->', posterior, centered, centered)
        total_second += p_action * r**2 * (var0 + np.dot(mean, mean))
        propagated = centered.copy()
        autocov = []
        for lag in range(64):
            autocov.append(np.einsum('i,id,id->', posterior, centered, propagated))
            propagated = transition @ propagated
        for m in conditional_trace:
            variance = (m * autocov[0] + 2 * sum((m - lag) * autocov[lag]
                                                for lag in range(1, m))) / m**2
            conditional_trace[m] += p_action * r**2 * variance
        for m in bad_mean:
            bad_posterior = np.full(len(indices), 1 / len(indices))
            for _ in range(m):
                bad_posterior = bad_posterior @ transition
            bad_mean[m] += p_action * r * (bad_posterior @ s)
        conditional.append({'action': action, 'probability': float(p_action),
                            'marginal_score': mean.tolist(),
                            'score_conditional_variance_trace': float(var0),
                            'kernel_stationarity_max_error': float(stationary_error),
                            'kernel_detailed_balance_max_error': float(detailed_error)})
    mean_square = np.dot(pg, pg)
    marginal_variance = marginal_second - mean_square
    original_variance = total_second - mean_square
    assert np.max(np.abs(pg - finite_difference)) < 1e-8
    assert np.max(np.abs(pg - marginal_mean)) < 1e-13
    rows = []
    for m, v in conditional_trace.items():
        variance = marginal_variance + v
        assert variance <= original_variance + 1e-12
        rows.append({'posterior_paths_averaged': m,
                     'expected_gradient': marginal_mean.tolist(),
                     'variance_trace': float(variance),
                     'variance_relative_to_single_path': float(variance / original_variance)})
    result = {
        'scope': 'Exact discrete latent-chain algorithm check, no VLA training or robot rollouts.',
        'parameters': THETA.tolist(),
        'internal_transitions': K,
        'enumerated_paths': len(PATHS),
        'observed_endpoint': 'first endpoint bit; remaining endpoint bit is latent',
        'joint_policy_gradient': pg.tolist(),
        'finite_difference_gradient': finite_difference.tolist(),
        'gradient_max_abs_error': float(np.max(np.abs(pg - finite_difference))),
        'posterior_marginal_mean_error': float(np.max(np.abs(pg - marginal_mean))),
        'conditionals': conditional,
        'single_path_variance_trace': float(original_variance),
        'ideal_exact_marginal_variance_trace': float(marginal_variance),
        'stationary_finite_chain_results': rows,
        'wrong_uniform_initialization_negative_control': [
            {'posterior_steps': m, 'gradient': g.tolist(),
             'bias_norm': float(np.linalg.norm(g - pg))} for m, g in bad_mean.items()],
        'not_verified': ['Continuous MALA implementation', 'Actual VLA mixing or memory cost',
                         'Closed-loop success gain', 'Novelty against all existing literature'],
    }
    (ROOT / 'estimator_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

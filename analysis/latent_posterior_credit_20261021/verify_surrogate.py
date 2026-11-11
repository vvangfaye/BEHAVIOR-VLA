"""Exact check of old-posterior importance weighting at new parameters."""
import json
from pathlib import Path

import numpy as np

from verify_estimator import ACTION, THETA, PATHS, joint, gibbs_kernel


old_probability, _ = joint(THETA)
new_theta = np.array([0.8, -1.2])
new_probability, new_score = joint(new_theta)
reward = ACTION.astype(float) - 0.3
weight = new_probability / old_probability
expected = ((new_probability * reward)[:, None] * new_score).sum(0)
importance_gradient = ((old_probability * reward * weight)[:, None] * new_score).sum(0)
eps = 1e-6
finite_difference = np.array([
    (joint(new_theta + eps * np.eye(2)[d])[0][ACTION == 1].sum()
     - joint(new_theta - eps * np.eye(2)[d])[0][ACTION == 1].sum()) / (2 * eps)
    for d in range(2)])
rows = []
clipped_joint, clipped_marginal = 0., 0.
for steps in (0, 1, 4, 16):
    mean = np.zeros(2)
    for u in (0, 1):
        indices = np.flatnonzero(ACTION == u)
        p_u = old_probability[indices].sum()
        posterior = old_probability[indices] / p_u
        kernel = gibbs_kernel(PATHS[indices], posterior)
        function = weight[indices, None] * new_score[indices] * (u - 0.3)
        propagated = function.copy()
        for _ in range(steps):
            propagated = kernel @ propagated
        mean += p_u * (posterior @ propagated)
    rows.append({'posterior_steps': steps, 'mean_gradient': mean.tolist(),
                 'max_error': float(np.abs(mean - expected).max())})
for u in (0, 1):
    indices = ACTION == u
    p_old = old_probability[indices].sum()
    marginal_ratio = new_probability[indices].sum() / p_old
    # Positive advantage illustrates clipping noncommutation directly.
    clipped_joint += (old_probability[indices] * np.minimum(weight[indices], 1.2)).sum()
    clipped_marginal += p_old * min(marginal_ratio, 1.2)
assert max(r['max_error'] for r in rows) < 1e-13
assert np.max(np.abs(expected - importance_gradient)) < 1e-13
assert np.max(np.abs(expected - finite_difference)) < 1e-8
assert abs(clipped_joint - clipped_marginal) > 1e-3
result = {'scope': 'Exact finite-bandit surrogate check; not an off-policy MDP return guarantee.',
          'old_theta': THETA.tolist(), 'new_theta': new_theta.tolist(),
          'exact_target_gradient': expected.tolist(),
          'joint_importance_gradient': importance_gradient.tolist(),
          'finite_difference_gradient': finite_difference.tolist(),
          'old_posterior_invariant_kernel_results': rows,
          'clipped_joint_ratio_expectation': float(clipped_joint),
          'clipped_marginal_ratio_expectation': float(clipped_marginal),
          'nonlinear_clipping_gap': float(clipped_marginal - clipped_joint)}
Path(__file__).with_name('surrogate_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

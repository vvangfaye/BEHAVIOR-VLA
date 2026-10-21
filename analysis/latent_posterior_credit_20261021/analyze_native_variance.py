"""Frozen paired second-moment analysis for native posterior path averaging.

Bootstrap units are original model draws, not correlated posterior samples.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / 'native_variance_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_ratio(numerator, denominator, rng, repeats=4000):
    draws = rng.integers(0, len(numerator), size=(repeats, len(numerator)))
    ratios = numerator[draws].mean(1) / denominator[draws].mean(1)
    return {'ratio': float(numerator.mean() / denominator.mean()),
            'paired_bootstrap_95_percentile_interval': np.quantile(ratios, [.025, .975]).tolist()}


def main():
    completion = json.loads((ROOT / 'collection_complete.json').read_text())
    plan = json.loads((ROOT / 'plan.json').read_text())
    if not completion['complete'] or completion['plan_sha256'] != sha(ROOT / 'plan.json'):
        raise ValueError('Unverified completion or plan')
    rng = np.random.default_rng(442810)
    rows = []
    by_context = {r['context']: r for r in completion['contexts']}
    for context in plan['contexts']:
        metadata = by_context[context]
        source = ROOT / metadata['file']
        if sha(source) != metadata['sha256']:
            raise ValueError('Corrupted raw score artifact')
        with np.load(source) as archive:
            scores = archive['scores'].astype(np.float64)
            weights = archive['action_space_weights'].astype(np.float64)
            acceptance = archive['acceptance']
            observed = archive['observed']
        if scores.shape[:2] != (plan['paths_per_context'], plan['retained_paths']):
            raise ValueError('Wrong sample dimensions')
        if not np.isfinite(scores).all() or observed.shape != (len(scores), 368):
            raise ValueError('Invalid score or observed prefix')
        base_norm = np.square(scores[:, 0]).sum(-1)
        cumulative = np.cumsum(scores, axis=1)
        measurements = []
        batches = plan['paths_per_context'] / plan['batch_size']
        score_cost = metadata['score_seconds'] / (plan['paths_per_context'] * plan['retained_paths'])
        sweep_cost = metadata['sweep_seconds'] / (plan['paths_per_context'] * (plan['retained_paths'] - 1))
        for m in (1, 4, 8, 16, 32):
            mean_score = cumulative[:, m - 1] / m
            norm = np.square(mean_score).sum(-1)
            plain = paired_ratio(norm, base_norm, rng)
            weighted = paired_ratio(norm * weights**2, base_norm * weights**2, rng)
            measurements.append({'paths_averaged': m,
                                 'unweighted_score_second_moment': plain,
                                 'artificial_action_reward_second_moment': weighted,
                                 'amortized_model_seconds_per_original_draw': m * score_cost + (m - 1) * sweep_cost})
        last_raw_norm = np.square(scores[:, -1]).sum(-1)
        raw_drift = paired_ratio(last_raw_norm, base_norm, rng)
        lagged_dot = (scores[:, 0] * scores[:, -1]).sum(-1)
        rows.append({'context': context, 'original_draws': len(scores),
                     'score_dimension': scores.shape[-1], 'rows': measurements,
                     'acceptance_by_latent_node': acceptance.mean((0, 1)).tolist(),
                     'last_vs_initial_raw_score_second_moment': raw_drift,
                     'first_last_raw_score_dot_ratio': float(lagged_dot.mean() / base_norm.mean()),
                     'score_parity_relative_error': metadata['score_parity_relative_error'],
                     'readout_score_batch_seconds': score_cost * plan['batch_size'],
                     'posterior_sweep_batch_seconds': sweep_cost * plan['batch_size']})
    gate = plan['decision_gate']
    end_points = [r['rows'][-1]['unweighted_score_second_moment'] for r in rows]
    reduced = sum(v['ratio'] <= gate['per_context_mean_second_moment_ratio_at_32_max'] for v in end_points)
    significant = sum(v['paired_bootstrap_95_percentile_interval'][1] < 1 for v in end_points)
    passed = reduced == gate['contexts_required'] and significant >= gate['bootstrap_ci_upper_below_one_required']
    result = {'status': 'READOUT_VARIANCE_SCREEN_PASS' if passed else 'READOUT_VARIANCE_SCREEN_FAIL',
              'gate': gate, 'contexts_meeting_effect_size': reduced,
              'contexts_with_interval_below_one': significant, 'contexts': rows,
              'collection_elapsed_seconds': completion['elapsed_seconds'],
              'scope': 'Three fixed contexts, 16 original draws each, readout parameters only.',
              'limitations': ['No full-actor variance or closed-loop reward measured.',
                              'The nonconstant action-space weight is artificial, not task score.',
                              '32 correlated samples do not establish posterior ESS.',
                              'Bootstrap intervals are conditional on these fixed contexts.',
                              'Second moments, not finite-sample centered covariance, are reported.'],
              'plan_sha256': sha(ROOT / 'plan.json'),
              'collection_sha256': sha(ROOT / 'collection_complete.json'),
              'analyzer_sha256': sha(Path(__file__))}
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

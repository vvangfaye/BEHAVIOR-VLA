"""Summarize the frozen paired panel, resampling whole instances, not frames."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def analyze():
    path = HERE / 'retrieved/comparison.json'
    comparison = json.loads(path.read_text())
    assert comparison['comparison_valid']
    pairs = comparison['pairs']
    assert len(pairs) == 30
    grouped = {}
    for row in pairs:
        assert row['seed'] in (0, 1, 2)
        grouped.setdefault(row['instance'], []).append(row)
    assert set(grouped) == set(range(301, 311))
    assert all({r['seed'] for r in rows} == {0, 1, 2} for rows in grouped.values())
    by_instance = []
    for instance, rows in sorted(grouped.items()):
        by_instance.append({'instance': instance,
            'baseline_mean_score': float(np.mean([r['baseline']['score'] for r in rows])),
            'final_mean_score': float(np.mean([r['final']['score'] for r in rows])),
            'score_delta': float(np.mean([r['score_delta'] for r in rows])),
            'success_rate_delta': float(np.mean([r['success_delta'] for r in rows]))})
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, 10, size=(10000, 10))
    intervals = {}
    for key in ('score_delta', 'success_rate_delta'):
        values = np.asarray([r[key] for r in by_instance])
        intervals[key] = np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()
    result = {'comparison_valid': True, 'baseline': comparison['baseline'], 'final': comparison['final'],
              'mean_score_delta': comparison['mean_score_delta'],
              'success_rate_delta': comparison['success_rate_delta'],
              'success_to_failure': comparison['success_to_failure'],
              'failure_to_success': comparison['failure_to_success'],
              'instances': by_instance, 'paired_instance_bootstrap_95_percent_intervals': intervals,
              'observed_gain_on_fixed_panel': comparison['mean_score_delta'] > 0 and comparison['success_rate_delta'] > 0,
              'both_interval_lower_bounds_positive': all(v[0] > 0 for v in intervals.values()),
              'limitations': 'One task, 10 fixed instances, 3 seeds each, one training run. Bootstrap resamples instances; it does not establish robustness across training seeds or tasks.'}
    (HERE / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    active_path = HERE.parent / 'active_run.json'
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if active.get('experiment') == 'sampling32_ppo_20261025':
            active.update(comparison_complete=True, analysis_path=str(HERE/'analysis.json'))
            active_path.write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(),ensure_ascii=False))

"""Deterministic sharding and paired reporting for the frozen 17-update model."""
import json
import math
from pathlib import Path


def assigned_cases(plan, rank):
    assert rank in (0, 1)
    return plan['evaluation']['cases'][rank::2]


def schedule(plan, rank):
    cases = assigned_cases(plan, rank)
    reused = {tuple(x) for x in plan['reuse']['baseline_cases']}
    return [('final', i, s) for i, s in cases] + [
        ('baseline', i, s) for i, s in cases if (i, s) not in reused]


def aggregate(rows):
    n = len(rows)
    return {'episodes': n, 'successes': sum(bool(x['success']) for x in rows),
            'success_rate': sum(bool(x['success']) for x in rows)/n if n else None,
            'mean_score': sum(x['score'] for x in rows)/n if n else None}


def paired_report(root, cases, tolerance=1e-4):
    root = Path(root)
    pairs = []
    available = {'baseline': [], 'final': []}
    for instance, seed in cases:
        key = f'i{instance}_s{seed}'
        rows = {}
        for phase in available:
            path = root/phase/key/'result.json'
            if path.is_file():
                row = json.loads(path.read_text())
                assert (row['instance'], row['seed']) == (instance, seed)
                assert math.isfinite(row['score'])
                available[phase].append(row)
                rows[phase] = row
        if len(rows) != 2:
            continue
        a, b = [json.loads((root/p/key/'initial.json').read_text()) for p in ('baseline', 'final')]
        assert a['physical_state'].keys() == b['physical_state'].keys()
        errors = [abs(x-y) for k in a['physical_state'] for part in (0, 1)
                  for x, y in zip(a['physical_state'][k][part], b['physical_state'][k][part])]
        assert len(a['proprio']) == len(b['proprio'])
        proprio_error = max((abs(x-y) for x,y in zip(a['proprio'], b['proprio'])), default=0.)
        error = max(errors, default=0.)
        times_match = all(a[k] == b[k] for k in ('physics_dt','rendering_dt','observation_dt'))
        pairs.append({'instance': instance, 'seed': seed, **rows,
                      'initial_max_error': error, 'proprio_max_error': proprio_error,
                      'initial_state_valid': error <= tolerance and proprio_error <= tolerance and times_match,
                      'score_delta': rows['final']['score']-rows['baseline']['score'],
                      'success_delta': int(rows['final']['success'])-int(rows['baseline']['success'])})
    before = aggregate([p['baseline'] for p in pairs])
    after = aggregate([p['final'] for p in pairs])
    return {'expected_pairs': len(cases), 'completed_pairs': len(pairs),
            'comparison_complete': len(pairs) == len(cases),
            'comparison_valid': bool(pairs) and all(p['initial_state_valid'] for p in pairs),
            'available': {p: aggregate(v) for p,v in available.items()},
            'baseline': before, 'final': after,
            'success_to_failure': sum(p['success_delta'] < 0 for p in pairs),
            'failure_to_success': sum(p['success_delta'] > 0 for p in pairs),
            'mean_score_delta': after['mean_score']-before['mean_score'] if pairs else None,
            'success_rate_delta': after['success_rate']-before['success_rate'] if pairs else None,
            'pairs': pairs, 'scope': 'Completed fixed pairs only; no checkpoint selection.'}

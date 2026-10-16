"""Compare accepted new5000 results with pinned historical results; no cloud operations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(bundle, expected_pairs, protocol, *, baseline=False):
    if bundle['protocol_sha256'] != protocol:
        raise ValueError('Evaluation protocol changed')
    expected = set(map(tuple, expected_pairs))
    rows = {}
    for row in bundle['records']:
        key = row['task'], row['instance_id']
        if key not in expected or key in rows or row.get('artifact_accepted') is not True:
            raise ValueError('Duplicate, unaccepted, or unexpected episode')
        if row.get('policy_seed') != 0 or row.get('environment_seed') != 0:
            raise ValueError('Seed differs from frozen comparison')
        if baseline:
            if row.get('policy_seed_source_verified') is not True:
                raise ValueError('Historical baseline seed source evidence missing')
        elif row.get('policy_seed_acknowledged') is not True:
            raise ValueError('Actual candidate policy seed ACK missing')
        if row.get('steps', 0) <= 0 or not (row.get('terminated') or row.get('truncated')):
            raise ValueError('Incomplete native episode')
        q = row['q_score']
        if type(row['model_success']) is not bool or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('Invalid native metrics')
        if row['model_success'] != (q == 1):
            raise ValueError('Success disagrees with native completion')
        rows[key] = row
    if set(rows) != expected:
        raise ValueError('Missing accepted episode; do not silently shrink the panel')
    return rows


def summary(rows, pairs):
    return {'episodes': len(pairs), 'successes': sum(rows[k]['model_success'] for k in pairs),
            'success_rate': mean(rows[k]['model_success'] for k in pairs),
            'mean_q': mean(rows[k]['q_score'] for k in pairs)}


def paired(old, new, pairs):
    differences = [new[k]['q_score'] - old[k]['q_score'] for k in pairs]
    rng = random.Random(1688977)
    draws = sorted(mean(rng.choices(differences, k=len(pairs))) for _ in range(10000))
    gains = [k for k in pairs if new[k]['model_success'] and not old[k]['model_success']]
    losses = [k for k in pairs if old[k]['model_success'] and not new[k]['model_success']]
    n, tail = len(gains) + len(losses), min(len(gains), len(losses))
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(tail + 1)) / 2**n) if n else 1.0
    return {'mean_q_delta': mean(differences), 'paired_q_bootstrap95': [draws[249], draws[9749]],
            'q_improved': [k for k, d in zip(pairs, differences) if d > 1e-9],
            'q_regressed': [k for k, d in zip(pairs, differences) if d < -1e-9],
            'success_gains': gains, 'success_losses': losses,
            'exact_mcnemar_two_sided_p': p}


def compare(candidate_path, *, expected_updates=5000):
    hist = ROOT / 'evaluation_history'
    manifest = json.loads((hist / 'manifest.json').read_text())
    candidate = json.loads(candidate_path.read_text())
    if (expected_updates not in (1000, 5000)
            or candidate.get('completed_updates') != expected_updates
            or candidate.get('run_id') != manifest['candidate_run_id']):
        raise ValueError('Candidate is not the expected quality-trial checkpoint')
    pairs = list(map(tuple, manifest['pairs']))
    new = validate(candidate, pairs, manifest['protocol_sha256'])
    result = {'kind': 'quality_trial_fixed_panel_comparison', 'candidate_sha256': digest(candidate_path),
              'candidate_completed_updates': expected_updates,
              'candidate': summary(new, pairs), 'comparisons': {}, 'historical_summaries': {},
              'protocol_sha256': manifest['protocol_sha256'], 'training_seen': True,
              'generalization_proven': False,
              'limitations': ['Single seed, small training-seen regression panel.',
                              'Bootstrap interval describes this panel only; it is not proof of generalization.',
                              'Old60:40 has18 accepted episodes; its two cancelled episodes are excluded, never scored zero.']}
    per_instance = {k: {'candidate': new[k]} for k in pairs}
    for model, source in manifest['models'].items():
        path = hist / source['file']
        if digest(path) != source['sha256']:
            raise ValueError('Pinned historical result changed: ' + model)
        old_bundle = json.loads(path.read_text())
        if candidate['checkpoint_manifest_sha256'] == old_bundle['checkpoint_manifest_sha256']:
            raise ValueError('Reused historical checkpoint is not a trained candidate')
        scope = list(map(tuple, source['pairs']))
        old = validate(old_bundle, scope, manifest['protocol_sha256'], baseline=model == '1688977')
        result['historical_summaries'][model] = summary(old, scope)
        result['comparisons'][model] = {'episodes': len(scope), 'candidate_same_scope': summary(new, scope),
                                        **paired(old, new, scope)}
        for key in scope:
            per_instance[key][model] = old[key]
    result['per_instance'] = [{'task': key[0], 'instance_id': key[1], 'models': values}
                              for key, values in per_instance.items()]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.candidate)
    a.output.mkdir(exist_ok=False, parents=True)
    (a.output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    models = ['candidate', *result['historical_summaries']]
    with (a.output / 'per_instance.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['task', 'instance_id'] + [m + suffix for m in models for suffix in ('_q', '_success')]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in result['per_instance']:
            values = {'task': row['task'], 'instance_id': row['instance_id']}
            for model, record in row['models'].items():
                values[model + '_q'] = record['q_score']
                values[model + '_success'] = int(record['model_success'])
            w.writerow(values)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_instance'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

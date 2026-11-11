"""Analyse only a complete, correctly bound new5000 full100 result set."""
import hashlib
import json
from pathlib import Path

from compare_candidate import paired, summary, validate

ROOT = Path(__file__).resolve().parent


def compare():
    folder = ROOT / 'full100_followup'
    plan = json.loads((folder / 'plan.json').read_text())
    path = folder / 'candidate_accepted.json'
    if not path.exists():
        return {'state': 'waiting_for_complete_new_candidate100', 'model_improvement_proven': False}
    candidate = json.loads(path.read_text())
    if (candidate.get('run_id') != plan['candidate_model_run'] or candidate.get('completed_updates') != 5000
            or not plan['candidate_manifest_sha256']
            or candidate['checkpoint_manifest_sha256'] != plan['candidate_manifest_sha256']):
        raise ValueError('Wrong candidate or checkpoint identity')
    pairs = list(map(tuple, plan['pairs']))
    if len(pairs) != len(set(pairs)) or len(pairs) != 100:
        raise ValueError('Full100 scope changed')
    current = validate(candidate, pairs, plan['protocol_sha256'])
    baseline_path = ROOT / 'evaluation_history/baseline100_accepted.json'
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != plan['baseline_accepted_sha256']:
        raise ValueError('Pinned baseline evidence changed')
    baseline = json.loads(baseline_path.read_text())
    old = {(r['task'], r['instance_id']): r for r in baseline['records']}
    if set(old) != set(pairs) or baseline['checkpoint'] != 1688977:
        raise ValueError('Wrong baseline scope')
    # The baseline has historical source-seed proof, not fabricated live ACKs.
    audit = json.loads((ROOT / 'evaluation_history/baseline100_audit.json').read_text())
    if audit['verified_records'] != 100 or not audit['protocol_equal_except_pairs']:
        raise ValueError('Historical full100 evidence missing')
    if hashlib.sha256((folder / 'baseline_accepted.json').read_bytes()).hexdigest() != hashlib.sha256(baseline_path.read_bytes()).hexdigest():
        raise ValueError('Baseline changed after preparing followup')
    cohorts = json.loads((ROOT / 'evaluation_history/cohorts.json').read_text())['rows']
    membership = {(r['task'], r['instance_id']): r for r in cohorts}
    scopes = {
        'all100': pairs,
        'instance_in_full_source': [k for k in pairs if membership[k]['instance_in_full_success_source']],
        'instance_not_in_full_source': [k for k in pairs if not membership[k]['instance_in_full_success_source']],
        'task_with_full_source': [k for k in pairs if membership[k]['task_has_full_success_source']],
        'task_without_full_source': [k for k in pairs if not membership[k]['task_has_full_success_source']],
    }
    return {'state': 'complete100_compared', 'candidate_run': candidate['run_id'],
            'candidate_manifest_sha256': candidate['checkpoint_manifest_sha256'],
            'candidate_result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cohorts': {name: {'baseline': summary(old, keys), 'candidate': summary(current, keys),
                              'paired': paired(old, current, keys)} for name, keys in scopes.items()},
            'training_seen_regression': True, 'unseen_generalization_proven': False,
            'limitations': ['Single seed per instance.',
                            'Full-source membership is selection, not a randomized treatment.',
                            'Confidence intervals describe the observed panel, not all future instances.']}


def main():
    result = compare()
    if result['state'] == 'complete100_compared':
        (ROOT / 'full100_followup/cohort_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Freeze the highest-Q candidate only after the complete tuning comparison."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
TUNING = ROOT.parent / 'huber50_heldout20'


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = load(ROOT / 'plan.json')
    immutable = dict(plan)
    immutable.update(candidate_checkpoint=None, expected_checkpoint_manifest_sha256=None,
                     formal_contract=None, selected_model=None, execution_enabled=False, deadline_epoch=0)
    proof = load(ROOT / 'runtime_preparation.json')
    if hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != proof['immutable_plan_sha256']:
        raise ValueError('Immutable selection options or confirmation protocol changed')
    policy_path = TUNING / 'model_selection_policy.json'
    if digest(policy_path) != plan['model_selection_policy_sha256']:
        raise ValueError('Preregistered selection policy changed')
    policy = load(policy_path)
    if policy['candidate_outcomes_available'] != 0 or policy['primary_metric'] != 'mean final Q-score over the fixed task-instance pairs':
        raise ValueError('Selection rule is not the preregistered mean-Q rule')
    spec = importlib.util.spec_from_file_location('complete_tuning_report', TUNING / 'build_report.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    data = report.collect()
    if data['missing']:
        print(json.dumps({'selected': False, 'missing_tuning_episodes': len(data['missing'])}))
        return
    # Preserve the incumbent on numerical ties; never select individual episodes.
    winner = 'expert_mse'
    for name in ('baseline', 'huber50'):
        if data['stats'][name]['mean_q'] > data['stats'][winner]['mean_q'] + 1e-12:
            winner = name
    candidate = plan['candidate_options'].get(winner)
    identity = {'selected_model': winner, 'checkpoint': candidate['checkpoint'] if candidate else plan['baseline_checkpoint'],
                'checkpoint_manifest_sha256': candidate['manifest_sha256'] if candidate else None,
                'formal_contract': candidate['formal_contract'] if candidate else None,
                'label': candidate['label'] if candidate else '原模型 1688977',
                'tuning_summaries': data['stats'], 'selection_policy_sha256': digest(policy_path),
                'tuning_comparison_sha256': digest(TUNING / 'comparison.json'),
                'tuning_status_sha256': digest(TUNING / 'status.json'),
                'confirmation_required': winner != 'baseline',
                'confirmation_protocol_sha256': plan['protocol_sha256'],
                'general_improvement_proven': False}
    path = ROOT / 'selection.json'
    if path.exists():
        saved = load(path)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError('Previously frozen selection or evidence changed')
    else:
        with path.open('x') as stream:
            json.dump({**identity, 'selected_at': time.time()}, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'selected': True, 'model': winner, 'mean_q': data['stats'][winner]['mean_q'],
                      'confirmation_required': identity['confirmation_required']}))


if __name__ == '__main__':
    main()

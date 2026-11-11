"""Freeze the next candidate's evaluation inputs; create no cloud jobs."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'heldout20_baseline_vs_expert1k'
DEST = ROOT / 'huber50_heldout20'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite a frozen evaluation definition')
    source = read(REFERENCE / 'plan.json')
    protocol = read(REFERENCE / 'protocol.json')
    canonical = json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != source['protocol_sha256']:
        raise ValueError('Reference protocol changed')
    if source['pairs'] != protocol['pairs'] or source['native_task_budgets'] != protocol['native_task_budgets']:
        raise ValueError('Reference instances or budgets changed')
    for name, expected in read(REFERENCE / 'frozen_sha256.json').items():
        if digest(REFERENCE / 'frozen' / name) != expected:
            raise ValueError('Reference runtime source changed')
    pairs = source['pairs']
    if len(pairs) != 20 or len(set(map(tuple, pairs))) != 20:
        raise ValueError('Expected exactly twenty unique instances')
    # Pair long and short native budgets to reduce the sequential tail.
    order = sorted(range(20), key=lambda i: (-source['native_task_budgets'][pairs[i][0]], i))
    groups = [{'id': f'batch-{i:02d}', 'members': [{'role': 'candidate', 'index': index}
               for index in (order[i], order[-1-i])]} for i in range(10)]
    if sorted(m['index'] for g in groups for m in g['members']) != list(range(20)):
        raise ValueError('Each instance must be evaluated exactly once')
    selected = ROOT / 'expert_huber50_candidate/selected_stage'
    job = read(selected / 'bundle/nebula/worker_job.json')
    if job['run_id'] != 'b1k-expert-huber50-1688977-20261017' or job['stage_end_step'] != 1000:
        raise ValueError('Wrong candidate training identity')
    plan = {
        'created_at': time.time(), 'run_id': 'b1k-huber50-heldout20-20261017',
        'training_iid': '241869', 'training_run_id': job['run_id'], 'candidate_step': 1000,
        'training_contract_sha256': digest(selected / 'bundle/inputs/formal_run_contract.json'),
        'selected_training_bundle_sha256': digest(selected / 'bundle/bundle_manifest.json'),
        'reference_campaign': REFERENCE.name, 'reference_plan_sha256': digest(REFERENCE / 'plan.json'),
        'reference_worker_manifest_sha256': digest(REFERENCE / 'frozen_sha256.json'),
        'protocol_sha256': source['protocol_sha256'], 'pairs': pairs,
        'native_task_budgets': source['native_task_budgets'], 'policy_seed': 0, 'environment_seed': 0,
        'groups': groups, 'candidate_episodes': 20, 'new_baseline_episodes': 0,
        'max_active_gpu_jobs': 10, 'quota_type': 'spot', 'automatic_failure_retry': False,
        'afs_root': '/mnt/afs/260010165/behavior_1k/huber50_heldout20_20261017',
        'object_prefix': 'behavior1k/huber50-heldout20-20261017',
        'execution_enabled': False, 'cloud_jobs_created': 0,
        'required_before_launch': ['committed step1000 with exact restore proof',
            'independently verified AFS inference export',
            'candidate-only controller and two-member batch assembly validation'],
        'required_before_comparison': ['all candidate and reference results accepted',
            'all relevant writers terminal', 'per-instance protocol/seed/weight verification'],
        'baseline_pretraining_exclusion_proven': False,
        'confirmation_panel': 'heldout20_confirmation_panel.json',
        'confirmation_panel_sha256': digest(ROOT / 'heldout20_confirmation_panel.json'),
    }
    DEST.mkdir()
    (DEST / 'protocol.json').write_bytes((REFERENCE / 'protocol.json').read_bytes())
    (DEST / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'instances': 20, 'groups': 10,
                      'protocol_unchanged': True, 'execution_enabled': False, 'cloud_jobs_created': 0}))


if __name__ == '__main__':
    main()

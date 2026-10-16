"""Prepare a disabled, finite80-episode extension of the new candidate's fixed20."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'retraining_20261015_step10000_task100'
DEST = ROOT / 'full100_followup'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=False)
    for name in ('frozen', 'private', 'intents', 'observations', 'reports'):
        (DEST / name).mkdir()
    (DEST / 'private').chmod(0o700)
    prior = json.loads((SOURCE / 'plan.json').read_text())
    first = json.loads((ROOT / 'paired_fixed20_step5000/plan.json').read_text())
    plan = {k: v for k, v in prior.items() if not k.startswith('candidate_') and k not in ('reuse_candidates', 'human_authorization')}
    plan.update(kind='quality5000_full100_regression', run_id='b1k-qbal5k-100-1016',
                afs_root='/mnt/afs/260010165/behavior_1k/quality_taskbalanced_full100_20261016',
                object_prefix='behavior1k/quality-taskbalanced-full100-20261016',
                candidate_step=5000, candidate_model_run=first['run_id'],
                candidate_checkpoint=first['afs_root'] + '/checkpoint5000',
                reused_candidate_afs_root=first['afs_root'],
                candidate_export_report_path=str(ROOT / 'paired_fixed20_step5000/reports/export.json'),
                candidate_export_report_sha256=None, candidate_manifest_sha256=None,
                candidate_reuse_source_sha256=None, candidate_files=[], reuse_candidates=[],
                candidate_reused_episodes=20, max_active_gpu_jobs=20, max_pending=5,
                maximum_job_seconds=prior['maximum_job_seconds'], deadline_epoch=None, execution_enabled=False,
                launch_preflight_complete=False, candidate_binding_complete=False,
                formal_contract=first['formal_contract'],
                execution_basis='Active model-improvement goal; one new candidate run, reuse fixed20 and baseline100.',
                first20_campaign=str(ROOT / 'paired_fixed20_step5000'))
    for path in (SOURCE / 'frozen').glob('*.py'):
        text = path.read_text()
        if path.name in ('worker.py', 'candidate_checks.py'):
            text = text.replace('10000', '5000')
        if path.name == 'candidate_checks.py':
            old = "assert str(path).startswith('/mnt/afs/260010165/behavior_1k/paired_quick20_reused1688977_step5000_20261014/')"
            new = "assert str(path).startswith(plan['reused_candidate_afs_root'].rstrip('/') + '/')"
            if text.count(old) != 1:
                raise ValueError('Unexpected old reuse-root guard')
            text = text.replace(old, new)
        ast.parse(text)
        (DEST / 'frozen' / path.name).write_text(text)
    helper = (SOURCE / 'submit_helpers.py').read_text()
    helper = helper.replace("REPO = ROOT.parents[1] / 'BEHAVIOR-VLAs'", "REPO = ROOT.parents[2] / 'BEHAVIOR-VLAs'")
    helper = helper.replace('checkpoint_step=10000', 'checkpoint_step=5000')
    old = "command = 'timeout --signal=TERM --kill-after=30 86400 '+shlex.quote(python)+' -c '+shlex.quote(startup)"
    new = "command = 'timeout --signal=TERM --kill-after=30 '+str(plan['maximum_job_seconds'])+' '+shlex.quote(python)+' -c '+shlex.quote(startup)"
    if helper.count(old) != 1:
        raise ValueError('Unexpected old timeout command')
    helper = helper.replace(old, new)
    controller = (SOURCE / 'controller.py').read_text().replace('10000', '5000')
    marker = "        raw=(ROOT/'plan.json').read_bytes();plan=json.loads(raw)"
    if controller.count(marker) != 1:
        raise ValueError('Unexpected controller plan load')
    controller = controller.replace(marker, marker + "\n        if (not plan.get('execution_enabled') or not plan.get('launch_preflight_complete')\n                or not plan.get('candidate_binding_complete') or not isinstance(plan.get('deadline_epoch'),(int,float))):\n            raise ValueError('Prepared only: bind actual new5000 results and validate launch before admission')")
    for name, text in [('submit_helpers.py', helper), ('controller.py', controller)]:
        ast.parse(text)
        (DEST / name).write_text(text)
    shutil.copyfile(ROOT / 'evaluation_history/baseline100_accepted.json', DEST / 'baseline_accepted.json')
    shutil.copyfile(SOURCE / 'protocol.json', DEST / 'protocol.json')
    plan['baseline_accepted_sha256'] = sha(DEST / 'baseline_accepted.json')
    (DEST / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (DEST / 'frozen_sha256.json').write_text(json.dumps({p.name: sha(p) for p in (DEST / 'frozen').glob('*.py')}, indent=2) + '\n')
    (DEST / 'preparation.json').write_text(json.dumps({
        'prepared_only': True, 'execution_enabled': False, 'candidate_binding_complete': False,
        'jobs_created': 0, 'reuse_new_fixed20': 20, 'new_episodes': 80, 'total_compared': 100,
        'old_candidate_weights_or_results_bound': False,
        'source_code': str(SOURCE), 'baseline_sha256': sha(DEST / 'baseline_accepted.json'),
        'next_step': 'Bind actual new5000 export and20 accepted results; verify first20 jobs terminal before enabling.'
    }, indent=2) + '\n')
    print(json.dumps({'destination': str(DEST), 'enabled': False, 'new_jobs': 0}))


if __name__ == '__main__':
    main()

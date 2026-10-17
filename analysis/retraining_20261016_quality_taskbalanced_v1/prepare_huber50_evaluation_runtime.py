"""Prepare an isolated, disabled candidate-only evaluation runtime."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'heldout20_baseline_vs_expert1k'
DEST = ROOT / 'huber50_heldout20'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (DEST / 'frozen').exists():
        raise FileExistsError('Do not overwrite a prepared runtime')
    source = json.loads((SOURCE / 'plan.json').read_text())
    plan = json.loads((DEST / 'plan.json').read_text())
    hashes = json.loads((SOURCE / 'frozen_sha256.json').read_text())
    for name, expected in hashes.items():
        if digest(SOURCE / 'frozen' / name) != expected:
            raise ValueError('Reference runtime changed')
    shutil.copytree(SOURCE / 'frozen', DEST / 'frozen', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = DEST / 'frozen/batch_runner.py'
    text = path.read_text()
    if text.count('len(results) == 4') != 1:
        raise ValueError('Unexpected batch acceptance condition')
    path.write_text(text.replace('len(results) == 4', "len(results) == len(config['members'])"))
    shutil.copy2(SOURCE / 'assemble.py', DEST / 'assemble.py')
    for key in ['code', 'runtime', 'baseline_checkpoint', 'baseline_files', 'repo_id', 'norm_sha256',
                'source_manifest_sha256', 'critical_files', 'supplemental_library', 'image',
                'native_task_budgets', 'task_metadata_sha256', 'instance_assets']:
        plan[key] = source[key]
    plan.update(candidate_checkpoint=plan['afs_root'] + '/checkpoint1000',
                expected_checkpoint_manifest_sha256=None,
                member_walltime_seconds=source['member_walltime_seconds'],
                job_walltime_seconds=2 * source['member_walltime_seconds'] + 4000,
                formal_contract=json.loads((ROOT / 'expert_huber50_candidate/selected_stage/bundle/inputs/formal_run_contract.json').read_text()))
    for group in plan['groups']:
        group['name'] = 'b1k-h50-heldout20-1017-' + group['id']
    for name in ['intents', 'reports', 'private', 'observations']:
        (DEST / name).mkdir(exist_ok=True)
    new_hashes = {name: digest(DEST / 'frozen' / name) for name in hashes}
    changed = [name for name in hashes if new_hashes[name] != hashes[name]]
    if changed != ['batch_runner.py']:
        raise ValueError('Inner evaluation worker must remain unchanged')
    (DEST / 'frozen_sha256.json').write_text(json.dumps(new_hashes, indent=2) + '\n')
    (DEST / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    proof = {'source_controller_sha256': digest(SOURCE / 'controller.py'),
             'assembler_sha256': digest(DEST / 'assemble.py'), 'source_frozen_hashes': hashes,
             'frozen_hashes': new_hashes, 'only_outer_batch_acceptance_changed': True,
             'execution_enabled': False, 'cloud_jobs_created': 0}
    (DEST / 'runtime_preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'worker_unchanged': True, 'execution_enabled': False}))


if __name__ == '__main__':
    main()

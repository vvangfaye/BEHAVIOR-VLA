"""Prepare, without submitting, a conservative threshold variant of Huber 5."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'expert_huber_candidate/bundle'
DEST = ROOT / 'expert_huber50_candidate/bundle'
RUN = 'b1k-expert-huber50-1688977-20261017'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite a prepared candidate')
    old = read(SOURCE / 'bundle_manifest.json')
    for name, expected in old.items():
        if digest(SOURCE / name) != expected:
            raise ValueError('Source candidate changed: ' + name)
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    path = DEST / 'inputs/compiled/training_argv.json'
    argv = read(path)
    for flag, value in [('--exp-name', RUN), ('--model.action-loss-huber-delta', '50.0')]:
        argv['argv'][argv['argv'].index(flag) + 1] = value
    assert argv['launch_authorized'] is False
    save(path, argv)
    path = DEST / 'inputs/formal_run_contract.json'
    contract = read(path)
    contract.update(run_id=RUN, action_loss={'schema_version': 1, 'kind': 'scaled_huber', 'delta': 50.0})
    save(path, contract)
    path = DEST / 'inputs/formal_authorization.json'
    authorization = read(path)
    assert authorization['formal_training_allowed'] is False
    authorization.update(action_loss=contract['action_loss'],
        rationale='Target rare extreme residuals with a conservative threshold; actual gradient and evaluation effects remain unmeasured.')
    save(path, authorization)
    path = DEST / 'nebula/worker_job.json'
    job = read(path)
    assert job['formal_training_authorized'] is False
    job.update(run_id=RUN, experiment='expert_only_huber50_control',
               output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/expert_huber/' + RUN)
    save(path, job)
    manifest = {name: digest(DEST / name) for name in old}
    changed = sorted(name for name in old if old[name] != manifest[name])
    expected = sorted(['inputs/compiled/training_argv.json', 'inputs/formal_run_contract.json',
                       'inputs/formal_authorization.json', 'nebula/worker_job.json'])
    if changed != expected:
        raise ValueError('Threshold variant changed code or training data')
    save(DEST / 'bundle_manifest.json', manifest)
    save(DEST.parent / 'preparation.json', {
        'run_id': RUN, 'delta': 50.0, 'updates': 1000, 'execution_enabled': False,
        'source_manifest_sha256': digest(SOURCE / 'bundle_manifest.json'),
        'candidate_manifest_sha256': digest(DEST / 'bundle_manifest.json'),
        'changed_files_vs_huber5': changed, 'same_model_loss_code': True,
        'same_sampling_lr_optimizer_seed_batch_normalization': True,
        'threshold_evidence_sha256': digest(ROOT / 'expert_huber_candidate/target_threshold_sensitivity.json'),
        'cloud_jobs_created': 0, 'remaining': 'CPU validation and selection after complete current evaluation.'})
    print(json.dumps({'prepared': True, 'delta': 50.0, 'changed_files': changed, 'cloud_jobs_created': 0}))


if __name__ == '__main__':
    main()

"""Local pre-submission verification (no upload, no job creation). Writes verification.json."""
import hashlib
import io
import json
import tarfile
import time

import deployment as d

config = d.campaign()
checks = {}


def bundle_names(bundle):
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        return {item.name: archive.extractfile(item).read() for item in archive}


signer = d.status_client()  # local SigV4 signing only; no request is sent


def dummy(method, item):
    """A real-length presigned URL for a verification-only key (never used by any job)."""
    return signer.generate_presigned_url(method, Params={'Bucket': '260010165',
        'Key': config['aoss_prefix'] + 'verification-only/' + item}, ExpiresIn=5 * 86400)

# Runtime integrity.
for name, digest in config['runtime_sha256'].items():
    assert d.sha_file(d.HERE / 'runtime' / name) == digest, name
checks['runtime_files'] = len(config['runtime_sha256'])

# Training arms.
for arm, spec in config['training_arms'].items():
    plan = d.worker_plan('train', arm)
    optimizer = plan['optimizer']
    assert optimizer['actor_lr'] == spec['actor_lr'] and optimizer['allow_kl_early_stop'] is True
    assert optimizer['value_lr'] == 1e-4 and optimizer['gamma'] == plan['shaping']['gamma'] == 0.99
    assert optimizer['updates_per_round'] == [17] and optimizer['kl_target'] == 0.01 and optimizer['clip_ratio'] == 0.2
    assert plan['comparison_variant'] == 'control' and plan['distributed']['world_size'] == 4
    assert plan['distributed']['kl_batch_size'] == 8 and plan['output'].endswith(f'/training/{arm}/results')
    assert len(plan['source_rollout_receipts']) == 32
    bundle = d.bundle_for('train', plan)
    files = bundle_names(bundle)
    assert files['run/driver.py'] == (d.HERE / 'runtime/distributed_driver.py').read_bytes()
    assert json.loads(files['run/plan.json'])['optimizer']['actor_lr'] == spec['actor_lr']
    proof = d.storage_proof(plan)
    name, command, limit = d.startup_command('train', d.identity('train', arm, None, 1), 1, bundle, proof, dummy)
    checks[f'train_{arm}'] = {'actor_lr': spec['actor_lr'], 'bundle_bytes': len(bundle), 'command_chars': len(command),
                              'job_name': name, 'timeout_seconds': limit}

# Evaluation arms: every case for every arm builds a valid plan; one bundle per arm is fully built.
for arm, spec in config['eval_arms'].items():
    cases = d.arm_cases(arm)
    for case in cases:
        plan = d.worker_plan('eval', arm, case)
        assert plan['evaluation']['cases'] == [case] and plan['eval_instances'] == [case[0]]
        assert plan['output'].endswith(f'/evaluations/{arm}/i{case[0]}_s{case[1]}/results')
    checkpoint = plan['comparison_checkpoint']
    if spec['checkpoint'] is None:
        assert checkpoint is None and plan['evaluate_original_policy'] is True
    elif spec['checkpoint'].startswith('training:'):
        assert checkpoint.endswith(f"/training/{spec['checkpoint'].split(':')[1]}/results/checkpoints/round_01")
        assert plan['evaluate_original_policy'] is False
    else:
        assert checkpoint == spec['checkpoint'] and plan['evaluate_original_policy'] is False
    bundle = d.bundle_for('eval', plan)
    files = bundle_names(bundle)
    assert files['run/driver.py'] == (d.HERE / 'runtime/comparison_driver.py').read_bytes()
    assert b'evaluate_original_policy' in files['run/driver.py']
    proof = d.storage_proof(plan)
    name, command, limit = d.startup_command('eval', d.identity('eval', arm, cases[-1], 3), 3, bundle, proof, dummy)
    checks[f'eval_{arm}'] = {'cases': len(cases), 'checkpoint': checkpoint, 'bundle_bytes': len(bundle),
                             'command_chars': len(command), 'longest_job_name': name, 'timeout_seconds': limit}

assert len(d.arm_cases('orig')) == 200 and len(d.arm_cases('origrep')) == 100
assert sorted({tuple(c) for c in d.arm_cases('origrep')}) == sorted({tuple(c) for c in d.arm_cases('orig') if c[1] < 5})
tests = json.loads((d.HERE / 'tests/cpu_tests_result.json').read_text())
assert tests['passed'] and tests['runtime_sha256'] == config['runtime_sha256']
checks['cpu_tests'] = {k: tests[k] for k in ['passed', 'tests', 'elapsed_seconds']}
checks['storage'] = {k: proof[k] for k in ['account_allocated_bytes', 'quota_lower_bound_bytes', 'required_remaining_bytes', 'passed']}
result = {'passed': True, 'at': time.time(), 'plan_sha256': d.sha_file(d.HERE / 'plan.json'), 'checks': checks}
d.save(d.HERE / 'verification.json', result)
print(json.dumps(result, indent=1)[:4000])

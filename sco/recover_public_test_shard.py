"""Resume a terminal infrastructure failure, preserving artifacts, attempts, and scope."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zlib

from submit_public_test_shard import build_payload, validate_capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--capacity', type=Path)
    parser.add_argument('--queue-spot', action='store_true')
    parser.add_argument('--admission-slot', type=int, default=1)
    parser.add_argument('--reconciliation-proof', type=Path)
    args = parser.parse_args()
    root = args.evidence.resolve()
    label = 'a' + str(args.index).zfill(3)
    submission_path = root / 'production_submissions' / (label + '.json')
    old = json.loads(submission_path.read_text())
    assert old['quota_type'] == 'spot' and old['state'] == 'submitted'
    generation = old['generation'] + 1
    assert 1 <= generation <= 3
    folder = root / 'production_recoveries'
    folder.mkdir(exist_ok=True)
    receipt = folder / (label + '_g' + str(generation).zfill(3) + '.json')
    assert not receipt.exists(), 'Inspect the exact existing recovery; never retry blindly'
    if args.queue_spot:
        capacity_sha = hashlib.sha256(b'user_authorized_finite_spot_scheduler_queue').hexdigest()
    else:
        capacity_bytes = args.capacity.read_bytes()
        capacity = json.loads(capacity_bytes)
        capacity_sha = hashlib.sha256(capacity_bytes).hexdigest()
        assert 1 <= args.admission_slot <= validate_capacity(capacity, now=datetime.now(timezone.utc))
        for path in (root / 'production_submissions').glob('a*.json'):
            prior = json.loads(path.read_text())
            assert (prior.get('capacity_sha256'), prior.get('admission_slot')) != (capacity_sha, args.admission_slot)
    proof_path = args.reconciliation_proof or (root / ('production_' + label + '_failure_afs_readback.json'))
    if args.reconciliation_proof is None and (root / (label + '_failure_afs_readback.json')).exists():
        proof_path = root / (label + '_failure_afs_readback.json')
    proof = json.loads(proof_path.read_text())
    assert 0 <= time.time() - proof['observed_at'] <= 120
    assert proof['assignment_root'] == old['assignment_root']
    assert proof['status.json']['source_manifest_sha256'] == old['source_manifest_sha256']
    assert {row['instance_id'] for row in proof['queue']} == set(old['instance_ids'])
    if args.reconciliation_proof is None:
        assert proof['status.json']['state'] == 'failed'
        assert proof['result_json_count'] == 0
        assert all(row['state'] == 'pending' and row['attempts'] == 0 for row in proof['queue'])
    else:
        assert proof['state'] == 'terminal_worker_queue_reconciled'
        assert proof['old_uid'] == old['uid'] and proof['old_generation'] == old['generation']
        assert proof['old_writer_terminal'] is True and proof['completed_artifacts_preserved'] is True
        assert proof['queue_source_sha256'] == '0e1c0a0ad0daae695119766731adc83981c98c6cacc4f726a7df3c27a0e0419f'
        assert proof['metadata'] == dict(run_id=old['run_id'], mode='public_test', num_rollouts=1,
            require_videos=False, checkpoint_step=1688977, source_manifest_sha256=old['source_manifest_sha256'])
        before = {row['instance_id']: row for row in proof['before_queue']}
        for row in proof['queue']:
            prior = before[row['instance_id']]
            assert row['task'] == old['task'] and row['state'] in ('pending', 'succeeded')
            assert row['attempts'] == prior['attempts'] and row['max_attempts'] == prior['max_attempts'] == 3
            if prior['state'] != 'running':
                assert row['state'] == prior['state']
    env = os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config', SCO_HOME='/private/tmp/behavior-sco-runtime',
               SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    cli = '/private/tmp/behavior-sco-runtime/bin/sco'

    def describe(name):
        return json.loads(subprocess.run([cli, 'acp', 'jobs', 'describe', name,
            '--workspace-name=share-space', '-o', 'json'], env=env, capture_output=True,
            text=True, check=True, timeout=30).stdout)

    terminal = describe(old['name'])
    assert terminal['uid'] == old['uid'] and terminal['ownership']['user_name'] == '260010165'
    assert terminal['state'] == 'FAILED'
    workers = subprocess.run([cli, 'acp', 'jobs', 'get-workers', old['name'],
        '--workspace-name=share-space'], env=env, capture_output=True, text=True, check=True, timeout=30).stdout
    assert ('Failed' in workers or 'Deleted' in workers) and old['uid'] in workers
    assert 'Running' not in workers and 'Pending' not in workers
    source = (root / 'pilot_driver_source.py').read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    assert source_sha == old['driver_sha256']
    name = 'behavior-public1688977-' + label + '-g' + str(generation).zfill(3) + '-' + datetime.now(
        timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    payload = build_payload(json.loads((root / 'preparation_ready.json').read_text()),
        json.loads((root / 'production_plan.json').read_text()),
        json.loads((root / 'pilot_accepted.json').read_text()), args.index, name, source_sha)
    for key in ('assignment_root', 'run_id', 'source_manifest_sha256', 'driver_sha256', 'task', 'instance_ids'):
        assert payload[key] == old[key]
    payload['generation'] = generation
    template = describe('pt-y91th8k9')
    assert template['resource_pool']['name'] == old['resource_pool']
    if not args.queue_spot:
        assert args.admission_slot <= validate_capacity(capacity, now=datetime.now(timezone.utc))
    from behavior_aoss_client import client
    report_key = 'behavior1k/test2000-1688977-20261010/production/' + label + '_g' + str(generation).zfill(3) + '_status.json'
    payload['report_put'] = client().generate_presigned_url('put_object',
        Params={'Bucket': '260010165', 'Key': report_key}, ExpiresIn=432000)
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    driver = base64.b64encode(zlib.compress(source)).decode()
    startup = 'import base64,zlib,json,hashlib;p=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + ')));s=zlib.decompress(base64.b64decode(' + repr(driver) + '));assert hashlib.sha256(s).hexdigest()==p["driver_sha256"];ns={};exec(compile(s,"public_test_shard","exec"),ns);ns["run"](p)'
    record = {key: value for key, value in payload.items() if key != 'report_put'}
    record.update(name=name, index=args.index, state='submitting', old_submission=old,
        old_writer_terminal=True, old_worker_phase='Deleted' if 'Deleted' in workers else 'Failed',
        queue_untouched=args.reconciliation_proof is None,
        generation=generation, capacity_sha256=capacity_sha, admission_slot=args.admission_slot,
        admission_mode='scheduler_queue' if args.queue_spot else 'capacity',
        resource_pool=old['resource_pool'], gpu_count=1, worker_spec='n12lp.nn.i10a.1',
        quota_type='spot', timeout_seconds=345600, max_gpu_hours=96, report_key=report_key)
    if args.reconciliation_proof is not None:
        # A current proof is refreshed for later failures; pin this generation's evidence.
        pinned_proof = folder / (label + '_g' + str(generation).zfill(3) + '_queue_reconciliation.json')
        proof_bytes = proof_path.read_bytes()
        assert json.loads(proof_bytes) == proof, 'Reconciliation proof changed during admission'
        if pinned_proof.exists():
            assert pinned_proof.read_bytes() == proof_bytes, 'Pinned proof must never change'
        else:
            with pinned_proof.open('xb') as stream:
                stream.write(proof_bytes)
        record['queue_reconciliation_proof'] = pinned_proof.relative_to(root).as_posix()
        record['queue_reconciliation_proof_sha256'] = hashlib.sha256(proof_bytes).hexdigest()
        record['attempt_counts_preserved'] = True
    with receipt.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    result = subprocess.run([cli, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
        '--copy-job-name=pt-y91th8k9', '--name=' + name, '--job-name=' + name,
        '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot',
        '--env=NVIDIA_DRIVER_CAPABILITIES:all',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 345600 python3 -c ' + shlex.quote(startup)],
        env=env, capture_output=True, text=True, timeout=90)
    record.update(state='submission_unknown' if result.returncode else 'submitted', exit_code=result.returncode)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    assert result.returncode == 0, 'Inspect the saved exact name before retrying'
    job = describe(name)
    assert job['ownership']['user_name'] == '260010165'
    assert job['resource_pool']['name'] == old['resource_pool'] and job['scheduling']['quota_type'] == 'SPOT'
    record.update(uid=job['uid'], platform_state=job['state'])
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    current = {key: value for key, value in record.items()
               if key not in ('old_submission', 'old_writer_terminal', 'old_worker_phase', 'queue_untouched')}
    current['recovery_receipt'] = receipt.relative_to(root).as_posix()
    submission_path.write_text(json.dumps(current, indent=2) + '\n')
    print(json.dumps({key: current[key] for key in ('index', 'name', 'uid', 'generation', 'platform_state')}))


if __name__ == '__main__':
    main()

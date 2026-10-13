"""Launch at most four 1-GPU reserved allocations, computing exclusively on CPU."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import zlib

from bounded_collection_admission import ACTIVE_STATES, requested_gpus
from build_retraining_parallel import balanced_parts, checked_receipts
from launch_retraining_cloud import frozen_tar, range_hash, sha, write
from submit_retraining_cloud_probe import client, CLI, PREFIX


def environment():
    return dict(os.environ, SCO_CONFIG='/private/tmp/behavior-sco-config',
        SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')


def reserved_admission(known, additional_gpus=1):
    assert 0 <= additional_gpus <= 4
    active = {}
    for state in ACTIVE_STATES:
        seen = set()
        for page in range(1, 101):
            result = subprocess.run([CLI, 'acp', 'jobs', 'list', '--workspace-name=share-space',
                '--user-name=260010165', '--state=' + state, '--page-size=20',
                '--page-token=' + str(page), '-o', 'json'], env=environment(),
                capture_output=True, text=True, timeout=40, check=True)
            text = result.stdout.strip()
            batch = [] if text in ('', 'No jobs found', 'No jobs found.') else json.loads(text)
            assert isinstance(batch, list)
            for job in batch:
                assert job['uid'] not in seen and job['ownership']['user_name'] == '260010165'
                assert job['state'] == state
                seen.add(job['uid']); active[job['uid']] = job
            if len(batch) < 20:
                break
        else:
            raise RuntimeError('Incomplete standard allocation pagination')
    active = list(active.values())
    reserved = [r for r in active if r['scheduling']['quota_type'] == 'RESERVED']
    visible = {r['name'] for r in reserved}
    charge = sum(requested_gpus(r) for r in reserved) + sum(r['name'] not in visible for r in known)
    if charge + additional_gpus > 4:
        raise RuntimeError('Four standard GPU allocations already committed; no additional submission')
    return {'reserved_gpus_before': charge, 'visible_names': sorted(visible),
            'observed_at': datetime.now(timezone.utc).isoformat()}


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    launch_path = out / 'parallel_launch.json'
    assert not launch_path.exists(), 'Existing launch must be reconciled; do not recreate'
    frozen = json.loads(args.sources.read_text())
    rows = frozen['source_archives']
    baseline = json.loads(args.baseline.read_text())['sources']
    accepted = checked_receipts(rows, baseline, require_complete=False)
    remaining = [r for r in rows if r['key'] not in accepted]
    parts = balanced_parts(remaining, 4)
    assert all(parts) and sum(map(len, parts)) == len(remaining)
    prior = json.loads(args.prior_receipt.read_text())
    assert prior['source_fingerprint'] == frozen['fingerprint']
    private = Path('/private/tmp/behavior-retraining-parallel-private')
    private.mkdir(mode=0o700, exist_ok=True)
    public, internal = client(), client(internal=True)
    stamp = datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
    prefix = PREFIX + 'parallel-' + stamp + '/'

    def sign(op, key, exclusive=False):
        params = {'Bucket': '260010165', 'Key': key}
        if exclusive:
            params['IfNoneMatch'] = '*'
        return internal.generate_presigned_url(op, Params=params, ExpiresIn=604800)

    repo = Path(__file__).resolve().parents[1]
    names = ['sco/prepare_retraining_data.py', 'sco/build_retraining_cloud.py',
        'sco/build_retraining_parallel.py', 'scripts/subtask_trajectory.py',
        'scripts/full_episode_trajectory.py', 'scripts/export_full_episode_collection.py',
        'openpi/src/openpi/__init__.py', 'openpi/src/openpi/training/recovery_index.py',
        'openpi/src/openpi/configs/tasks/b1k.py']
    code = private / 'parallel_code.tar.gz'
    frozen_tar(code, [(name, repo / name) for name in names])

    def upload(path, name, extract_to=None):
        digest, size = sha(path), path.stat().st_size
        key = prefix + 'inputs/' + digest + '/' + name
        try:
            head = public.head_object(Bucket='260010165', Key=key)
        except public.exceptions.ClientError as error:
            if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                raise
            with path.open('rb') as stream:
                public.put_object(Bucket='260010165', Key=key, Body=stream,
                                  IfNoneMatch='*', Metadata={'sha256': digest})
            head = public.head_object(Bucket='260010165', Key=key)
        assert head['ContentLength'] == size
        assert range_hash(public, key, size, head['ETag']) == digest
        result = {'name': name, 'key': key, 'get': sign('get_object', key), 'sha256': digest, 'bytes': size}
        if extract_to:
            result['extract_to'] = extract_to
        return result

    common = [upload(code, 'cloud_code.tar.gz', 'code')]
    for name, target in [('lerobot_source.tar.gz', 'lerobot'), ('cpu_data_wheels.tar.gz', 'wheels')]:
        asset = next(a for a in prior['assets'] if a['name'] == name)
        head = public.head_object(Bucket='260010165', Key=asset['key'])
        assert head['ContentLength'] == asset['bytes'] and head['Metadata']['sha256'] == asset['sha256']
        common.append(dict(asset, get=sign('get_object', asset['key']), extract_to=target))
    baseline_copy = out / 'accepted_before_cutover.json'
    write(baseline_copy, {'sources': list(accepted.values()), 'source_fingerprint': frozen['fingerprint']})
    plan = {'mode': 'four_reserved_allocations_cpu_only', 'source_fingerprint': frozen['fingerprint'],
        'baseline_file': str(baseline_copy.resolve()), 'baseline_sha256': sha(baseline_copy),
        'remaining_sources': len(remaining), 'source_count': len(rows), 'shards': [],
        'training_allowed': False, 'gpu_computation_enabled': False,
        'gpu_allocations': 4, 'cpu_processes_per_job': 4, 'timeout_seconds': 86400,
        'created_at': datetime.now(timezone.utc).isoformat()}
    # Freeze every assignment before creating the first remote resource.
    payloads = []
    for i, assigned in enumerate(parts):
        name = 'behavior-data-parallel-' + stamp + '-s' + str(i)
        subprefix = prefix + 'shard-' + str(i) + '/'
        scope = {'shard': i, 'shard_count': 4, 'cpu_processes': 4, 'assigned_keys': [r['key'] for r in assigned]}
        inputs = dict(frozen, source_archives=assigned, parallel_execution=scope)
        inputs['authorization'] = dict(frozen['authorization'], cpu_only=True,
            maximum_scratch_bytes=32 * 1024**3, gpu_allocations=1, gpu_computation_enabled=False)
        for row in inputs['source_archives']:
            row['signed_get_url'] = sign('get_object', row['key'])
        storage = {'prefix': subprefix, 'status_put': sign('put_object', subprefix + 'status.json'),
                   'outputs': {}, 'already_transferred': {},
                   'child_status_put': [sign('put_object', subprefix + 'process-' + str(j) + '.json') for j in range(4)]}
        keys = [subprefix + 'cloud_prepared_manifest.json']
        for row in assigned:
            identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
            keys += [subprefix + 'receipts/' + identity + '.json',
                     subprefix + 'indexes/' + row['kind'] + '_' + identity + '.json']
            if row['kind'] == 'full_episode':
                keys.append(subprefix + 'full_episode/exports/' + row['sha256'] + '.tar')
        for key in keys:
            storage['outputs'][key] = {'put': sign('put_object', key, True), 'get': sign('get_object', key)}
        directory = private / ('shard-' + str(i)); directory.mkdir(mode=0o700, exist_ok=True)
        for filename, value in [('private_sources.json', inputs), ('private_storage.json', storage)]:
            write(directory / filename, value); (directory / filename).chmod(0o600)
        assets = common + [upload(directory / n, n) for n in ['private_sources.json', 'private_storage.json']]
        payload = {'root': '/mnt/afs/260010165/behavior_1k/retraining_20261013/parallel-' + stamp + '/shard-' + str(i),
                   'assets': assets, 'status_put': storage['status_put'],
                   'entrypoint': 'build_retraining_parallel.py', 'cpu_threads': 1}
        record = {'name': name, 'state': 'prepared_not_submitted', 'gpu_count': 1,
            'gpu_computation_enabled': False, 'cpu_count': 22, 'memory_gib': 240,
            'cpu_processes': 4, 'worker_spec': 'n6ls.iu.i40.1', 'quota_type': 'reserved',
            'resource_pool': 'share-cluster', 'workspace': 'share-space', 'root': payload['root'],
            'timeout_seconds': 86400, 'status_key': subprefix + 'status.json',
            'manifest_key': subprefix + 'cloud_prepared_manifest.json', 'source_keys': scope['assigned_keys'],
            'source_count': len(assigned), 'source_bytes': sum(r['bytes'] for r in assigned),
            'source_frames': sum(r['frames'] for r in assigned), 'training_allowed': False}
        plan['shards'].append(record); payloads.append(payload)
    write(launch_path, plan)
    source = Path(__file__).with_name('bootstrap_retraining_cloud.py').read_text()
    known = []
    for i, (record, payload) in enumerate(zip(plan['shards'], payloads)):
        record['admission'] = reserved_admission(known)
        record['state'] = 'submission_intent'
        write(launch_path, plan)
        program = source + '\nrun(' + repr(payload) + ')\n'
        encoded = base64.b64encode(zlib.compress(program.encode())).decode()
        startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(encoded) + ')),"data_bootstrap","exec"))'
        command = [CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space', '--aec2-name=share-cluster',
            '--name=' + record['name'], '--job-name=' + record['name'], '--training-framework=pytorch',
            '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
            '--worker-spec=n6ls.iu.i40.1', '--worker-nodes=1', '--quota-type=reserved',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--command=timeout --signal=TERM --kill-after=30 86400 python3 -c ' + shlex.quote(startup)]
        result = subprocess.run(command, env=environment(), capture_output=True, text=True, timeout=90)
        record.update(state='created_needs_describe' if result.returncode == 0 else 'submission_unknown',
                      cli_exit_code=result.returncode)
        write(launch_path, plan)
        if result.returncode:
            raise RuntimeError('Inspect exact recorded name; do not retry creation blindly')
        result = subprocess.run([CLI, 'acp', 'jobs', 'describe', record['name'],
            '--workspace-name=share-space', '-o', 'json'], env=environment(), capture_output=True,
            text=True, timeout=40, check=True)
        job = json.loads(result.stdout)
        assert job['ownership']['user_name'] == '260010165'
        assert job['scheduling']['quota_type'] == 'RESERVED'
        assert requested_gpus(job) == 1 and job['resource_pool']['name'] == 'share-cluster'
        record.update(uid=job['uid'], platform_state=job['state'], state='submitted')
        known.append(record)
        write(launch_path, plan)
        print(json.dumps({k:record[k] for k in ['name','uid','state','source_count','gpu_count','cpu_processes']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--prior-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())

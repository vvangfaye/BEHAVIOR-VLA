"""Freeze and launch a finite, isolated source-side CPU data build."""
import argparse
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import zlib

from submit_retraining_cloud_probe import client, CLI, PREFIX


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def frozen_tar(path, entries):
    """Reuse identical tool bundles after an interrupted transport check."""
    expected = {name: source for name, source in entries}
    if path.exists():
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            if (len(members) == len(expected) and all(m.isfile() and m.name in expected for m in members)
                    and all(hashlib.sha256(archive.extractfile(m).read()).hexdigest() == sha(expected[m.name])
                            for m in members)):
                return
    with tarfile.open(path, 'w:gz') as archive:
        for name, source in entries:
            archive.add(source, arcname=name, recursive=False)


def range_hash(store, key, size, etag):
    """Hash exact ordered ranges with a bounded eight-request window."""
    part = 1024**2
    def fetch(start):
        end = min(start + part, size) - 1
        response = store.get_object(Bucket='260010165', Key=key,
            Range='bytes=%d-%d' % (start, end), IfMatch=etag)
        body = response['Body']
        try:
            data = body.read()
        finally:
            body.close()
        assert response['ContentRange'] == 'bytes %d-%d/%d' % (start, end, size)
        assert len(data) == end - start + 1
        return data
    digest = hashlib.sha256()
    starts = iter(range(0, size, part))
    with ThreadPoolExecutor(max_workers=8) as executor:
        pending = deque(executor.submit(fetch, start) for start in list(range(0, min(size, part * 8), part)))
        for _ in range(len(pending)):
            next(starts)
        while pending:
            digest.update(pending.popleft().result())
            following = next(starts, None)
            if following is not None:
                pending.append(executor.submit(fetch, following))
    return digest.hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / 'cloud_build_submission.json'
    assert not receipt.exists(), 'Reconcile the existing exact cloud build before a new submission'
    inputs = json.loads(args.sources.read_text())
    assert inputs['authorization']['training_allowed'] is False
    assert len(inputs['source_archives']) == 463
    probe = json.loads((args.output / 'cloud_probe_report.json').read_text())
    assert probe['routes']['internal']['verified'] and probe['training_executed'] is False
    private = Path('/private/tmp/behavior-retraining-cloud-private')
    private.mkdir(mode=0o700, exist_ok=True); private.chmod(0o700)
    public_client, internal = client(), client(internal=True)
    prior_assets = {}
    if args.prior_receipt:
        previous = json.loads(args.prior_receipt.read_text())
        assert previous['source_fingerprint'] == inputs['fingerprint']
        prior_assets = {a['key']: a for a in previous['assets']}
    def sign(operation, key, exclusive=False):
        params = {'Bucket': '260010165', 'Key': key}
        if exclusive:
            params['IfNoneMatch'] = '*'
        return internal.generate_presigned_url(operation, Params=params, ExpiresIn=604800)
    prefix = PREFIX + 'build-v1/'
    status_put = sign('put_object', prefix + 'status.json')
    prior = {}
    with tarfile.open(args.output / 'cutover_receipts.tar.gz') as archive:
        for member in archive:
            if (member.isfile() and member.name.startswith('receipts/') and member.name.endswith('.json')
                    and not Path(member.name).name.startswith('._')):
                record = json.load(archive.extractfile(member))
                assert record['accepted'] and record['index_target']
                prior[record['source_key']] = record
    storage = {'prefix': prefix, 'status_put': status_put, 'outputs': {}, 'already_transferred': prior}
    for row in inputs['source_archives']:
        row['signed_get_url'] = sign('get_object', row['key'])
        identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
        keys = [prefix + 'indexes/' + row['kind'] + '_' + identity + '.json',
                prefix + 'receipts/' + identity + '.json']
        if row['kind'] == 'full_episode':
            keys.append(prefix + 'full_episode/exports/' + row['sha256'] + '.tar')
        for key in keys:
            storage['outputs'][key] = {'put': sign('put_object', key, exclusive=True), 'get': sign('get_object', key)}
    final_key = prefix + 'cloud_prepared_manifest.json'
    storage['outputs'][final_key] = {'put': sign('put_object', final_key, exclusive=True),
                                     'get': sign('get_object', final_key)}
    write(private / 'private_sources.json', inputs)
    write(private / 'private_storage.json', storage)
    for path in private.glob('private_*.json'):
        path.chmod(0o600)
    repo = Path(__file__).resolve().parents[1]
    code = private / 'cloud_code.tar.gz'
    selected = ['sco/prepare_retraining_data.py', 'sco/build_retraining_cloud.py',
        'scripts/subtask_trajectory.py', 'scripts/full_episode_trajectory.py',
        'scripts/export_full_episode_collection.py', 'openpi/src/openpi/__init__.py',
        'openpi/src/openpi/training/recovery_index.py', 'openpi/src/openpi/configs/tasks/b1k.py']
    frozen_tar(code, [(name, repo / name) for name in selected])
    wheels = private / 'cpu_data_wheels.tar.gz'
    files = sorted(args.wheels.glob('*.whl'))
    assert len(files) >= 20 and any(p.name.startswith('h5py-') for p in files)
    frozen_tar(wheels, [(path.name, path) for path in files])
    assets = []
    def upload_asset(path, name, extract_to=None):
        digest, size = sha(path), path.stat().st_size
        key = prefix + 'inputs/' + digest + '/' + name
        reused = prior_assets.get(key)
        if reused and reused['sha256'] == digest and reused['bytes'] == size:
            existing = public_client.head_object(Bucket='260010165', Key=key)
            assert existing['ContentLength'] == size and existing['Metadata']['sha256'] == digest
        else:
            try:
                existing = public_client.get_object(Bucket='260010165', Key=key)
            except public_client.exceptions.NoSuchKey:
                with path.open('rb') as stream:
                    public_client.put_object(Bucket='260010165', Key=key, Body=stream, IfNoneMatch='*',
                        Metadata={'sha256': digest})
                existing = public_client.get_object(Bucket='260010165', Key=key)
            existing['Body'].close()
            assert existing['ContentLength'] == size
            assert range_hash(public_client, key, size, existing['ETag']) == digest
        asset = {'name': name, 'key': key, 'get': sign('get_object', key), 'bytes': size, 'sha256': digest}
        if extract_to:
            asset['extract_to'] = extract_to
        assets.append(asset)
    upload_asset(code, 'cloud_code.tar.gz', 'code')
    upload_asset(args.output / 'lerobot_source.safe.tar.gz', 'lerobot_source.tar.gz', 'lerobot')
    upload_asset(wheels, 'cpu_data_wheels.tar.gz', 'wheels')
    upload_asset(private / 'private_sources.json', 'private_sources.json')
    upload_asset(private / 'private_storage.json', 'private_storage.json')
    payload = {'root': '/mnt/afs/260010165/behavior_1k/retraining_20261013/source_side_v1',
               'assets': assets, 'status_put': status_put}
    source = Path(__file__).with_name('bootstrap_retraining_cloud.py').read_text()
    program = source + '\nrun(' + repr(payload) + ')\n'
    name = 'behavior-data-cpu-build-' + datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
    record = {'name': name, 'state': 'submission_intent', 'gpu_count': 0, 'cpu_count': 8,
        'memory_gib': 32, 'worker_spec': 'n6ls.iu.i40.8c32g', 'quota_type': 'spot',
        'resource_pool': 'share-cluster', 'workspace': 'share-space', 'timeout_seconds': 86400,
        'status_key': prefix + 'status.json', 'manifest_key': final_key,
        'source_fingerprint': inputs['fingerprint'], 'training_allowed': False,
        'source_count': 463, 'already_transferred': len(prior),
        'root': payload['root'], 'bootstrap_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'assets': [{k: v for k, v in a.items() if k != 'get'} for a in assets],
        'rule_snapshot': '2026-09-11; live rule page inaccessible; 24h CPU job below 128h snapshot limit'}
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    encoded = base64.b64encode(zlib.compress(program.encode())).decode()
    startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(encoded) + ')),"data_bootstrap","exec"))'
    env = dict(os.environ, SCO_CONFIG='/private/tmp/behavior-sco-config',
        SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    command = [CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space', '--aec2-name=share-cluster',
        '--name=' + name, '--job-name=' + name, '--training-framework=pytorch',
        '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
        '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 86400 python3 -c ' + shlex.quote(startup)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    record.update(state='created_needs_describe' if result.returncode == 0 else 'submission_unknown',
                  cli_exit_code=result.returncode)
    write(receipt, record)
    if result.returncode:
        raise RuntimeError('Inspect exact saved build name before any retry')
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
        env=env, capture_output=True, text=True, timeout=40, check=True)
    job = json.loads(result.stdout)
    assert job['ownership']['user_name'] == '260010165' and job['scheduling']['quota_type'] == 'SPOT'
    specs = [s for role in job['roles'] for s in role['resource_spec']]
    assert len(specs) == 1 and specs[0]['name'] == record['worker_spec']
    assert int(specs[0]['limits'].get('nvidia.com/gpu', 0)) == 0
    record.update(state='submitted', uid=job['uid'], platform_state=job['state'])
    write(receipt, record)
    print(json.dumps({k: record[k] for k in ['name', 'uid', 'platform_state', 'gpu_count', 'cpu_count',
        'already_transferred', 'source_count', 'training_allowed']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--prior-receipt', type=Path)
    main(parser.parse_args())

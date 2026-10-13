"""Wait for one finite cloud data build, then relay only verified train exports."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from prepare_retraining_data import download, save
from relay_retraining_sources import REMOTE_CACHE, REMOTE_OUTPUT, SOCKET
from submit_retraining_cloud_probe import client, CLI
from build_retraining_parallel import checked_receipts


def merge_parallel(frozen, baseline, completed, shards):
    all_receipts = list(baseline)
    occupied = {r['source_key'] for r in baseline}
    for shard in shards:
        manifest = completed[shard['name']]
        assert manifest['source_fingerprint'] == frozen['fingerprint']
        assert manifest['all_assigned_sources_accounted_for']
        assert manifest['training_allowed'] is False and manifest['training_executed'] is False
        assigned = set(shard['source_keys'])
        assert not assigned & occupied
        rows = [r for r in frozen['source_archives'] if r['key'] in assigned]
        assert len(rows) == len(assigned)
        checked_receipts(rows, manifest['sources'])
        occupied.update(assigned)
        all_receipts.extend(manifest['sources'])
    verified = checked_receipts(frozen['source_archives'], all_receipts)
    return {'kind': 'source_cloud_prepared_training_exports', 'source_fingerprint': frozen['fingerprint'],
        'sources': list(verified.values()), 'expected_sources': len(verified),
        'all_sources_accounted_for': True, 'cloud_exports_accepted': True,
        'destination_transfer_pending': True, 'data_ready': False,
        'training_allowed': False, 'training_executed': False}


def wait_parallel(store, launch, frozen, a, environment, end, state_path):
    data = Path(launch['baseline_file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == launch['baseline_sha256']
    baseline = json.loads(data)['sources']
    checked_receipts(frozen['source_archives'], baseline, require_complete=False)
    completed = {}
    def read(key):
        r = store.get_object(Bucket='260010165', Key=key)
        try:
            return r['Body'].read()
        finally:
            r['Body'].close()
    while time.monotonic() < end:
        if (a.output / 'CLOUD_STOP').exists():
            raise RuntimeError('Cloud relay STOP is present')
        observations = []
        for shard in launch['shards']:
            if shard['name'] in completed:
                observations.append({'name': shard['name'], 'phase': 'all_cloud_exports_verified'})
                continue
            try:
                status = json.loads(read(shard['status_key']))
            except store.exceptions.NoSuchKey:
                status = {'phase': 'waiting_for_cloud_worker'}
            observations.append({'name': shard['name'], **status})
            if status['phase'] in ('needs_diagnostic', 'bootstrap_needs_diagnostic'):
                raise RuntimeError('Parallel shard requires diagnosis: ' + shard['name'])
            if status['phase'] == 'all_cloud_exports_verified':
                body = read(shard['manifest_key'])
                assert len(body) == status['manifest']['bytes']
                assert hashlib.sha256(body).hexdigest() == status['manifest']['sha256']
                completed[shard['name']] = json.loads(body)
                continue
            reply = subprocess.run([CLI, 'acp', 'jobs', 'describe', shard['name'],
                '--workspace-name=share-space', '-o', 'json'], env=environment,
                capture_output=True, text=True, timeout=40, check=True)
            job = json.loads(reply.stdout)
            assert job['uid'] == shard['uid'] and job['ownership']['user_name'] == '260010165'
            if job['state'] in ('FAILED', 'SUSPENDED', 'DELETED'):
                raise RuntimeError('Parallel shard ended before its manifest: ' + shard['name'])
            if job['state'] == 'SUCCEEDED':
                latest = json.loads(read(shard['status_key']))
                assert latest['phase'] == 'all_cloud_exports_verified', 'Successful job lacks data manifest'
        save(state_path, {'pid': os.getpid(), 'phase': 'waiting_for_parallel_cloud_build',
            'cloud_shards': observations, 'baseline_archives': len(baseline),
            'updated_at': time.time(), 'training_executed': False})
        if len(completed) == len(launch['shards']):
            return merge_parallel(frozen, baseline, completed, launch['shards'])
        time.sleep(30)
    raise RuntimeError('Finite parallel data preparation deadline reached')


def remote(row, prepare_only=False):
    command = [REMOTE_CACHE + '/env/bin/python', REMOTE_CACHE + '/code/sco/accept_cloud_prepared.py',
        '--cache', REMOTE_CACHE, '--output', REMOTE_OUTPUT, '--key', row['key']]
    if prepare_only:
        command.append('--prepare-only')
    paths = ':'.join(REMOTE_CACHE + '/' + p for p in ['code/sco', 'code/scripts', 'code/openpi/src', 'lerobot/src'])
    result = subprocess.run(['ssh', '-S', SOCKET, 'macpro', shlex.join(['env', 'PYTHONPATH=' + paths, *command])],
        capture_output=True, text=True)
    if result.returncode:
        # Local decoder errors contain no signed URLs; preserve the remote evidence.
        raise RuntimeError('Destination acceptance failed: ' + result.stderr[-1200:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def run(a):
    a.cache.mkdir(parents=True, exist_ok=True); a.cache.chmod(0o700)
    lock = (a.output / 'cloud_relay.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads(a.sources.read_text())
    launch_path = a.launch or a.output / 'cloud_build_submission.json'
    launch = json.loads(launch_path.read_text())
    not_before = launch_path.stat().st_mtime - 5
    assert launch['source_fingerprint'] == frozen['fingerprint'] and launch['training_allowed'] is False
    store = client()
    environment = dict(os.environ, SCO_CONFIG='/private/tmp/behavior-sco-config',
        SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    end = time.monotonic() + 26 * 3600
    state_path = a.output / 'cloud_relay_status.json'
    def read(key):
        response = store.get_object(Bucket='260010165', Key=key)
        try:
            if key == launch.get('status_key') and response['LastModified'].timestamp() < not_before:
                return {'phase': 'waiting_for_this_cloud_execution'}
            return json.loads(response['Body'].read())
        finally:
            response['Body'].close()
    if 'shards' in launch:
        manifest = wait_parallel(store, launch, frozen, a, environment, end, state_path)
    while 'shards' not in launch:
        if (a.output / 'CLOUD_STOP').exists() or time.monotonic() > end:
            raise RuntimeError('Cloud relay stopped or finite time limit reached')
        try:
            status = read(launch['status_key'])
        except store.exceptions.NoSuchKey:
            status = {'phase': 'waiting_for_cloud_worker'}
        save(state_path, {'pid': os.getpid(), 'phase': 'waiting_for_cloud_build',
            'cloud_status': status, 'updated_at': time.time(), 'training_executed': False})
        if status['phase'] in {'bootstrap_needs_diagnostic', 'needs_diagnostic'}:
            raise RuntimeError('Cloud data build requires diagnosis')
        if status['phase'] == 'all_cloud_exports_verified':
            response = store.get_object(Bucket='260010165', Key=launch['manifest_key'])
            try:
                body = response['Body'].read()
            finally:
                response['Body'].close()
            assert len(body) == status['manifest']['bytes']
            assert hashlib.sha256(body).hexdigest() == status['manifest']['sha256']
            manifest = json.loads(body); break
        described = subprocess.run([CLI, 'acp', 'jobs', 'describe', launch['name'],
            '--workspace-name=share-space', '-o', 'json'], env=environment,
            capture_output=True, text=True, timeout=40)
        if described.returncode == 0:
            job = json.loads(described.stdout)
            assert job['uid'] == launch['uid'] and job['ownership']['user_name'] == '260010165'
            if job['state'] == 'SUCCEEDED':
                # The worker may finish between the status read and describe.
                latest = read(launch['status_key'])
                if latest['phase'] == 'all_cloud_exports_verified':
                    continue
                raise RuntimeError('Cloud worker succeeded without a verified data manifest')
            if job['state'] in ('FAILED', 'SUSPENDED'):
                raise RuntimeError('Cloud worker ended before publishing a verified manifest')
        time.sleep(30)
    assert manifest['source_fingerprint'] == frozen['fingerprint'] and manifest['all_sources_accounted_for']
    assert manifest['training_allowed'] is False and manifest['training_executed'] is False
    proofs = {r['source_key']: r for r in manifest['sources']}
    assert set(proofs) == {r['key'] for r in frozen['source_archives']}
    save(a.output / 'cloud_prepared_manifest.json', manifest)
    import ssl
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    for number, row in enumerate(frozen['source_archives']):
        if (a.output / 'CLOUD_STOP').exists():
            raise RuntimeError('Cloud relay STOP is present')
        prep = remote(row, True)
        if prep['already_accepted']:
            continue
        proof = proofs[row['key']]
        assert proof['accepted'] and proof['source_sha256'] == row['sha256']
        assert proof['execution_location'] == 'source_cloud_cpu'
        identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
        work = a.cache / identity; work.mkdir(exist_ok=True)
        save(state_path, {'pid': os.getpid(), 'phase': 'relaying_prepared_training_export',
            'source_id': row['id'], 'index': number, 'raw_bytes_not_downloaded': row['bytes'],
            'training_bytes': proof['target']['bytes'], 'training_executed': False})
        for field, filename in [('target', 'data.tar'), ('index_target', 'cloud_index.json')]:
            target = proof[field]
            head = store.head_object(Bucket='260010165', Key=target['key'])
            assert head['ContentLength'] == target['bytes']
            url = store.generate_presigned_url('get_object', Params={'Bucket': '260010165', 'Key': target['key']}, ExpiresIn=3600)
            transfer = {**target, 'etag': head['ETag'], 'signed_get_url': url}
            download(transfer, work / filename, opener)
        save(work / 'cloud_receipt.json', proof)
        destination = prep['work']
        for name in ['data.tar', 'cloud_index.json', 'cloud_receipt.json']:
            subprocess.run(['scp', '-o', 'ControlPath=' + SOCKET, str(work / name),
                'macpro:' + shlex.quote(destination + '/' + name)], check=True, stdout=subprocess.DEVNULL)
        accepted = remote(row)
        assert accepted['accepted']
        import shutil
        shutil.rmtree(work)
    paths = ':'.join(REMOTE_CACHE + '/' + p for p in ['code/sco', 'code/scripts', 'code/openpi/src', 'lerobot/src'])
    while time.monotonic() < end:
        result = subprocess.run(['ssh', '-S', SOCKET, 'macpro', shlex.join(['env', 'PYTHONPATH=' + paths,
            REMOTE_CACHE + '/env/bin/python', REMOTE_CACHE + '/code/sco/finalize_retraining_data.py',
            REMOTE_CACHE, REMOTE_OUTPUT])], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Final data acceptance requires diagnosis')
        report = json.loads(result.stdout)
        save(state_path, {'pid': os.getpid(), 'phase': 'complete' if report['data_ready'] else 'waiting_original_reader',
            'acceptance': report, 'training_executed': False})
        if report['data_ready']:
            return
        if report.get('missing_archives') or report.get('duplicates'):
            raise RuntimeError('Final manifest found missing or duplicate sources; inspect before marking ready')
        time.sleep(60)
    raise RuntimeError('Finite data preparation deadline reached')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', type=Path)
    a = parser.parse_args()
    try:
        run(a)
    except Exception as error:
        save(a.output / 'cloud_relay_failure.json', {'error_type': type(error).__name__,
            'training_executed': False, 'time': time.time()})
        raise SystemExit(1) from None

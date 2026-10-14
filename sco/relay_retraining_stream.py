"""Bounded streaming transfer of accepted cloud packages with parallel CPU readback."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import ssl
import subprocess
import time
import urllib.request

from build_retraining_parallel import checked_receipts
from prepare_retraining_data import download, error_detail, save
from relay_cloud_prepared import merge_parallel
from relay_retraining_sources import REMOTE_CACHE, REMOTE_OUTPUT, SOCKET
from submit_retraining_cloud_probe import client, CLI
from launch_retraining_parallel import environment


def collect_new(store, launch, frozen, proofs):
    """Only accept assigned, fully validated receipts from these exact four roots."""
    rows = {r['key']: r for r in frozen['source_archives']}
    for shard in launch['shards']:
        prefix = shard['manifest_key'].rsplit('/', 1)[0] + '/receipts/'
        assigned = set(shard['source_keys'])
        for page in store.get_paginator('list_objects_v2').paginate(Bucket='260010165', Prefix=prefix):
            for obj in page.get('Contents', []):
                name = obj['Key']
                if not name.endswith('.json') or name in proofs['seen_objects']:
                    continue
                with store.get_object(Bucket='260010165', Key=name)['Body'] as body:
                    receipt = json.loads(body.read())
                key = receipt['source_key']
                if key not in assigned:
                    raise ValueError('Receipt is outside frozen shard assignment')
                checked_receipts([rows[key]], [receipt])
                if key in proofs['by_key'] and proofs['by_key'][key] != receipt:
                    raise ValueError('Conflicting source receipt')
                proofs['by_key'][key] = receipt
                proofs['seen_objects'].add(name)
    return proofs['by_key']


def command_remote(script, args, *, timeout=1800):
    paths = ':'.join(REMOTE_CACHE + '/' + p for p in ['code/sco', 'code/scripts', 'code/openpi/src', 'lerobot/src'])
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
               'PYTHONPATH=' + paths, REMOTE_CACHE + '/env/bin/python', REMOTE_CACHE + '/code/sco/' + script, *args]
    result = subprocess.run(['ssh', '-S', SOCKET, 'macpro', shlex.join(command)], capture_output=True,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Destination CPU readback failed: ' + result.stderr[-2000:])
    # Finalization emits an indented JSON object; other commands may emit
    # progress lines followed by a compact JSON result.
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return json.loads(result.stdout.strip().splitlines()[-1])


def accepted_keys(output):
    code = "import json;from pathlib import Path;p=Path(" + repr(REMOTE_OUTPUT) + ");print(json.dumps([json.loads(f.read_text()) for f in (p/'receipts').glob('*.json')]))"
    result = subprocess.run(['ssh', '-S', SOCKET, 'macpro', shlex.join(['python3', '-c', code])],
                            capture_output=True, text=True, timeout=40, check=True)
    return json.loads(result.stdout)


def transfer(row, proof, args):
    identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
    work = args.cache / identity
    work.mkdir(parents=True, exist_ok=True)
    log = args.output / 'stream_transfers' / (identity + '.json')
    def phase(name, **details):
        save(log, {'source_key': row['key'], 'phase': name, 'updated_at': time.time(),
                   'training_executed': False, **details})
    try:
        phase('preparing_destination')
        prep = command_remote('accept_cloud_prepared.py', ['--cache', REMOTE_CACHE, '--output', REMOTE_OUTPUT,
                              '--key', row['key'], '--prepare-only'])
        if prep['already_accepted']:
            phase('already_accepted'); return row['key']
        store = client()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                             urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        targets = {}
        for field, filename in [('target', 'data.tar'), ('index_target', 'cloud_index.json')]:
            target = proof[field]
            head = store.head_object(Bucket='260010165', Key=target['key'])
            if head['ContentLength'] != target['bytes']:
                raise ValueError('Accepted target size changed')
            url = store.generate_presigned_url('get_object', Params={'Bucket': '260010165', 'Key': target['key']}, ExpiresIn=3600)
            transfer_input = {**target, 'etag': head['ETag'], 'signed_get_url': url}
            if args.direct_macpro:
                targets[filename] = transfer_input
            else:
                phase('downloading_' + field, bytes=target['bytes'])
                download(transfer_input, work / filename, opener, workers=4)
        if args.direct_macpro:
            phase('macpro_direct_download', bytes=proof['target']['bytes'])
            executable = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python'
            command = [executable, REMOTE_CACHE + '/code/sco/download_prepared_targets.py']
            reply = subprocess.run(['ssh', '-S', SOCKET, 'macpro', shlex.join(command)],
                input=json.dumps({'identity': identity, 'targets': targets, 'proof': proof}),
                capture_output=True, text=True, timeout=1800)
            if reply.returncode:
                raise RuntimeError('Direct prepared download failed: ' + reply.stdout[-2000:])
            assert json.loads(reply.stdout.strip().splitlines()[-1])['downloaded']
        else:
            save(work / 'cloud_receipt.json', proof)
            phase('sending_to_macpro')
            for name in ['data.tar', 'cloud_index.json', 'cloud_receipt.json']:
                subprocess.run(['scp', '-o', 'ControlPath=' + SOCKET, str(work / name),
                                'macpro:' + shlex.quote(prep['work'] + '/' + name)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
        phase('destination_reader_and_upload')
        result = command_remote('accept_cloud_prepared.py', ['--cache', REMOTE_CACHE, '--output', REMOTE_OUTPUT, '--key', row['key']])
        if result.get('accepted') is not True:
            raise ValueError('Destination did not accept the prepared package')
        phase('accepted', completed_at=time.time(), bytes=proof['target']['bytes'])
        shutil.rmtree(work)
        return row['key']
    except BaseException as error:
        phase('needs_diagnostic', **error_detail(error))
        raise


def run(args):
    assert 1 <= args.workers <= 8
    args.cache.mkdir(parents=True, exist_ok=True); args.cache.chmod(0o700)
    lock = (args.output / 'cloud_relay.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads(args.sources.read_text()); launch = json.loads(args.launch.read_text())
    assert frozen['fingerprint'] == launch['source_fingerprint'] and not launch['training_allowed']
    rows = {r['key']: r for r in frozen['source_archives']}
    data = Path(launch['baseline_file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == launch['baseline_sha256']
    baseline = json.loads(data)['sources']; checked_receipts(list(rows.values()), baseline, require_complete=False)
    proofs = {'by_key': {r['source_key']: r for r in baseline}, 'seen_objects': set()}
    destination = accepted_keys(args.output)
    checked_receipts(list(rows.values()), destination, require_complete=False)
    accepted = {r['source_key'] for r in destination}
    initial_accepted = len(accepted)
    store = client(); futures = {}; completed = {}; last_refresh = float('-inf'); observations = []
    deadline = time.monotonic() + args.hours * 3600
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while time.monotonic() < deadline:
            if (args.output / 'CLOUD_STOP').exists():
                raise RuntimeError('Streaming transfer STOP is present')
            if time.monotonic() - last_refresh >= 30:
                collect_new(store, launch, frozen, proofs)
                observations = []
                for shard in launch['shards']:
                    if shard['name'] in completed: continue
                    with store.get_object(Bucket='260010165', Key=shard['status_key'])['Body'] as body:
                        status = json.loads(body.read())
                    observations.append({'name': shard['name'], **status})
                    if status['phase'] in ('needs_diagnostic', 'bootstrap_needs_diagnostic'):
                        raise RuntimeError('Source cloud shard needs diagnosis: ' + shard['name'])
                    if status['phase'] == 'all_cloud_exports_verified':
                        with store.get_object(Bucket='260010165', Key=shard['manifest_key'])['Body'] as body:
                            manifest_bytes = body.read()
                        assert len(manifest_bytes) == status['manifest']['bytes']
                        assert hashlib.sha256(manifest_bytes).hexdigest() == status['manifest']['sha256']
                        completed[shard['name']] = json.loads(manifest_bytes)
                    else:
                        result = subprocess.run([CLI, 'acp', 'jobs', 'describe', shard['name'], '--workspace-name=share-space', '-o', 'json'],
                            env=environment(), capture_output=True, text=True, timeout=40, check=True)
                        job = json.loads(result.stdout)
                        assert job['uid'] == shard['uid'] and job['ownership']['user_name'] == '260010165'
                        if job['state'] in ('FAILED', 'SUSPENDED', 'DELETED'):
                            raise RuntimeError('Source cloud job ended before manifest: ' + shard['name'])
                last_refresh = time.monotonic()
            pending = set(futures.values())
            candidates = [key for key in proofs['by_key'] if key not in accepted and key not in pending]
            # Small packages finish first; each slot remains occupied through final readback.
            candidates.sort(key=lambda k: (proofs['by_key'][k]['target']['bytes'], k))
            occupied_bytes = sum(proofs['by_key'][k]['target']['bytes'] for k in futures.values())
            for key in candidates:
                if len(futures) >= args.workers: break
                size = proofs['by_key'][key]['target']['bytes']
                if size > args.inflight_bytes:
                    raise ValueError('One prepared package exceeds the bounded receiver budget')
                if occupied_bytes + size > args.inflight_bytes: continue
                if shutil.disk_usage(args.cache).free < size + 16 * 1024**3: continue
                futures[pool.submit(transfer, rows[key], proofs['by_key'][key], args)] = key
                occupied_bytes += size
            done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                key = futures.pop(future)
                assert future.result() == key
                accepted.add(key)
            save(args.output / 'cloud_relay_status.json', {'phase': 'streaming_parallel_transfer', 'pid': os.getpid(),
                'updated_at': time.time(), 'workers': args.workers, 'source_accepted_archives': len(proofs['by_key']),
                'destination_accepted_archives': len(accepted), 'initial_destination_archives': initial_accepted,
                'in_flight': list(futures.values()), 'cloud_shards': observations,
                'cloud_complete_shards': len(completed), 'training_executed': False})
            if len(completed) == len(launch['shards']) and not futures and len(accepted) == len(rows):
                manifest = merge_parallel(frozen, baseline, completed, launch['shards'])
                save(args.output / 'cloud_prepared_manifest.json', manifest)
                break
            if not futures: time.sleep(5)
        else:
            raise RuntimeError('Finite streaming receiver deadline reached')
    while time.monotonic() < deadline:
        result = command_remote('finalize_retraining_data.py', [REMOTE_CACHE, REMOTE_OUTPUT])
        save(args.output / 'cloud_relay_status.json', {'phase': 'complete' if result['data_ready'] else 'waiting_original_reader',
             'pid': os.getpid(), 'acceptance': result, 'updated_at': time.time(), 'training_executed': False})
        if result['data_ready']: return
        if result.get('missing_archives') or result.get('duplicates'):
            raise RuntimeError('Final data audit has missing or duplicate sources')
        time.sleep(30)
    raise RuntimeError('Final acceptance deadline reached')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--launch', type=Path, required=True); p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--hours', type=float, default=12)
    p.add_argument('--inflight-bytes', type=int, default=8 * 1024**3)
    p.add_argument('--direct-macpro', action='store_true')
    args = p.parse_args()
    try: run(args)
    except BaseException as error:
        save(args.output / 'cloud_relay_failure.json', error_detail(error)); raise

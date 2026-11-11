"""Prefix-cache probe (user 2026-11-03: "前缀缓存的实测还在等你确认。我确认").

One Spot H100 (share-cluster, n6ls.iu.i40.1: 1 GPU, 22 vCPU, 240 GiB), measurement only; see README.md.
The learner model, rollouts and runtime are the frozen rl_multiround_20261102 ones (hash-verified); only
driver.py is replaced by prefix_probe_driver.py and the bootstrap accepts world_size 1.

Usage: python probe.py submit|status|collect
"""
import base64
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import zlib

HERE = Path(__file__).resolve().parent
MR = HERE.parent / 'rl_multiround_20261102'
sys.path.insert(0, str(MR))
import deployment as d  # noqa: E402
import validation as val  # noqa: E402

REMOTE = '/mnt/afs/260010165/behavior_1k/prefix_cache_probe_20261103'
PREFIX = 'behavior1k/prefix-cache-probe-20261103/'
LIMIT = 2400                 # seconds per attempt (expected ~15 min)
CAP_GPU_SECONDS = 3600       # whole probe, all attempts (user-approved cap: 1 GPU-hour)
PROBE = {'episodes': [0, 1, 2], 'rows': 384, 'train_rows': 84, 'host_rows': 64, 'kl_batch_size': 8, 'microbatch': 4}
JOBS = HERE / 'jobs'


def bootstrap_source():
    source = (MR / 'runtime' / 'bootstrap.py').read_text()
    edits = [('if len(gpu) != 4 or', 'if len(gpu) != int(plan["distributed"].get("world_size", 4)) or'),
             ('"--nproc_per_node=4"', '"--nproc_per_node=" + str(int(plan["distributed"].get("world_size", 4)))')]
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def probe_plan(ident):
    plan = val.learner_plan('val-learn-h100-probe', 'h100', d.campaign()['fast_learner'])
    plan.update(experiment='prefix_cache_probe_20261103', output=f'{REMOTE}/results/{ident}', probe=PROBE)
    plan.pop('reference_checkpoint', None)
    plan['distributed'].update(world_size=1)
    plan['phase_limits_seconds']['train1'] = LIMIT
    plan['runtime_storage']['max_persistent_bytes'] = 1024**3
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


def bundle(plan):
    config = d.campaign()
    for name, digest in config['runtime_sha256'].items():
        assert d.sha_file(MR / 'runtime' / name) == digest, name
    contents = {n: (MR / 'runtime' / n).read_bytes() for n in d.TRAIN_FILES + d.PACKAGE if n != 'driver.py'}
    contents['driver.py'] = (HERE / 'prefix_probe_driver.py').read_bytes()
    contents = {n: d.minify_source(body.decode()).encode() for n, body in contents.items()}
    wire = dict(plan)
    for key in ['authorization', 'resume_authorization', 'dependencies', 'dependency_status_file',
                'dependency_campaign_directory', 'previous_generation_directory', 'startup_repair', 'rules',
                'comparisons', 'baseline_sources']:
        wire.pop(key, None)
    contents['plan.json'] = json.dumps(wire, sort_keys=True).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=9) as archive:
        for name, body in contents.items():
            item = tarfile.TarInfo('run/' + name)
            item.size = len(body)
            item.mtime = 0
            archive.addfile(item, io.BytesIO(body))
    return stream.getvalue()


def startup(ident, attempt, body, proof, signed):
    """As rl_multiround_20261102/deployment.startup_command, with this probe's root, name and bootstrap."""
    assets = json.loads((d.ADAPT / 'cloud_smoke/artifacts.json').read_text())
    before = json.loads((d.ADAPT / 'cloud_smoke/local_preflight.json').read_text())
    reuse = {'checkpoint/' + k: item['sha256'] for k, item in assets['checkpoint'].items()}
    reuse['bundle.tar.gz'] = assets['bundle']['sha256']
    reuse.update({k: digest for k, digest in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for path in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py', 'rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/' + path] = d.sha_file(d.ADAPT / 'upstream' / path)
    name = 'b1k-pcp-1103-' + ident
    config = {'root': f'{REMOTE}/jobs/{ident}', 'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
              'task_phase': 'train1', 'rank': 0, 'attempt': attempt, 'job_name': name,
              'bundle': {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}, 'reuse_sha256': reuse,
              'status_put': signed('put_object', 'status.json'), 'result_put': signed('put_object', 'result.tar.gz'),
              'artifact_puts': {}, 'storage_proof': proof, 'bundle_inline': base64.b64encode(body).decode()}
    source = bootstrap_source() + '\nguarded_run(' + repr(config) + ')\n'
    packed = base64.b64encode(zlib.compress(source.encode(), level=9)).decode()
    command = f'timeout --signal=TERM --kill-after=60s {LIMIT - 60}s python3 -u -c ' + shlex.quote(
        'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
    assert len(command) < 64000, len(command)
    return name, command


def rows():
    found = []
    for path in sorted(JOBS.glob('*/submission.json')):
        row = json.loads(path.read_text())
        latest = path.with_name('latest.json')
        found.append(json.loads(latest.read_text()) if latest.exists() else row)
    return found


def used_gpu_seconds(found):
    from datetime import datetime
    parse = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    total = 0.0
    for row in found:
        if row.get('start_time') and row.get('complete_time'):
            total += (parse(row['complete_time']) - parse(row['start_time'])).total_seconds()
        elif row.get('state') not in d.TERMINAL | {'NOT_CREATED'}:
            total += row['max_seconds']
    return total


def submit():
    found = rows()
    assert all(r.get('state') in d.TERMINAL | {'NOT_CREATED'} for r in found), 'An attempt is still active'
    assert not any(r.get('state') == 'SUCCEEDED' for r in found), 'Already succeeded; run collect'
    assert used_gpu_seconds(found) + LIMIT <= CAP_GPU_SECONDS, 'GPU-hour cap reached'
    attempt = len(found) + 1
    ident = f'h100-a{attempt}'
    folder = JOBS / ident
    prefix = PREFIX + 'jobs/' + ident + '/'
    signer, client = d.status_client(), d.aoss_client()
    try:
        signed = lambda method, item: signer.generate_presigned_url(
            method, Params={'Bucket': '260010165', 'Key': prefix + item}, ExpiresIn=2 * 86400)
        plan = probe_plan(ident)
        body = bundle(plan)
        proof = d.storage_proof(plan)
        name, command = startup(ident, attempt, body, proof, signed)
        client.put_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz', Body=body)
        folder.mkdir(parents=True, exist_ok=False)
        d.save(folder / 'worker_plan.json', plan)
        (folder / 'bootstrap.py').write_text(bootstrap_source())
        pool = d.POOLS['h100']
        row = {'id': ident, 'job': name, 'role': 'probe', 'pool': 'h100', 'gpu_count': 1, 'quota': 'spot',
               'aoss_prefix': prefix, 'output': plan['output'], 'max_seconds': LIMIT, 'state': 'submitting',
               'submitted_at': time.time(), 'command_chars': len(command), 'bundle_bytes': len(body),
               'startup_sha256': hashlib.sha256(command.encode()).hexdigest()}
        with (folder / 'submission.json').open('x') as handle:
            json.dump(row, handle, indent=2)
        assert d.describe(pool['template'])['resource_pool']['name'] == pool['pool']
        with (d.ANALYSIS / 'rl_gpu_submit.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run([d.CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
                                     '--copy-job-name=' + pool['template'], '--name=' + name, '--job-name=' + name,
                                     '--worker-spec=' + pool['spec'][1], '--worker-nodes=1', '--quota-type=spot',
                                     '--retry-times=0', '--enable-fault-tolerance=false',
                                     '--enable-anomaly-detection=false', '--command=' + command],
                                    env=d.ENV, capture_output=True, text=True, timeout=90)
        row['submit_returncode'] = result.returncode
        if result.returncode:
            row['submit_error'] = re.sub(r'https?://\S+', '<url>', result.stderr or result.stdout)[-600:]
        d.save(folder / 'submission.json', row)
        job = d.describe(name)
        assert job['scheduling']['quota_type'] == 'SPOT' and job['resource_pool']['name'] == pool['pool']
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
        row.update(uid=job['uid'], state=job['state'])
        d.save(folder / 'submission.json', row)
        print(json.dumps({k: row[k] for k in ['job', 'uid', 'state', 'gpu_count', 'command_chars', 'bundle_bytes']}))
    finally:
        client.close()
        signer.close()


def status():
    client = d.aoss_client()
    try:
        for path in sorted(JOBS.glob('*/submission.json')):
            row = json.loads(path.read_text())
            if 'uid' in row:
                job = d.describe(row['job'])
                assert job['uid'] == row['uid']
                row.update(state=job['state'], start_time=job.get('start_time'), complete_time=job.get('complete_time'))
            try:
                body = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'status.json')['Body'].read()
                row['progress'] = json.loads(body)
            except client.exceptions.ClientError:
                pass
            d.save(path.with_name('latest.json'), row)
            progress = row.get('progress') or {}
            print(row['id'], row['state'], progress.get('stage'), progress.get('error_type'))
    finally:
        client.close()


def collect():
    client = d.aoss_client()
    try:
        for path in sorted(JOBS.glob('*/latest.json')):
            row = json.loads(path.read_text())
            if row['state'] not in d.TERMINAL:
                continue
            try:
                body = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'result.tar.gz')['Body'].read()
            except client.exceptions.ClientError:
                print(row['id'], 'no result')
                continue
            root = path.parent / 'retrieved'
            with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
                for item in archive:
                    name = Path(item.name)
                    assert item.isfile() and not name.is_absolute() and '..' not in name.parts and name.suffix == '.json'
                    data = archive.extractfile(item).read()
                    json.loads(data)
                    (root / name).parent.mkdir(parents=True, exist_ok=True)
                    (root / name).write_bytes(data)
            print(row['id'], 'collected', sorted(str(p.relative_to(root)) for p in root.rglob('*.json')))
    finally:
        client.close()


if __name__ == '__main__':
    {'submit': submit, 'status': status, 'collect': collect}[sys.argv[1]]()

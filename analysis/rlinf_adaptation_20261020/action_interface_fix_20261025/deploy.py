"""Deploy two independent frozen-checkpoint pairs, with fresh quota admission."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLI = str(Path.home() / '.local/bin/sco')
IMAGE = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def describe(name):
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', name,
                             '--workspace-name=share-space', '-o', 'json'],
                            capture_output=True, text=True, check=True, timeout=45)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    rank = parser.parse_args().rank
    target = HERE / 'attempt02' / f'r{rank}'
    target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / 'submission.json'
    assert not receipt_path.exists(), 'Resolve the existing receipt; never submit twice.'
    plan = json.loads((HERE / 'plan.json').read_text())
    assert plan['validation']['no_training'] and plan['validation']['max_new_episodes'] == 4
    assert plan['validation']['max_active_gpus'] == 2 and plan['max_job_seconds'] == 7200
    assert json.loads((HERE / 'test_result.json').read_text())['passed']
    instance = plan['validation']['instances'][rank]
    prior = json.loads((HERE / f'r{rank}/submission.json').read_text())
    terminal = describe(prior['job'])
    assert terminal['uid'] == prior['uid'] and terminal['state'] == 'FAILED'
    status = json.loads((HERE / f'r{rank}/status.json').read_text())
    assert status['state'] == 'bootstrap_failed' and status['phase'] == 'verify_original_checkpoint'
    used = (datetime.fromisoformat(terminal['complete_time'].replace('Z', '+00:00'))
            - datetime.fromisoformat(terminal['start_time'].replace('Z', '+00:00'))).total_seconds()
    assert 0 <= used <= 200, 'Recalculate remaining total resource budget.'
    job = f'b1k-iface-1025-a2-r{rank}'
    remote = f'/mnt/afs/260010165/behavior_1k/rlinf_interface_fix_20261025/pilot01/attempt02/r{rank}'
    prefix = f'behavior1k/action-interface-fix-20261025/pilot01/attempt02/r{rank}/'
    plan.update(job_name=job, output=remote + '/results', gpu_count=1, eval_instances=[instance])
    plan.update(max_job_seconds=7000, compute_timeout_seconds=6400)
    plan['recovery'] = {'attempt': 2, 'previous_uid': prior['uid'], 'previous_state': 'FAILED',
                        'previous_gpu_seconds': used, 'previous_controls': 0,
                        'reason': 'Bootstrap checked the train split for public evaluation instances.',
                        'automatic_further_retries': False}
    plan['validation']['instance'] = instance
    plan['evaluation']['cases'] = [[instance, 0]]
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    capacity = query_standard_capacity(CLI, dict(os.environ))
    write(target / 'capacity.json', capacity)
    quota = 'reserved' if capacity['used_gpus'] < capacity['standard_limit'] else 'spot'
    plan['quota_type'] = quota
    write(target / 'plan.json', plan)
    files = ['common.py', 'training.py', 'driver.py', 'experiment.py', 'helpers.py', 'simulator.py']
    files += [str(p.relative_to(HERE)) for p in sorted((HERE / 'b1k_rlinf').glob('*.py'))]
    buffer = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in files + ['plan.json']:
            raw = (target / name if name == 'plan.json' else HERE / name).read_bytes()
            item = tarfile.TarInfo('run/' + name)
            item.size = len(raw)
            item.mtime = 0
            archive.addfile(item, io.BytesIO(raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
    bundle = buffer.getvalue()
    (target / 'bundle.tar.gz').write_bytes(bundle)
    previous = json.loads((HERE.parent / 'cloud_smoke/artifacts.json').read_text())
    reuse = {'checkpoint/' + k: v['sha256'] for k, v in previous['checkpoint'].items()}
    reuse['bundle.tar.gz'] = previous['bundle']['sha256']
    before = json.loads((HERE.parent / 'cloud_smoke/local_preflight.json').read_text())
    reuse.update({k: v for k, v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py',
                 'rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/' + name] = hashlib.sha256((HERE.parent / 'upstream' / name).read_bytes()).hexdigest()
    receipt = {'job': job, 'state': 'preparing', 'rank': rank, 'instance': instance,
               'gpu_count': 1, 'max_seconds': 7000, 'quota': quota, 'source_hashes': hashes,
               'bundle_bytes': len(bundle), 'bundle_sha256': hashlib.sha256(bundle).hexdigest(),
               'output': remote + '/results', 'aoss_prefix': prefix,
               'submitted_at': datetime.now(timezone.utc).isoformat()}
    write(receipt_path, receipt)
    client = aoss_client()
    try:
        key = prefix + 'bundle.tar.gz'
        client.put_object(Bucket='260010165', Key=key, Body=bundle)
        assert client.get_object(Bucket='260010165', Key=key)['Body'].read() == bundle

        def signed(key):
            return client.generate_presigned_url('put_object',
                Params={'Bucket': '260010165', 'Key': prefix + key}, ExpiresIn=86400)

        config = {'root': remote, 'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
                  'rank': rank, 'job_name': job, 'bundle_inline': base64.b64encode(bundle).decode(),
                  'bundle': {'bytes': len(bundle), 'sha256': receipt['bundle_sha256']},
                  'reuse_sha256': reuse, 'status_put': signed('status.json'),
                  'result_put': signed('report.json'), 'artifact_puts': {}}
        source = (HERE / 'bootstrap.py').read_text() + '\nguarded_run(' + repr(config) + ')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        cmd = 'timeout --signal=TERM --kill-after=30s 6970s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
        receipt.update(startup_sha256=hashlib.sha256(cmd.encode()).hexdigest(), state='submitting')
        write(receipt_path, receipt)
        try:
            response = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00', '--name=' + job, '--job-name=' + job,
                '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=' + quota,
                '--retry-times=0', '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url=' + IMAGE, '--command=' + cmd],
                capture_output=True, text=True, timeout=90)
            receipt['submit_returncode'] = response.returncode
        except subprocess.TimeoutExpired:
            receipt['submit_returncode'] = 'unknown_timeout'
        actual = describe(job)
        assert actual['ownership']['user_name'] == '260010165'
        assert actual['scheduling']['quota_type'] == quota.upper()
        assert actual['roles'][0]['total_replicas'] == 1
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest() == receipt['startup_sha256']
        receipt.update(uid=actual['uid'], state=actual['state'])
        write(receipt_path, receipt)
        print(json.dumps({k: receipt[k] for k in ['job', 'uid', 'state', 'quota', 'instance', 'bundle_bytes']}))
    finally:
        client.close()


if __name__ == '__main__':
    main()

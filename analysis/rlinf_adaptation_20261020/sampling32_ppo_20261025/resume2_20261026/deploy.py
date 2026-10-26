"""Submit exactly one verified worker per rank for the 32-trajectory experiment."""
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
ROOT = HERE.parents[3]
BASE = HERE.parent.parent
CLI = str(Path.home() / '.local/bin/sco')
IMAGE = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
PREFIX = 'behavior1k/rlinf-sampling32-20261025/resume2-20261026/pilot01/'
REMOTE = '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/resume2_20261026/pilot01'


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
    target = HERE / f'r{rank}'; target.mkdir(parents=True, exist_ok=True)
    receipt_path = target / 'submission.json'
    assert not receipt_path.exists(), 'Resolve the existing receipt; never submit twice.'
    for name in ('resume_test_result.json', 'configuration_equivalence.json', 'preflight_result.json'):
        assert json.loads((HERE / name).read_text())['passed']
    plan = json.loads((HERE / 'plan.json').read_text())
    assert plan['gpu_count'] == 2 and plan['protocol']['episodes_per_round'] == 32
    assert plan['optimizer']['updates_per_round'] == [17, 13] and plan['max_job_seconds'] == 51900
    assert plan['action_interface']['gripper_open_tolerance'] == 0
    previous = plan['resume']['previous_jobs'][rank]
    stopped = describe(previous['name'])
    assert stopped['uid'] == previous['uid'] and stopped['state'] == 'FAILED'
    workers = subprocess.run([CLI,'acp','jobs','get-workers',previous['name'],'--workspace-name=share-space'],capture_output=True,text=True,check=True).stdout
    assert 'Failed' in workers and 'Running' not in workers
    old_receipt = json.loads((HERE.parent/f'attempt02/r{rank}/submission.json').read_text())
    for name, expected in old_receipt['source_hashes'].items():
        if name in ('training.py', 'helpers.py') or name.startswith('b1k_rlinf/'):
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, 'Recovery changed runtime logic: '+name
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name'] == '260010165'
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    assert template['roles'][0]['image_path'] == IMAGE
    assert template['roles'][0]['resource_spec'][0]['name'] == 'n12lp.nn.i10a.1'
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    capacity = query_standard_capacity(CLI, dict(os.environ))
    write(target / 'capacity.json', capacity)
    quota = 'reserved' if capacity['used_gpus'] < capacity['standard_limit'] else 'spot'
    job = f'b1k-s32-resume2-1026-r{rank}'
    files = ['common.py', 'training.py', 'driver.py', 'experiment.py', 'parallel.py',
             'helpers.py', 'simulator.py', 'plan.json']
    files += [str(p.relative_to(HERE)) for p in sorted((HERE / 'b1k_rlinf').glob('*.py'))]
    buffer = io.BytesIO(); hashes = {}
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in files:
            raw = (HERE / name).read_bytes()
            item = tarfile.TarInfo('run/' + name); item.size = len(raw); item.mtime = 0
            archive.addfile(item, io.BytesIO(raw))
            hashes[name] = hashlib.sha256(raw).hexdigest()
    bundle = buffer.getvalue()
    (target / 'bundle.tar.gz').write_bytes(bundle)
    previous = json.loads((BASE / 'cloud_smoke/artifacts.json').read_text())
    reuse = {'checkpoint/' + k: v['sha256'] for k, v in previous['checkpoint'].items()}
    reuse['bundle.tar.gz'] = previous['bundle']['sha256']
    before = json.loads((BASE / 'cloud_smoke/local_preflight.json').read_text())
    reuse.update({k: v for k, v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py',
                 'rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/' + name] = hashlib.sha256((BASE / 'upstream' / name).read_bytes()).hexdigest()
    receipt = {'job': job, 'state': 'preparing', 'rank': rank, 'gpu_count': 1, 'max_seconds': 51900,
               'quota': quota, 'source_hashes': hashes, 'output': plan['output'],
               'bundle_bytes': len(bundle), 'bundle_sha256': hashlib.sha256(bundle).hexdigest(),
               'aoss_prefix': PREFIX, 'submitted_at': datetime.now(timezone.utc).isoformat()}
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    client = aoss_client()
    try:
        key = PREFIX + f'bundle-r{rank}.tar.gz'
        client.put_object(Bucket='260010165', Key=key, Body=bundle)
        assert client.get_object(Bucket='260010165', Key=key)['Body'].read() == bundle
        def signed(key):
            return client.generate_presigned_url('put_object',
                Params={'Bucket': '260010165', 'Key': PREFIX + key}, ExpiresIn=604800)
        artifacts = {}
        if rank == 0:
            for relative in ['checkpoints/round_02/trainable.pt', 'checkpoints/round_02/optimizer.pt']:
                artifacts[relative] = {'url': signed(relative), 'key': PREFIX + relative}
        config = {'root': REMOTE + f'/worker_{rank}',
                  'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
                  'rank': rank, 'job_name': job, 'bundle_inline': base64.b64encode(bundle).decode(),
                  'bundle': {'bytes': len(bundle), 'sha256': receipt['bundle_sha256']},
                  'reuse_sha256': reuse, 'status_put': signed(f'status_rank{rank}.json'),
                  'result_put': signed(f'result_rank{rank}.tar.gz'), 'artifact_puts': artifacts}
        source = (HERE / 'bootstrap.py').read_text() + '\nguarded_run(' + repr(config) + ')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        cmd = 'timeout --signal=TERM --kill-after=60s 51840s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
        assert len(cmd) < 64000
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
        print(json.dumps({k: receipt[k] for k in ['job', 'uid', 'state', 'quota', 'bundle_bytes']}))
    finally:
        client.close()


if __name__ == '__main__':
    main()

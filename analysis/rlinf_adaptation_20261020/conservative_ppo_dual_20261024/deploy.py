"""Submit exactly two cooperating one-GPU workers after retiring the old writer."""
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
BASE = '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01'
REMOTE = '/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/dual01'
PREFIX = 'behavior1k/rlinf-conservative-20261024/dual01/'
IMAGE = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
JOBS = ['b1k-ppoc2-d1-1024-r0', 'b1k-ppoc2-d1-1024-r1']


def write(name, value):
    temporary = HERE / (name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(HERE / name)


def describe(name):
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
                            capture_output=True, text=True, check=True, timeout=45)
    return json.loads(result.stdout)


def main():
    for name in ('training_test_result.json', 'parallel_test_result.json', 'configuration_equivalence.json'):
        assert json.loads((HERE / name).read_text())['passed']
    assert not (HERE / 'submission_intent.json').exists(), 'Resolve existing receipt before resubmitting'
    old = describe('b1k-ppoc2-1024-a03')
    assert old['uid'] == '6a26775e09ef45dca42200545b48170f'
    assert old['ownership']['user_name'] == '260010165'
    assert old['state'] in {'SUSPENDED', 'FAILED', 'TERMINATED', 'CANCELED', 'STOPPED', 'SUCCEEDED'}
    workers = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', old['name'], '--workspace-name=share-space'],
                             capture_output=True, text=True, check=True, timeout=45)
    assert old['uid'] in workers.stdout and 'Deleted' in workers.stdout
    write('previous_writer_verified.json', {'job': old['name'], 'uid': old['uid'],
          'state': old['state'], 'worker_phase': 'Deleted', 'writer_stopped': True})
    plan = json.loads((HERE / 'plan.json').read_text())
    plan['migration']['previous_writer_stopped'] = True
    write('plan.json', plan)
    reference = json.loads((HERE.parent / 'conservative_ppo_storage_20261024/plan.json').read_text())
    identical = json.loads((HERE / 'configuration_equivalence.json').read_text())['identical_fields']
    assert all(plan[key] == reference[key] for key in identical)
    assert plan['gpu_count'] == 2 and plan['rounds'] == 2
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name'] == '260010165'
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    assert template['roles'][0]['image_path'] == IMAGE
    assert template['roles'][0]['resource_spec'][0]['name'] == 'n12lp.nn.i10a.1'
    files = ['common.py', 'simulator.py', 'helpers.py', 'training.py', 'experiment.py', 'parallel.py', 'driver.py', 'plan.json']
    files += [str(p.relative_to(HERE)) for p in sorted((HERE / 'b1k_rlinf').glob('*.py'))]
    blob = io.BytesIO()
    members = {}
    with tarfile.open(fileobj=blob, mode='w:gz') as tar:
        for name in files:
            data = (HERE / name).read_bytes()
            item = tarfile.TarInfo('run/' + name)
            item.size = len(data)
            item.mtime = 0
            tar.addfile(item, io.BytesIO(data))
            members[name] = hashlib.sha256(data).hexdigest()
    bundle = blob.getvalue()
    (HERE / 'bundle.tar.gz').write_bytes(bundle)
    assert len(bundle) < 1024**2
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(), 'gpu_count': 2,
               'max_worker_hours': 6, 'max_gpu_hours': 12, 'source_sha256': members,
               'bundle_sha256': hashlib.sha256(bundle).hexdigest(), 'workers': [], 'state': 'preparing'}
    with (HERE / 'submission_intent.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    client = aoss_client()
    try:
        key = PREFIX + 'bundle-' + receipt['bundle_sha256'] + '.tar.gz'
        client.put_object(Bucket='260010165', Key=key, Body=bundle,
                          Metadata={'sha256': receipt['bundle_sha256']})
        assert client.get_object(Bucket='260010165', Key=key)['Body'].read(len(bundle) + 1) == bundle
        old_art = json.loads((HERE.parent / 'cloud_smoke/artifacts.json').read_text())
        reuse = {'checkpoint/' + name: item['sha256'] for name, item in old_art['checkpoint'].items()}
        reuse['bundle.tar.gz'] = old_art['bundle']['sha256']
        initial = json.loads((HERE.parent / 'cloud_smoke/local_preflight.json').read_text())
        reuse.update({k: v for k, v in initial['execution_source_sha256'].items() if k.startswith('adapter/')})
        for relative in ('rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py', 'rlinf/models/embodiment/openpi/tasks/rl.py'):
            reuse['upstream/' + relative] = hashlib.sha256((HERE.parent / 'upstream' / relative).read_bytes()).hexdigest()

        def signed(method, key):
            return client.generate_presigned_url(method, Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=604800)

        for rank, job in enumerate(JOBS):
            capacity = query_standard_capacity(CLI, dict(os.environ))
            write(f'admission_rank{rank}.json', capacity)
            # Each step rechecks all outstanding applications; never evict another project.
            quota = 'reserved' if capacity['used_gpus'] < capacity['standard_limit'] else 'spot'
            artifacts = {}
            if rank == 0:
                for index in (1, 2):
                    for filename in ('trainable.pt', 'optimizer.pt'):
                        relative = f'checkpoints/round_{index:02d}/{filename}'
                        target = PREFIX + relative
                        artifacts[relative] = {'url': signed('put_object', target), 'key': target}
            config = {'root': REMOTE + f'/worker_{rank}', 'base': BASE, 'rank': rank, 'job_name': job,
                      'bundle_inline': base64.b64encode(bundle).decode(), 'reuse_sha256': reuse,
                      'bundle': {'bytes': len(bundle), 'sha256': receipt['bundle_sha256']},
                      'status_put': signed('put_object', PREFIX + f'status_rank{rank}.json'),
                      'result_put': signed('put_object', PREFIX + f'result_rank{rank}.tar.gz'),
                      'artifact_puts': artifacts}
            source = (HERE / 'bootstrap.py').read_text() + '\nguarded_run(' + repr(config) + ')\n'
            encoded = base64.b64encode(zlib.compress(source.encode())).decode()
            command = 'timeout --signal=TERM --kill-after=60s 21540s python3 -u -c ' + shlex.quote(
                'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(encoded) + ')))')
            assert len(command) < 64000
            intent = {'rank': rank, 'job_name': job, 'quota_type': quota, 'state': 'submitting',
                      'startup_sha256': hashlib.sha256(command.encode()).hexdigest(),
                      'submission_attempted': True}
            receipt['workers'].append(intent)
            write('submission_intent.json', receipt)
            argv = [CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
                    '--copy-job-name=b1k-repeat1688-1019-batch-00', '--name=' + job, '--job-name=' + job,
                    '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=' + quota,
                    '--retry-times=0', '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                    '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                    '--container-image-url=' + IMAGE, '--command=' + command]
            try:
                completed = subprocess.run(argv, capture_output=True, text=True, timeout=90)
                intent['submit_returncode'] = completed.returncode
            except subprocess.TimeoutExpired:
                intent['submit_error'] = 'timeout_outcome_unknown'
            actual = describe(job)  # An uncertain submission is resolved only by this exact name.
            assert actual['ownership']['user_name'] == '260010165'
            assert actual['scheduling']['quota_type'] == quota.upper()
            assert actual['resource_pool']['name'] == 'computing-cluster-5090-01g'
            assert len(actual['roles']) == 1 and actual['roles'][0]['total_replicas'] == 1
            spec = actual['roles'][0]['resource_spec'][0]
            assert spec['name'] == 'n12lp.nn.i10a.1' and int(spec['limits']['nvidia.com/gpu']) == 1
            assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest() == intent['startup_sha256']
            intent.update(uid=actual['uid'], state=actual['state'])
            write('submission_intent.json', receipt)
            print(json.dumps({k: intent[k] for k in ('rank', 'job_name', 'uid', 'quota_type', 'state')}), flush=True)
        receipt['state'] = 'submitted'
        write('submission_intent.json', receipt)
    finally:
        client.close()


if __name__ == '__main__':
    main()

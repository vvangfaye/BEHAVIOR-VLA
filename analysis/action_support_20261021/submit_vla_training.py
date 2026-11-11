"""Submit each frozen pilot variant once, after data and GPU gates pass."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PREFIX = 'behavior1k/action-support-20261021/vla-pilot-v1/'
AFS = '/mnt/afs/260010165/behavior_1k/action_support_20261021'
OLD = '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01'
IMAGE = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
TEMPLATE = 'b1k-repeat1688-1019-batch-00'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def cli(*arguments, timeout=45):
    return subprocess.run([str(Path.home() / '.local/bin/sco'), *arguments],
                          capture_output=True, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['standard', 'conditioned', 'projected'], required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    data = HERE / 'vla_pilot'
    plan_path = data / 'pilot_training_plan.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['source_sha256'].items():
        assert digest(HERE / name) == expected, 'Frozen training source changed'
    if not (data / 'image_receipt.json').exists():
        print(json.dumps({'ready': False, 'waiting_for': 'immutable image cache'}))
        return
    image_receipt = json.loads((data / 'image_receipt.json').read_text())
    assert image_receipt['complete'] and image_receipt['data_plan_sha256'] == plan['data_plan_sha256']
    assert json.loads((data / 'full_data_verification.json').read_text())['passed']
    assert json.loads((data / 'timestamp_alignment.json').read_text())['passed']
    if not (HERE / 'gpu_probe_job/result.json').exists():
        print(json.dumps({'ready': False, 'waiting_for': 'real GPU optimizer result'}))
        return
    gpu = json.loads((HERE / 'gpu_probe_job/result.json').read_text())
    assert gpu['passed'] and gpu['worker_returncode'] == 0 and gpu['completed_updates'] == 2
    assert json.loads((HERE / 'vla_data_verification.json').read_text())['passed']
    assert json.loads((HERE / 'sft_equivalence.json').read_text())['passed']
    objects = {}
    for name in ['data_plan.json', 'raw_targets.npz', 'rgb.npy', 'image_receipt.json']:
        path = data / name
        objects[name] = {'bytes': path.stat().st_size, 'sha256': digest(path), 'key': PREFIX + 'data/' + name}
    assert objects['rgb.npy']['sha256'] == image_receipt['image_sha256']
    work = data / ('submission_seed17_' + args.method)
    work.mkdir(exist_ok=True)
    archive = work / 'code.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as tar:
            for name in plan['source_sha256']:
                tar.add(HERE / name, arcname=name, recursive=False)
            tar.add(plan_path, arcname=plan_path.name, recursive=False)
    artifact = {'bytes': archive.stat().st_size, 'sha256': digest(archive),
                'key': PREFIX + 'code/' + digest(archive) + '.tar.gz'}
    job_name = 'b1k-support-1021-' + args.method + '-s17'
    if not args.submit:
        print(json.dumps({'ready': True, 'job_name': job_name, 'data_bytes': sum(x['bytes'] for x in objects.values())}))
        return
    intent_path = work / 'submission_intent.json'
    if intent_path.exists():
        print(json.dumps({'already_has_submission_intent': True, **json.loads(intent_path.read_text())}))
        return
    sys.path.insert(0, str(PROJECT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    try:
        if args.method != 'standard':
            try:
                response = client.get_object(Bucket='260010165', Key=PREFIX + 'data_ready.json')
            except ClientError:
                print(json.dumps({'ready': False, 'waiting_for': 'first variant shared-AFS data staging'}))
                return
            try:
                ready = json.loads(response['Body'].read(128 * 1024))
            finally:
                response['Body'].close()
            assert ready['ready'] and ready['data_plan_sha256'] == plan['data_plan_sha256']
        response = cli('acp', 'jobs', 'describe', TEMPLATE, '--workspace-name=share-space', '-o', 'json')
        assert response.returncode == 0
        template = json.loads(response.stdout)
        assert template['uid'] == '168d5223752b415caf67bd27ed504a83'
        assert template['ownership']['user_name'] == '260010165'
        assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
        assert template['roles'][0]['image_path'] == IMAGE
        intent = {'job_name': job_name, 'method': args.method, 'seed': 17, 'submission_attempted': False,
                  'plan_sha256': digest(plan_path), 'gpu_count': 1, 'max_seconds': 14400}
        with intent_path.open('x') as stream:
            json.dump(intent, stream, indent=2)
        def upload(path, item):
            try:
                existing = client.head_object(Bucket='260010165', Key=item['key'])
            except ClientError as error:
                if error.response.get('Error', {}).get('Code') not in ('404', 'NoSuchKey', 'NotFound'):
                    raise
                existing = None
            if existing is not None:
                assert existing['ContentLength'] == item['bytes'] and existing['Metadata']['sha256'] == item['sha256']
                return
            client.upload_file(str(path), '260010165', item['key'], ExtraArgs={'Metadata': {'sha256': item['sha256']}})
            head = client.head_object(Bucket='260010165', Key=item['key'])
            assert head['ContentLength'] == item['bytes'] and head['Metadata']['sha256'] == item['sha256']
        for name, item in objects.items():
            upload(data / name, item)
        upload(archive, artifact)
        def signed(item):
            return {**item, 'url': client.generate_presigned_url('get_object', Params={'Bucket': '260010165', 'Key': item['key']}, ExpiresIn=86400)}
        config = {'root': AFS + '/pilot-seed17-' + args.method, 'method': args.method, 'seed': 17,
                  'code': signed(artifact), 'data': {name: signed(item) for name, item in objects.items()},
                  'upstream': OLD + '/upstream', 'adapter': OLD + '/adapter', 'site': OLD + '/site',
                  'python': '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python',
                  'data_ready_put': client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': PREFIX + 'data_ready.json'}, ExpiresIn=86400),
                  'progress_put': client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': PREFIX + args.method + '-s17-progress.json'}, ExpiresIn=86400),
                  'result_put': client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': PREFIX + args.method + '-s17-result.json'}, ExpiresIn=86400)}
        source = (HERE / 'vla_training_bootstrap.py').read_text() + '\nrun(' + repr(config) + ')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        command = 'timeout --signal=TERM --kill-after=60s 14340s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
        intent.update(submission_attempted=True, startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent_path, intent)
        try:
            response = cli('acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=' + TEMPLATE,
                           '--name=' + job_name, '--job-name=' + job_name, '--worker-spec=n12lp.nn.i10a.1',
                           '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
                           '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                           '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                           '--container-image-url=' + IMAGE, '--command=' + command, timeout=120)
            intent['submit_returncode'] = response.returncode
        except subprocess.TimeoutExpired:
            intent['submit_outcome'] = 'unknown_timeout'
        write(intent_path, intent)
        response = cli('acp', 'jobs', 'describe', job_name, '--workspace-name=share-space', '-o', 'json')
        if response.returncode == 0:
            job = json.loads(response.stdout)
            intent.update(uid=job['uid'], state=job['state'])
        else:
            intent['state'] = 'requires_exact_name_query'
        write(intent_path, intent)
        print(json.dumps({k: intent.get(k) for k in ['job_name', 'uid', 'state', 'submit_returncode']}))
    finally:
        client.close()


if __name__ == '__main__':
    main()

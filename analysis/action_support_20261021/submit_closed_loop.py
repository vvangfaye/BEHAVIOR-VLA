"""Submit at most one exact panel episode; gate expansion on actual results."""
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
PREFIX = 'behavior1k/action-support-20261021/closed-loop-v1/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cli(*args, timeout=45):
    return subprocess.run([str(Path.home() / '.local/bin/sco'), *args], capture_output=True,
                          text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    directory = HERE / 'closed_loop_v1'
    plan = json.loads((directory / 'plan.json').read_text())
    pair = [args.task, args.instance]
    if args.method not in plan['protocol']['methods'] or pair not in plan['pairs']:
        raise ValueError('Episode is outside the frozen panel')
    for name, expected in plan['source_sha256'].items():
        assert sha(directory / 'frozen' / name) == expected
    assert json.loads((HERE / 'native_websocket_verification.json').read_text())['passed']
    assert json.loads((HERE / 'vla_interface_v2.json').read_text())['passed']
    first = [args.method, *pair] == plan['resource_bound']['first_job']
    index = plan['pairs'].index(pair)
    tag = {'original1688977': 'orig', 'standard': 'std', 'conditioned': 'cond', 'projected': 'proj'}[args.method]
    name = f'b1k-support-1021-eval-{tag}-{index:02d}'
    work = directory / name
    work.mkdir(exist_ok=True)
    archive = work / 'code.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(directory / 'plan.json', arcname='plan.json', recursive=False)
            for source in sorted((directory / 'frozen').glob('*.py')):
                tar.add(source, arcname=source.name, recursive=False)
    intent_path = work / 'submission_intent.json'
    if intent_path.exists():
        print(json.dumps({'existing_intent': True, **json.loads(intent_path.read_text())}))
        return
    if not args.submit:
        print(json.dumps({'job_name': name, 'first_episode': first, 'max_gpu_hours': 8,
                          'bundle_bytes': archive.stat().st_size, 'submitted': False}))
        return
    sys.path.insert(0, str(PROJECT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    def get_json(key):
        value = client.get_object(Bucket='260010165', Key=key)
        try:
            raw = value['Body'].read(256 * 1024 + 1)
            assert len(raw) <= 256 * 1024
            return json.loads(raw)
        finally:
            value['Body'].close()
    try:
        if not first:
            first_result = get_json(PREFIX + 'b1k-support-1021-eval-orig-00/result.json')
            assert first_result['accepted'] and first_result['worker_returncode'] == 0
            assert first_result['protocol_sha256'] == plan['protocol_sha256']
            for method in ['standard', 'conditioned', 'projected']:
                training = get_json('behavior1k/action-support-20261021/vla-pilot-v1/' + method + '-s17-result.json')
                assert training['complete'] and training['completed_steps'] == 1000 and training['worker_returncode'] == 0
        response = cli('acp', 'jobs', 'describe', 'b1k-repeat1688-1019-batch-00', '--workspace-name=share-space', '-o', 'json')
        assert response.returncode == 0
        template = json.loads(response.stdout)
        assert template['uid'] == '168d5223752b415caf67bd27ed504a83'
        assert template['ownership']['user_name'] == '260010165'
        assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
        image = template['roles'][0]['image_path']
        expected_image = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
        assert image == expected_image
        intent = {'job_name': name, 'method': args.method, 'pair': pair, 'max_gpu_seconds': 28800,
                  'plan_sha256': sha(directory / 'plan.json'), 'submission_attempted': False}
        with intent_path.open('x') as stream:
            json.dump(intent, stream, indent=2)
        key = PREFIX + 'code/' + sha(archive) + '.tar.gz'
        try:
            head = client.head_object(Bucket='260010165', Key=key)
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') not in ('NoSuchKey', 'NotFound', '404'):
                raise
            client.upload_file(str(archive), '260010165', key, ExtraArgs={'Metadata': {'sha256': sha(archive)}})
            head = client.head_object(Bucket='260010165', Key=key)
        assert head['ContentLength'] == archive.stat().st_size and head['Metadata']['sha256'] == sha(archive)
        signed = lambda operation, key: client.generate_presigned_url(operation,
            Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=86400)
        converted = json.loads((PROJECT / 'analysis/rlinf_adaptation_20261020/cloud_smoke/artifacts.json').read_text())['checkpoint']
        config = {'job_root': plan['afs_root'] + '/jobs/' + name,
                  'method': args.method, 'pair': pair, 'plan_sha256': intent['plan_sha256'],
                  'checkpoint_files': {k: {f: v[f] for f in ('bytes', 'sha256')} for k, v in converted.items()},
                  'bundle_bytes': archive.stat().st_size, 'bundle_sha256': sha(archive),
                  'bundle_url': signed('get_object', key), 'result_put': signed('put_object', PREFIX + name + '/result.json')}
        source = (HERE / 'closed_loop_bootstrap.py').read_text() + '\nrun(' + repr(config) + ')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        command = 'timeout --signal=TERM --kill-after=60s 28740s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
        intent.update(submission_attempted=True, startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        intent_path.write_text(json.dumps(intent, indent=2) + '\n')
        try:
            response = cli('acp', 'jobs', 'copy', '--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00', '--name=' + name, '--job-name=' + name,
                '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot',
                '--retry-times=0', '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url=' + image, '--command=' + command, timeout=120)
            intent['submit_returncode'] = response.returncode
        except subprocess.TimeoutExpired:
            intent['submit_outcome'] = 'unknown_timeout'
        intent_path.write_text(json.dumps(intent, indent=2) + '\n')
        response = cli('acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json')
        if response.returncode == 0:
            job = json.loads(response.stdout)
            intent.update(uid=job['uid'], state=job['state'])
        else:
            intent['state'] = 'requires_exact_name_query'
        intent_path.write_text(json.dumps(intent, indent=2) + '\n')
        print(json.dumps({k: intent.get(k) for k in ('job_name', 'uid', 'state', 'submit_returncode')}))
    finally:
        client.close()


if __name__ == '__main__':
    main()

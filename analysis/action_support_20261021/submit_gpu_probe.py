"""Prepare or submit one bounded Spot GPU precision/optimizer verification."""
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
JOB = 'b1k-support-1021-gpusmoke01'
ROOT = '/mnt/afs/260010165/behavior_1k/action_support_20261021/gpu-smoke01'
OLD = '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01'
PREFIX = 'behavior1k/action-support-20261021/' + JOB + '/'
TEMPLATE = 'b1k-repeat1688-1019-batch-00'
IMAGE = 'registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
FILES = ['gpu_probe.py', 'vla_runtime.py', 'vla_support_adapter.py', 'support_flow.py']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cli(*arguments, timeout=45):
    return subprocess.run([str(Path.home() / '.local/bin/sco'), *arguments],
                          capture_output=True, text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    output = HERE / 'gpu_probe_job'
    output.mkdir(exist_ok=True)
    source_cache = PROJECT / 'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1'
    record = json.loads((source_cache / 'cache_manifest.json').read_text())['records'][0]
    sample = source_cache / record['npz_path']
    assert checksum(sample) == record['npz_sha256']
    plan = {'job_name': JOB, 'workspace': 'share-space', 'root': ROOT,
            'checkpoint': OLD + '/checkpoint', 'upstream': OLD + '/upstream',
            'adapter': OLD + '/adapter', 'site': OLD + '/site',
            'python': '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python',
            'tokenizer': '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/openpi-cache/big_vision/paligemma_tokenizer.model',
            'sample_sha256': record['npz_sha256'], 'max_job_seconds': 1200, 'gpu_count': 1,
            'quota_type': 'spot', 'scope': 'two projected SFT AdamW updates on one repeated existing observation; no full training or simulator',
            'rules_snapshot': '2026-09-11; online source page inaccessible; finite20minute budget',
            'authorization': 'research_autonomy_authorization_20261020.json',
            'source_sha256': {f: checksum(HERE / f) for f in FILES}}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        write(plan_path, plan)
    archive = output / 'bundle.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as tar:
            for f in FILES:
                tar.add(HERE / f, arcname=f, recursive=False)
            tar.add(plan_path, arcname='plan.json', recursive=False)
            tar.add(sample, arcname='sample.npz', recursive=False)
    with tarfile.open(archive) as tar:
        for name, expected in {**plan['source_sha256'], 'sample.npz': record['npz_sha256'],
                               'plan.json': checksum(plan_path)}.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected
    artifacts = {'bundle_bytes': archive.stat().st_size, 'bundle_sha256': checksum(archive),
                 'bundle_key': PREFIX + 'bundle.tar.gz', 'result_key': PREFIX + 'result.json'}
    assert artifacts['bundle_bytes'] < 32 * 1024**2
    write(output / 'artifacts.json', artifacts)
    if not args.submit:
        print(json.dumps({'plan': str(plan_path), **artifacts}))
        return
    intent_path = output / 'submission_intent.json'
    if intent_path.exists():
        raise RuntimeError('Submission intent already exists; reconcile exact job before any retry')
    intent = {'job_name': JOB, 'workspace': 'share-space', 'submission_attempted': False,
              'plan_sha256': checksum(plan_path), 'artifacts': artifacts}
    with intent_path.open('x') as stream:
        json.dump(intent, stream, indent=2)
    response = cli('acp', 'jobs', 'describe', TEMPLATE, '--workspace-name=share-space', '-o', 'json')
    if response.returncode:
        raise RuntimeError('Template read failed; no job was submitted')
    template = json.loads(response.stdout)
    assert template['uid'] == '168d5223752b415caf67bd27ed504a83'
    assert template['ownership']['user_name'] == '260010165'
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    assert template['roles'][0]['image_path'] == IMAGE
    assert template['roles'][0]['resource_spec'][0]['name'] == 'n12lp.nn.i10a.1'
    write(output / 'template_verified.json', {k: template[k] for k in ['uid', 'name', 'state', 'resource_pool', 'ownership']})
    sys.path.insert(0, str(PROJECT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client = aoss_client()
    try:
        client.upload_file(str(archive), '260010165', artifacts['bundle_key'],
                           ExtraArgs={'Metadata': {'sha256': artifacts['bundle_sha256']}})
        head = client.head_object(Bucket='260010165', Key=artifacts['bundle_key'])
        assert head['ContentLength'] == artifacts['bundle_bytes'] and head['Metadata']['sha256'] == artifacts['bundle_sha256']
        config = {'root': ROOT, 'bundle_bytes': artifacts['bundle_bytes'], 'bundle_sha256': artifacts['bundle_sha256'],
                  'bundle_url': client.generate_presigned_url('get_object', Params={'Bucket': '260010165', 'Key': artifacts['bundle_key']}, ExpiresIn=86400),
                  'result_put': client.generate_presigned_url('put_object', Params={'Bucket': '260010165', 'Key': artifacts['result_key']}, ExpiresIn=86400)}
        source = (HERE / 'gpu_probe_bootstrap.py').read_text() + '\nrun(' + repr(config) + ')\n'
        encoded = base64.b64encode(zlib.compress(source.encode())).decode()
        command = 'timeout --signal=TERM --kill-after=30s 1170s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(encoded) + ')))')
        intent.update(submission_attempted=True, startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent_path, intent)
        try:
            response = cli('acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=' + TEMPLATE,
                           '--name=' + JOB, '--job-name=' + JOB, '--worker-spec=n12lp.nn.i10a.1',
                           '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
                           '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                           '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                           '--container-image-url=' + IMAGE, '--command=' + command, timeout=120)
            intent['submit_returncode'] = response.returncode
        except subprocess.TimeoutExpired:
            intent['submit_outcome'] = 'unknown_timeout'
        write(intent_path, intent)
        response = cli('acp', 'jobs', 'describe', JOB, '--workspace-name=share-space', '-o', 'json')
        if response.returncode == 0:
            job = json.loads(response.stdout)
            intent.update(uid=job['uid'], state=job['state'])
            write(output / 'job_status.json', {k: job.get(k) for k in ['uid', 'name', 'state', 'resource_pool', 'create_time', 'start_time', 'complete_time']})
        else:
            intent['state'] = 'requires_exact_name_query'
        write(intent_path, intent)
        print(json.dumps({k: intent.get(k) for k in ['job_name', 'uid', 'state', 'submit_returncode']}))
    finally:
        client.close()


if __name__ == '__main__':
    main()

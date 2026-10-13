"""One explicit s0 download recovery, reusing immutable assignment and AFS byte prefixes."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import zlib

from bounded_collection_admission import requested_gpus
from launch_retraining_cloud import frozen_tar, range_hash, sha, write
from launch_retraining_parallel import environment, reserved_admission
from recover_retraining_parallel import startup_payload
from submit_retraining_cloud_probe import CLI, client


def resume(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    old = dict(plan['shards'][0])
    assert old['name'] == 'behavior-data-parallel-1013-124351-s0'
    assert old['uid'] == 'dff8df5609504116b48213bf1353a2da' and not old.get('generation')
    assert plan['training_allowed'] is False and old['source_count'] == 85
    untouched = json.dumps(plan['shards'][1:], sort_keys=True)
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', old['name'],
        '--workspace-name=share-space', '-o', 'json'], env=environment(), capture_output=True,
        text=True, timeout=45, check=True)
    job = json.loads(result.stdout)
    assert job['uid'] == old['uid'] and job['state'] == 'FAILED'
    assert job['ownership']['user_name'] == '260010165'
    workers = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', old['name'],
        '--workspace-name=share-space'], env=environment(), capture_output=True,
        text=True, timeout=45, check=True).stdout
    assert re.search(re.escape('pt-' + old['uid'] + '-worker-0') + r'[^\n]*\|\s*(Failed|Deleted)\s*\|', workers)
    payload = startup_payload(job['roles'][0]['startup_script'])
    assert payload['root'] == old['root'] and payload['entrypoint'] == 'build_retraining_parallel.py'
    source_asset = next(a for a in payload['assets'] if a['name'] == 'private_sources.json')
    payload.update(resume_source_sha256=source_asset['sha256'], recovery_generation='generation-1-download-retry')
    public, internal = client(), client(internal=True)
    prefix = old['status_key'].rsplit('/', 1)[0] + '/generation-1/'
    def sign(op, key):
        return internal.generate_presigned_url(op, Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=604800)
    def asset(file, name, extract_to=None):
        digest = sha(file); size = file.stat().st_size
        key = prefix + 'inputs/' + digest + '/' + name
        try:
            head = public.head_object(Bucket='260010165', Key=key)
        except public.exceptions.ClientError as error:
            if error.response['ResponseMetadata']['HTTPStatusCode'] != 404: raise
            with file.open('rb') as stream:
                public.put_object(Bucket='260010165', Key=key, Body=stream,
                                  IfNoneMatch='*', Metadata={'sha256': digest})
            head = public.head_object(Bucket='260010165', Key=key)
        assert head['ContentLength'] == size and range_hash(public, key, size, head['ETag']) == digest
        value = {'name': name, 'key': key, 'get': sign('get_object', key), 'bytes': size, 'sha256': digest}
        if extract_to: value['extract_to'] = extract_to
        return value
    local = Path('/private/tmp/behavior-retraining-parallel-private/shard-0')
    storage_file = local / 'private_storage.json'
    old_storage = next(a for a in payload['assets'] if a['name'] == 'private_storage.json')
    assert sha(storage_file) == old_storage['sha256']
    storage = json.loads(storage_file.read_text())
    status_key = prefix + 'status.json'
    storage['status_put'] = sign('put_object', status_key)
    storage['child_status_put'] = [sign('put_object', prefix + 'process-' + str(i) + '.json') for i in range(4)]
    # New private input is separate; the original local and object manifests remain untouched.
    storage_file = local / 'generation-1-private_storage.json'
    write(storage_file, storage); storage_file.chmod(0o600)
    repo = Path(__file__).resolve().parents[1]
    names = ['sco/prepare_retraining_data.py', 'sco/build_retraining_cloud.py',
        'sco/build_retraining_parallel.py', 'scripts/subtask_trajectory.py',
        'scripts/full_episode_trajectory.py', 'scripts/export_full_episode_collection.py',
        'openpi/src/openpi/__init__.py', 'openpi/src/openpi/training/recovery_index.py',
        'openpi/src/openpi/configs/tasks/b1k.py']
    code = local / 'generation-1-code.tar.gz'
    frozen_tar(code, [(name, repo / name) for name in names])
    new_assets = [asset(code, 'cloud_code.tar.gz', 'code'), asset(storage_file, 'private_storage.json')]
    payload['assets'] = [a for a in payload['assets'] if a['name'] not in ('cloud_code.tar.gz', 'private_storage.json')] + new_assets
    payload['status_put'] = storage['status_put']
    with public.get_object(Bucket='260010165', Key=old['status_key'])['Body'] as body:
        old_status = json.loads(body.read())
    source = repo.joinpath('sco/bootstrap_retraining_cloud.py').read_text()
    record = dict(old, name=old['name'] + '-g1', generation=1,
        attempt_history=[dict(old, state='FAILED', failure=old_status)],
        status_key=status_key, recovery='bounded_download_retry_preserved_afs_prefixes',
        source_manifest_sha256=source_asset['sha256'], code_sha256=sha(code),
        bootstrap_sha256=hashlib.sha256(source.encode()).hexdigest(),
        recovery_at=datetime.now(timezone.utc).isoformat())
    record.pop('uid'); record.pop('platform_state', None)
    record['admission'] = reserved_admission(plan['shards'][1:])
    record['state'] = 'submission_intent'
    plan['shards'][0] = record
    assert json.dumps(plan['shards'][1:], sort_keys=True) == untouched
    write(path, plan)
    encoded = base64.b64encode(zlib.compress((source + '\nrun(' + repr(payload) + ')\n').encode())).decode()
    startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(encoded) + ')),"data_bootstrap","exec"))'
    command = [CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space', '--aec2-name=share-cluster',
        '--name=' + record['name'], '--job-name=' + record['name'], '--training-framework=pytorch',
        '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
        '--worker-spec=n6ls.iu.i40.1', '--worker-nodes=1', '--quota-type=reserved',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 86400 python3 -c ' + shlex.quote(startup)]
    result = subprocess.run(command, env=environment(), capture_output=True, text=True, timeout=90)
    record.update(state='created_needs_describe' if result.returncode == 0 else 'submission_unknown', cli_exit_code=result.returncode)
    write(path, plan)
    if result.returncode: raise RuntimeError('Reconcile exact name; never resubmit an unknown create')
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', record['name'], '--workspace-name=share-space', '-o', 'json'],
        env=environment(), capture_output=True, text=True, timeout=45, check=True)
    job = json.loads(result.stdout)
    assert job['ownership']['user_name'] == '260010165' and job['scheduling']['quota_type'] == 'RESERVED'
    assert requested_gpus(job) == 1 and job['resource_pool']['name'] == 'share-cluster'
    record.update(uid=job['uid'], platform_state=job['state'], state='submitted')
    write(path, plan)
    print(json.dumps({k: record[k] for k in ['name', 'uid', 'platform_state', 'root', 'code_sha256']}), flush=True)


if __name__ == '__main__':
    resume(sys.argv[1])

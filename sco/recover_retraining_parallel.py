"""One bounded replacement for the two diagnosed pre-data DNS bootstrap failures."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import zlib

from launch_retraining_parallel import environment, reserved_admission
from launch_retraining_cloud import write, range_hash
from submit_retraining_cloud_probe import client, CLI


def startup_payload(command):
    words = shlex.split(command)
    tree = ast.parse(words[words.index('-c') + 1])
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    program = zlib.decompress(base64.b64decode(ast.literal_eval(call.args[0]))).decode()
    last = ast.parse(program).body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
    assert isinstance(last.value.func, ast.Name) and last.value.func.id == 'run'
    return ast.literal_eval(last.value.args[0])


def recover(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    public, internal = client(), client(internal=True)
    def sign(op, key):
        return internal.generate_presigned_url(op, Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=604800)
    known = [plan['shards'][i] for i in [0, 1]]
    for i in [2, 3]:
        old = dict(plan['shards'][i])
        assert not old.get('generation') and old['name'].endswith('-s' + str(i))
        result = subprocess.run([CLI, 'acp', 'jobs', 'describe', old['name'],
            '--workspace-name=share-space', '-o', 'json'], env=environment(),
            capture_output=True, text=True, timeout=40, check=True)
        job = json.loads(result.stdout)
        assert job['uid'] == old['uid'] and job['ownership']['user_name'] == '260010165'
        assert job['state'] in ('FAILED', 'SUSPENDED'), 'Old allocation must be released first'
        workers = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', old['name'],
            '--workspace-name=share-space'], env=environment(), capture_output=True,
            text=True, timeout=40, check=True).stdout
        assert re.search(re.escape('pt-' + old['uid'] + '-worker-0') + r'[^\n]*\|\s*(Failed|Deleted)\s*\|', workers)
        response = public.get_object(Bucket='260010165', Key=old['status_key'])
        try:
            failure = json.loads(response['Body'].read())
        finally:
            response['Body'].close()
        assert failure['phase'] == 'bootstrap_needs_diagnostic'
        assert failure['reason_type'] == 'gaierror' and failure['current_asset'] is None
        payload = startup_payload(job['roles'][0]['startup_script'])
        assert payload['root'] == old['root']
        record = dict(old, name=old['name'] + '-g1', generation=1, state='preparing_replacement',
            attempt_history=[{**old, 'state': job['state'], 'failure': failure}],
            root=old['root'] + '-g1',
            status_key=old['status_key'].rsplit('/', 1)[0] + '/generation-1/status.json')
        record.pop('uid'); record.pop('platform_state', None)
        payload['root'] = record['root']; payload['status_put'] = sign('put_object', record['status_key'])
        storage_path = Path('/private/tmp/behavior-retraining-parallel-private') / ('shard-' + str(i)) / 'private_storage.json'
        storage = json.loads(storage_path.read_text())
        storage['status_put'] = payload['status_put']
        prefix = record['status_key'].rsplit('/', 1)[0]
        storage['child_status_put'] = [sign('put_object', prefix + '/process-' + str(j) + '.json') for j in range(4)]
        body = (json.dumps(storage, indent=2) + '\n').encode()
        digest = hashlib.sha256(body).hexdigest()
        key = prefix + '/inputs/' + digest + '/private_storage.json'
        public.put_object(Bucket='260010165', Key=key, Body=body, IfNoneMatch='*', Metadata={'sha256': digest})
        head = public.head_object(Bucket='260010165', Key=key)
        assert range_hash(public, key, len(body), head['ETag']) == digest
        payload['assets'] = [a for a in payload['assets'] if a['name'] != 'private_storage.json'] + [
            {'name': 'private_storage.json', 'key': key, 'get': sign('get_object', key),
             'bytes': len(body), 'sha256': digest}]
        source = Path(__file__).with_name('bootstrap_retraining_cloud.py').read_text()
        record['bootstrap_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        record['admission'] = reserved_admission(known)
        record['state'] = 'submission_intent'
        plan['shards'][i] = record; write(path, plan)
        encoded = base64.b64encode(zlib.compress((source + '\nrun(' + repr(payload) + ')\n').encode())).decode()
        startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(encoded) + ')),"data_bootstrap","exec"))'
        command = [CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space', '--aec2-name=share-cluster',
            '--name=' + record['name'], '--job-name=' + record['name'], '--training-framework=pytorch',
            '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
            '--worker-spec=n6ls.iu.i40.1', '--worker-nodes=1', '--quota-type=reserved',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--command=timeout --signal=TERM --kill-after=30 86400 python3 -c ' + shlex.quote(startup)]
        result = subprocess.run(command, env=environment(), capture_output=True, text=True, timeout=90)
        record.update(state='created_needs_describe' if result.returncode == 0 else 'submission_unknown',
                      cli_exit_code=result.returncode)
        write(path, plan)
        if result.returncode:
            raise RuntimeError('Creation result needs exact-name reconciliation')
        result = subprocess.run([CLI, 'acp', 'jobs', 'describe', record['name'],
            '--workspace-name=share-space', '-o', 'json'], env=environment(),
            capture_output=True, text=True, timeout=40, check=True)
        current = json.loads(result.stdout)
        assert current['ownership']['user_name'] == '260010165'
        assert current['scheduling']['quota_type'] == 'RESERVED'
        record.update(state='submitted', uid=current['uid'], platform_state=current['state'])
        known.append(record); write(path, plan)
        print(json.dumps({k:record[k] for k in ['name','uid','generation','platform_state']}), flush=True)


if __name__ == '__main__':
    recover(sys.argv[1])

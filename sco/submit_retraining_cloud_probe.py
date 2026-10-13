"""Submit one finite zero-GPU probe for the authorized data preparation."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import zlib

import boto3
from botocore.config import Config

CLI = '/private/tmp/behavior-sco-runtime/bin/sco'
PREFIX = 'behavior1k/retraining_20261013/source_side_v1/'


def client(internal=False):
    text = (Path.home() / '.config/sco/profiles/default.toml').read_text()
    values = {key: ast.literal_eval(value) for key, value in re.findall(
        r'^(access_key_id|access_key_secret)\s*=\s*([\"\'].*[\"\'])\s*$', text, re.M)}
    endpoint = ('aoss-internal' if internal else 'aoss') + '.cn-sh-01b.sensecoreapi-oss.cn'
    return boto3.client('s3', endpoint_url='https://' + endpoint, region_name='cn-sh-01b',
        aws_access_key_id=values['access_key_id'], aws_secret_access_key=values['access_key_secret'],
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
            proxies={}, connect_timeout=15, read_timeout=90,
            max_pool_connections=16,
            retries={'max_attempts': 2, 'mode': 'standard'}))


def main(args):
    manifest = json.loads(args.sources.read_text())
    assert manifest['authorization']['training_allowed'] is False
    assert not args.receipt.exists(), 'Reconcile the exact saved probe before resubmitting'
    row = next(r for r in manifest['source_archives'] if r['kind'] == 'full_episode')
    name = 'behavior-data-cpu-probe-' + datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
    report_key = PREFIX + name + '/report.json'
    public, internal = client(), client(internal=True)
    def sign(c, operation, key):
        return c.generate_presigned_url(operation, Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=86400)
    payload = {'source_routes': {label: {'url': sign(c, 'get_object', row['key']),
        'bytes': row['bytes'], 'etag': row['etag']} for label, c in [('public', public), ('internal', internal)]},
        'report_put': sign(internal, 'put_object', report_key)}
    source = Path(__file__).with_name('probe_retraining_cloud.py').read_text()
    program = source + '\nrun(' + repr(payload) + ')\n'
    encoded = base64.b64encode(zlib.compress(program.encode())).decode()
    startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(encoded) + ')),"data_probe","exec"))'
    record = {'name': name, 'state': 'submission_intent', 'gpu_count': 0, 'cpu_count': 8,
        'memory_gib': 32, 'worker_spec': 'n6ls.iu.i40.8c32g', 'quota_type': 'spot',
        'resource_pool': 'share-cluster', 'workspace': 'share-space', 'timeout_seconds': 900,
        'report_key': report_key, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'source_fingerprint': manifest['fingerprint'], 'training_allowed': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    env = dict(os.environ, SCO_CONFIG='/private/tmp/behavior-sco-config',
        SCO_HOME='/private/tmp/behavior-sco-runtime', SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    command = [CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space', '--aec2-name=share-cluster',
        '--name=' + name, '--job-name=' + name, '--training-framework=pytorch',
        '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
        '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=15 900 python3 -c ' + shlex.quote(startup)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    record.update(state='created_needs_describe' if result.returncode == 0 else 'submission_unknown',
                  cli_exit_code=result.returncode)
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode != 0:
        raise RuntimeError('Inspect exact probe name before any further creation')
    result = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
        env=env, capture_output=True, text=True, timeout=40, check=True)
    job = json.loads(result.stdout)
    assert job['ownership']['user_name'] == '260010165'
    assert job['scheduling']['quota_type'] == 'SPOT'
    specs = [s for role in job['roles'] for s in role['resource_spec']]
    assert len(specs) == 1 and specs[0]['name'] == record['worker_spec']
    assert int(specs[0]['limits'].get('nvidia.com/gpu', 0)) == 0
    record.update(state='submitted', uid=job['uid'], platform_state=job['state'])
    args.receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    main(parser.parse_args())

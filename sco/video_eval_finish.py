"""One CPU-only final packaging job after all 1000 video pairs are accepted."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

from video_eval_adapter import aoss_client, cli_json, environment, build_startup, CLI, BUCKET
from video_eval_finalize import make_finalization_payload, summarize
from video_eval_queue import write_json
from video_eval_submission_readiness import image_packaging_readiness


def step(root):
    root = Path(root)
    if (root / 'STOP').exists():
        return {'state': 'stopped'}
    summary = summarize(root)
    if not summary['all_results_verified']:
        return {'state': 'incomplete_results', 'completed': summary['completed']}
    plan = json.loads((root / 'plan.json').read_text())
    ready = json.loads((root / 'preparation_ready.json').read_text())
    client = aoss_client()
    prefix = plan['object_prefix'].rstrip('/')
    record_path = root / 'finalization_submission.json'
    receipt_key = prefix + '/submission/VERIFICATION.json'
    if not record_path.exists():
        readiness = image_packaging_readiness(root)
        if readiness['state'] != 'ready':
            return readiness
        signer = lambda op, key: client.generate_presigned_url(op,
            Params={'Bucket': BUCKET, 'Key': key}, ExpiresIn=604800)
        payload = make_finalization_payload(root, plan['afs_output_root'] + '/final_submission',
                                           (root / 'SUBMISSION_README.md').read_text(), signer)
        body = json.dumps(payload, sort_keys=True).encode()
        key = prefix + '/finalization/startup.private.json'
        client.put_object(Bucket=BUCKET, Key=key, Body=body)
        source = Path(__file__).with_name('video_eval_finalize.py').read_bytes()
        frozen = root / 'frozen/video_eval_finalize.py'
        if frozen.exists():
            assert frozen.read_bytes() == source
        else:
            frozen.write_bytes(source)
        startup = ('import sys\nsys.path.insert(0,' + repr(ready['code'] + '/sco') + ')\n'
                   + build_startup(signer('get_object', key), hashlib.sha256(body).hexdigest(), source))
        name = plan['run_id'] + '-package'
        record = {'name': name, 'state': 'UNKNOWN', 'gpu_count': 0,
                  'payload_sha256': hashlib.sha256(body).hexdigest(),
                  'worker_sha256': hashlib.sha256(source).hexdigest(), 'receipt_key': receipt_key}
        if (root / 'STOP').exists():
            return {'state': 'stopped'}
        write_json(record_path, record, exclusive=True)
        result = subprocess.run([CLI, 'acp', 'jobs', 'create', '--workspace-name=share-space',
            '--aec2-name=share-cluster', '--name=' + name, '--job-name=' + name,
            '--training-framework=pytorch',
            '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
            '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--command=timeout --signal=TERM --kill-after=15 86400 python3 -c ' + shlex.quote(startup)],
            env=environment(), capture_output=True, text=True, timeout=90)
        record['create_exit_code'] = result.returncode
        write_json(record_path, record)
    record = json.loads(record_path.read_text())
    job = cli_json(['acp', 'jobs', 'describe', record['name'], '--workspace-name=share-space', '-o', 'json'])
    assert job['name'] == record['name'] and job['ownership']['user_name'] == BUCKET
    assert job['resource_pool']['name'] == 'share-cluster'
    if record.get('uid'):
        assert record['uid'] == job['uid']
    record.update(uid=job['uid'], state=job['state'])
    write_json(record_path, record)
    if job['state'] == 'SUCCEEDED':
        with client.get_object(Bucket=BUCKET, Key=receipt_key)['Body'] as stream:
            final = json.loads(stream.read())
        assert final['verified'] and final['completed'] == 1000 and final['run_id'] == plan['run_id']
        assert final['source_manifest_sha256'] == ready['source_manifest_sha256']
        assert final['checkpoint_ready_sha256'] == ready['checkpoint_ready_sha256']
        assert final['zip_full_readback_verified'] and final['direct_videos_uploaded']
        write_json(root / 'FINAL_VERIFICATION.json', final)
        return {**final, 'artifact_state': final['state'], 'state': 'complete'}
    if job['state'] in ('FAILED', 'STOPPED', 'DELETED', 'CANCELLED', 'CANCELED'):
        return {'state': 'finalization_failed', 'uid': job['uid']}
    return {'state': 'finalizing', 'uid': job['uid'], 'platform_state': job['state']}

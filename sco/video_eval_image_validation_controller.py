"""One durable auxiliary image-validation job sharing the evaluation pending cap."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED', 'DELETED', 'CANCELLED', 'CANCELED'}


def step(adapter, allow_submit=False):
    from video_eval_adapter import cli_json, CLI, TEMPLATE, TEMPLATE_UID, POOL, BUCKET, environment
    from video_eval_queue import write_json
    bundle = adapter.root / 'serving_bundle'
    path = bundle / 'GPU_VALIDATION_REQUEST.json'
    if not path.exists():
        return dict(state='NOT_REQUESTED', pending_gpus=0, query_complete=True)
    request = json.loads(path.read_text())
    assert request['name'] == adapter.plan['run_id'] + '-imagegpu-g000'
    intent = bundle / 'GPU_VALIDATION_INTENT.json'
    record_path = bundle / 'GPU_VALIDATION_JOB.json'
    if not intent.exists():
        if not allow_submit:
            return dict(state='PLANNED', pending_gpus=0, query_complete=True)
        assert not (adapter.root / 'STOP').exists()
        template = cli_json(['acp', 'jobs', 'describe', TEMPLATE, '--workspace-name=share-space', '-o', 'json'])
        assert template['uid'] == TEMPLATE_UID and template['ownership']['user_name'] == BUCKET
        assert template['resource_pool']['name'] == POOL
        command_path = bundle / request['command_file']
        command = command_path.read_text()
        assert hashlib.sha256(command.encode()).hexdigest() == request['command_sha256']
        write_json(intent, dict(name=request['name'], image=request['image_reference'],
                               created_epoch=time.time(), state='UNKNOWN', gpu_count=1), exclusive=True)
        result = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=' + TEMPLATE, '--name=' + request['name'], '--job-name=' + request['name'],
            '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot',
            '--container-image-url=' + request['image_reference'],
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/outputs',
            '--env=NVIDIA_DRIVER_CAPABILITIES:all', '--command=' + command],
            env=environment(), capture_output=True, text=True, timeout=90)
        write_json(bundle / 'GPU_VALIDATION_COPY_EXIT.json', dict(exit_code=result.returncode, observed_epoch=time.time()))
        # A persisted intent is never submitted again, including on timeout.
        submitted_now = True
    else:
        submitted_now = False
    try:
        job = cli_json(['acp', 'jobs', 'describe', request['name'], '--workspace-name=share-space', '-o', 'json'])
        assert job['name'] == request['name'] and job['ownership']['user_name'] == BUCKET
        assert job['resource_pool']['name'] == POOL
        if record_path.exists():
            assert json.loads(record_path.read_text())['uid'] == job['uid']
        record = dict(name=job['name'], uid=job['uid'], state=job['state'], observed_epoch=time.time(),
                      gpu_count=1, image_reference=request['image_reference'])
        write_json(record_path, record)
        # New requests reserve one full slot for the current pass even if running.
        pending = int(submitted_now or job['state'] not in TERMINAL | {'RUNNING'})
        try:
            response = adapter.client.get_object(Bucket=BUCKET, Key=request['report_key'])
            with response['Body'] as stream:
                raw = stream.read(20 * 1024**2 + 1)
            assert len(raw) <= 20 * 1024**2
            report = json.loads(raw)
            assert report['image_reference'] == request['image_reference']
            assert report['image_config_digest'] == request['config_digest']
            write_json(bundle / 'GPU_VALIDATION_RESULT.json', report)
        except adapter.client.exceptions.ClientError as error:
            if error.response.get('Error', {}).get('Code') not in ('404', 'NoSuchKey'):
                raise
        return dict(state=job['state'], pending_gpus=pending, uid=job['uid'], query_complete=True)
    except Exception as error:
        return dict(state='UNKNOWN', pending_gpus=1, query_complete=False, error_type=type(error).__name__)

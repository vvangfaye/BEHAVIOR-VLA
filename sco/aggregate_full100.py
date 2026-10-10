"""Strictly aggregate all SCO assignments and export the full task reports."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback
import urllib.request


def verify_full_report(summary, success):
    expected = {'expected_episode_jobs': 400, 'expected_segment_rollouts': 7223,
                'expected_annotated_segments': 8112}
    if summary.get('complete') is not True:
        raise ValueError('The full panel is incomplete')
    for key, count in expected.items():
        if summary.get(key) != count or success.get(key) != count:
            raise ValueError('The final report changed the frozen full100 scope')
    if (len(summary.get('per_task', {})) != 100 or
            summary.get('valid_segment_rollouts') != 7223 or
            summary.get('accounted_annotations') != 8112 or
            summary.get('overall', {}).get('unsupported') != 889 or
            summary.get('panel_fingerprint') != success.get('panel_fingerprint')):
        raise ValueError('Full100 accounting or panel identity is invalid')
    fidelity = summary.get('state_fidelity', {})
    if fidelity.get('contract') != 'recorded_native_fields_v1' or fidelity.get('verified') is not True:
        raise ValueError('Recorded native state fidelity is unverified')
    video = summary.get('video_completeness', {})
    if video.get('required') is None or video.get('completed') != video['required']:
        raise ValueError('Required rollout videos are incomplete')


def run(payload):
    runtime, code, root = (Path(payload[k]) for k in ['runtime_root', 'code_root', 'run_root'])
    source = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(source).hexdigest() == payload['source_manifest_sha256']
    for relative, expected in json.loads(source)['overlay_files_sha256'].items():
        assert hashlib.sha256((code / relative).read_bytes()).hexdigest() == expected
    report_root = Path(payload['report_root'])
    report_root.mkdir(parents=True, exist_ok=False)
    log = (report_root / 'driver.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status = {'state': 'running', 'gpu_count': 0, 'run_root': str(root),
              'started_at': time.time()}

    def upload(url, data):
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method='PUT'), timeout=120):
            pass

    def publish():
        status['updated_at'] = time.time()
        (report_root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('report_put'):
            report = dict(status)
            with (report_root / 'driver.log').open('rb') as stream:
                stream.seek(max(0, os.fstat(stream.fileno()).st_size - 12000))
                report['log_tail'] = stream.read().decode('utf-8', errors='replace')
            try:
                upload(payload['report_put'], json.dumps(report).encode())
            except Exception:
                print('aggregate_status_upload_failed', flush=True)

    try:
        publish()
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment['PYTHONNOUSERSITE'] = '1'
        subprocess.run([str(runtime / 'envs/b1k-eval/bin/python'),
            str(code / 'sco/full100_shards.py'), 'aggregate', '--run-root', str(root)],
            check=True, env=environment, cwd=code)
        summary = json.loads((root / 'summary.json').read_text())
        success = json.loads((root / '_SUCCESS').read_text())
        verify_full_report(summary, success)
        names = ['summary.json', 'task_success.csv', 'subtask_success.csv',
                 'panel.json', 'manifest.json', 'assignments.json',
                 'SOURCE_PROVENANCE.json', 'result_manifest.json', '_SUCCESS', '_REPORTED']
        hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
        export = io.BytesIO()
        with tarfile.open(fileobj=export, mode='w:gz') as archive:
            for name in names:
                archive.add(root / name, arcname=name, recursive=False)
            data = (json.dumps(hashes, indent=2) + '\n').encode()
            info = tarfile.TarInfo('REPORT_SHA256.json')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        blob = export.getvalue()
        (report_root / 'reports.tar.gz').write_bytes(blob)
        if payload.get('bundle_put'):
            upload(payload['bundle_put'], blob)
        status.update(state='complete', completed_at=time.time(),
            panel_fingerprint=summary['panel_fingerprint'], tasks=100, episode_jobs=400,
            segment_rollouts=7223, overall=summary['overall'],
            bundle_sha256=hashlib.sha256(blob).hexdigest(), bundle_bytes=len(blob),
            report_files_sha256=hashes)
        publish()
    except BaseException as error:
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__)
        publish()
        raise

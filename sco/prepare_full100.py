"""Freeze an evaluation source snapshot and prepare all400 task assignments."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request


def run(payload):
    runtime = Path(payload['runtime_root'])
    assert (runtime / '_RUNTIME_READY').is_file()
    assert (runtime / 'graphics-runtime/_READY').is_file()
    graphics_hash = hashlib.sha256((runtime / 'graphics-runtime/_READY').read_bytes()).hexdigest()
    validation = json.loads(Path(payload['validation_success']).read_text())
    assert validation['graphics_runtime_manifest_sha256'] == graphics_hash
    code = Path(payload['code_root'])
    assert not code.exists(), 'A source snapshot requires a fresh directory'
    report_root = Path(payload['report_root'])
    report_root.mkdir(parents=True, exist_ok=False)
    log = (report_root / 'driver.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status = {'state': 'running', 'gpu_count': 0, 'run_root': payload['run_root'],
              'code_root': str(code), 'started_at': time.time()}

    def publish(stage):
        status.update(stage=stage, updated_at=time.time())
        (report_root / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('report_put'):
            report = dict(status)
            with (report_root / 'driver.log').open('rb') as source:
                source.seek(max(0, os.fstat(source.fileno()).st_size - 12000))
                report['log_tail'] = source.read().decode('utf-8', errors='replace')
            try:
                request = urllib.request.Request(payload['report_put'],
                    data=json.dumps(report).encode(), method='PUT')
                with urllib.request.urlopen(request, timeout=30):
                    pass
            except Exception:
                print('prepare_report_upload_failed', flush=True)

    try:
        publish('source_copy_started')
        digest = hashlib.sha256()
        with (runtime / 'inputs/code.tar.gz').open('rb') as archive:
            for block in iter(lambda: archive.read(8 * 1024**2), b''):
                digest.update(block)
        assert digest.hexdigest() == payload['source_archive_sha256']
        shutil.copytree(runtime / 'code', code, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        overlay = base64.b64decode(payload['overlay'])
        assert hashlib.sha256(overlay).hexdigest() == payload['overlay_sha256']
        with tarfile.open(fileobj=io.BytesIO(overlay), mode='r:gz') as archive:
            for item in archive.getmembers():
                relative = Path(item.name)
                assert item.isfile() and not relative.is_absolute() and '..' not in relative.parts
                assert relative.name != 'config.sh' and '.private.' not in str(relative)
                target = code / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(item).read())
                target.chmod(item.mode & 0o777)
        for relative, expected in payload['overlay_files_sha256'].items():
            assert hashlib.sha256((code / relative).read_bytes()).hexdigest() == expected
        assert not (code / 'nebula/config.sh').exists()
        source = {'source_archive_sha256': payload['source_archive_sha256'],
                  'graphics_runtime_manifest_sha256': graphics_hash,
                  'overlay_sha256': payload['overlay_sha256'],
                  'overlay_files_sha256': payload['overlay_files_sha256'],
                  'code_root': str(code), 'created_at': time.time()}
        source_bytes = (json.dumps(source, indent=2) + '\n').encode()
        (code / '_CODE_SNAPSHOT_READY').write_bytes(source_bytes)
        status['source_manifest_sha256'] = hashlib.sha256(source_bytes).hexdigest()
        publish('source_snapshot_ready')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment.update(PYTHONNOUSERSITE='1',
            OPENPI_DATA_HOME=str(runtime / 'openpi-cache'),
            MAX_SUBTASK_LEN='64', EXECUTE_HORIZON='16', PI05_PAIRED_RETEST_SEED='1',
            OMNIGIBSON_DISABLE_TORCH_COMPILE='1',
            LD_LIBRARY_PATH=str(runtime / 'graphics-runtime/usr/lib/x86_64-linux-gnu') + ':' +
                str(runtime / 'envs/b1k-eval/lib') + ':' + environment.get('LD_LIBRARY_PATH', ''))
        command = [runtime / 'envs/b1k-eval/bin/python', code / 'sco/full100_shards.py', 'prepare',
            '--run-root', payload['run_root'], '--run-id', payload['run_id'],
            '--panel', code / 'eval_results/subtask100_native_baseline_1375000_20261007_091214/panel.json',
            '--raw-data-root', payload['raw_data_root'], '--checkpoint', runtime / 'checkpoint',
            '--eval-env', runtime / 'envs/b1k-eval', '--pi05-env', runtime / 'envs/b1k-pi05',
            '--omnigibson-data-path', runtime / 'omnigibson-data',
            '--validation-success', payload['validation_success'],
            '--repo-id', 'b1k_full100_joint_subtask_v1_drop_episode_s1',
            '--nodes', str(payload['nodes']), '--gpus-per-node', str(payload['gpus_per_node']),
            '--base-port', '8810']
        publish('full400_state_preflight_and_queue_preparation')
        subprocess.run([str(v) for v in command], check=True, env=environment, cwd=code)
        prepared = json.loads((Path(payload['run_root']) / '_PREPARED').read_text())
        assert prepared['tasks'] == 100 and prepared['episode_jobs'] == 400
        assert prepared['segment_rollouts'] == 7223
        (Path(payload['run_root']) / 'SOURCE_PROVENANCE.json').write_bytes(source_bytes)
        status.update(state='complete', prepared=prepared, completed_at=time.time())
        publish('full100_prepared')
    except BaseException as error:
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__)
        publish('failed')
        raise

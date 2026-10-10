"""Start a verified full-panel task assignment on its allocated SCO GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request


def run(payload):
    runtime = Path(payload['runtime_root'])
    assert (runtime / '_RUNTIME_READY').is_file(), 'CPU restoration is incomplete'
    assert (runtime / 'graphics-runtime/_READY').is_file(), 'GL loader restoration is incomplete'
    code = Path(payload['code_root'])
    source_bytes = (code / '_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == payload['source_manifest_sha256']
    graphics_bytes = (runtime / 'graphics-runtime/_READY').read_bytes()
    assert hashlib.sha256(graphics_bytes).hexdigest() == json.loads(source_bytes)['graphics_runtime_manifest_sha256']
    graphics_ready = json.loads(graphics_bytes)
    for name, expected in graphics_ready['libraries_sha256'].items():
        assert hashlib.sha256((Path(graphics_ready['library_path']) / name).read_bytes()).hexdigest() == expected
    for relative, expected in json.loads(source_bytes)['overlay_files_sha256'].items():
        assert hashlib.sha256((code / relative).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(code / 'sco'))
    import full100_shards as shards
    panel, assignment, manifest, queue = shards.load_assignment(payload['run_root'], payload['index'])
    shards.gate(payload['validation_success'], runtime / 'checkpoint', panel)
    root = Path(manifest['queue_db']).parent
    execution_name = payload['execution_name']
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,80}', execution_name)
    cache_root = Path('/var/tmp/behavior-1k-runtime') / execution_name
    cache_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    assert cache_root.stat().st_dev != runtime.stat().st_dev, 'Worker cache must be outside AFS'
    local_free_bytes = shutil.disk_usage(cache_root).free
    assert local_free_bytes >= 40 * 1024**3, 'Worker temporary disk lacks texture-cache capacity'
    os.environ['SCO_EPHEMERAL_RUNTIME_ROOT'] = str(cache_root)
    log = (root / 'driver.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status = {'state': 'preflight', 'assignment_index': assignment['index'],
              'panel_fingerprint': panel['fingerprint'], 'tasks': assignment['tasks'],
              'episode_jobs': len(assignment['episode_keys']),
              'segment_rollouts': len(assignment['sample_ids']), 'started_at': time.time()}
    status.update(ephemeral_runtime_cache_root=str(cache_root),
                  ephemeral_disk_free_bytes_at_start=local_free_bytes,
                  ephemeral_cache_on_different_filesystem=True)

    def publish():
        status['updated_at'] = time.time()
        (root / 'job_status.json').write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('report_put'):
            report = dict(status)
            try:
                request = urllib.request.Request(payload['report_put'],
                    data=json.dumps(report).encode(), method='PUT')
                with urllib.request.urlopen(request, timeout=30):
                    pass
            except Exception:
                print('assignment_report_upload_failed', flush=True)

    def command(args, **kwargs):
        subprocess.run([str(v) for v in args], check=True, **kwargs)

    try:
        publish()
        rows = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,driver_version,memory.total',
            '--format=csv,noheader'], text=True).strip().splitlines()
        assert len(rows) == len(manifest['gpus']), 'Allocation differs from the immutable assignment'
        assert all(any(name in row for name in ('5090', 'N12LP')) for row in rows), rows
        drivers = {row.split(',')[2].strip() for row in rows}
        assert len(drivers) == 1, 'Unexpected mixed drivers within one node'
        driver = next(iter(drivers))
        candidates = [p for parent in ['/usr/local/nvidia/lib64', '/usr/lib/x86_64-linux-gnu',
            '/usr/lib64', '/lib/x86_64-linux-gnu', '/lib64']
            for p in Path(parent).glob('libEGL_nvidia.so.' + driver)]
        assert candidates, 'Graphics driver libraries are missing'
        graphics = root / 'graphics'
        graphics.mkdir(exist_ok=True)
        library = str(candidates[0])
        vulkan = graphics / 'nvidia_icd.json'
        egl = graphics / '10_nvidia.json'
        assert manifest['child_environment']['VK_ICD_FILENAMES'] == str(vulkan)
        assert manifest['child_environment']['__EGL_VENDOR_LIBRARY_FILENAMES'] == str(egl)
        vulkan.write_text(json.dumps({'file_format_version': '1.0.0',
            'ICD': {'library_path': library, 'api_version': '1.3.242'}}) + '\n')
        egl.write_text(json.dumps({'file_format_version': '1.0.0',
            'ICD': {'library_path': library}}) + '\n')
        environment = os.environ.copy()
        environment.pop('PYTHONHOME', None)
        environment.pop('PYTHONPATH', None)
        environment.update(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES',
            OMNIGIBSON_DATA_PATH=str(runtime / 'omnigibson-data'), OMNIGIBSON_HEADLESS='1',
            OMNIGIBSON_DISABLE_TORCH_COMPILE='1', OPENPI_DATA_HOME=str(runtime / 'openpi-cache'),
            XLA_PYTHON_CLIENT_PREALLOCATE='false', VK_ICD_FILENAMES=str(vulkan),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(egl),
            LD_LIBRARY_PATH=str(runtime / 'graphics-runtime/usr/lib/x86_64-linux-gnu') + ':' +
                str(runtime / 'envs/b1k-eval/lib') + ':' + environment.get('LD_LIBRARY_PATH', ''))
        for gpu in manifest['gpus']:
            gpu_environment = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
            command([runtime / 'envs/b1k-pi05/bin/python', '-c',
                'import jax;d=jax.devices();assert len(d)==1 and d[0].platform=="gpu",d;print(d)'],
                env=gpu_environment)
            command([runtime / 'envs/b1k-eval/bin/python',
                code / 'nebula/check_eval_runtime.py',
                '--cache-dir', root / f'warp-preflight-{gpu}'], env=gpu_environment)
        status.update(state='running', allocated_gpus=rows)
        publish()
        shards.run_assignment(argparse.Namespace(run_root=Path(payload['run_root']), index=payload['index']))
        proof = json.loads((root / '_ASSIGNED_SUCCESS').read_text())
        assert proof['complete_full100'] is False
        status.update(state='complete', completed_at=time.time(), assigned_success=proof)
        publish()
    except BaseException as error:
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__)
        publish()
        raise

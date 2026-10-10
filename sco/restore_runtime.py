"""Restore the verified Linux runtime on personal AFS before allocating GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile


def run(payload):
    root = Path(payload['root'])
    root.mkdir(parents=True, exist_ok=True)
    log = (root / 'restore.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status = {'state': 'running', 'root': str(root), 'gpu_count': 0,
              'started_at': time.time(), 'completed_stages': []}

    def publish():
        (root / 'restore_status.json').write_text(json.dumps(status, indent=2) + '\n')
        if payload.get('restore_report_put'):
            report = dict(status)
            with (root / 'restore.log').open('rb') as source:
                source.seek(max(0, os.fstat(source.fileno()).st_size - 12000))
                report['log_tail'] = source.read().decode('utf-8', errors='replace')
            try:
                request = urllib.request.Request(payload['restore_report_put'],
                    data=json.dumps(report).encode(), method='PUT')
                with urllib.request.urlopen(request, timeout=30):
                    pass
            except Exception:
                print('restore_report_upload_failed', flush=True)

    def stage(name):
        status['completed_stages'].append(name)
        status['updated_at'] = time.time()
        print(name, flush=True)
        publish()

    def command(args, **kwargs):
        subprocess.run([str(v) for v in args], check=True, **kwargs)

    try:
        stage('download_started')
        exec(compile((root / 'download_signed_objects.py').read_text(),
                     'download_signed_objects.py', 'exec'), namespace := {})
        namespace['run'](payload['downloads'])
        stage('all_inputs_verified')
        code = root / 'code'
        if not (root / '_CODE_READY').exists():
            stage('code_extract_started')
            code.mkdir(exist_ok=True)
            command(['tar', '--no-same-owner', '-xzf', root / 'inputs/code.tar.gz', '-C', code])
            (root / '_CODE_READY').touch()
        stage('code_ready')
        for name, archive in payload['environments'].items():
            env = root / 'envs' / name
            if (env / '_RESTORED').exists():
                continue
            stage(name + '_extract_started')
            env.mkdir(parents=True, exist_ok=True)
            command(['tar', '--no-same-owner', '-xzf', archive, '-C', env])
            python = env / 'bin/python'
            assert python.is_file(), 'Archive has no Python executable'
            unpack = env / 'bin/conda-unpack'
            if unpack.is_file():
                command([python, unpack])
            site = subprocess.check_output([str(python), '-c',
                'import site; print(site.getsitepackages()[0])'], text=True).strip()
            sources = (['openpi/src', 'openpi/packages/openpi-client/src'] if name == 'b1k-pi05'
                       else ['BEHAVIOR-1K/OmniGibson', 'BEHAVIOR-1K/bddl3', 'BEHAVIOR-1K/joylo'])
            (Path(site) / '00-sco-b1k-source.pth').write_text(
                ''.join(str(code / p) + '\n' for p in sources))
            (env / '_RESTORED').touch()
            stage(name + '_restored')
        policy = root / 'envs/b1k-pi05'
        evaluation = root / 'envs/b1k-eval'
        link = code / 'openpi/.venv'
        if not link.exists():
            link.symlink_to(policy, target_is_directory=True)
        command([evaluation / 'bin/python', code / 'nebula/eval_runtime.py',
                 'verify', '--wheel-dir', code / 'nebula/runtime-wheels'])
        command([evaluation / 'bin/python', '-m', 'pip', 'install', '--no-index',
                 '--no-deps', '--no-cache-dir', '--force-reinstall', '--require-hashes',
                 '--find-links', code / 'nebula/runtime-wheels',
                 '-r', code / 'nebula/requirements-eval-runtime.txt'])
        stage('pinned_warp_installed')
        assets = root / 'inputs/assets'
        stage('assets_checksum_started')
        command(['sha256sum', '-c', 'SHA256SUMS'], cwd=assets)
        data = root / 'omnigibson-data'
        for archive, destination in [
            ('behavior-1k-assets-3.9.0.zip', 'behavior-1k-assets'),
            ('omnigibson-robot-assets-3.8.2.zip', 'omnigibson-robot-assets'),
            ('2026-challenge-task-instances.zip', '2026-challenge-task-instances')]:
            target = data / destination
            marker = root / ('_' + destination + '_EXTRACTED')
            if not marker.exists():
                stage(destination + '_extract_started')
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(assets / archive) as zipped:
                    for item in zipped.infolist():
                        p = Path(item.filename)
                        assert not p.is_absolute() and '..' not in p.parts
                    zipped.extractall(target)
                marker.touch()
            stage(destination + '_extracted')
        license_target = data / 'omnigibson.key'
        license_target.write_bytes((assets / 'omnigibson.key').read_bytes())
        license_target.chmod(0o600)
        cache = root / 'openpi-cache'
        if not (cache / 'big_vision/paligemma_tokenizer.model').is_file():
            command([evaluation / 'bin/python', code / 'nebula/eval_storage.py',
                     'tokenizer', '--archive', payload['cache_archive'],
                     '--destination', cache])
        command(['sha256sum', '-c', 'SHA256SUMS.inference'], cwd=root / 'checkpoint')
        ready = json.loads((root / 'checkpoint/_READY').read_text())
        assert ready['step'] == 1375000
        assert ready['manifest_sha256'] == payload['checkpoint_manifest_sha256']
        checks = [data / 'behavior-1k-assets/VERSION',
                  data / 'omnigibson-robot-assets/VERSION',
                  data / '2026-challenge-task-instances/metadata/task.jsonl']
        assert all(p.is_file() and p.stat().st_size for p in checks)
        stage('runtime_inputs_ready')
        status['state'] = 'complete'
        status['completed_at'] = time.time()
        publish()
        (root / '_RUNTIME_READY').write_text(json.dumps(status, indent=2) + '\n')
    except BaseException as error:
        import traceback
        traceback.print_exc()
        status.update(state='failed', error_type=type(error).__name__, updated_at=time.time())
        publish()
        raise

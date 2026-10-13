"""Install isolated, hashed data-reader dependencies and run the CPU export."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import subprocess
import tarfile
import time
import urllib.error
import urllib.request


def open_with_dns_retry(opener, request, timeout):
    for attempt in range(3):
        try:
            return opener.open(request, timeout=timeout)
        except urllib.error.URLError as error:
            reason = getattr(error, 'reason', None)
            if not isinstance(reason, (socket.gaierror, TimeoutError, ConnectionError)) or attempt == 2:
                raise
            time.sleep(1 + attempt)


def run(payload):
    root = Path(payload['root'])
    assert str(root).startswith('/mnt/afs/260010165/behavior_1k/retraining_20261013/')
    root.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    def status(phase, **details):
        value = {'phase': phase, 'training_executed': False, **details}
        (root / 'bootstrap_status.json').write_text(json.dumps(value, indent=2))
        with open_with_dns_retry(opener, urllib.request.Request(payload['status_put'],
                data=json.dumps(value).encode(), method='PUT'), timeout=60):
            pass
        print(json.dumps(value), flush=True)
    def extract(path, target):
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as archive:
            seen = set()
            for member in archive.getmembers():
                rel = Path(member.name)
                assert not rel.is_absolute() and '..' not in rel.parts
                assert member.name not in seen and (member.isfile() or member.isdir())
                seen.add(member.name)
            archive.extractall(target)
    try:
        if payload.get('resume_source_sha256'):
            previous = root / 'private_sources.json'
            assert hashlib.sha256(previous.read_bytes()).hexdigest() == payload['resume_source_sha256'], 'Frozen resume source manifest differs'
            assert not (root / 'output/STOP').exists(), 'Resume STOP is present'
            for child_stop in (root / 'processes').glob('*/output/STOP'):
                raise RuntimeError('Child resume STOP is present')
            audit = root / 'recovery' / payload['recovery_generation']
            audit.mkdir(parents=True, exist_ok=False)
            names = ['bootstrap_status.json', 'cloud_code.tar.gz', 'dependency_install.log',
                     'runtime_import_error.log', 'output/cloud_status.json', 'output/failure.json']
            names += [str(p.relative_to(root)) for p in (root / 'processes').glob('*/output/*.json')]
            for name in names:
                path = root / name
                if path.is_file():
                    target = audit / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        status('fetching_hashed_data_toolchain')
        for item in payload['assets']:
            current_asset = item['name']
            path = root / item['name']
            for attempt in range(3):
                try:
                    with opener.open(item['get'], timeout=60) as source, path.open('wb') as target:
                        digest = hashlib.sha256(); count = 0
                        while True:
                            block = source.read(8 * 1024**2)
                            if not block:
                                break
                            target.write(block); digest.update(block); count += len(block)
                    break
                except urllib.error.URLError as error:
                    reason = getattr(error, 'reason', None)
                    if not isinstance(reason, (socket.gaierror, TimeoutError, ConnectionError)) or attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            assert count == item['bytes'] and digest.hexdigest() == item['sha256']
            if item.get('extract_to'):
                extract(path, root / item['extract_to'])
            if item['name'].startswith('private_'):
                path.chmod(0o600)
        python = '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python'
        assert Path(python).is_file()
        deps = root / 'deps'; deps.mkdir(exist_ok=True)
        status('installing_isolated_cpu_data_dependencies')
        wheels = sorted(map(str, (root / 'wheels').glob('*.whl')))
        assert wheels
        result = subprocess.run([python, '-m', 'pip', 'install', '--no-index', '--no-deps',
            '--target', str(deps), *wheels], stdout=(root / 'dependency_install.log').open('w'),
            stderr=subprocess.STDOUT, timeout=900)
        assert result.returncode == 0, 'Offline data dependency installation failed'
        cpu_threads = str(payload.get('cpu_threads', 4))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=cpu_threads,
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS=cpu_threads, HF_HUB_OFFLINE='1',
            HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONNOUSERSITE='1',
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
            LD_LIBRARY_PATH=str(Path(python).parent.parent / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
            PYTHONPATH=':'.join(str(root / p) for p in ['deps', 'code/sco', 'code/scripts',
                'code/openpi/src', 'lerobot/src']))
        check = subprocess.run([python, '-c',
            'import torch,numpy,pyarrow,datasets,h5py,av;from lerobot.datasets import LeRobotDataset;'
            'assert not torch.cuda.is_available(), "Data preparation must not use CUDA";'
            'import json;print(json.dumps({x.__name__:x.__version__ for x in [torch,numpy,pyarrow,datasets,h5py,av]}))'],
            env=env, stdout=subprocess.PIPE, stderr=(root / 'runtime_import_error.log').open('w'),
            text=True, timeout=120)
        assert check.returncode == 0, 'Isolated real LeRobot import failed'
        status('reader_runtime_verified', versions=json.loads(check.stdout))
        entrypoint = payload.get('entrypoint', 'build_retraining_cloud.py')
        assert entrypoint in ('build_retraining_cloud.py', 'build_retraining_parallel.py')
        result = subprocess.run([python, str(root / 'code/sco' / entrypoint),
            str(root), str(root / 'output')], env=env, timeout=23 * 3600)
        if result.returncode:
            raise RuntimeError('Cloud data export reported failure')
    except Exception as error:
        diagnostics = {}
        for name in ['runtime_import_error.log', 'dependency_install.log']:
            path = root / name
            if path.exists():
                text = path.read_text(errors='replace')[-1800:]
                diagnostics[name] = re.sub(r'https?://[^\s]+', '<URL>', text)
        reason = getattr(error, 'reason', None)
        status('bootstrap_needs_diagnostic', error_type=type(error).__name__,
            error_detail=re.sub(r'https?://[^\s]+', '<URL>', str(error))[-1000:],
            reason_type=type(reason).__name__, reason_errno=getattr(reason, 'errno', None),
            http_status=getattr(error, 'code', None), current_asset=locals().get('current_asset'),
            diagnostics=diagnostics)
        raise SystemExit(1) from None

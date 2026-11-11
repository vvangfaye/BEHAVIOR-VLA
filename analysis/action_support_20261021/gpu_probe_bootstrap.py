"""Bounded worker bootstrap. Presigned URLs remain only in job memory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


def run(config):
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    archive = root / 'bundle.tar.gz'
    with opener.open(config['bundle_url'], timeout=60) as response:
        payload = response.read(config['bundle_bytes'] + 1)
    assert len(payload) == config['bundle_bytes']
    assert hashlib.sha256(payload).hexdigest() == config['bundle_sha256']
    archive.write_bytes(payload)
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    plan = json.loads((root / 'plan.json').read_text())
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               RLINF_ROOT=plan['upstream'], B1K_ADAPTER_ROOT=plan['adapter'],
               PYTHONPATH=':'.join([str(root), plan['site'], plan['adapter'], plan['upstream']]))
    result_path = root / 'gpu_probe_result.json'
    try:
        with (root / 'worker.log').open('w') as log:
            proc = subprocess.run([plan['python'], '-u', str(root / 'gpu_probe.py'),
                                   '--plan', str(root / 'plan.json'), '--sample', str(root / 'sample.npz'),
                                   '--output', str(result_path)], env=env, cwd=root,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=1000)
        code = proc.returncode
        result = json.loads(result_path.read_text()) if result_path.exists() else {'passed': False, 'returncode': code}
    except Exception as error:
        code = 1
        result = {'passed': False, 'error_type': type(error).__name__}
    result['worker_returncode'] = code
    result['log_tail'] = (root / 'worker.log').read_text()[-12000:] if (root / 'worker.log').exists() else ''
    request = urllib.request.Request(config['result_put'], data=json.dumps(result).encode(), method='PUT')
    with opener.open(request, timeout=60) as response:
        assert response.status in (200, 201, 204)
    print(json.dumps({k: v for k, v in result.items() if k != 'log_tail'}), flush=True)
    raise SystemExit(code)

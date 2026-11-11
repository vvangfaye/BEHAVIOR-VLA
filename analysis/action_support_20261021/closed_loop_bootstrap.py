"""Install a bounded evaluation bundle and publish one episode receipt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


def run(config):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root = Path(config['job_root'])
    root.mkdir(parents=True, exist_ok=False)
    code = 1
    result = {'accepted': False, 'method': config['method'], 'pair': config['pair']}
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(chunk)
        return h.hexdigest()
    try:
        archive = root / 'code.tar.gz'
        with opener.open(config['bundle_url'], timeout=120) as response, archive.open('xb') as stream:
            shutil.copyfileobj(response, stream, length=1024**2)
        assert archive.stat().st_size == config['bundle_bytes'] and sha(archive) == config['bundle_sha256']
        with tarfile.open(archive) as tar:
            for member in tar:
                path = Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                with tar.extractfile(member) as source, (root / path).open('xb') as target:
                    shutil.copyfileobj(source, target)
        assert sha(root / 'plan.json') == config['plan_sha256']
        plan = json.loads((root / 'plan.json').read_text())
        for name, ref in config['checkpoint_files'].items():
            path = Path(plan['checkpoint']) / name
            assert path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256']
        env = dict(os.environ)
        env.update(PYTHONPATH=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        with (root / 'worker.log').open('x') as log:
            process = subprocess.Popen(['python3', '-u', str(root / 'closed_loop_worker.py'),
                '--plan', str(root / 'plan.json'), '--method', config['method'],
                '--task', config['pair'][0], '--instance', str(config['pair'][1])],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=28560)
            except subprocess.TimeoutExpired:
                # Let the worker's finally block stop both process groups and
                # preserve a failed receipt before the outer eight-hour cap.
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        code = process.returncode
        episode = Path(plan['afs_root']) / config['method'] / (config['pair'][0] + '-' + str(config['pair'][1]))
        if (episode / 'receipt.json').exists():
            result = json.loads((episode / 'receipt.json').read_text())
        if not result.get('accepted'):
            for name in ['policy.log', 'eval.log']:
                path = episode / name
                if path.exists():
                    result[name] = path.read_text(errors='replace')[-16000:]
    except Exception as error:
        result.update(error_type=type(error).__name__)
    result['worker_returncode'] = code
    if (root / 'worker.log').exists():
        result['worker_log_tail'] = (root / 'worker.log').read_text(errors='replace')[-12000:]
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    request = urllib.request.Request(config['result_put'], data=json.dumps(result).encode(), method='PUT')
    with opener.open(request, timeout=60) as response:
        assert response.status in (200, 201, 204)
    print(json.dumps({k: result.get(k) for k in ('accepted', 'method', 'pair', 'worker_returncode', 'record')}))
    raise SystemExit(0 if code == 0 and result.get('accepted') else 1)

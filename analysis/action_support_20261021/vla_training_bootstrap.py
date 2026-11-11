"""Stage verified inputs and run one finite pilot variant on shared AFS."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request


def run(config):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=False)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(chunk)
        return h.hexdigest()
    def put(url, data):
        req = urllib.request.Request(url, data=json.dumps(data).encode(), method='PUT')
        with opener.open(req, timeout=60) as response:
            assert response.status in (200, 201, 204)
    def download(item, target):
        if target.exists():
            assert target.stat().st_size == item['bytes'] and digest(target) == item['sha256']
            return
        partial = target.with_suffix(target.suffix + '.part')
        size = 0
        h = hashlib.sha256()
        with opener.open(item['url'], timeout=180) as response, partial.open('wb') as output:
            for chunk in iter(lambda: response.read(8 * 1024**2), b''):
                size += len(chunk)
                if size > item['bytes']:
                    raise ValueError('Remote object exceeded committed size')
                h.update(chunk)
                output.write(chunk)
        assert size == item['bytes'] and h.hexdigest() == item['sha256']
        partial.replace(target)
    result = {'complete': False, 'method': config['method'], 'seed': config['seed']}
    code = 1
    try:
        if shutil.disk_usage(root).free < 40 * 1024**3:
            raise RuntimeError('Insufficient free disk for bounded checkpoint staging')
        archive = root / 'code.tar.gz'
        download(config['code'], archive)
        with tarfile.open(archive) as tar:
            for member in tar:
                path = Path(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination)
        plan = json.loads((root / 'pilot_training_plan.json').read_text())
        for name, expected in plan['source_sha256'].items():
            assert digest(root / name) == expected
        data = Path(plan['data_directory'])
        data.mkdir(parents=True, exist_ok=True)
        ready = data / '_READY.json'
        if config['method'] == 'standard' and not ready.exists():
            with (data / '.stage.lock').open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                for name, item in config['data'].items():
                    assert Path(name).name == name
                    download(item, data / name)
                status = {'data_plan_sha256': plan['data_plan_sha256'], 'ready': True,
                          'objects': {name: {k: item[k] for k in ('bytes', 'sha256')}
                                      for name, item in config['data'].items()}}
                ready.write_text(json.dumps(status, indent=2) + '\n')
                put(config['data_ready_put'], status)
        status = json.loads(ready.read_text())
        assert status['ready'] and status['data_plan_sha256'] == plan['data_plan_sha256']
        for name, item in status['objects'].items():
            assert (data / name).stat().st_size == item['bytes']
        env = dict(os.environ)
        env.pop('PYTHONHOME', None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                   RLINF_ROOT=config['upstream'], B1K_ADAPTER_ROOT=config['adapter'],
                   PYTHONPATH=':'.join([str(root), config['site'], config['adapter'], config['upstream']]))
        output = root / 'output'
        def progress(event):
            value = {'method': config['method'], 'seed': config['seed'],
                     'observed_at': time.time(), **event}
            (root / 'progress.json').write_text(json.dumps(value, indent=2) + '\n')
            try:
                put(config['progress_put'], value)
            except Exception as error:
                print(json.dumps({'progress_upload_error': type(error).__name__}), flush=True)
        progress({'event': 'model_loading', 'verified_data_plan': plan['data_plan_sha256']})
        with (root / 'worker.log').open('w') as log:
            proc = subprocess.Popen([config['python'], '-u', str(root / 'train_vla_pilot.py'),
                                     '--plan', str(root / 'pilot_training_plan.json'),
                                     '--method', config['method'], '--seed', str(config['seed']),
                                     '--output', str(output)], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors='replace')
            timer = threading.Timer(14100, proc.terminate)
            timer.daemon = True
            timer.start()
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get('event') in ('training_start', 'update', 'complete'):
                        progress(event)
                code = proc.wait(timeout=60)
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        if (output / 'result.json').exists():
            result = json.loads((output / 'result.json').read_text())
        result.update(method=config['method'], seed=config['seed'], output=str(output), worker_returncode=code)
    except Exception as error:
        result.update(error_type=type(error).__name__, worker_returncode=code)
    result['log_tail'] = (root / 'worker.log').read_text()[-16000:] if (root / 'worker.log').exists() else ''
    put(config['result_put'], result)
    print(json.dumps({k: v for k, v in result.items() if k != 'log_tail'}), flush=True)
    raise SystemExit(code)

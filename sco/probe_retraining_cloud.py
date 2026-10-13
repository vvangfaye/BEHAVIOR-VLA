"""CPU-only source-side connectivity and installed-runtime probe.

The caller supplies short-lived URLs privately. Reports never contain URLs or
credentials, and this probe does not import a policy or start training.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request


def run(payload):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    report = {'kind': 'source_side_data_preparation_probe', 'training_executed': False,
              'gpu_requested': 0, 'started_at': time.time(), 'python': sys.version,
              'cpu_count_visible': os.cpu_count(), 'routes': {}, 'runtimes': {}}
    root = Path('/mnt/afs/260010165/behavior_1k/retraining_20261013')
    root.mkdir(parents=True, exist_ok=True)
    report['afs_free_bytes'] = shutil.disk_usage(root).free
    code = '''import json,sys,importlib.util
names=['torch','numpy','pyarrow','datasets','h5py','av','lerobot','oss2']
print(json.dumps({'python':sys.version,'modules':{n:bool(importlib.util.find_spec(n)) for n in names}}))
'''
    for executable in [sys.executable,
        '/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python']:
        if not Path(executable).exists():
            continue
        result = subprocess.run([executable, '-c', code], capture_output=True,
                                text=True, timeout=60)
        report['runtimes'][executable] = json.loads(result.stdout) if result.returncode == 0 else {
            'returncode': result.returncode}
    for name, sample in payload['source_routes'].items():
        started = time.monotonic()
        def get_range(i):
            start = i * 1024**2
            end = start + 1024**2 - 1
            request = urllib.request.Request(sample['url'], headers={
                'Range': 'bytes=%d-%d' % (start, end), 'If-Match': sample['etag']})
            with opener.open(request, timeout=30) as response:
                assert response.status == 206
                assert response.headers['Content-Range'] == 'bytes %d-%d/%d' % (start, end, sample['bytes'])
                data = response.read()
                assert len(data) == 1024**2
                return len(data)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                size = sum(pool.map(get_range, range(8)))
            elapsed = time.monotonic() - started
            report['routes'][name] = {'verified': True, 'bytes': size,
                'seconds': elapsed, 'MB_per_second': size / elapsed / 1e6}
        except Exception as error:
            report['routes'][name] = {'verified': False, 'error_type': type(error).__name__,
                'http_status': getattr(error, 'code', None)}
    report['completed_at'] = time.time()
    report['source_access_verified'] = any(r['verified'] for r in report['routes'].values())
    data = (json.dumps(report, indent=2) + '\n').encode()
    (root / 'connectivity_probe.json').write_bytes(data)
    with opener.open(urllib.request.Request(payload['report_put'], data=data, method='PUT'), timeout=60):
        pass
    print(json.dumps(report), flush=True)
    if not report['source_access_verified']:
        raise SystemExit(2)

"""Bounded read-only environment inspection plus an owned-path write probe."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def run(config):
    root = Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025')
    root.mkdir(parents=True, exist_ok=True)
    runtime = Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
    result = {'at': time.time(), 'root': str(root), 'runtime_exists': runtime.exists()}
    probe = root / 'write_probe.tmp'
    with probe.open('xb') as stream:
        stream.write(b'0' * 16384)
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()
    result['write_probe'] = True
    result['volume_free_not_personal_quota'] = shutil.disk_usage(root).free
    assets = runtime / 'omnigibson-data'
    result['asset_roots'] = sorted(p.name for p in assets.iterdir()) if assets.exists() else []
    result['versions'] = {}
    for folder in ('behavior-1k-assets', 'omnigibson-robot-assets'):
        path = assets / folder / 'VERSION'
        result['versions'][folder] = path.read_text().strip() if path.exists() else None
    result['instances_2025_present'] = (assets / '2025-challenge-task-instances/metadata/test_instances.csv').is_file()
    result['instances_2026_present'] = (assets / '2026-challenge-task-instances/metadata/test_instances.csv').is_file()
    result['environments'] = {}
    for name in ('b1k-eval', 'b1k-pi05'):
        env = runtime / 'envs' / name
        libs = list(env.glob('lib/python*/site-packages'))
        installed = {}
        for distribution in importlib.metadata.distributions(path=[str(p) for p in libs]):
            key = distribution.metadata.get('Name', '')
            if key.lower().replace('_', '-') in {'jax','jaxlib','flax','torch','numpy','scipy','isaacsim','open3d','yourdfpy','hydra-core','pandas','h5py'}:
                installed[key] = distribution.version
        result['environments'][name] = {'python':str(env/'bin/python'),'present':(env/'bin/python').exists(),'packages':installed}
    try:
        measured = subprocess.run(['du','-sx','--block-size=1','/mnt/afs/260010165'], capture_output=True, text=True, timeout=120)
        result['owner_usage'] = {'returncode': measured.returncode, 'stdout': measured.stdout[:1000],
                                 'permission_errors': len(measured.stderr.splitlines()),
                                 'complete': measured.returncode == 0}
    except subprocess.TimeoutExpired:
        result['owner_usage'] = {'complete':False,'timeout':True}
    result['completed_at'] = time.time()
    payload = json.dumps(result, indent=2).encode()
    (root/'preflight.json').write_bytes(payload)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(config['put_url'], data=payload, method='PUT'),timeout=60) as response:
        assert response.status in (200,201,204)
    print(json.dumps(result),flush=True)

"""Restore the exact 32 archived rollouts to a new task-owned input directory."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def run(config):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=True)
    done = []
    mutex = threading.Lock()
    started = time.time()
    def upload(url, value):
        body = json.dumps(value, allow_nan=False).encode()
        req = urllib.request.Request(url, data=body, method='PUT')
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=60) as response:
            assert response.status in [200, 201, 204]
    def publish(stage):
        upload(config['status_put'], {'stage': stage, 'completed_files': len(done), 'total_files': 32,
                                      'elapsed': time.time() - started})
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        return h.hexdigest()
    def restore(item):
        target = root / f"episode_{item['episode']:02}.pt"
        assert item['archive_complete_verified']
        if target.exists():
            assert target.stat().st_size == item['size'] and digest(target) == item['expected_sha256']
        else:
            part = target.with_name(target.name + '.part')
            assert not part.exists(), 'Keep incomplete output for explicit recovery'
            h = hashlib.sha256()
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(item['get'], timeout=90) as response:
                assert response.status == 200 and response.geturl() == item['get']
                assert int(response.headers['Content-Length']) == item['size']
                with part.open('xb') as stream:
                    for block in iter(lambda: response.read(8 * 1024**2), b''):
                        stream.write(block)
                        h.update(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            assert part.stat().st_size == item['size'] and h.hexdigest() == item['expected_sha256']
            part.rename(target)
        # Verify the persisted file, rather than accepting only the transfer hash.
        assert digest(target) == item['expected_sha256']
        value = {k: item[k] for k in ['episode', 'path', 'key', 'size', 'expected_sha256']}
        value['restored_path'] = str(target)
        with mutex:
            done.append(value)
            publish('restoring')
        return value
    with (root / 'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert config['total_bytes'] < 16 * 1024**3
            publish('starting')
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(restore, config['files']))
            result = {'passed': True, 'files': rows, 'total_bytes': sum(r['size'] for r in rows),
                      'complete': len(rows) == 32, 'original_files_unchanged': True,
                      'all_sha256_verified': True, 'elapsed_seconds': time.time() - started}
            body = json.dumps(result, indent=2).encode()
            (root / 'restore_manifest.json').write_bytes(body)
            upload(config['result_put'], result)
            publish('complete')
        except BaseException as error:
            upload(config['status_put'], {'stage': 'failed', 'error_type': type(error).__name__,
                                          'completed_files': len(done), 'total_files': 32})
            raise

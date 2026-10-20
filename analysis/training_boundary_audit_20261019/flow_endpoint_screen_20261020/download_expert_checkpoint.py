"""Download the exact accepted inference checkpoint; keep signed URLs in memory."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
from urllib.parse import urlsplit
import urllib.request

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def save(value):
    tmp = HERE / 'checkpoint_download_status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(HERE / 'checkpoint_download_status.json')


def main():
    plan = json.loads((HERE / 'checkpoint_download_plan.json').read_text())
    prefix = plan['remote_source_prefix'] + '/'
    rows = plan['files']
    for row in rows:
        rel = PurePosixPath(row['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        assert row['source_key'] == prefix + str(rel)
        assert row['bytes'] > 0 and len(row['sha256']) == 64
    assert len({r['relative_path'] for r in rows}) == len(rows) == 28
    assert sum(r['bytes'] for r in rows) == plan['total_bytes']
    destination = Path(plan['destination'])
    assert destination == HERE / 'expert1000_checkpoint'
    destination.mkdir(exist_ok=True)
    state = {'status': 'signing', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'expected_bytes': plan['total_bytes'], 'expected_files': len(rows), 'completed': [],
             'urls_persisted': False, 'credentials_copied': False}
    save(state)
    # The existing remote config stays on macpro. No credentials are returned.
    remote = ('source "/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs/nebula/config.sh" && '
              'export OSS_ENDPOINT=https://oss-cn-zhangjiakou.aliyuncs.com && '
              'export PYTHONPATH="/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs" && '
              '"/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python" -')
    code = ('import json\nfrom nebula.submit_oss import OSSClient\n'
            + 'rows=json.loads(' + repr(json.dumps(rows)) + ')\n'
            + 'c=OSSClient()\n'
            + 'out=[{"relative_path":r["relative_path"],"url":c.bucket.sign_url("GET",r["source_key"],3600)} for r in rows]\n'
            + 'print(json.dumps(out))\n')
    signed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', remote],
                            input=code, text=True, capture_output=True, timeout=90)
    if signed.returncode:
        state.update(status='signing_failed', ssh_returncode=signed.returncode)
        save(state)
        raise RuntimeError('Remote signing failed; output deliberately not persisted')
    urls = json.loads(signed.stdout)
    url_map = {r['relative_path']: r['url'] for r in urls}
    assert set(url_map) == {r['relative_path'] for r in rows}
    for value in url_map.values():
        parsed = urlsplit(value)
        assert parsed.scheme == 'https' and parsed.hostname.endswith('.aliyuncs.com')
    state['status'] = 'downloading'
    save(state)
    started = time.monotonic()

    def download(row):
        target = destination / row['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
            return {**row, 'reused': True}
        if time.monotonic() - started > plan['maximum_wall_seconds']:
            raise TimeoutError('Transfer wall budget exhausted')
        partial = target.with_name(target.name + '.download-part')
        if partial.exists():
            raise RuntimeError('Incomplete earlier transfer needs explicit resume verification')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url_map[row['relative_path']],
                                     headers={'If-Match': '"' + row['etag'] + '"'})
        digest = hashlib.sha256()
        size = 0
        with opener.open(req, timeout=60) as response:
            assert response.status == 200
            assert int(response.headers['Content-Length']) == row['bytes']
            assert response.headers['ETag'].strip('"') == row['etag']
            with partial.open('xb') as stream:
                while True:
                    if time.monotonic() - started > plan['maximum_wall_seconds']:
                        raise TimeoutError('Transfer wall budget exhausted')
                    data = response.read(4 * 1024**2)
                    if not data:
                        break
                    size += len(data)
                    if size > row['bytes']:
                        raise ValueError('Oversized response')
                    digest.update(data)
                    stream.write(data)
        assert size == row['bytes'] and digest.hexdigest() == row['sha256']
        partial.replace(target)
        # Read the written file independently of the received-stream digest.
        assert sha(target) == row['sha256']
        return {**row, 'reused': False}

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for future in as_completed([pool.submit(download, row) for row in rows]):
                row = future.result()
                state['completed'].append(row)
                state['elapsed_seconds'] = time.monotonic() - started
                save(state)
                print(json.dumps({'completed_files': len(state['completed']),
                                  'completed_bytes': sum(r['bytes'] for r in state['completed']),
                                  'elapsed_seconds': round(state['elapsed_seconds'], 2)}), flush=True)
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__)
        save(state)
        raise RuntimeError('Transfer failed; consult unsigned status, no URL logged') from None
    assert len(state['completed']) == 28
    state.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                 local_content_sha256_verified=True)
    save(state)
    (destination / '_LOCAL_VERIFIED.json').write_text(json.dumps({
        'plan_sha256': sha(HERE / 'checkpoint_download_plan.json'),
        'files': rows, 'all_content_verified': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()

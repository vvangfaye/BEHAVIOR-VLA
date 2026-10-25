"""Bounded two-worker sampling-size experiment; immutable reports and checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tarfile
import time
import traceback
import urllib.request

from common import ExecutionStopped, write
from parallel import coordination, wait_json


def upload(url, path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    size = Path(path).stat().st_size
    assert size < 5 * 1024**3
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with Path(path).open('rb') as stream:
        request = urllib.request.Request(url, data=stream, method='PUT', headers={'Content-Length': str(size)})
        with opener.open(request, timeout=1800):
            pass
    return {'bytes': size, 'sha256': digest.hexdigest()}


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here / 'plan.json').read_text())
    rank = int(os.environ['B1K_RANK'])
    root = Path(plan['output'])
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'plan.json', plan)
        write(coordination(plan) / 'leader_ready.json', {'rank': 0})
    else:
        wait_json(coordination(plan) / 'leader_ready.json', plan, timeout=1800)
    accepted = json.loads(Path(plan['accepted_gate']).read_text())
    assert accepted['passed'] and all(accepted['checks'].values())
    sources = {str(p.relative_to(here)): hashlib.sha256(p.read_bytes()).hexdigest() for p in here.rglob('*.py')}
    write(root / f'execution_source_sha256_rank{rank}.json', sources)
    state = {'state': 'running', 'stage': 'loading', 'rank': rank, 'job_name': os.environ['B1K_JOB_NAME'],
             'started_at': time.time(), 'plan': '32 trajectories/round, 17+13 updates, 30 fixed eval cases'}
    status_path = root / f'status_rank{rank}.json'
    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(status_path, state)
        try:
            upload(os.environ['B1K_STATUS_PUT'], status_path)
        except Exception as error:
            print('Status upload unavailable: ' + type(error).__name__, flush=True)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    def deadline(*args):
        raise ExecutionStopped('Bounded run stopped; preserve completed artifacts')
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    success = False
    try:
        from parallel import run
        publish(stage='loading')
        report = run(plan, publish)
        assert report['comparison_valid']
        success = True
    except BaseException as error:
        details = re.sub(r'https?://[^\s\"\']+', '[URL omitted]', traceback.format_exc())
        write(coordination(plan) / f'failure_rank{rank}.json', {'error_type': type(error).__name__, 'traceback': details})
        publish(state='failed', stage='compute_failed', error_type=type(error).__name__, error=str(error)[:600])
    finally:
        signal.alarm(0)
        try:
            artifacts = {}
            if rank == 0:
                publish(stage='uploading_artifacts')
                for relative, target in json.loads(os.environ['B1K_ARTIFACT_PUTS']).items():
                    path = root / relative
                    if path.is_file():
                        artifacts[relative] = {**upload(target['url'], path), 'key': target['key']}
                write(root / 'artifact_manifest.json', artifacts)
            archive = here.parent / 'result.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                paths = root.rglob('*') if rank == 0 else [status_path]
                for path in paths:
                    if path.is_file() and path.suffix in ('.json', '.log'):
                        tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            upload(os.environ['B1K_RESULT_PUT'], archive)
        except BaseException as error:
            success = False
            state['upload_error'] = type(error).__name__
        publish(state='complete' if success else 'failed', stage='complete' if success else 'failed')
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()

"""One finite single-GPU experiment, with separate report and checkpoint uploads."""
import hashlib
import json
import os
import re
import signal
import tarfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from common import write


def fingerprints(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            sha.update(block); md5.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def upload(url, path):
    signature = fingerprints(path)
    assert signature['bytes'] < 5*1024**3, 'Split artifacts must stay below a single PUT limit'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(4):
        try:
            with Path(path).open('rb') as stream:
                request = urllib.request.Request(url, data=stream, method='PUT', headers={'Content-Length': str(signature['bytes'])})
                with opener.open(request, timeout=1800) as response:
                    assert response.status in (200, 201, 204)
                    etag = response.headers.get('ETag', '').strip('"')
                    if re.fullmatch('[0-9a-f]{32}', etag):assert etag == signature['md5'], 'Upload MD5 mismatch'
                    signature['etag'] = etag
            return signature
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:raise
            time.sleep(2**attempt)


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'plan.json').read_text())
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'plan.json', plan)
    accepted = json.loads(Path(plan['accepted_gate']).read_text())
    assert accepted['passed'] and all(accepted['checks'].values())
    write(root/'prior_acceptance_gate.json', accepted)
    sources = {str(p.relative_to(here)): fingerprints(p)['sha256'] for p in here.rglob('*.py')}
    write(root/'execution_source_sha256.json', sources)
    started = time.time()
    state = {'state': 'running', 'stage': 'loading', 'started_at': started, 'job_name': plan['job_name']}

    def publish(**fields):
        state.update(fields, updated_at=time.time())
        write(root/'status.json', state)
        try:upload(os.environ['B1K_STATUS_PUT'], root/'status.json')
        except Exception as e:print('Status upload unavailable: '+type(e).__name__, flush=True)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def deadline(signum, frame):
        raise TimeoutError('Frozen compute budget exhausted; preserving partial results')

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(plan['compute_timeout_seconds'])
    try:
        from experiment import run
        comparison = run(plan, publish)
        assert comparison['comparison_valid'], 'Paired initial-state check failed'
        state.update(state='complete', stage='compute_complete')
    except BaseException as error:
        details = re.sub(r'https?://[^\s\"\']+', '[URL omitted]', traceback.format_exc())
        state.update(state='failed', stage='compute_failed', error_type=type(error).__name__, error=str(error)[:1000])
        write(root/'failure.json', {'error_type':type(error).__name__, 'traceback':details})
    finally:
        signal.alarm(0)
        completed_state = state.copy()
        receipts = {}
        try:
            state["state"] = "running"
            destinations = json.loads(os.environ['B1K_ARTIFACT_PUTS'])
            for relative, target in destinations.items():
                path = root/relative
                if not path.is_file():continue
                publish(stage='uploading_checkpoint', artifact=relative)
                receipts[relative] = {**upload(target['url'], path), 'bucket':'260010165', 'key':target['key']}
            write(root/'artifact_manifest.json', receipts)
        except BaseException as error:
            completed_state.update(state='failed', upload_error=type(error).__name__+': '+str(error)[:600])
        state.clear(); state.update(completed_state, elapsed_seconds=time.time()-started)
        write(root/'status.json', state)
        archive = root.parent/'result.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for path in root.rglob('*'):
                rel = path.relative_to(root)
                if path.is_file() and path.suffix in {'.json','.log'} and not any(part in {'appdata','cache','config','warp'} for part in rel.parts):
                    tar.add(path, arcname=str(rel), recursive=False)
        try:
            report_receipt = upload(os.environ['B1K_RESULT_PUT'], archive)
            write(root.parent/'report_upload_receipt.json', report_receipt)
        except BaseException as error:
            state.update(state='failed', report_upload_error=type(error).__name__)
        publish(**state)
    raise SystemExit(0 if state['state']=='complete' else 1)


if __name__=='__main__':main()

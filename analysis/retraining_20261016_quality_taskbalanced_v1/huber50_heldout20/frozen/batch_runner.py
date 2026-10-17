"""Four fixed episodes sequentially on one GPU; never retry a rollout."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import zlib


class Cancelled(BaseException):
    pass


def run(config):
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    child = None

    def send(url, data):
        raw = json.dumps(data, allow_nan=False).encode()
        for attempt in range(4):
            try:
                with opener.open(urllib.request.Request(url, data=raw, method='PUT'), timeout=25):
                    return True
            except OSError:
                if attempt < 3:
                    time.sleep(2 ** attempt)
        return False

    def stop_child():
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=40)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)

    def stop(signum, frame):
        raise Cancelled('Batch stopped')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    results = []
    status = dict(id=config['id'], state='running', protocol_sha256=config['protocol_sha256'],
                  worker_bundle_sha256=config['worker_bundle_sha256'], members=results,
                  started_at=time.time(), training_modified=False)

    def publish():
        status['observed_at'] = time.time()
        tmp = root / 'status.json.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(root / 'status.json')
        send(config['report_url'], status)

    try:
        for member in config['members']:
            status['current_member'] = member['id']
            publish()
            source = zlib.decompress(base64.b64decode(member['source']))
            if hashlib.sha256(source).hexdigest() != member['sha256']:
                raise ValueError('Member source changed')
            script = root / (member['id'] + '.py')
            script.write_bytes(source)
            os.chmod(script, 0o600)
            error = None
            try:
                with (root / (member['id'] + '.supervisor.log')).open('x') as log:
                    child = subprocess.Popen([config['python'], str(script)], stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    child.wait(timeout=config['member_walltime_seconds'])
            except subprocess.TimeoutExpired:
                error = 'MemberTimeout'
                stop_child()
            finally:
                stop_child()
            report_path = Path(member['report_path'])
            if report_path.is_file():
                report = json.loads(report_path.read_text())
            else:
                report = dict(id=member['id'], state='failed', error_type=error or 'MemberExitedWithoutReport',
                              protocol_sha256=config['protocol_sha256'],
                              worker_bundle_sha256=config['worker_bundle_sha256'], finished_at=time.time())
            # Preserve each accepted result immediately, even if a later member fails.
            uploaded = send(member['report_url'], report)
            results.append(dict(id=member['id'], state=report['state'],
                                returncode=child.returncode, report_uploaded=uploaded,
                                report_path=str(report_path), error_type=error))
            child = None
            publish()
        status['state'] = ('accepted' if len(results) == len(config['members']) and
                           all(r['state'] == 'accepted' and r['report_uploaded'] for r in results)
                           else 'failed')
        status['finished_at'] = time.time()
        publish()
        return 0 if status['state'] == 'accepted' else 1
    finally:
        stop_child()

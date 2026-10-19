"""Supervise one unchanged frozen worker; persist diagnostics, never retry it."""
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


def supervisor(config):
    root = Path(config['monitor_root'])
    root.mkdir(parents=True, exist_ok=False)
    child_source = zlib.decompress(base64.b64decode(config['child_source']))
    assert hashlib.sha256(child_source).hexdigest() == config['child_sha256']
    (root/'child.startup.py').write_bytes(child_source)
    os.chmod(root/'child.startup.py', 0o600)
    episode = Path(config['episode_root'])
    worker = Path(config['worker_root'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def send(url, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        for attempt in range(4):
            try:
                with opener.open(urllib.request.Request(url, data=raw, method='PUT'), timeout=20):
                    return True
            except OSError:
                if attempt != 3:
                    time.sleep(2**attempt)
        return False

    def memory():
        data = {}
        for name in ('memory.current', 'memory.max', 'memory.peak', 'memory.events',
                     'memory/memory.usage_in_bytes', 'memory/memory.limit_in_bytes',
                     'memory/memory.max_usage_in_bytes', 'memory/memory.failcnt',
                     'memory/memory.oom_control'):
            path = Path('/sys/fs/cgroup') / name
            if path.is_file():
                try:
                    data[name] = path.read_text()[:2000]
                except OSError as error:
                    data[name] = {'error_type': type(error).__name__}
        processes = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                text = (proc/'status').read_text()
                fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
                if fields.get('Uid', '').split()[0] == str(os.getuid()):
                    processes.append({'pid': int(proc.name), 'name': fields.get('Name', '').strip(),
                                      'VmRSS': fields.get('VmRSS', '').strip()})
            except (OSError, IndexError):
                continue
        return {'cgroup': data, 'processes': processes[:100]}

    started = time.time()
    with (root/'supervisor.log').open('x') as log:
        child = subprocess.Popen([config['python'], str(root/'child.startup.py')],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def stop(signum, frame):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            rc = child.poll()
            heartbeat = dict(observed_at=time.time(), started_at=started, child_pid=child.pid,
                             child_returncode=rc, state='running' if rc is None else 'exited',
                             evaluation_id=config['id'], attempt=1, generation=0,
                             original_uid=config['original_uid'], training_modified=False,
                             checkpoint_step=config['checkpoint_step'], resource=memory())
            paths = ['policy.log', 'eval.log', 'seed_ack.json', 'acceptance.json',
                     'results/json/'+config['result_name']+'.json',
                     'results/videos/'+config['result_name']+'.mp4']
            heartbeat['episode_files'] = {name: {'exists': (episode/name).is_file(),
                                                 'bytes': (episode/name).stat().st_size if (episode/name).is_file() else None}
                                           for name in paths}
            (root/'heartbeat.json').write_text(json.dumps(heartbeat, indent=2)+'\n')
            with (root/'resource_history.jsonl').open('a') as history:
                history.write(json.dumps(heartbeat)+'\n')
            send(config['heartbeat_put_url'], heartbeat)
            if rc is not None:
                failure = dict(heartbeat)
                failure['log_contents_exported'] = False
                (root/'exit_diagnostics.json').write_text(json.dumps(failure, indent=2)+'\n')
                send(config['diagnostics_put_url'], failure)
                return rc
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass

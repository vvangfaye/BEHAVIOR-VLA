"""Read-only 45-second monitor for the single authorized CCR upload."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

B = Path(__file__).resolve().parents[1] / 'eval_results/sco_submission1000_2000000_20261013/serving_bundle'
REMOTE = '/home/faye/behavior-video2m-20261013/serving_bundle'

def save(name, data):
    p = B / name
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(data, indent=2) + '\n')
    t.replace(p)

def main():
    with (B / 'ccr_direct_watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('ccr_direct_watch_process.json', {'pid': os.getpid(), 'started_epoch': time.time(), 'poll_seconds': 45})
        code = "import pathlib,json; b=pathlib.Path(%r); names=['ccr_direct_status.json','ccr_direct_process.json','ccr_publish_status.json','CCR_IMAGE_PUBLISHED.json']; print(json.dumps({n:json.loads((b/n).read_text()) for n in names if (b/n).exists()}))" % REMOTE
        while True:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'slai-wy-local', 'python3 -c ' + shlex.quote(code)], capture_output=True, text=True, timeout=35)
            if p.returncode == 0:
                data = json.loads(p.stdout)
                for name, value in data.items():
                    save(name, value)
                state = data.get('ccr_direct_status.json', {}).get('state')
                if state in ('published', 'attention_required'):
                    save('ccr_direct_watch_exit.json', {'state': state, 'observed_epoch': time.time()})
                    return
            else:
                save('ccr_direct_watch_status.json', {'state': 'read_temporarily_unavailable', 'observed_epoch': time.time()})
            time.sleep(45)

if __name__ == '__main__':
    main()

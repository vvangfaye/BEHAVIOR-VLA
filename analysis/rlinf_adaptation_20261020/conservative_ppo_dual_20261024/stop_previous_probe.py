"""Request graceful termination of this exact previous experiment driver."""
import base64
import json
import os
from pathlib import Path
import signal
import time

root = Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/resume03')
output = root / 'results'
driver = str(root / 'run/driver.py')
pids = []
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().split(b'\0')
        if driver.encode() in args:
            pids.append(int(path.parent.name))
    except (OSError, ProcessLookupError):
        pass
assert len(pids) == 1, pids
status = json.loads((output / 'status.json').read_text())
progress = {}
for path in sorted((output / 'training').glob('round_*/episode_*/progress.json')):
    progress[str(path.relative_to(output))] = json.loads(path.read_text())
receipt = {'requested_at': time.time(), 'driver_pid': pids[0],
           'reason': 'User requested two-GPU migration; preserve completed cases',
           'status_before': status, 'training_progress_before': progress,
           'completed': [str(p.relative_to(output)) for p in sorted(
               (output / 'training').glob('round_*/episode_*/result.json'))]}
(root / 'migration_stop_request.json').write_text(json.dumps(receipt, indent=2))
os.kill(pids[0], signal.SIGTERM)
summary = {k: receipt[k] for k in ('requested_at', 'driver_pid', 'completed', 'training_progress_before')}
summary['stage_before'] = status.get('stage')
print('B1KPROBE_BEGIN' + base64.b64encode(json.dumps(summary).encode()).decode() + 'B1KPROBE_END', flush=True)

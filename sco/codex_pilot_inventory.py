"""Bounded read-only inventory of the pilot's existing dataset/runtime sources."""
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

root = Path('/mnt/afs/260010165/behavior_1k')
assert root.is_dir()
result = dict(state='inventory_complete', observed_at=time.time(), directories=[], datasets=[], source_files=[])
result['disk_usage'] = shutil.disk_usage(root)._asdict()
for p in sorted(root.iterdir()):
    if p.is_dir():
        result['directories'].append(p.name)
count = 0
for parent, dirs, files in os.walk(root):
    relative = Path(parent).relative_to(root)
    # Inspect metadata only, bounded depth/count; never traverse simulatorassets or checkpoints.
    dirs[:] = [d for d in dirs if d not in {'omnigibson-data','assets','params','optimizer','videos',
                'site-packages','pkgs','envs','raw_full100_20261007_1308','node_modules','.git'}]
    if len(relative.parts) > 6:
        dirs[:] = []
        continue
    count += 1
    if count > 20000:
        result['bounded_scan'] = True
        break
    if 'info.json' in files and Path(parent).name == 'meta':
        p = Path(parent)/'info.json'
        try:
            info=json.loads(p.read_text())
            result['datasets'].append(dict(root=str(p.parent.parent), total_episodes=info.get('total_episodes'),
                                           total_frames=info.get('total_frames'), fps=info.get('fps')))
        except (OSError,ValueError):
            pass
    for name in files:
        if name in {'subtask_segments.parquet','_READY','_RUNTIME_READY'} and len(result['source_files'])<150:
            result['source_files'].append(str(Path(parent)/name))
runtime = root/'runtime_20261007_1306'
result['runtime']={str(p):p.exists() for p in [runtime/'envs/b1k-eval/bin/python',runtime/'envs/b1k-pi05/bin/python',
    root/'checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010/_READY']}
raw = json.dumps(result).encode()
req = urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'], data=raw, method='PUT')
with urllib.request.urlopen(req, timeout=30):
    pass
print(json.dumps(dict(state=result['state'], datasets=len(result['datasets']), scanned_directories=count)),flush=True)

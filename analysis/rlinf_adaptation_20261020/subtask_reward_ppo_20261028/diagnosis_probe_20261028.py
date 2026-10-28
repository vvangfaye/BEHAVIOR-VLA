"""Read existing training report; no model execution or result changes."""
import base64
import json
import math
from pathlib import Path

root = Path('/mnt/afs/260010165/behavior_1k/rlinf_midreward_20261028/pilot01/results')
path = root / 'training_updates/train_eval_r0_a1/result.json'
assert path.is_file() and path.stat().st_size < 64 * 1024**2
report = json.loads(path.read_text())
out = {'training': {k: v for k, v in report.items() if k not in ('metrics', 'order')}}
metrics = report.get('metrics', [])
out['metric_keys'] = sorted(set().union(*(set(row) for row in metrics)))
out['metric_mean'] = {}
for key in out['metric_keys']:
    if key == 'chunk':
        continue
    values = [float(row[key]) for row in metrics
              if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
    if values:
        out['metric_mean'][key] = sum(values) / len(values)
for name in ('training_complete.json', 'checkpoints/round_01/ready.json'):
    source = root / name
    if source.is_file():
        out[name] = json.loads(source.read_text())
print('B1KPROBE_BEGIN' + base64.b64encode(json.dumps(out).encode()).decode() + 'B1KPROBE_END')

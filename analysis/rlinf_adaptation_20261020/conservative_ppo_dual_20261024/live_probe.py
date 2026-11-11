import base64
import json
from pathlib import Path
import subprocess
import time

base = Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/dual01')
root = base / 'results'
result = {'at': time.time(), 'root_exists': root.exists(), 'files': {}}
paths = list(root.glob('*.json')) + list((root / 'coordination').glob('*.json'))
paths += list((root / 'training').glob('round_*/result.json'))
paths += list((root / 'training').glob('round_*/episode_*/result.json'))
paths += list((root / 'training').glob('round_*/episode_*/progress.json'))
paths += list((root / 'training').glob('round_*/episode_*/rollout/initialization.json'))
paths += list((root / 'final').glob('*/result.json'))
for path in sorted(set(paths)):
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and 'update' in data:
            data['update'].pop('metrics', None)
            data['update'].pop('order', None)
        result['files'][str(path.relative_to(root))] = data
    except FileNotFoundError:pass
for rank in (0, 1):
    for name in ('bootstrap_error.json', 'runtime_receipt.json', 'resume_preflight.json'):
        path = base / f'worker_{rank}' / name
        if path.is_file():result[f'worker_{rank}/{name}'] = json.loads(path.read_text())
try:
    result['local_gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True, timeout=10).strip()
except Exception as error:result['gpu_error'] = type(error).__name__
print('B1KPROBE_BEGIN' + base64.b64encode(json.dumps(result).encode()).decode() + 'B1KPROBE_END', flush=True)

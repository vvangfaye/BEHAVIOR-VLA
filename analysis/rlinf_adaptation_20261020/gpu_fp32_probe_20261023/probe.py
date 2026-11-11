"""Measure FP32 JAX beside the already-running simulator, without driving it."""
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

ROOT = Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry07')
OUT = ROOT / 'results/gpu_fp32_probe'
OUT.mkdir(exist_ok=True)
samples = []
stopping = threading.Event()
started = time.monotonic()


def write(name, value):
    p = OUT / name
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(p)


def sample():
    while not stopping.is_set():
        try:
            raw = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ], text=True, timeout=5).strip()
            used, total, util = map(int, raw.split(','))
            samples.append({'seconds': time.monotonic()-started, 'used_mib': used, 'total_mib': total, 'utilization': util})
        except Exception:
            pass
        stopping.wait(0.3)


thread = threading.Thread(target=sample, daemon=True)
thread.start()
report = {'passed': False, 'state': 'starting', 'jax_platform': os.environ.get('JAX_PLATFORMS')}
try:
    import numpy as np
    import jax
    from rollout import jax_policy, infer_jax
    jax.config.update('jax_default_matmul_precision', 'highest')
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == 'gpu', devices
    plan = json.loads((ROOT/'run/plan.json').read_text())
    report.update(state='loading', device=str(devices[0]))
    write('progress.json', report)
    policy = jax_policy(plan)
    report.update(state='infer', model_loaded_seconds=time.monotonic()-started)
    write('progress.json', report)
    stats = json.loads((Path(plan['converted'])/'norm_stats.json').read_text())['norm_stats']['actions']
    std = np.array(stats['std'])[:23] + 1e-6
    paths = sorted((ROOT/'results/precheck_reference').glob('*/probe_*.npz'))
    paths += sorted((ROOT/'results/jax/302').glob('probe_*.npz'))
    comparisons = []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as d:
            raw = {'state': d['state'], 'images': list(d['images']), 'prompt': str(d['prompt'])}
            before = time.monotonic()
            actions, text = infer_jax(policy, raw, d['noise'])
            error = np.abs(actions-d['actions'])/std[None]
            comparisons.append({'path': str(path.relative_to(ROOT/'results')),
                                'seconds': time.monotonic()-before,
                                'subtask_equal': text == str(d['subtask']),
                                'mean_error': float(error.mean()), 'max_error': float(error.max()),
                                'finite': bool(np.isfinite(actions).all())})
        write('progress.json', {**report, 'completed': index+1, 'total': len(paths), 'last': comparisons[-1]})
    report.update(comparisons=comparisons, checks={
        'enough_inputs': len(comparisons) >= 12,
        'all_finite': all(x['finite'] for x in comparisons),
        'same_subtasks': all(x['subtask_equal'] for x in comparisons),
        'same_input_mean_error': max(x['mean_error'] for x in comparisons) <= plan['gate']['mean_action_error'],
        'same_input_max_error': max(x['max_error'] for x in comparisons) <= plan['gate']['max_action_error'],
    })
    report['passed'] = all(report['checks'].values())
    report['state'] = 'complete'
except BaseException as error:
    report.update(state='failed', error_type=type(error).__name__, error=str(error)[:2000], traceback=traceback.format_exc()[-6000:])
finally:
    stopping.set()
    thread.join(timeout=6)
    report.update(elapsed_seconds=time.monotonic()-started,
                  peak_device_used_mib=max((x['used_mib'] for x in samples),default=None),
                  memory_samples=samples)
    write('result.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'memory_samples','traceback','comparisons'}}), flush=True)

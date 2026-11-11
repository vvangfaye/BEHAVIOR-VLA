"""Compare completed, matched training updates; these are not task evaluations."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('h50_window_reader', ROOT / 'inspect_extreme_window.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    source = reader.REMOTE.replace("790<=row['completed_updates']<=840", "0<row['completed_updates']<=1000")
    command = shlex.quote('/Users/faye/code/Behavior-1k Challenge/.venv/bin/python') + ' -c ' + shlex.quote(source)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                             '-o', 'ConnectionAttempts=1', 'macpro', command],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only numeric training observation unavailable')
    data = json.loads(result.stdout)
    by_name = {name: {r['completed_updates']: r for r in value['rows']} for name, value in data.items()}
    if any(1000 not in rows for rows in by_name.values()):
        raise ValueError('Both runs must reach 1000 updates')
    steps = sorted(set(by_name['expert_mse']) & set(by_name['huber50']))
    comparisons = []
    for step in steps:
        old, new = [by_name[name][step] for name in ('expert_mse', 'huber50')]
        if old['mixture_fingerprint'] != new['mixture_fingerprint']:
            raise ValueError('Training data fingerprint differs')
        a, b = old['metrics'], new['metrics']
        for key in a:
            if key.startswith('source_count/') and a[key] != b[key]:
                raise ValueError('Source counts differ at matched updates')
        comparisons.append({'completed_updates': step,
            'old_action_mse': a.get('action_mse', a['action_loss']),
            'new_action_mse': b['action_mse'], 'old_grad_norm': a['grad_norm'],
            'new_grad_norm': b['grad_norm'], 'new_robust_tail_fraction': b['action_robust_tail_fraction']})
    nonfinite = {name: sum(r['metrics'].get('nonfinite_count', 0) for r in value['rows'])
                 for name, value in data.items()}
    active = [row for row in comparisons if row['new_robust_tail_fraction'] > 0]
    last100 = [row for row in comparisons if row['completed_updates'] > 900]
    report = {'completed_updates': 1000, 'matched_logged_windows': len(steps),
              'old_logged_windows': len(by_name['expert_mse']), 'new_logged_windows': len(by_name['huber50']),
              'nonfinite_counts': nonfinite, 'robust_active_windows': active,
              'maximum_logged_grad_norm': {name: max(value['rows'], key=lambda r: r['metrics']['grad_norm'])
                                           for name, value in data.items()},
              'last100_logged_windows': last100,
              'task_ability_improvement_proven': False,
              'interpretation': 'Training aggregates only; robust loss differs from MSE. Grad norms precede unchanged optimizer clipping. Task improvement requires paired evaluation.',
              'source_sha256': {name: value['metrics_snapshot_sha256'] for name, value in data.items()}}
    (ROOT / 'completed_training_dynamics_inputs.json').write_text(json.dumps(data, indent=2) + '\n')
    (ROOT / 'completed_training_dynamics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'matched_logged_windows': len(steps), 'robust_active_updates': [r['completed_updates'] for r in active],
                      'nonfinite_counts': nonfinite, 'task_ability_improvement_proven': False}))


if __name__ == '__main__':
    main()

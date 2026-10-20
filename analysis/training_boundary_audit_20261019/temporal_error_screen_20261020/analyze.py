"""Exact within-chunk MSE decomposition on previously paired predictions."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import action_change_calibration as calibration


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    paths = {k: (HERE/plan[k]).resolve() for k in ('baseline', 'candidate')}
    docs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    pairing, _ = calibration.paired_samples(docs['baseline'], docs['candidate'])
    maps = {name: {(r['episode_index'], r['frame_index']): r for r in doc['sample_keys']}
            for name, doc in docs.items()}
    rows = []
    max_error = 0.0
    for key, base in maps['baseline'].items():
        mask = np.asarray(base['raw_actions']['valid_mask'], dtype=bool)
        if not mask[:16].all():
            continue
        row = {'task': base['task'], 'episode_index': key[0], 'frame_index': key[1]}
        for name in paths:
            raw = maps[name][key]['raw_actions']
            error = (np.asarray(raw['prediction'], dtype=np.float64) -
                     np.asarray(raw['ground_truth'], dtype=np.float64))[:16]
            dc = error.mean(axis=0)**2
            ac = ((error-error.mean(axis=0))**2).mean(axis=0)
            mse = (error**2).mean(axis=0)
            max_error = max(max_error, float(np.max(np.abs(mse-dc-ac))))
            row[name] = {'mse': mse.tolist(), 'dc': dc.tolist(), 'ac': ac.tolist(),
                         'block_mean_error_power': {str(n): (error.reshape(16//n, n, 23).mean(axis=1)**2).mean(axis=0).tolist()
                                                    for n in (1,2,4,8,16)}}
        rows.append(row)
    assert len(rows) == 27
    assert max_error < 1e-12
    tasks = sorted({r['task'] for r in rows})
    aggregates = []
    for task in tasks + ['equal_task_macro']:
        metrics = {}
        for name in paths:
            metrics[name] = {}
            for field in ('mse','dc','ac'):
                per_task = [np.mean([r[name][field] for r in rows if r['task']==t], axis=0)
                            for t in (tasks if task=='equal_task_macro' else [task])]
                metrics[name][field] = np.mean(per_task, axis=0)
        coords = []
        for d in range(23):
            item = {'dimension': d}
            for field in ('mse','dc','ac'):
                b,c = (float(metrics[n][field][d]) for n in ('baseline','candidate'))
                item[field] = {'baseline': b, 'candidate': c, 'delta': c-b}
            item['mse_down_dc_up'] = item['mse']['delta'] < 0 and item['dc']['delta'] > 0
            coords.append(item)
        aggregates.append({'task': task, 'coordinates': coords,
                           'mse_down_dimensions': sum(c['mse']['delta'] < 0 for c in coords),
                           'dc_up_dimensions': sum(c['dc']['delta'] > 0 for c in coords),
                           'mse_down_dc_up_dimensions': sum(c['mse_down_dc_up'] for c in coords)})
    result = {'status': 'complete', 'paired_samples_validated': pairing['exact_samples'],
              'complete_prefix_samples': len(rows), 'tasks': len(tasks),
              'episodes': len({r['episode_index'] for r in rows}),
              'inputs': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for k,p in paths.items()},
              'max_decomposition_identity_error': max_error,
              'rows': rows, 'aggregates': aggregates, 'limits': plan['limits']}
    with (HERE/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='coordinates'} for r in aggregates]))


if __name__ == '__main__':
    main()

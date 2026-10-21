"""Descriptive task/episode table after the frozen primary analysis is ready.

No new selection rule, threshold, or checkpoint comparison is introduced.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from analyze_vla_pilot import score, sha

HERE = Path(__file__).resolve().parent
PILOT = HERE / 'posterior_support_pilot_ada01'


def main():
    gate_path = PILOT / 'training_contribution_result.json'
    if not gate_path.exists():
        print(json.dumps(dict(ready=False, waiting_for=str(gate_path))))
        return
    gate = json.loads(gate_path.read_text())
    assert gate['ready']
    data = json.loads((HERE / 'vla_pilot/data_plan.json').read_text())
    paths = dict(control=PILOT / 'formal_ambient_aux/test.npz',
                 trained=PILOT / 'formal_posterior/test.npz',
                 posthoc=HERE / 'posterior_posthoc_control_ada01/posthoc.npz')
    arrays = {}
    for method, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            arrays[method] = {k: z[k].copy() for k in ('index', 'episode', 'frame', 'target', 'samples')}
    reference = arrays['control']
    for method in ('trained', 'posthoc'):
        for key in ('index', 'episode', 'frame', 'target'):
            np.testing.assert_array_equal(reference[key], arrays[method][key])
    metrics = {m: score(a['samples'], a['target']) for m, a in arrays.items()}
    tasks = []
    for i, episode, frame in zip(reference['index'], reference['episode'], reference['frame']):
        sample = data['samples'][int(i)]
        assert sample['episode_index'] == int(episode) and sample['frame_index'] == int(frame)
        tasks.append(sample['task_name'])
    tasks = np.asarray(tasks)
    task_rows, episode_rows = [], []
    for task in sorted(set(tasks)):
        selected = tasks == task
        row = dict(task=task, anchors=int(selected.sum()),
                   episodes=len(np.unique(reference['episode'][selected])))
        for method in metrics:
            for name, values in metrics[method].items():
                row[method+'_'+name] = float(values[selected].mean())
        task_rows.append(row)
        for episode in np.unique(reference['episode'][selected]):
            mask = selected & (reference['episode'] == episode)
            row = dict(task=task, episode=int(episode), anchors=int(mask.sum()))
            for method in metrics:
                for name, values in metrics[method].items():
                    row[method+'_'+name] = float(values[mask].mean())
            episode_rows.append(row)
    for name, rows in [('by_task', task_rows), ('by_episode', episode_rows)]:
        with (PILOT / (name+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(ready=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  training_contribution_gate_passed=gate['training_contribution_gate_passed'],
                  source_gate_sha256=sha(gate_path),
                  source_arrays_sha256={m: sha(p) for m, p in paths.items()},
                  tasks=task_rows, episodes=episode_rows,
                  scope='Descriptive fixed-final-checkpoint breakdown; does not change either preregistered gate. Expert-subtask offline observations, not task success.',
                  goal_complete=False)
    (PILOT / 'descriptive_breakdown.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(tasks=task_rows, gate_passed=gate['training_contribution_gate_passed']), indent=2))


if __name__ == '__main__':
    main()

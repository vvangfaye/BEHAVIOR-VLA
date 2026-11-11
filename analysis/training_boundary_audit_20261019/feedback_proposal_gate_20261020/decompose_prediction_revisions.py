"""Separate common action updates from changed old/fresh response contrasts.

An algebraic diagnostic on saved predictions, not an environment intervention.
"""
import hashlib
import json
from pathlib import Path

from compare_observation_revisions import FILES, GROUPS, average


def main():
    data = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    by_model = {
        name: {(r['probe_metadata']['boundary_id'], r['probe_metadata']['bucket']): r
               for r in model['sample_keys']}
        for name, model in data.items()
    }
    norm_path = Path(data['finetuned']['artifacts']['norm_stats']['path'])
    std = json.loads(norm_path.read_text())['norm_stats']['actions']['std']
    previous_path = Path(__file__).with_name('compare_observation_revisions.json')
    previous = json.loads(previous_path.read_text())
    records = []
    for row in previous['per_boundary']:
        boundary = row['boundary_id']
        samples = {
            name: [by_model[name][(boundary, bucket)] for bucket in ['edge_H', 'inside_K']]
            for name in by_model
        }
        results = {}
        for group, dims in GROUPS.items():
            common_energy, differential_energy = 0., 0.
            losses = {name: [[], []] for name in ['parent', 'finetuned', 'common_only']}
            for t in range(16):
                for dim in dims:
                    parent, tuned, truth = [], [], []
                    for old_or_fresh, offset in enumerate([16, 0]):
                        b = samples['parent'][old_or_fresh]['raw_actions']
                        s = samples['finetuned'][old_or_fresh]['raw_actions']
                        assert b['valid_mask'][t+offset] and s['valid_mask'][t+offset]
                        assert abs(b['ground_truth'][t+offset][dim]-s['ground_truth'][t+offset][dim]) < 1e-10
                        scale = std[dim] + 1e-6
                        parent.append(b['prediction'][t+offset][dim]/scale)
                        tuned.append(s['prediction'][t+offset][dim]/scale)
                        truth.append(b['ground_truth'][t+offset][dim]/scale)
                    assert abs(truth[0]-truth[1]) < 1e-7
                    updates = [tuned[j]-parent[j] for j in [0, 1]]
                    common = (updates[0]+updates[1])/2
                    differential = (updates[1]-updates[0])/2
                    common_energy += common**2
                    differential_energy += differential**2
                    assert abs(sum(v*v for v in updates)-2*(common**2+differential**2)) < 1e-8
                    for j in [0, 1]:
                        losses['parent'][j].append((parent[j]-truth[j])**2)
                        losses['finetuned'][j].append((tuned[j]-truth[j])**2)
                        losses['common_only'][j].append((parent[j]+common-truth[j])**2)
            losses = {name: {'old': average(v[0]), 'fresh': average(v[1])}
                      for name, v in losses.items()}
            total_energy = common_energy + differential_energy
            results[group] = {
                'common_update_energy': common_energy,
                'differential_update_energy': differential_energy,
                'differential_fraction': differential_energy/total_energy if total_energy else None,
                'losses': losses,
            }
        records.append({'task': row['task'], 'episode_index': row['episode_index'],
                        'boundary_id': boundary, 'groups': results})
    per_task = {}
    for task in sorted({r['task'] for r in records}):
        selected = [r for r in records if r['task'] == task]
        per_task[task] = {}
        for group in GROUPS:
            common = sum(r['groups'][group]['common_update_energy'] for r in selected)
            diff = sum(r['groups'][group]['differential_update_energy'] for r in selected)
            losses = {name: {age: average([r['groups'][group]['losses'][name][age] for r in selected])
                             for age in ['old', 'fresh']}
                      for name in ['parent', 'finetuned', 'common_only']}
            def pattern(name):
                return losses[name]['old'] < losses['parent']['old'] and losses[name]['fresh'] > losses['parent']['fresh']
            per_task[task][group] = {'differential_fraction': diff/(common+diff) if common+diff else None,
                                     'losses': losses, 'observed_asymmetry': pattern('finetuned'),
                                     'asymmetry_with_common_update_only': pattern('common_only')}
    asymmetric = [(task, group, result) for task, groups in per_task.items()
                  for group, result in groups.items() if result['observed_asymmetry']]
    # Concrete counterexample: both outputs shift equally, changing no response contrast.
    old, fresh, shift = 1., -1., -.1
    assert (fresh+shift)-(old+shift) == fresh-old
    assert (old+shift)**2 < old**2 and (fresh+shift)**2 > fresh**2
    report = {
        'status': 'complete',
        'definition': 'common=(update_old+update_fresh)/2; differential=(update_fresh-update_old)/2',
        'inputs': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in FILES.items()},
        'summary': {'asymmetric_task_groups': len(asymmetric),
                    'asymmetry_preserved_with_common_update_only': sum(r['asymmetry_with_common_update_only'] for _, _, r in asymmetric)},
        'per_task': per_task, 'per_boundary': records,
        'interpretation_limits': [
            'common_only is an offline algebraic reconstruction, not a trained or deployed policy.',
            'A different common shift is fitted for every target action and coordinate; it is not a single globally fitted calibration.',
            'The common component is common to two observations only; it is not proved state-independent elsewhere.',
            'The differential component is a finite prediction contrast along expert states, not a causal feedback derivative.',
            'Noise differs between old and fresh observations, though paired checkpoint noise identities match.',
            'No physical success or feedback-retention claim is established.'
        ],
        'counterexample': {'truth': 0, 'parent_old': old, 'parent_fresh': fresh, 'common_shift': shift,
                           'parent_squared_errors': [old**2, fresh**2],
                           'shifted_squared_errors': [(old+shift)**2, (fresh+shift)**2],
                           'response_contrast_unchanged': True},
        'goal_complete': False,
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['summary']))
    print(json.dumps({'unloading_right_arm': per_task['unloading_the_car']['right_arm']}))


if __name__ == '__main__':
    main()

"""Exploratory position-conditioned extension with whole-task holdout."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import action_change_calibration as prior


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    data = {}
    for name, source in plan['sources'].items():
        path = Path(source['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        data[name] = json.loads(path.read_text())
    baseline = data['baseline1688977']
    holds = {}
    for row in baseline['sample_keys']:
        raw = prior._raw_arrays(row, baseline['horizon'], 'baseline')
        selected = raw['valid_mask'] & (np.arange(baseline['horizon']) < prior.PREFIX_STEPS)
        holds[row['episode_index'], row['frame_index']] = raw['hold_action'][selected][:, prior.DIMENSIONS]
    results = []
    for name in ('mixed1000', 'mixed5000', 'expert1000'):
        pairing, samples = prior.paired_samples(baseline, data[name])
        tasks = sorted({s.task for s in samples})
        for task in tasks:
            train = [s for s in samples if s.task != task]
            test = [s for s in samples if s.task == task]
            prior_models, _ = prior.fit_models(train)
            x_train, y_train, w_train = prior._stack(train)
            h_train = np.concatenate([holds[s.episode_index, s.frame_index] for s in train])
            x_test, y_test, w_test = prior._stack(test)
            h_test = np.concatenate([holds[s.episode_index, s.frame_index] for s in test])
            candidates = {'zero': np.zeros_like(y_test),
                          'constant': np.broadcast_to(prior_models['constant_bias'][1], y_test.shape),
                          'prior_affine': x_test*prior_models['affine'][0]+prior_models['affine'][1],
                          'position_affine': np.zeros_like(y_test)}
            fit_details = []
            for j, dimension in enumerate(prior.DIMENSIONS):
                features = np.stack([x_train[:, j], h_train[:, j]], axis=-1)
                means = (w_train[:, None] * features).sum(axis=0)
                scales = np.sqrt((w_train[:, None]*(features-means)**2).sum(axis=0))
                scales[scales < 1e-12] = 1
                design = np.column_stack([(features-means)/scales, np.ones(len(features))])
                beta, _, rank, singular = np.linalg.lstsq(design*np.sqrt(w_train[:, None]),
                                                         y_train[:, j]*np.sqrt(w_train), rcond=1e-10)
                held_features = np.stack([x_test[:, j], h_test[:, j]], axis=-1)
                held_design = np.column_stack([(held_features-means)/scales, np.ones(len(held_features))])
                candidates['position_affine'][:, j] = held_design@beta
                fit_details.append({'coordinate': dimension, 'rank': int(rank),
                                    'scaled_coefficients': beta.tolist(),
                                    'singular_values': singular.tolist()})
            denom = (w_test[:, None]*y_test**2).sum(axis=0)
            residuals = {m: (w_test[:, None]*(pred-y_test)**2).sum(axis=0)
                         for m,pred in candidates.items()}
            for group, coords in prior.GROUPS.items():
                indices = [prior.DIMENSIONS.index(i) for i in coords]
                valid = [j for j in indices if denom[j] > prior.MIN_DELTA_RMS**2]
                explained = {m: float(np.mean([1-res[j]/denom[j] for j in valid]))
                             if valid else None for m,res in residuals.items()}
                all_finite = [v for v in explained.values() if v is not None]
                assert all(np.isfinite(all_finite))
                results.append({'checkpoint': name, 'held_out_task': task, 'group': group,
                                'coordinates_with_signal': len(valid),
                                'explained_fraction_macro': explained,
                                'position_beats_zero_and_prior': bool(valid and
                                    explained['position_affine'] > max(0, explained['prior_affine'])),
                                'fit_details': [fit_details[j] for j in indices]})
    summaries = {}
    for name in ('mixed1000','mixed5000','expert1000'):
        rows = [x for x in results if x['checkpoint'] == name]
        summaries[name] = {'groups': len(rows),
                           'position_beats_zero_and_prior': sum(x['position_beats_zero_and_prior'] for x in rows),
                           'position_positive_explained_groups': sum(
                               x['explained_fraction_macro']['position_affine'] is not None and
                               x['explained_fraction_macro']['position_affine'] > 0 for x in rows)}
    report = {'kind':'exploratory_whole_task_holdout_position_affine',
              'plan_sha256':hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest(),
              'strict_existing_raw_pairing':True,'summary':summaries,'results':results,
              'limits':plan['limits']}
    with (HERE/'result.json').open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()

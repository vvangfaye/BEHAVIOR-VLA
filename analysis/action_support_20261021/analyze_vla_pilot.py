"""Recompute fixed held-out VLA scores and paired episode uncertainty on CPU."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE / 'local_vla_pilot_ada01'
METHODS = ('standard', 'conditioned', 'projected')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def score(samples, target):
    samples, target = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64)
    assert samples.ndim == target.ndim + 1 and samples.shape[0] == target.shape[0]
    assert samples.shape[2:] == target.shape[1:] and samples.shape[-1] == 23
    assert np.isfinite(samples).all() and np.isfinite(target).all()
    flat, truth = samples.reshape(len(samples), samples.shape[1], -1), target.reshape(len(target), -1)
    draws = flat.shape[1]
    assert draws > 1
    first = np.linalg.norm(flat - truth[:, None], axis=-1).mean(axis=1)
    second = sum(np.linalg.norm(flat[:, i] - flat[:, j], axis=-1)
                 for i in range(draws) for j in range(draws) if i != j) / (draws * (draws - 1))
    es = (first - .5 * second) / np.sqrt(truth.shape[-1])
    mse = np.square(samples.mean(axis=1) - target).reshape(len(target), -1).mean(axis=1)
    return {'energy_score': es, 'mean_action_mse': mse}


def compare(left, right, episode):
    """Left-minus-right, with complete episodes as resampling units."""
    delta = left - right
    ids = np.unique(episode)
    means = np.array([delta[episode == i].mean() for i in ids])
    counts = np.array([np.sum(episode == i) for i in ids])
    rng = np.random.default_rng(673)
    selection = rng.integers(0, len(ids), size=(10000, len(ids)))
    episode_draws = means[selection].mean(axis=1)
    # Cluster bootstrap retaining each selected episode's original anchors.
    anchor_draws = (means[selection] * counts[selection]).sum(axis=1) / counts[selection].sum(axis=1)
    return {'anchor_mean_delta': float(delta.mean()),
            'anchor_weighted_cluster_bootstrap95': np.quantile(anchor_draws, [.025, .975]).tolist(),
            'episode_mean_delta': float(means.mean()),
            'episode_bootstrap95': np.quantile(episode_draws, [.025, .975]).tolist(),
            'episodes_better': int(np.sum(means < 0)), 'episodes_tied': int(np.sum(means == 0)),
            'episodes_worse': int(np.sum(means > 0)), 'episodes': len(ids),
            'scope': 'Same fixed episodes and one training seed; no task or training-seed uncertainty'}


def main():
    missing = [m for m in METHODS if not (TRAIN / m / 'completion.json').exists()]
    if missing:
        print(json.dumps({'ready': False, 'waiting_for_completed_methods': missing}))
        return
    plan_hash = sha(TRAIN / 'plan.json')
    arrays, metrics, sources, declared = {}, {}, {}, {}
    for method in METHODS:
        folder = TRAIN / method
        completion, run, result = [load(folder / name) for name in ('completion.json', 'run.json', 'result.json')]
        assert completion['accepted'] and completion['worker_returncode'] == 0
        assert completion['result_sha256'] == sha(folder / 'result.json')
        assert run['plan_sha256'] == plan_hash and run['method'] == method and run['seed'] == 17
        assert result['complete'] and result['completed_steps'] == 1000
        with np.load(folder / 'test.npz', allow_pickle=False) as z:
            arrays[method] = {key: z[key].copy() for key in ('samples', 'target', 'index', 'episode')}
        metrics[method] = score(arrays[method]['samples'], arrays[method]['target'])
        declared[method] = load(folder / 'test.json')['means']
        for metric, values in metrics[method].items():
            assert abs(float(values.mean()) - declared[method][metric]) < 2e-5, (method, metric)
        sources[method] = {'test_npz_sha256': sha(folder / 'test.npz'), 'result_sha256': sha(folder / 'result.json')}
    reference = arrays['standard']
    for method in METHODS[1:]:
        for key in ('index', 'episode', 'target'):
            np.testing.assert_array_equal(arrays[method][key], reference[key])
    comparisons = {}
    for baseline in ('standard', 'conditioned'):
        comparisons['projected_minus_' + baseline] = {
            key: compare(metrics['projected'][key], metrics[baseline][key], reference['episode'])
            for key in metrics['projected']}
    posthoc = HERE / 'vla_posthoc_result'
    selection = HERE / 'posthoc_result_source.json'
    if selection.exists():
        selected = load(selection)
        posthoc = HERE / selected['directory']
        assert posthoc.resolve().parent == HERE.resolve()
        assert sha(posthoc / 'result.json') == selected['result_sha256']
    if (posthoc / 'result.json').exists() and load(posthoc / 'result.json').get('complete'):
        with np.load(posthoc / 'samples.npz', allow_pickle=False) as z:
            for key in ('index', 'episode', 'target'):
                np.testing.assert_array_equal(z[key], reference[key])
            metrics['conditioned_posthoc'] = score(z['posthoc'], z['target'])
            replayed = score(z['conditioned'], z['target'])
        comparisons['posthoc_minus_replayed_conditioned'] = {
            key: compare(metrics['conditioned_posthoc'][key], replayed[key], reference['episode'])
            for key in replayed}
        comparisons['projected_minus_conditioned_posthoc'] = {
            key: compare(metrics['projected'][key], metrics['conditioned_posthoc'][key], reference['episode'])
            for key in metrics['projected']}
        sources['conditioned_posthoc'] = {'samples_sha256': sha(posthoc / 'samples.npz'),
                                         'report_sha256': sha(posthoc / 'result.json')}
    report = {'as_of_utc': datetime.now(timezone.utc).isoformat(), 'ready': True,
              'training_plan_sha256': plan_hash, 'seed': 17, 'steps': 1000,
              'anchors': len(reference['index']), 'episodes': len(np.unique(reference['episode'])),
              'means': {method: {key: float(value.mean()) for key, value in items.items()}
                        for method, items in metrics.items()},
              'comparisons': comparisons, 'source_hashes': sources,
              'posthoc_complete': 'conditioned_posthoc' in metrics,
              'limits': ['Existing expert subtask supplied during offline scoring',
                         'Actions use checkpoint normalization; these scores are not raw physical-unit errors',
                         'Single training seed and three tasks; resampling does not establish broader generalization',
                         'No task-success or closed-loop improvement follows from these scores']}
    (HERE / 'vla_pilot_reanalysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('ready', 'anchors', 'episodes', 'means', 'posthoc_complete')}, indent=2))


if __name__ == '__main__':
    main()

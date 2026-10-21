"""Paired spectral / output-only controls, with no target masks at inference."""
import json
from pathlib import Path

import numpy as np
import torch

from analyze_generative import energy
from train_dct_control import prepare, ranks, basis, METHODS
from train_generative import Flow

HERE = Path(__file__).resolve().parent


def predicted_masks(model, x, test_ids):
    # Replay exactly the RNG consumption in train_generative.py's evaluation.
    torch.manual_seed(761)
    masks = []
    with torch.inference_mode():
        for ids in test_ids.split(64):
            h = model.context(x[ids])
            probs = model.mask(h).sigmoid().repeat_interleave(8, 0)
            masks.append(torch.bernoulli(probs).reshape(len(ids), 8, 23))
            torch.randn(len(ids) * 8, 32, 23)
    return torch.cat(masks).numpy()


def main():
    torch.set_num_threads(4)
    rows, x, _, active, train_ids, test_ids = prepare()
    rank, _ = ranks(active, train_ids)
    b = basis().numpy().astype(np.float64)
    keep = np.arange(32)[:, None] < rank[None]
    result = {}
    for seed in [17, 29, 43]:
        folder = HERE / ('generative_runs' if seed == 17 else f'generative_runs_seed{seed}')
        support = np.load(folder / 'support_flow.npz')
        target = support['target']
        ep = support['episode']
        expected = np.array([rows[i]['episode'] for i in test_ids])
        np.testing.assert_array_equal(ep, expected)
        scores = {}
        samples = {'support_flow': support['samples']}
        support_masks_verified = False
        for method in ['full_flow', 'mask_conditioned_flow', 'support_flow']:
            with np.load(folder / f'{method}.npz') as f:
                np.testing.assert_array_equal(f['episode'], ep)
                np.testing.assert_array_equal(f['target'], target)
                a = f['samples'].reshape(-1, 8, 32, 23)
            model = Flow(x.shape[-1])
            model.load_state_dict(torch.load(folder / f'{method}.pt', weights_only=True))
            model.eval()
            masks = predicted_masks(model, x, test_ids)
            if method == 'support_flow':
                residual = np.ptp(a, axis=2)[masks == 0]
                assert residual.max(initial=0) < 1e-5
                support_masks_verified = True
                continue
            samples[method] = a.reshape(-1, 8, 736)
            mean = a.mean(2, keepdims=True)
            samples[method + '_posthoc_mask'] = (mean + masks[:, :, None, :] * (a - mean)).reshape(-1, 8, 736)
            coef = np.einsum('kt,nstd->nskd', b, a)
            samples[method + '_posthoc_dct'] = np.einsum('kt,nskd->nstd', b, coef * keep).reshape(-1, 8, 736)
        available = []
        dct_dir = HERE / f'dct_controls_seed{seed}'
        for method in METHODS:
            path = dct_dir / f'{method}.npz'
            if not path.exists():
                continue
            with np.load(path) as f:
                np.testing.assert_array_equal(f['episode'], ep)
                np.testing.assert_array_equal(f['target'], target)
                samples[method] = f['samples']
            available.append(method)
        groups = {}
        for group, coords in [('all', list(range(23))),
                              ('body_without_constant_and_grippers', [j for j in range(23) if j not in [6, 14, 22]])]:
            scores = {m: energy(v, target, coords) for m, v in samples.items()}
            episodes = np.unique(ep)
            comparisons = {}
            for method in samples:
                if method == 'support_flow':
                    continue
                delta = np.array([(scores['support_flow'] - scores[method])[ep == e].mean() for e in episodes])
                rng = np.random.default_rng(20261021)
                boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
                comparisons[method] = {'support_minus_control_episode_mean': float(delta.mean()),
                                       'episode_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
                                       'episodes_support_better': int((delta < 0).sum())}
            groups[group] = {'energy_score': {m: float(v.mean()) for m, v in scores.items()},
                             'mean_action_mse': {m: float(np.square(v.mean(1) - target).mean()) for m, v in samples.items()} if group == 'all' else None,
                             'paired_comparisons': comparisons}
        result[str(seed)] = {'groups': groups, 'dct_training_available': available,
                              'support_rng_replay_verified': support_masks_verified}
    output = {'scope': 'Same 14 held-out episodes; no target activity masks used in any inference control; no VLA/closed-loop claim',
              'posthoc_definition': 'Apply original model predicted masks or train-only fixed DCT lowpass to its saved output, without retraining',
              'seeds': result}
    (HERE / 'control_comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({s: r['groups']['all']['energy_score'] for s, r in result.items()}, indent=2))


if __name__ == '__main__':
    main()

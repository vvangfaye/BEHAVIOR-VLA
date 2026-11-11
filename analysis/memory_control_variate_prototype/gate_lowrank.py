"""Low-rank gate-writer case for the event-baseline residual estimator."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from prototype import estimate, summarize
from residual_prototype import aligned_baseline, residual_estimate


def problem(seed, length, eps, dim=16, features=64, events=4):
    rng = np.random.default_rng(seed)
    p = dict(weight=rng.normal(0, .15, (1, features)),
             recurrent=rng.normal(0, .3 / np.sqrt(dim), (dim, dim)),
             basis=rng.normal(0, .5, (dim, 1)),
             x=rng.normal(0, .2 / np.sqrt(features), (length, features)),
             events=np.unique(np.linspace(0, length - 1, events, dtype=int)),
             alpha=np.full(length, eps),
             readout=rng.normal(0, 1 / np.sqrt(dim), dim))
    p['x'][p['events']] = rng.normal(0, 1 / np.sqrt(features), (events, features))
    p['x'][0, 0] = rng.choice([-1., 1.])
    p['alpha'][p['events']] = .2
    cue = rng.choice([-1., 1.])
    p['readout'] *= cue
    p['target'] = float(cue * p['x'][0, 0])
    return p


def trajectory(p):
    dim, nparam = len(p['basis']), p['weight'].size
    h, s = np.zeros(dim), np.zeros((dim, nparam))
    js, bs = [], []
    for x, a in zip(p['x'], p['alpha']):
        value = np.tanh(p['recurrent'] @ h + p['basis'] @ (p['weight'] @ x))
        deriv = a * (1 - value * value)
        j = np.eye(dim) + deriv[:, None] * p['recurrent']
        # B = dF/dg * dg/dtheta: rank at most one, P=64 (not P=1).
        b = (deriv[:, None] * p['basis']) @ x[None]
        h = h + a * value
        s = j @ s + b
        js.append(j); bs.append(b)
    err = p['readout'] @ h - p['target']
    q = err * p['readout']
    return dict(j=np.asarray(js), b=np.asarray(bs), adjoint=q, exact=q @ s,
                sensitivity=s, loss=float(.5 * err ** 2))


def verify(p, full):
    w = torch.tensor(p['weight'], dtype=torch.float64, requires_grad=True)
    r = torch.tensor(p['recurrent'], dtype=torch.float64)
    v = torch.tensor(p['basis'], dtype=torch.float64)
    h = torch.zeros(len(v), dtype=torch.float64)
    for x, a in zip(p['x'], p['alpha']):
        h = h + float(a) * torch.tanh(r @ h + v @ (w @ torch.tensor(x, dtype=torch.float64)))
    loss = .5 * (torch.tensor(p['readout'], dtype=torch.float64) @ h - p['target']) ** 2
    loss.backward()
    err = float(np.max(np.abs(w.grad.numpy().ravel() - full['exact'])))
    assert err < 1e-10
    return err


def main():
    start = time.time()
    rows = []
    for length in [256, 1024]:
        for eps in [0., .002, .01]:
            for seed in range(3):
                p = problem(seed, length, eps)
                f = trajectory(p)
                baseline = aligned_baseline(f, p['events'])
                samples = 512
                u, _ = estimate(f, baseline, samples, 2000 + seed, coupled=False, block_size=4)
                res, exact_base, rank = residual_estimate(f, p['events'], samples, 3000 + seed)
                error = float(np.max(np.abs(exact_base - baseline['exact'])))
                assert error < 1e-10
                metrics = dict(uoro_T=summarize(u, f['exact'], 4),
                               uoro_T_matched_factor_storage=summarize(u, f['exact'], 8),
                               residual_uoro=summarize(res, f['exact'], 4))
                full_rank = int(np.linalg.matrix_rank(f['sensitivity'], tol=1e-10))
                baseline_rank = int(np.linalg.matrix_rank(baseline['sensitivity'], tol=1e-10))
                rows.append(dict(length=length, gap_strength=eps, seed=seed,
                                 event_indices=list(map(int, p['events'])),
                                 parameter_count=64, state_dimension=16, write_output_rank=1,
                                 full_sensitivity_rank=full_rank, baseline_sensitivity_rank=baseline_rank,
                                 max_factor_rank=rank, torch_bptt_max_error=verify(p, f),
                                 factored_baseline_max_error=error, metrics=metrics))
    out = dict(description='Low-rank gate network writer. Scalar output with 64 independently trained weights.',
               caution='Synthetic CPU gradient test, no robotics claims, no matched compute or matched bytes claim.',
               samples_per_setting=512, probe_count=4, seeds=3,
               wall_seconds=time.time() - start, rows=rows)
    path = Path(__file__).with_name('gate_lowrank_results.json')
    path.write_text(json.dumps(out, indent=2) + '\n')
    for length in [256, 1024]:
        for eps in [0., .002, .01]:
            r = [x for x in rows if x['length'] == length and x['gap_strength'] == eps]
            ratio = sum(x['metrics']['residual_uoro']['mse'] for x in r) / sum(x['metrics']['uoro_T']['mse'] for x in r)
            print(f'T={length}, epsilon={eps}: residual/UORO-4 MSE={ratio:.6g}, '
                  f'K-ranks={[x["baseline_sensitivity_rank"] for x in r]}, '
                  f'full-ranks={[x["full_sensitivity_rank"] for x in r]}')
    print(f'Results: {path}; CPU wall seconds: {out["wall_seconds"]:.2f}')


if __name__ == '__main__':
    main()

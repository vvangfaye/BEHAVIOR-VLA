"""Forward-aligned event baseline plus factored residual UORO.

The exact baseline K propagates only selected full-trajectory Jacobians.
Its derivative factors are updated only at selected events. The residual
R=S-K obeys R'=A R + (A-As)K + (B-Bs); only R is randomly compressed.
No perturbed physical world, trained teacher, or alternate forward state is used.
The tiny CPU reference materializes local A/B for verification; the factored
implementation of the residual injection does not materialize K or E.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from prototype import make_problem, trajectory, check_torch, estimate, summarize, factors


def aligned_baseline(full, event_indices):
    keep = set(map(int, event_indices))
    dim, nparam = full['b'].shape[1:]
    js, bs = [], []
    k = np.zeros((dim, nparam))
    for t, (a, b) in enumerate(zip(full['j'], full['b'])):
        aa = a if t in keep else np.eye(dim)
        bb = b if t in keep else np.zeros_like(b)
        k = aa @ k + bb
        js.append(aa); bs.append(bb)
    return dict(j=np.asarray(js), b=np.asarray(bs), adjoint=full['adjoint'],
                exact=full['adjoint'] @ k, sensitivity=k)


def advance_factors(u, v, a, b):
    """Exact K=U V^T factor update, compressing only on event boundaries."""
    u = a @ u
    bu, singular, bvh = np.linalg.svd(b, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-13))
    if rank:
        u = np.concatenate([u, bu[:, :rank] * singular[:rank]], axis=1)
        v = np.concatenate([v, bvh[:rank].T], axis=1)
    if u.shape[1]:
        q, r = np.linalg.qr(u, mode='reduced')
        u, v = q, v @ r.T
    return u, v


def residual_estimate(full, event_indices, samples, seed):
    keep = set(map(int, event_indices))
    dim, nparam = full['b'].shape[1:]
    rng = np.random.default_rng(seed)
    # K=left @ right.T is exact; only the residual has a rank-one sketch.
    left = np.zeros((dim, 0)); right = np.zeros((nparam, 0))
    u = np.zeros((samples, dim)); v = np.zeros((samples, nparam))
    max_rank = 0
    for t, (a, b) in enumerate(zip(full['j'], full['b'])):
        is_event = t in keep
        aa = a if is_event else np.eye(dim)
        bb = b if is_event else np.zeros_like(b)
        noise = rng.choice([-1., 1.], (samples, dim))
        # E^T nu = V U^T (A-As)^T nu + (B-Bs)^T nu.
        injection = ((noise @ (a - aa)) @ left) @ right.T + noise @ (b - bb)
        transported = u @ a.T
        rho0, rho1 = factors([transported], [v], [injection], noise)
        u = rho0 * transported + rho1 * noise
        v = v / rho0 + np.divide(injection, rho1, out=np.zeros_like(injection), where=rho1 != 0)
        if is_event:
            left, right = advance_factors(left, right, aa, bb)
            max_rank = max(max_rank, left.shape[1])
    exact_baseline = (full['adjoint'] @ left) @ right.T
    gradient = (u @ full['adjoint'])[:, None] * v + exact_baseline
    return gradient, exact_baseline, max_rank


def run(args):
    started = time.time()
    rows = []
    for length in args.lengths:
        for gap in args.gaps:
            for seed in range(args.seeds):
                problem = make_problem(length, gap, seed)
                full = trajectory(problem, False)
                keep = problem['events']
                if args.omit_event:
                    # Deliberately miss the first, task-relevant historical write.
                    keep = keep[1:]
                baseline = aligned_baseline(full, keep)
                bptt_error = check_torch(problem, full['exact'])
                paired_f, paired_s = estimate(full, baseline, args.samples, 10000 + seed, coupled=True)
                uoro_1, _ = estimate(full, baseline, args.samples, 20000 + seed, coupled=False)
                uoro_t, _ = estimate(full, baseline, args.samples, 30000 + seed,
                                     block_size=len(keep), coupled=False)
                residual, exact_factored, max_rank = residual_estimate(full, keep, args.samples, 40000 + seed)
                factored_error = float(np.max(np.abs(exact_factored - baseline['exact'])))
                assert factored_error < 1e-10, factored_error
                methods = dict(uoro_1=uoro_1, uoro_T=uoro_t,
                               aligned_paired_cv=paired_f - paired_s + baseline['exact'],
                               residual_uoro=residual)
                metrics = {m: summarize(g, full['exact'], args.probes) for m, g in methods.items()}
                rows.append(dict(length=length, gap_strength=gap, seed=seed,
                                 selected_event_indices=list(map(int, keep)),
                                 omitted_relevant_event=args.omit_event,
                                 exact_gradient_difference=float(np.linalg.norm(full['exact'] - baseline['exact'])),
                                 exact_gradient_norm=float(np.linalg.norm(full['exact'])),
                                 exact_full_loss=full['loss'], torch_bptt_max_error=bptt_error,
                                 factored_baseline_max_error=factored_error, max_baseline_factor_rank=max_rank,
                                 metrics=metrics))
    output = dict(description='Synthetic CPU diagnostic for forward-aligned event-baseline residual RTRL.',
                  caution='Not a robotics result. Fixed parameters, detached deterministic event selection. Equal probe counts; computational costs are not equal.',
                  parameter_count=12, state_dimension=4, probe_count=args.probes,
                  samples_per_setting=args.samples, seeds=args.seeds,
                  omitted_relevant_event=args.omit_event, wall_seconds=time.time() - started, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for length in args.lengths:
        for gap in args.gaps:
            group = [r for r in rows if r['length'] == length and r['gap_strength'] == gap]
            means = {m: np.mean([r['metrics'][m]['mse'] for r in group]) for m in methods}
            ratio = means['residual_uoro'] / max(means['uoro_T'], 1e-30)
            print(f'T={length:4d} eps={gap:7.4f} residual/UORO-T MSE={ratio:9.4g} '
                  + ' '.join(f'{m}={means[m]:.3g}' for m in methods))
    print(f'Results: {args.output}; CPU wall seconds: {output["wall_seconds"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', nargs='+', type=int, default=[64, 256, 1024])
    parser.add_argument('--gaps', nargs='+', type=float, default=[0., .0005, .002, .01])
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--probes', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--omit-event', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('residual_results.json'))
    run(parser.parse_args())

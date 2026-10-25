"""CPU-only diagnostic for event-skeleton control variates for UORO.

This is synthetic gradient-estimator validation, not a robotics result. All
parameters are held fixed while a full/skeleton pair is evaluated. The skeleton
is a detached, noncontiguous subset of the real sequence. Missing transitions
are identities. The two UORO estimators use common Rademacher draws and common
even-in-current-noise scaling; both marginals remain unbiased.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def make_problem(length: int, gap_strength: float, seed: int, dim: int = 4):
    rng = np.random.default_rng(seed)
    nin = 3
    weight = rng.normal(0, 0.25, (dim, nin))
    recurrent = rng.normal(0, 0.12, (dim, dim))
    events = np.unique(np.linspace(0, length - 1, 8, dtype=int))
    x = rng.normal(0, 0.2, (length, nin))
    x[events] = rng.choice([-1., 1.], (len(events), nin))
    alpha = np.full(length, gap_strength)
    alpha[events] = 0.2
    cue = rng.choice([-1., 1.])
    readout = cue * rng.normal(0, 0.5, dim)
    target = cue * float(x[events[0], 0] + 0.4 * x[events[3], 1])
    return dict(weight=weight, recurrent=recurrent, x=x, alpha=alpha,
                events=events, readout=readout, target=target)


def trajectory(problem, skeleton: bool):
    weight, recurrent, x = (problem[k] for k in ('weight', 'recurrent', 'x'))
    dim, nin = weight.shape
    state = np.zeros(dim)
    sensitivity = np.zeros((dim, dim * nin))
    keep = set(map(int, problem['events']))
    js, bs, states = [], [], [state.copy()]
    for t, xt in enumerate(x):
        alpha = problem['alpha'][t] if not skeleton or t in keep else 0.
        value = np.tanh(recurrent @ state + weight @ xt)
        diag = alpha * (1. - value * value)
        jac = np.eye(dim) + diag[:, None] * recurrent
        direct = np.zeros((dim, dim * nin))
        for row in range(dim):
            direct[row, row * nin:(row + 1) * nin] = diag[row] * xt
        state = state + alpha * value
        sensitivity = jac @ sensitivity + direct
        js.append(jac)
        bs.append(direct)
        states.append(state.copy())
    error = float(problem['readout'] @ state - problem['target'])
    adjoint = error * problem['readout']
    exact = adjoint @ sensitivity
    return dict(j=np.asarray(js), b=np.asarray(bs), state=np.asarray(states),
                adjoint=adjoint, exact=exact, loss=0.5 * error * error)


def blocks(traj, size):
    j, b = traj['j'], traj['b']
    dim, nparam = b.shape[1:]
    out_j, out_b = [], []
    for start in range(0, len(j), size):
        aj, bb = np.eye(dim), np.zeros((dim, nparam))
        for jt, bt in zip(j[start:start + size], b[start:start + size]):
            aj, bb = jt @ aj, jt @ bb + bt
        out_j.append(aj)
        out_b.append(bb)
    return np.asarray(out_j), np.asarray(out_b)


def norms(a):
    return np.sqrt(np.sum(a * a, axis=-1))


def factors(a_list, v_list, d_list, noise):
    # Shared positive scales, invariant to flipping all current noise signs.
    # The algebraic rank-one estimator is unbiased for any such scales.
    tiny = 1e-30
    na2 = sum(norms(a) ** 2 for a in a_list)
    nv2 = sum(norms(v) ** 2 for v in v_list)
    nd2 = sum(norms(d) ** 2 for d in d_list)
    rho0 = np.where((na2 > tiny) & (nv2 > tiny),
                    (np.maximum(nv2, tiny) / np.maximum(na2, tiny)) ** .25, 1.)
    rho1 = np.where(nd2 > tiny,
                    (nd2 / (len(d_list) * np.maximum(norms(noise) ** 2, tiny))) ** .25,
                    0.)
    return rho0[:, None], rho1[:, None]


def estimate(full, short, samples, seed, block_size=1, coupled=True):
    jf, bf = blocks(full, block_size)
    js, bs = blocks(short, block_size)
    dim, nparam = bf.shape[1:]
    rng = np.random.default_rng(seed)
    uf = np.zeros((samples, dim)); vf = np.zeros((samples, nparam))
    us = np.zeros_like(uf); vs = np.zeros_like(vf)
    for jt, bt, ks, ds in zip(jf, bf, js, bs):
        noise = rng.choice([-1., 1.], (samples, dim))
        noise_s = noise if coupled else rng.choice([-1., 1.], (samples, dim))
        af, ass = uf @ jt.T, us @ ks.T
        df, dd = noise @ bt, noise_s @ ds
        if coupled:
            r0, r1 = factors([af, ass], [vf, vs], [df, dd], noise)
            s0, s1 = r0, r1
        else:
            r0, r1 = factors([af], [vf], [df], noise)
            s0, s1 = factors([ass], [vs], [dd], noise_s)
        uf = r0 * af + r1 * noise
        us = s0 * ass + s1 * noise_s
        vf = vf / r0 + np.divide(df, r1, out=np.zeros_like(df), where=r1 != 0)
        vs = vs / s0 + np.divide(dd, s1, out=np.zeros_like(dd), where=s1 != 0)
    gf = (uf @ full['adjoint'])[:, None] * vf
    gs = (us @ short['adjoint'])[:, None] * vs
    return gf, gs


def summarize(est, exact, probes):
    # Independent sets of probes are averaged before evaluating diagnostics.
    count = (len(est) // probes) * probes
    grouped = est[:count].reshape(-1, probes, est.shape[-1]).mean(axis=1)
    error = grouped - exact
    mean_error = error.mean(axis=0)
    norm2 = float(exact @ exact)
    return dict(mse=float(np.mean(np.sum(error * error, axis=1))),
                normalized_mse=float(np.mean(np.sum(error * error, axis=1)) / max(norm2, 1e-30)),
                bias_norm=float(np.linalg.norm(mean_error)),
                mean_gradient_norm=float(np.linalg.norm(grouped.mean(axis=0))),
                exact_gradient_norm=float(np.sqrt(norm2)),
                standard_error_norm=float(np.sqrt(np.var(grouped, axis=0, ddof=1).sum() / len(grouped))))


def check_torch(problem, exact):
    import torch
    dtype = torch.float64
    w = torch.tensor(problem['weight'], dtype=dtype, requires_grad=True)
    r = torch.tensor(problem['recurrent'], dtype=dtype)
    h = torch.zeros(len(r), dtype=dtype)
    for xt, a in zip(problem['x'], problem['alpha']):
        h = h + float(a) * torch.tanh(r @ h + w @ torch.tensor(xt, dtype=dtype))
    loss = .5 * (torch.tensor(problem['readout'], dtype=dtype) @ h - problem['target']) ** 2
    loss.backward()
    err = float(np.max(np.abs(w.grad.detach().numpy().ravel() - exact)))
    assert err < 1e-10, err
    return err


def run(args):
    started = time.time()
    rows = []
    for length in args.lengths:
        for gap in args.gaps:
            for seed in range(args.seeds):
                problem = make_problem(length, gap, seed)
                full, short = trajectory(problem, False), trajectory(problem, True)
                autograd_error = check_torch(problem, full['exact'])
                cf, cs = estimate(full, short, args.samples, 10000 + seed, coupled=True)
                independent_f, independent_s = estimate(full, short, args.samples, 20000 + seed, coupled=False)
                # memory-T UORO baseline: the same number of stored steps as the
                # noncontiguous exact skeleton, with the same number of probes.
                tf, _ = estimate(full, short, args.samples, 30000 + seed,
                                 block_size=len(problem['events']), coupled=False)
                methods = {
                    'uoro_1': independent_f,
                    'uoro_T': tf,
                    'paired_cv': cf - cs + short['exact'],
                    'independent_cv': independent_f - independent_s + short['exact'],
                }
                stats = {k: summarize(v, full['exact'], args.probes) for k, v in methods.items()}
                rows.append(dict(length=length, gap_strength=gap, seed=seed,
                                 event_indices=list(map(int, problem['events'])),
                                 exact_full_loss=full['loss'], exact_skeleton_loss=short['loss'],
                                 exact_gradient_difference=float(np.linalg.norm(full['exact'] - short['exact'])),
                                 state_difference=float(np.linalg.norm(full['state'][-1] - short['state'][-1])),
                                 torch_bptt_max_error=autograd_error, metrics=stats))
    output = dict(description='Synthetic CPU estimator diagnostic; no robotics training or cluster use.',
                  parameter_count=12, state_dimension=4, probe_count=args.probes,
                  samples_per_setting=args.samples, seeds=args.seeds,
                  wall_seconds=time.time() - started, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for length in args.lengths:
        for gap in args.gaps:
            group = [r for r in rows if r['length'] == length and r['gap_strength'] == gap]
            means = {m: np.mean([r['metrics'][m]['mse'] for r in group]) for m in methods}
            ratio = means['paired_cv'] / max(means['uoro_T'], 1e-30)
            print(f'T={length:4d} eps={gap:7.4f} CV/UORO-T MSE={ratio:9.4g} '
                  + ' '.join(f'{m}={means[m]:.3g}' for m in methods))
    print(f'Results: {args.output}; CPU wall seconds: {output["wall_seconds"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', nargs='+', type=int, default=[64, 256, 1024])
    parser.add_argument('--gaps', nargs='+', type=float, default=[0., .0005, .002, .01])
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--probes', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    run(parser.parse_args())

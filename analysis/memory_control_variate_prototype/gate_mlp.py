"""P >> d, rank-one MLP gate: a small persistent-state budget comparison.

This deliberately remains a gradient-estimator diagnostic, not a training
benchmark. UORO with 8 probes matches the (4 exact + 4 stochastic) vector-factor
storage of the proposed estimator, ignoring local working activations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from prototype import estimate, summarize
from residual_prototype import aligned_baseline, residual_estimate

D, Q, H = 32, 32, 32
P = H * Q + H + H + 1


def unpack(theta):
    return theta[:H*Q].reshape(H, Q), theta[H*Q:H*Q+H], theta[H*Q+H:H*Q+2*H], theta[-1]


def gate(theta, x):
    w, b, v, c = unpack(theta)
    z = np.tanh(w @ x + b)
    back = v * (1 - z*z)
    return float(v @ z + c), np.concatenate([(back[:, None] * x).ravel(), back, z, np.ones(1)])


def make(seed, length, eps):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, .1, P)
    events = np.unique(np.linspace(0, length - 1, 4, dtype=int))
    x = rng.normal(0, .2/np.sqrt(Q), (length, Q))
    x[events] = rng.normal(0, 1/np.sqrt(Q), (4, Q))
    x[0, 0] = rng.choice([-1., 1.])
    alpha = np.full(length, eps); alpha[events] = .2
    cue = rng.choice([-1., 1.])
    return dict(theta=theta, recurrent=rng.normal(0, .3/np.sqrt(D), (D, D)),
                basis=rng.normal(0, .5, D), readout=cue*rng.normal(0, 1/np.sqrt(D), D),
                x=x, alpha=alpha, events=events, target=float(cue*x[0, 0]))


def trajectory(p):
    h = np.zeros(D); sensitivity = np.zeros((D, P)); js=[]; bs=[]
    for x, alpha in zip(p['x'], p['alpha']):
        g, grad = gate(p['theta'], x)
        z = np.tanh(p['recurrent'] @ h + p['basis']*g)
        deriv = alpha*(1-z*z)
        a = np.eye(D) + deriv[:, None]*p['recurrent']
        b = np.outer(deriv*p['basis'], grad)
        h += alpha*z
        sensitivity = a @ sensitivity + b
        js.append(a); bs.append(b)
    error = float(p['readout'] @ h - p['target'])
    adjoint = error*p['readout']
    return dict(j=np.asarray(js), b=np.asarray(bs), adjoint=adjoint,
                exact=adjoint @ sensitivity, sensitivity=sensitivity, loss=.5*error*error)


def verify(p, f):
    theta = torch.tensor(p['theta'], dtype=torch.float64, requires_grad=True)
    w, b, v, c = unpack(theta)
    recurrent=torch.tensor(p['recurrent'], dtype=torch.float64)
    basis=torch.tensor(p['basis'], dtype=torch.float64)
    h=torch.zeros(D, dtype=torch.float64)
    for x, alpha in zip(p['x'], p['alpha']):
        gate_value = v @ torch.tanh(w @ torch.tensor(x, dtype=torch.float64)+b)+c
        h=h+float(alpha)*torch.tanh(recurrent@h+basis*gate_value)
    loss=.5*(torch.tensor(p['readout'], dtype=torch.float64)@h-p['target'])**2
    loss.backward()
    error=float(np.max(np.abs(theta.grad.numpy()-f['exact'])))
    assert error<1e-10
    return error


def main():
    started=time.time(); rows=[]
    for eps in [.002, .01]:
        for seed in range(1):
            p=make(seed, 128, eps); f=trajectory(p)
            skeleton=aligned_baseline(f, p['events'])
            u, _=estimate(f, skeleton, 128, 5000+seed, block_size=4, coupled=False)
            res, base, rank=residual_estimate(f, p['events'], 128, 6000+seed)
            factor_error=float(np.max(np.abs(base-skeleton['exact'])))
            assert factor_error<1e-10
            metrics=dict(uoro_T_4=summarize(u, f['exact'], 4),
                         uoro_T_8_matched_factor_storage=summarize(u, f['exact'], 8),
                         residual_4=summarize(res, f['exact'], 4))
            rows.append(dict(length=128, gap_strength=eps, seed=seed,
                             parameter_count=P, state_dimension=D, gate_output_dimension=1,
                             skeleton_events=4, max_exact_factor_rank=rank,
                             exact_rtrl_persistent_scalars=D*P,
                             residual_persistent_scalars=(rank+4)*(D+P),
                             matched_uoro_persistent_scalars=8*(D+P),
                             torch_bptt_max_error=verify(p, f),
                             factor_max_error=factor_error, metrics=metrics))
    output=dict(description='32-state / 1089-parameter scalar MLP write gate, 4 separated events; one-seed small diagnostic.',
                caution='Synthetic gradient fidelity only. Persistent vector-factor storage matched; neither total memory nor FLOPs/walltime matched. OK/KF-RTRL not implemented.',
                wall_seconds=time.time()-started, rows=rows)
    path=Path(__file__).with_name('gate_mlp_results.json')
    path.write_text(json.dumps(output, indent=2)+'\n')
    for eps in [.002, .01]:
        rr=[x for x in rows if x['gap_strength']==eps]
        a=sum(x['metrics']['residual_4']['mse'] for x in rr)
        b=sum(x['metrics']['uoro_T_8_matched_factor_storage']['mse'] for x in rr)
        print(f'eps={eps}, residual4/UORO-T8 MSE={a/b:.6g}; ranks={[x["max_exact_factor_rank"] for x in rr]}')
    print(f'Results: {path}; CPU wall seconds={output["wall_seconds"]:.2f}')


if __name__=='__main__': main()

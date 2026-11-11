"""Matrix-free local IBP gradient prototype; synthetic CPU checks only.

Does not load a VLA, launch training, collect data, or establish novelty.
Uses implicit differentiation of a linear solve rather than differentiating
through CG. No full generator Jacobian or parameter Jacobian in the estimator.
"""
import json
import time
from pathlib import Path

import torch
from torch.func import jacrev, jvp, vjp

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def generator(params, z):
    w1, b1, w2, b2 = params
    return z[:len(b2)] + w2 @ torch.tanh(w1 @ z + b1) + b2


def parameters(n, m, h, seed, scale=.2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(h, n, generator=g)/n**.5,
            torch.randn(h, generator=g)*.1,
            torch.randn(m, h, generator=g)*scale/h**.5,
            torch.randn(m, generator=g)*.1)


def cg(operator, b, tol=1e-11, maxiter=200):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = r @ r
    norm = max(float(b.norm()), 1e-30)
    for i in range(maxiter):
        ap = operator(p)
        curvature = p @ ap
        if curvature <= 0:
            raise RuntimeError('CG operator lost positive definiteness')
        alpha = rr/curvature
        x = x+alpha*p
        r = r-alpha*ap
        if float(r.norm())/norm <= tol:
            break
        next_rr = r @ r
        p = r+(next_rr/rr)*p
        rr = next_rr
    residual = float((operator(x)-b).norm())/norm
    if residual > tol*10:
        raise RuntimeError(f'CG failed to converge: {residual}')
    return x.detach(), dict(iterations=i+1, relative_residual=residual)


def coefficients(ref_params, z, probe):
    """Return p, u, D_probe u using three solves with C=J J^T."""
    ref = lambda q: generator(ref_params, q)
    def j_product(q, direction):
        return jvp(ref, (q,), (direction,))[1]
    def c_product(q, a):
        jt_a = vjp(ref, q)[1](a)[0]
        return j_product(q, jt_a)
    operator = lambda a: c_product(z, a)
    p, rp = cg(operator, j_product(z, z))
    u, ru = cg(operator, j_product(z, probe))
    # C Du = D(J probe) - (DC)u. u is fixed in this directional derivative.
    rhs = jvp(lambda q: j_product(q, probe)-c_product(q, u),
              (z,), (probe,))[1]
    du, rd = cg(operator, rhs)
    return p.detach(), u.detach(), du.detach(), [rp, ru, rd]


def matrixfree_objective(params, ref_params, z, probe, advantage):
    p, u, du, receipts = coefficients(ref_params, z, probe)
    delta = generator(params, z)-generator(ref_params, z)
    delta_jvp = jvp(lambda q: generator(params, q)-generator(ref_params, q),
                    (z,), (probe,))[1]
    return advantage*((p-du)@delta-u@delta_jvp), receipts


def explicit_objective(params, ref_params, z, probe, advantage):
    ref = lambda q: generator(ref_params, q)
    j = jacrev(ref)(z)
    b = j.T @ torch.linalg.solve(j@j.T, torch.eye(j.shape[0]))
    h = b @ (generator(params, z)-ref(z))
    directional_div = torch.autograd.grad(probe@h, z, create_graph=True)[0]@probe
    return advantage*(z@h-directional_div)


def small_check(seed):
    n, m, h = 7, 4, 9
    torch.manual_seed(seed)
    ref_params = parameters(n, m, h, seed)
    params = tuple(x.clone().requires_grad_(True) for x in ref_params)
    z = torch.randn(n, requires_grad=True)
    probe = torch.randint(0, 2, (n,))*2.-1.
    loss, receipts = matrixfree_objective(params, ref_params, z, probe, .73)
    g1 = torch.autograd.grad(loss, params)
    oracle = explicit_objective(params, ref_params, z, probe, .73)
    g2 = torch.autograd.grad(oracle, params)
    flat1, flat2 = torch.cat([x.flatten() for x in g1]), torch.cat([x.flatten() for x in g2])
    return dict(seed=seed, max_parameter_gradient_error=float((flat1-flat2).abs().max()),
                relative_parameter_gradient_error=float((flat1-flat2).norm()/flat2.norm()),
                linear_solves=receipts)


def shape_check(scale):
    # Native-like dimensions, but a small residual MLP, NOT the native model.
    n, m, h = 1024, 368, 128
    torch.manual_seed(20)
    ref_params = parameters(n, m, h, 20, scale)
    params = tuple(x.clone().requires_grad_(True) for x in ref_params)
    z = torch.randn(n)
    probe = torch.randint(0, 2, (n,))*2.-1.
    start = time.perf_counter()
    loss, receipts = matrixfree_objective(params, ref_params, z, probe, .73)
    grad = torch.autograd.grad(loss, params)
    elapsed = time.perf_counter()-start
    assert all(torch.isfinite(x).all() for x in grad)
    return dict(noise_dimensions=n, output_dimensions=m,
                parameter_count=sum(x.numel() for x in params), residual_scale=scale,
                cpu_seconds=elapsed, linear_solves=receipts,
                caveat='Dimension check on a small residual MLP only; timing does not extrapolate to VLA.')


if __name__ == '__main__':
    result = dict(small_exact_comparison=[small_check(i) for i in range(4)],
                  larger_shape_check=[shape_check(s) for s in [.2, 1., 3.]],
                  scope='Synthetic CPU prototype only; no native VLA feasibility claim.')
    dest = Path(__file__).with_name('native_flow_matrixfree_check_20261020.json')
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

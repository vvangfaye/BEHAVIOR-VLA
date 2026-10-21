"""Reward-free least-squares fitting of action-conditional control coefficients.

Features must be functions only of the pre-query context and executed action.
The fitted coefficients must be independent of the gradient batch on which they
are subsequently used (a held-out batch or valid cross-fitting).
"""
import torch


def fit_linear(features, base_control_products, control_squared_norms, ridge=0.):
    """Minimize sum ||H_b + (x @ weights_b) C_b||^2 per parameter block.

    The inputs are detached Gram entries, avoiding differentiation through
    full VLA parameter gradients. Features are [samples, features]; Gram
    entries are [samples, blocks]. The H squared term is constant in weights.
    """
    x, hc, cc = [v.detach().to(torch.float64) for v in
                 (features, base_control_products, control_squared_norms)]
    if x.ndim != 2 or hc.ndim != 2 or cc.shape != hc.shape or x.shape[0] != hc.shape[0]:
        raise ValueError('Expected features [n,k] and Gram entries [n,b]')
    if x.shape[0] == 0 or ridge < 0 or not all(torch.isfinite(v).all() for v in (x, hc, cc)):
        raise ValueError('Finite nonempty inputs and nonnegative ridge required')
    if (cc < 0).any():
        raise ValueError('Squared norms must be nonnegative')
    gram = torch.einsum('ni,nb,nj->bij', x, cc, x)/x.shape[0]
    rhs = -torch.einsum('ni,nb->bi', x, hc)/x.shape[0]
    if ridge:
        gram = gram+ridge*torch.eye(x.shape[1], dtype=x.dtype)[None]
    # A zero control or redundant feature is valid; the least-norm solution
    # gives zero coefficients on unsupported directions.
    weights = torch.linalg.lstsq(gram, rhs[..., None]).solution[..., 0].T
    return weights.detach()


def coefficients(features, weights, max_abs=None):
    result = features.detach().to(weights.dtype)@weights.detach()
    if max_abs is not None:
        if max_abs <= 0:
            raise ValueError('max_abs must be positive')
        result = result.clamp(-max_abs, max_abs)
    return result.detach()

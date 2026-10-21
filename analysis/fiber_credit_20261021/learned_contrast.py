"""Trainable, translation-cancelling contrasts for conditional-null controls.

The fixed global-antithetic contrast misses every even noise response. Two
opposite pair rotations keep that contrast's translation cancellation while
allowing mixed even responses. This is a richer control family, not a new
Stein identity or a claim of native VLA effectiveness.
"""
import torch
from torch.func import jvp


def pair_rotate(noise, front, tail, angles, sign=1.):
    """Orthogonal Givens rotations on disjoint front/tail coordinate pairs."""
    x, y = noise[front], noise[tail]
    c, s = angles.cos(), sign*angles.sin()
    result = noise.clone()
    result = result.index_copy(0, front, c*x-s*y)
    result = result.index_copy(0, tail, s*x+c*y)
    return result


def contrast(function, noise, front, tail, angles):
    return .5*(function(pair_rotate(noise, front, tail, angles, 1.))
               - function(pair_rotate(noise, front, tail, angles, -1.)))


def projected_coefficients(parts, noise):
    """Return p=N z-div N and q=N e using existing native HVPs.

    This allows L to remain differentiable without differentiating the SVD.
    Geometry is frozen at the actor reference, as in the native gradient.
    """
    j, b, e = parts['j'], parts['b'], parts['probe']
    z = noise.to(j.dtype)
    div_n = -j.T@parts['div_b']-parts['hessian_weighted']
    return (z-b@(j@z)-div_n).detach(), (e-b@(j@e)).detach()


def learned_surrogate(feature, noise, parts, injection):
    """delta(N L feature_theta), differentiable in actor and L/feature.

    L can also be computed from reference context and reference executed A.
    Such features must be frozen in the actor derivative. Along a fiber,
    derivatives of L(c,A) do not contribute because N J.T=0. Arbitrary latent-
    dependent L would require additional divergence terms and is not supported.
    """
    p, q = projected_coefficients(parts, noise)
    value, tangent = jvp(feature, (noise,), (parts['probe'].to(noise.dtype),))
    l = injection.to(p.dtype)
    return ((l.T@p)*value.to(p.dtype)).sum()-((l.T@q)*tangent.to(p.dtype)).sum()


def directional_variance_objective(surrogate, parameters, direction):
    """2 D_theta surrogate[stop(g)], whose phi derivative is d_phi ||g||^2.

    Caller supplies g=grad_theta(S_base+S_control), detached. This evaluates
    a parameter JVP instead of retaining a full parameter Hessian graph.
    surrogate(parameters) must still depend differentiably on control phi.
    """
    frozen_direction = tuple(g.detach() for g in direction)
    return 2*jvp(surrogate, (tuple(parameters),), (frozen_direction,))[1]

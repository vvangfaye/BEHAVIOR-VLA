"""Implicit derivatives of a Gaussian right-inverse integration-by-parts weight.

Matrix calculus implementation, not a new integration-by-parts theorem.
The dense Jacobian is a value-only input. No differentiated SVD/inverse graph.
"""
import torch
from torch.func import jvp, grad


def geometry(jacobian):
    """Full-row-rank right inverse, without damping/truncation."""
    u, s, vh = torch.linalg.svd(jacobian.detach(), full_matrices=False)
    if len(s) != jacobian.shape[0] or s[-1] <= 0:
        raise ValueError('A full-row-rank Jacobian is required')
    inverse_gram = (u / s.square()[None, :]) @ u.T
    right_inverse = (vh.T / s[None, :]) @ u.T
    return right_inverse, inverse_gram, s


def divergence_probe(function, noise, jacobian, right_inverse, inverse_gram, probe):
    """One unbiased trace probe for div_z B, where B=J.T (J J.T)^-1.

    E probe probe.T must be I. Inputs J,B,M,probe are held fixed inside HVPs.
    The equality follows by differentiating the right-inverse formula.
    """
    j, b, m = [x.detach() for x in (jacobian, right_inverse, inverse_gram)]
    e = probe.detach().to(j.dtype)
    u = (m @ (j @ e)).detach()
    null = (e - j.T @ u).detach()
    tangent = e.to(noise.dtype)
    null_tangent = null.to(noise.dtype)
    mixed = jvp(lambda z: jvp(function, (z,), (null_tangent,))[1], (noise,), (tangent,))[1]
    weighted_hvp = jvp(grad(lambda z: (function(z).to(u.dtype) * u).sum()), (noise,), (tangent,))[1]
    return m @ (mixed.to(j.dtype) - j @ weighted_hvp.to(j.dtype))


def local_surrogate(function, noise, right_inverse, divergence, probe):
    """Gradient surrogate at the reference parameters, not a log probability.

    function must retain parameter gradients. Geometry must be independently
    evaluated at those same reference parameters, then frozen for one update.
    The same or an independent isotropic probe can estimate the second trace.
    """
    b, d, e = [x.detach() for x in (right_inverse, divergence, probe)]
    output, tangent = jvp(function, (noise,), (e.to(noise.dtype),))
    return ((b.T @ noise.to(b.dtype) - d) * output.to(b.dtype)).sum() - ((b.T @ e.to(b.dtype)) * tangent.to(b.dtype)).sum()

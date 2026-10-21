"""Reward-conditional null-space transport controls for implicit policies.

All geometry is evaluated at reference parameters; these expressions are
gradient surrogates, not action likelihoods. Requires full-row-rank J.
"""
import torch
from torch.func import jvp, grad


def reference_parts(function, noise, jacobian, right_inverse, inverse_gram, probe):
    j,b,m=[v.detach() for v in (jacobian,right_inverse,inverse_gram)]
    e=probe.detach().to(j.dtype)
    u=(b.T@e).detach()
    n=(e-j.T@u).to(noise.dtype).detach()
    ee=e.to(noise.dtype)
    mixed=jvp(lambda z:jvp(function,(z,),(n,))[1],(noise,),(ee,))[1]
    hu=jvp(grad(lambda z:(function(z).to(u.dtype)*u).sum()),(noise,),(ee,))[1]
    div_b=m@(mixed.to(j.dtype)-j@hu.to(j.dtype))
    return {'j':j,'b':b,'probe':e,'div_b':div_b.detach(),'hessian_weighted':hu.detach().to(j.dtype)}


def null_parts(parts, injection):
    """C=(I-BJ)L and one consistent trace estimate of div C.

    L is a fixed matrix, independent of current noise and actor parameters.
    """
    j,b,e=parts['j'],parts['b'],parts['probe']
    l=injection.detach().to(j.dtype)
    jl=j@l
    c=l-b@jl
    div_c=-jl.T@parts['div_b']-l.T@parts['hessian_weighted']
    return c.detach(),div_c.detach()


def null_surrogate(function, noise, field, field_divergence, probe, rotation):
    """One antithetic sampler-based null control, differentiable in actor.

    rotation is fixed; orthogonality is convenient but not required for the
    zero-conditional-mean identity. Antithetic evaluation removes translations
    in a simple affine example; it is not required for unbiasedness.
    """
    c,d,e=[v.detach() for v in (field,field_divergence,probe)]
    s=rotation.detach().to(noise.dtype)
    def odd(z):
        return .5*(function(s@z)-function(-s@z))
    output,tangent=jvp(odd,(noise,),(e.to(noise.dtype),))
    return ((c.T@noise.to(c.dtype)-d)*output.to(c.dtype)).sum()-((c.T@e.to(c.dtype))*tangent.to(c.dtype)).sum()

"""Posterior-path MALA for nondegenerate diagonal Gaussian action samplers.

This module is a training estimator, not a deployment controller. It conditions
on a fixed pre-decode action prefix. The terminal unused coordinates are
integrated out; internal coordinates are retained in full.

`transition(x, step)` must return the exact sampling mean/std at the frozen
collection parameters. No ODE steps, dropout, clipping, or unrecorded randomness
are allowed inside this callback.
"""
from collections.abc import Callable
import math

import torch


Transition = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]


def normal_logprob(x, mean, std):
    if not torch.all(torch.isfinite(std) & (std > 0)):
        raise ValueError('Every Gaussian coordinate must have finite positive std')
    return -0.5 * ((x - mean) / std).square() - torch.log(std) - 0.5 * math.log(2 * math.pi)


class GaussianChainPosterior:
    def __init__(self, transition: Transition, steps: int, observed_mask: torch.Tensor):
        if steps < 1 or observed_mask.ndim != 1 or observed_mask.dtype != torch.bool:
            raise ValueError('Expected steps >= 1 and a flattened boolean action mask')
        if not observed_mask.any():
            raise ValueError('At least one executed coordinate must be conditioned on')
        self.transition = transition
        self.steps = steps
        self.mask = observed_mask

    def _check(self, path, observed):
        if path.ndim != 3 or path.shape[1] != self.steps:
            raise ValueError('Latent path is [batch, steps, flattened_action_dim], excluding terminal output')
        if path.shape[-1] != self.mask.numel():
            raise ValueError('Action mask and path dimensions differ')
        if observed.shape != (path.shape[0], int(self.mask.sum())):
            raise ValueError('Observed action has wrong shape')

    def log_joint(self, path, observed):
        """Exact joint log density with terminal unused coordinates integrated out.

        Detach `path` and `observed` before differentiating this in actor weights.
        Do not differentiate the posterior Markov-chain trajectory.
        """
        self._check(path, observed)
        total = normal_logprob(path[:, 0], torch.zeros_like(path[:, 0]),
                               torch.ones_like(path[:, 0])).sum(-1)
        for k in range(self.steps):
            mean, std = self.transition(path[:, k], k)
            if k == self.steps - 1:
                total = total + normal_logprob(observed, mean[:, self.mask],
                                               std[:, self.mask]).sum(-1)
            else:
                total = total + normal_logprob(path[:, k + 1], mean, std).sum(-1)
        return total

    def conditional_logprob(self, x, path, observed, site):
        """Unnormalized node conditional: only the two neighboring factors."""
        if not 0 <= site < self.steps:
            raise ValueError('Invalid latent node index')
        if site == 0:
            total = normal_logprob(x, torch.zeros_like(x), torch.ones_like(x)).sum(-1)
        else:
            mean, std = self.transition(path[:, site - 1], site - 1)
            total = normal_logprob(x, mean, std).sum(-1)
        mean, std = self.transition(x, site)
        if site == self.steps - 1:
            total = total + normal_logprob(observed, mean[:, self.mask], std[:, self.mask]).sum(-1)
        else:
            total = total + normal_logprob(path[:, site + 1], mean, std).sum(-1)
        return total

    def node_value_grad(self, x, path, observed, site):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            value = self.conditional_logprob(x, path.detach(), observed.detach(), site)
            gradient, = torch.autograd.grad(value.sum(), x)
        return value.detach(), gradient.detach()

    @torch.no_grad()
    def mala_step(self, path, observed, site, step_size, *, generator=None):
        """One exact MH-adjusted MALA node update at fixed collection weights.

        A constant step size must be selected before observing this chain's
        realization. Returned path and observed action are detached.
        """
        self._check(path, observed)
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError('step_size must be a finite positive constant')
        current = path[:, site].detach()
        lp, gradient = self.node_value_grad(current, path, observed, site)
        forward_mean = current + 0.5 * step_size**2 * gradient
        noise = torch.randn(current.shape, dtype=current.dtype, device=current.device,
                            generator=generator)
        proposal = forward_mean + step_size * noise
        lp_proposal, reverse_gradient = self.node_value_grad(proposal, path, observed, site)
        reverse_mean = proposal + 0.5 * step_size**2 * reverse_gradient
        sigma = torch.full_like(current, step_size)
        log_forward = normal_logprob(proposal, forward_mean, sigma).sum(-1)
        log_reverse = normal_logprob(current, reverse_mean, sigma).sum(-1)
        log_acceptance = lp_proposal - lp + log_reverse - log_forward
        uniforms = torch.rand(lp.shape, dtype=lp.dtype, device=lp.device, generator=generator)
        accepted = torch.log(uniforms) < torch.minimum(log_acceptance, torch.zeros_like(log_acceptance))
        new_path = path.detach().clone()
        new_path[:, site] = torch.where(accepted[:, None], proposal, current)
        return new_path, accepted

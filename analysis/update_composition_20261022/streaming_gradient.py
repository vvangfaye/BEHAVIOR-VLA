"""Compose accumulated old/new-policy gradients through one inner update.

Each outer log-density graph can be freed after its local score gradient.
The expensive inner-update VJP is evaluated once for the complete batch.
"""
import torch


def compose_meta_gradient(initial, updated, gradient_before, gradient_after):
    names=tuple(initial)
    if not (tuple(updated)==tuple(gradient_before)==tuple(gradient_after)==names):
        raise ValueError('Parameter and gradient orders differ')
    pulled_back=torch.autograd.grad(tuple(updated.values()),tuple(initial.values()),
        grad_outputs=tuple(gradient_after[n].detach() for n in names))
    return {n:g+gradient_before[n].detach() for n,g in zip(names,pulled_back)}

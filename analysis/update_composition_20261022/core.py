"""Differentiable demonstration update and on-policy cross-version objective.

No simulator, reward model, replay approximation, or optimizer hidden here.
Callers must provide log densities of the actual sampled stochastic process.
"""
from collections import OrderedDict

import torch


def inner_sgd(params, loss_fn, lr, steps=1):
    """Functional SGD; retain the Hessian-vector route to the initial params."""
    if steps < 1 or lr <= 0:
        raise ValueError("Expected positive steps and learning rate")
    updated = OrderedDict(params)
    for _ in range(steps):
        loss = loss_fn(updated)
        gradients = torch.autograd.grad(loss, tuple(updated.values()), create_graph=True)
        updated = OrderedDict(
            (name, parameter - lr * gradient)
            for (name, parameter), gradient in zip(updated.items(), gradients)
        )
    return updated


def inner_adam_first_step(params, loss_fn, lr=1e-3, epsilon=1e-6):
    """Exact first Adam step with fresh zero moments and no weight decay.

    Bias correction makes m_hat=g, v_hat=g**2, independently of beta1/beta2.
    abs(g) avoids the undefined autograd sqrt derivative at an exactly zero
    gradient. Epsilon is part of the frozen optimizer configuration.
    """
    if lr<=0 or epsilon<=0:
        raise ValueError('Positive Adam learning rate and epsilon required')
    gradients=torch.autograd.grad(loss_fn(params),tuple(params.values()),create_graph=True)
    return OrderedDict((name,p-lr*g/(g.abs()+epsilon))
                       for (name,p),g in zip(params.items(),gradients))


def on_policy_loss(logp_before, logp_after, use_after, valid, advantage):
    """Negative REINFORCE objective, summed in time and averaged over episodes.

    Inputs logp_* are [episodes, replanning decisions], already SUMMED over
    action and sampler dimensions. Routes and advantage must not depend on
    the sampled action at that decision. No PPO reuse is implied.
    """
    if not (logp_before.shape == logp_after.shape == use_after.shape == valid.shape):
        raise ValueError("Expected matching [episode, decision] shapes")
    if logp_before.ndim != 2 or advantage.shape != logp_before.shape[:1]:
        raise ValueError("Expected one detached return/baseline difference per episode")
    if use_after.dtype != torch.bool or valid.dtype != torch.bool:
        raise TypeError("Route and padding masks must be boolean")
    chosen = torch.where(use_after, logp_after, logp_before)
    selected = torch.where(valid, chosen, torch.zeros_like(chosen))
    return -(selected.sum(-1) * advantage.detach()).mean()


def mixture_loss(full_updated_loss, mixed_loss, weight=0.1):
    if not 0 <= weight < 1:
        raise ValueError("Compatibility weight must lie in [0, 1)")
    return (1 - weight) * full_updated_loss + weight * mixed_loss


def training_episode_weights(method, compatibility_weight):
    """Match the conservative control to symmetric temporal composition.

    F = (1-w) J++ + w/2 (J+- + J-+)
      = (1-w/2) J++ + w/2 J-- - w/2 (J++ + J-- - J+- - J-+).
    Thus the before/after-only control needs w/2 old-policy weight.
    """
    if not 0 <= compatibility_weight < 1:
        raise ValueError('Invalid compatibility weight')
    if method == 'before_after':
        return [1-compatibility_weight/2, compatibility_weight/2]
    if method in ('meta', 'fixed_crossplay'):
        return [1-compatibility_weight, compatibility_weight]
    if method in ('maml', 'rl_bc'):
        return [.5, .5]
    raise ValueError('Unknown training method')

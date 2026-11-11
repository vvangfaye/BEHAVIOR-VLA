"""Memory-bounded actor gradient accumulation for fixed posterior particles.

Collection must be on policy and particles must use the same fixed actor.
This helper deliberately does not construct PPO ratios or step an optimizer.
"""
import torch

from posterior import normal_logprob


def accumulate_actor_gradient(engine, particles, observed, advantages, parameters, *, scale=1.0):
    """Accumulate -A times averaged joint score, one transition graph at a time.

    particles: [M, B, K, D], detached posterior paths without terminal output.
    observed: [B, D_executed], fixed normalized action prefix.
    advantages: [B], detached return-minus-baseline for actual physical behavior.
    Existing parameter gradients are accumulated into, never silently cleared.
    `scale` is an external minibatch aggregation weight, not an importance ratio.
    For a complete finite-horizon rollout batch, the episodic objective averages
    sums over episodes. If this call contains all N recorded decisions from E
    episodes, use scale=N/E. If sampling decision minibatches from that batch,
    use the same N/E to obtain an unbiased estimate of the empirical episodic
    gradient. Do not normalize a variable-length return by episode length.
    """
    if particles.ndim != 4 or particles.requires_grad or observed.requires_grad:
        raise ValueError('Expected detached posterior paths and observed actions')
    count, batch, steps, _ = particles.shape
    if count < 1 or batch < 1 or steps != engine.steps:
        raise ValueError('Invalid posterior particle dimensions')
    if advantages.shape != (batch,) or advantages.requires_grad:
        raise ValueError('Advantages must be a detached batch vector')
    if not torch.isfinite(advantages).all():
        raise ValueError('Nonfinite advantages')
    parameters = tuple(parameters)
    versions = tuple(p._version for p in parameters)
    if not any(p.requires_grad for p in parameters):
        raise ValueError('No actor parameters enabled for training')
    normalizer = scale / (count * batch)
    loss_value = 0.0
    factor_count = 0
    for particle in particles:
        engine._check(particle, observed)
        for step in range(steps):
            mean, std = engine.transition(particle[:, step], step)
            if step == steps - 1:
                log_factor = normal_logprob(observed, mean[:, engine.mask], std[:, engine.mask]).sum(-1)
            else:
                log_factor = normal_logprob(particle[:, step + 1], mean, std).sum(-1)
            loss = -(advantages * log_factor).sum() * normalizer
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite actor loss')
            if loss.requires_grad:
                loss.backward()
                factor_count += 1
            loss_value += float(loss.detach())
    if tuple(p._version for p in parameters) != versions:
        raise RuntimeError('Actor parameters changed during score accumulation')
    return {'loss_without_fixed_initial_density': loss_value,
            'backward_factors': factor_count, 'posterior_particles': count,
            'physical_decisions': batch}

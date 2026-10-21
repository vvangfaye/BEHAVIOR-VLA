"""Check streaming posterior-gradient accumulation against joint autodiff."""
import json
from pathlib import Path

import torch

from posterior import GaussianChainPosterior
from training import accumulate_actor_gradient


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(7335)
theta = torch.nn.Parameter(torch.tensor([0.17, -0.28]))
matrix = torch.randn((3, 4, 4), generator=generator) * 0.2
offset = torch.randn((3, 4), generator=generator)


def transition(x, step):
    mu = torch.tanh(x @ matrix[step].T + theta[0] * offset[step]) + theta[1]
    std = torch.full_like(x, 0.5 + 0.1 * step)
    return mu, std


engine = GaussianChainPosterior(transition, 3, torch.tensor([True, False, True, False]))
particles = torch.randn((5, 7, 3, 4), generator=generator)
observed = torch.randn((7, 2), generator=generator)
advantages = torch.randn((7,), generator=generator)
versions = theta._version
reference_loss = sum(-(advantages * engine.log_joint(p, observed)).mean() for p in particles) / len(particles)
reference_gradient, = torch.autograd.grad(reference_loss, theta)
info = accumulate_actor_gradient(engine, particles, observed, advantages, [theta])
error = float((theta.grad - reference_gradient).abs().max())
assert error < 1e-11 and theta._version == versions
result = {'streaming_vs_joint_gradient_max_abs_error': error,
          'reference_gradient': reference_gradient.tolist(),
          'streaming_gradient': theta.grad.tolist(), 'actor_not_updated': True,
          'scope': 'Nonlinear CPU gradient-accumulation check; arbitrary detached particles, not a posterior-quality test.',
          'accumulation': info}
Path(__file__).with_name('training_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))

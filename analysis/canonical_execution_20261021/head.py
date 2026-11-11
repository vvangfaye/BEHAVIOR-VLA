"""A bounded conditional log-sum-exp gradient generator.

Classical convex-potential construction; this module alone is not a new method.
All convexity assertions are in the declared normalized action/noise coordinates.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CanonicalExecutionHead(nn.Module):
    def __init__(self, context_dim, action_dim, prototypes, center=None, radius=None,
                 hidden=128, temperature=1.):
        super().__init__()
        if prototypes < 2 or action_dim < 1 or temperature <= 0:
            raise ValueError('Invalid head dimensions/temperature')
        self.action_dim = action_dim
        self.prototypes = prototypes
        self.context = nn.Sequential(nn.Linear(context_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.shape = nn.Linear(hidden, 2 * action_dim)
        self.bias = nn.Linear(hidden, prototypes)
        self.raw_vectors = nn.Parameter(torch.randn(prototypes, action_dim) * .3)
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(temperature))))
        self.register_buffer('center', torch.zeros(action_dim) if center is None else center.clone())
        self.register_buffer('radius', torch.ones(action_dim) if radius is None else radius.clone())
        if self.center.shape != (action_dim,) or self.radius.shape != (action_dim,) or (self.radius < 0).any():
            raise ValueError('Invalid coordinate bounds')

    def coefficients(self, context):
        h = self.context(context)
        shift, scale = self.shape(h).chunk(2, dim=-1)
        vectors = self.center + self.radius * torch.tanh(
            shift[:, None, :] + F.softplus(scale)[:, None, :] * self.raw_vectors[None])
        temperature = F.softplus(self.raw_temperature) + 1e-5
        return vectors, self.bias(h), temperature

    def potential(self, context, noise):
        vectors, bias, temperature = self.coefficients(context)
        logits = torch.einsum('bjd,bd->bj', vectors, noise) + bias
        return temperature * torch.logsumexp(logits / temperature, dim=-1)

    def forward(self, context, noise):
        vectors, bias, temperature = self.coefficients(context)
        logits = torch.einsum('bjd,bd->bj', vectors, noise) + bias
        weights = torch.softmax(logits / temperature, dim=-1)
        return torch.einsum('bj,bjd->bd', weights, vectors)


def two_sample_energy(first, second, target):
    """Unbiased two-draw energy-score estimate, up to fixed dimension scaling."""
    dimension = first.shape[-1]
    return ((first-target).norm(dim=-1) + (second-target).norm(dim=-1)
            - (first-second).norm(dim=-1)).mean() / (2 * math.sqrt(dimension))

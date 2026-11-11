"""Candidate action-distribution loss; no environment or policy calls.

The signature is of integrated controls, NOT of the sampled joint-position
curve. For absolute target commands this is an algebraic feature map; any
physical response interpretation additionally requires a valid control model.
"""
from __future__ import annotations

import torch


def control_signature2(actions: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Level 1 and level 2 of time-augmented, piecewise-constant controls.

    actions: [..., K, D] in a fixed, documented control coordinate system.
    Returns [..., D+1] and [..., D+1, D+1].
    Time is the first channel. The initial integrated-control value is zero.
    """
    if actions.ndim < 2 or actions.shape[-2] < 1 or dt <= 0:
        raise ValueError("Need nonempty control sequence and positive dt")
    time = torch.ones_like(actions[..., :1])
    increments = torch.cat((time, actions), dim=-1) * dt
    past = increments.cumsum(dim=-2) - increments
    level1 = increments.sum(dim=-2)
    level2 = torch.einsum("...ki,...kj->...ij", past, increments)
    level2 = level2 + 0.5 * torch.einsum("...ki,...kj->...ij", increments, increments)
    return level1, level2


def order_features(actions: torch.Tensor, dt: float) -> torch.Tensor:
    """Only antisymmetric level-two terms: order/time moments, not DC terms."""
    _, level2 = control_signature2(actions, dt)
    area = 0.5 * (level2 - level2.transpose(-1, -2))
    i, j = torch.triu_indices(area.shape[-1], area.shape[-1], offset=1, device=area.device)
    return area[..., i, j]


def energy_score_two(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unbiased two-independent-sample estimate, with sample axis -2."""
    if samples.shape[-2] != 2:
        raise ValueError("Exactly two independent generated samples are required")
    attraction = torch.linalg.vector_norm(samples - target.unsqueeze(-2), dim=-1).mean(-1)
    repulsion = 0.5 * torch.linalg.vector_norm(samples[..., 0, :] - samples[..., 1, :], dim=-1)
    return attraction - repulsion


def split_order_features(actions: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate linear time moments from quadratic control-control areas.

    The first group is a fixed linear temporal projection of actions, not a
    nonlinear interaction. Neither group is automatically physical response.
    """
    _, level2 = control_signature2(actions, dt)
    area = 0.5 * (level2 - level2.transpose(-1, -2))
    temporal = area[..., 0, 1:]
    control_area = area[..., 1:, 1:]
    i, j = torch.triu_indices(control_area.shape[-1], control_area.shape[-1], 1, device=area.device)
    return temporal, control_area[..., i, j]


def control_order_score(
    samples: torch.Tensor, target: torch.Tensor, dt: float, order_weight: float = 1.0
) -> torch.Tensor:
    """Raw energy score plus an order-feature energy score.

    samples: [B, 2, K, D]; target: [B, K, D].
    Positive raw weight (fixed at one) retains strict propriety on the raw
    action law under the usual finite-first-moment assumptions. Truncated
    order features alone do not identify that law. This says nothing about
    closed-loop success or finite-parameter optimization improvement.
    """
    if samples.ndim != 4 or target.ndim != 3 or samples.shape[1] != 2:
        raise ValueError("Expected [B,2,K,D] samples and [B,K,D] targets")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("Shapes disagree")
    if order_weight < 0:
        raise ValueError("Negative order weight does not define a proper score")
    raw = energy_score_two(samples.flatten(-2), target.flatten(-2))
    ordered = energy_score_two(order_features(samples, dt), order_features(target, dt))
    return (raw + order_weight * ordered).mean()

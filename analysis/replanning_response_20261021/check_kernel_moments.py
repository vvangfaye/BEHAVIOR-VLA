"""Check a policy-law-invariant replacement for the failed pathwise surrogate.

This is an analytic toy validation, not a VLA or simulator experiment. Gaussian
closure is exact in the affine, constant-variance case only. Cubature accuracy
for general physical kernels is deliberately not inferred from these checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)


def moments_from_samples(mean, variance, parameters):
    """Moment-match x_next=a using conditional action moments then state cubature."""
    gain, bias, log_sigma, omega, curvature, heteroskedasticity = parameters
    state_nodes = torch.tensor([-3.0**0.5, 0.0, 3.0**0.5])
    state_weights = torch.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    states = mean + variance.sqrt() * state_nodes
    # These four points exactly integrate degree-two polynomials of z~N(0,I).
    noise = torch.tensor([[2.0**0.5, 0.0], [-2.0**0.5, 0.0],
                          [0.0, 2.0**0.5], [0.0, -2.0**0.5]])
    angle = omega * states
    projected_noise = (angle.cos()[:, None] * noise[:, 0]
                       + angle.sin()[:, None] * noise[:, 1])
    conditional_mean = bias + gain * states + curvature * states.square()
    sigma = (log_sigma + heteroskedasticity * states).exp()
    actions = conditional_mean[:, None] + sigma[:, None] * projected_noise
    first = actions.mean(dim=1)
    second = actions.square().mean(dim=1)
    next_mean = (state_weights * first).sum()
    next_variance = (state_weights * second).sum() - next_mean.square()
    return next_mean, next_variance


def propagate(parameters, steps=4):
    mean, variance = torch.tensor(0.02), torch.tensor(0.0001)
    for _ in range(steps):
        mean, variance = moments_from_samples(mean, variance, parameters)
    return mean.square() + variance


def analytic_affine(parameters, steps=4):
    gain, bias, log_sigma = parameters[:3]
    mean, variance = torch.tensor(0.02), torch.tensor(0.0001)
    for _ in range(steps):
        mean = gain * mean + bias
        variance = gain.square() * variance + (2 * log_sigma).exp()
    return mean.square() + variance


def evaluate(values):
    parameters = torch.tensor(values, requires_grad=True)
    value = propagate(parameters)
    gradient, = torch.autograd.grad(value, parameters)
    return value.item(), gradient.tolist()


def main():
    rows = []
    errors = []
    for gain, bias in [(0.0, 0.0), (0.4, 0.03), (0.8, -0.02)]:
        for omega in [0.0, 10.0, 20.0]:
            p = torch.tensor([gain, bias, -2.302585092994046, omega, 0.0, 0.0],
                             requires_grad=True)
            value = propagate(p)
            gradient, = torch.autograd.grad(value, p)
            expected = analytic_affine(p)
            expected_gradient, = torch.autograd.grad(expected, p)
            # Curvature / heteroskedasticity derivatives are outside the
            # affine oracle's parameter family, so compare only its first four.
            errors.append(max(abs(value.item() - expected.item()),
                              (gradient[:4] - expected_gradient[:4]).abs().max().item()))
            rows.append({"gain": gain, "bias": bias, "omega": omega,
                         "cost": value.item(), "omega_gradient": gradient[3].item()})
    nonlinear = [evaluate([0.4, 0.03, -2.302585092994046, omega, 0.2, 0.3])
                 for omega in [0.0, 10.0, 20.0]]
    nonlinear_value_spread = max(v for v, _ in nonlinear) - min(v for v, _ in nonlinear)
    nonlinear_gradient_spread = max(
        abs(g[j] - nonlinear[0][1][j]) for _, g in nonlinear for j in range(6))
    passed = (max(errors) < 1e-11 and nonlinear_value_spread < 1e-11
              and nonlinear_gradient_spread < 1e-10)
    result = {
        "scope": "Analytic kernel-moment and autodiff checks only; no VLA efficacy claim",
        "affine_constant_variance_exact_oracle_max_error": max(errors),
        "affine_rows": rows,
        "nonlinear_gaussian_closure_value_spread_under_noise_rotation": nonlinear_value_spread,
        "nonlinear_gaussian_closure_gradient_spread_under_noise_rotation": nonlinear_gradient_spread,
        "checks_passed": passed,
        "not_verified": ["general nonlinear closure accuracy", "native visual policy integration",
                         "physical model identification", "novelty", "closed-loop improvement"],
    }
    Path(__file__).with_name("kernel_moments_check.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

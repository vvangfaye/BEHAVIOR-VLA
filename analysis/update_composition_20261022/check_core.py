"""Exact finite-state gradient verification, not a robotics benefit experiment."""
import itertools
import json
from pathlib import Path

import torch
from core import inner_sgd, mixture_loss, on_policy_loss

torch.set_default_dtype(torch.float64)


def probabilities(theta):
    p = {"theta": theta}
    target = torch.tensor([0.8, -0.6, 0.5])
    # A non-diagonal Hessian ensures this detects loss of the inner-update graph.
    matrix = torch.tensor([[1., .3, 0.], [.3, 1.2, .2], [0., .2, .8]])
    def demo_loss(q):
        delta = q["theta"] - target
        return 0.5 * delta @ matrix @ delta
    updated = inner_sgd(p, demo_loss, lr=0.25, steps=2)["theta"]
    return theta.sigmoid(), updated.sigmoid()


def terms(theta):
    before, after = probabilities(theta)
    full = torch.zeros(())
    estimate = torch.zeros(())
    for route in ((1, 1), (0, 1), (1, 0)):
        mode_weight = .9 if route == (1, 1) else .05
        for a, b in itertools.product((0, 1), repeat=2):
            # Second-stage observation is the first-stage action.
            p0 = (after if route[0] else before)[0]
            p1 = (after if route[1] else before)[1 + a]
            mass = (p0 if a else 1-p0) * (p1 if b else 1-p1)
            reward = torch.tensor([[.2, .9], [1., .1]])[a, b]
            full = full + mode_weight * mass * reward
            l0 = torch.stack([before[0] if a else 1-before[0], before[1+a] if b else 1-before[1+a]]).log()[None]
            l1 = torch.stack([after[0] if a else 1-after[0], after[1+a] if b else 1-after[1+a]]).log()[None]
            loss = on_policy_loss(l0, l1, torch.tensor([route], dtype=torch.bool),
                                  torch.ones(1, 2, dtype=torch.bool), reward[None])
            estimate = estimate - mode_weight * mass.detach() * loss
    return full, estimate


def main():
    theta = torch.tensor([.2, -.3, .4], requires_grad=True)
    exact, estimate = terms(theta)
    analytic = torch.autograd.grad(exact, theta, retain_graph=True)[0]
    policy_gradient = torch.autograd.grad(estimate, theta)[0]
    eps = 1e-5
    finite = []
    for i in range(3):
        delta = torch.zeros_like(theta); delta[i] = eps
        finite.append((terms((theta.detach()+delta).requires_grad_())[0] -
                       terms((theta.detach()-delta).requires_grad_())[0]).detach() / (2*eps))
    finite = torch.stack(finite)
    error = max(float((analytic-policy_gradient).abs().max()), float((analytic-finite).abs().max()))
    assert error < 1e-8
    # Negative control: drop all inner Hessian terms while preserving forward values.
    original_probabilities = globals()["probabilities"]
    def first_order(q):
        old, new = original_probabilities(q)
        # Recover adapted logits, but make d(theta')/d(theta) the identity.
        logits = torch.logit(new)
        return old, (q + (logits-q).detach()).sigmoid()
    globals()["probabilities"] = first_order
    first_order_grad = torch.autograd.grad(terms(theta)[1], theta)[0]
    globals()["probabilities"] = original_probabilities
    mismatch = float((first_order_grad-analytic).abs().max())
    assert mismatch > 1e-4
    result = {"passed": True, "scope": "Exact two-decision MDP gradient only; no VLA benefit claim",
              "max_error": error, "first_order_negative_control_error": mismatch,
              "expected_return": float(exact.detach()), "gradient": analytic.tolist()}
    Path(__file__).with_name("core_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

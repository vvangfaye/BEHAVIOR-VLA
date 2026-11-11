"""Exact state-dependent MDP check of the scientific control, not VLA evidence."""
import itertools
import json
from pathlib import Path

import torch

from core import training_episode_weights
from check_core import probabilities

torch.set_default_dtype(torch.float64)


def returns(before, after):
    values = {}
    for route in itertools.product((0, 1), repeat=2):
        total = torch.zeros(())
        for a, b in itertools.product((0, 1), repeat=2):
            # Stage two depends on the actual stage-one state; no resetting.
            p0 = (after if route[0] else before)[0]
            p1 = (after if route[1] else before)[1+a]
            mass = (p0 if a else 1-p0) * (p1 if b else 1-p1)
            total += mass * torch.tensor([[.2, .9], [1., .1]])[a, b]
        values[route] = total
    return values


def components(values, weight=.1):
    old, new, left, right = (values[k] for k in ((0,0),(1,1),(1,0),(0,1)))
    candidate = (1-weight)*new + weight/2*(left+right)
    conservative = (1-weight/2)*new + weight/2*old
    interaction = new + old - left - right
    return candidate, conservative, interaction


def main():
    theta = torch.tensor([.2, -.3, .4], requires_grad=True)
    before, after = probabilities(theta)
    candidate, conservative, interaction = components(returns(before, after))
    residual = candidate - conservative + .05*interaction
    value_error = float(residual.detach().abs())
    gradient_error = float(torch.autograd.grad(residual, theta)[0].abs().max())
    assert value_error < 1e-14 and gradient_error < 1e-14
    delta = torch.tensor(0., requires_grad=True)
    before = theta.detach().sigmoid()
    after = (theta.detach()+delta*torch.tensor([.3,-.4,.8])).sigmoid()
    values = returns(before, after)
    candidate, conservative, interaction = components(values)
    matched_first_order = float(torch.autograd.grad(candidate-conservative, delta, retain_graph=True)[0])
    old_control = .9*values[(1,1)] + .1*values[(0,0)]
    mismatched_first_order = float(torch.autograd.grad(candidate-old_control, delta)[0])
    assert abs(matched_first_order) < 1e-14
    assert abs(mismatched_first_order) > 1e-5
    assert training_episode_weights('before_after', .1) == [.95,.05]
    assert training_episode_weights('meta', .1) == [.9,.1]
    result = dict(passed=True, identity_value_error=value_error,
        identity_meta_gradient_error=gradient_error,
        matched_control_first_order=matched_first_order,
        previous_control_first_order=mismatched_first_order,
        scope='Exact two-stage state-dependent MDP identity and first-order control matching; no VLA benefit claim')
    Path(__file__).with_name('temporal_interaction_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

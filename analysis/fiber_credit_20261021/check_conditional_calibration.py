"""Known-answer test of reward-free fitting and conditional reward transfer.

This checks the fitter and the statistical premise, not VLA effectiveness.
H=A+(A^2-1)B is the native IBP weight at theta=0 for
F_theta(A,B)=A+theta*(1+A*B). Its conditional mean is the true score A.
C=B is a valid conditional-null control from the tangent field h_theta=theta e2.
"""
from pathlib import Path
import json
import torch
from calibration import fit_linear, coefficients

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
gen = torch.Generator().manual_seed(29180)


def batch(n):
    a, b = torch.randn(n, 2, generator=gen).T
    return a, a+(a.square()-1)*b, b


a, h, c = batch(16384)
x = torch.stack((torch.ones_like(a), a.square()), 1)
w = fit_linear(x, (h*c)[:, None], c.square()[:, None])
constant_w = fit_linear(x[:, :1], (h*c)[:, None], c.square()[:, None])
assert (w[:, 0]-torch.tensor([1., -1.])).abs().max() < .06

a, h, c = batch(131072)
x = torch.stack((torch.ones_like(a), a.square()), 1)
corrected = h+coefficients(x, w)[:, 0]*c
constant = h+coefficients(x[:, :1], constant_w)[:, 0]*c
results = []
for name, reward in [('outer_action', (a.abs() > 1).double()),
                     ('inner_action', (a.abs() < .5).double()),
                     ('smooth', torch.sigmoid(a))]:
    target = reward*a
    raw, fitted, scalar = reward*h, reward*corrected, reward*constant
    variance = lambda t: float(t.var(unbiased=True))
    results.append(dict(reward=name, exact_score_sample_mean=float(target.mean()),
                        base_mean=float(raw.mean()), corrected_mean=float(fitted.mean()),
                        base_variance=variance(raw), constant_variance=variance(scalar),
                        conditional_variance=variance(fitted), exact_score_variance=variance(target)))
    assert variance(fitted) < variance(raw)*.8
    assert abs(float(fitted.mean()-target.mean())) < .005

# Ordinary E[C]=0 is insufficient: C=A is not zero conditional on executed A.
bad_shift = float(((a > 0).double()*a).mean())
assert bad_shift > .35
result = dict(status='conditional_fitter_known_answer_passed',
              calibration_samples=16384, independent_test_samples=131072,
              fitted_weights=w[:, 0].tolist(), expected_weights=[1., -1.],
              constant_weights=constant_w[:, 0].tolist(), rewards=results,
              unconditional_zero_control_reward_bias=bad_shift,
              scope='Analytic scalar example only. No VLA training or variance result.')
(Path(__file__).parent/'conditional_calibration_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))

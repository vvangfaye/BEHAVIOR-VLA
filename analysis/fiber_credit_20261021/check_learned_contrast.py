"""Check a missing parity mode and its learned correction; no VLA calls."""
from pathlib import Path
import json
import torch
from torch.func import jacrev
from learned_contrast import contrast, learned_surrogate, directional_variance_objective

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(21773)
front, tail = torch.tensor([0]), torch.tensor([1])
theta = torch.zeros(2, requires_grad=True)
phi = torch.tensor([-0.2, 1.3], requires_grad=True)  # L amplitude, rotation angle
z = torch.tensor([.7, -.4])
e = torch.tensor([1., -1.])
parts = dict(j=torch.tensor([[1., 0.]]), b=torch.tensor([[1.], [0.]]),
             probe=e, div_b=torch.zeros(1), hessian_weighted=torch.zeros(2))


def f(t, x):
    return (x[0]+t[0]+t[1]*x[1].square()).reshape(1)


def loss_surrogate(t):
    # Base: delta(e1 * F_theta), estimated with this same trace probe.
    base = z[0]*f(t, z)[0]-e[0]*(jacrev(lambda x:f(t, x))(z)@e)[0]
    feature = lambda x:contrast(lambda v:f(t, v), x, front, tail, phi[1:])
    injection = torch.stack((phi[0]*0, phi[0])).reshape(2, 1)
    return base+learned_surrogate(feature, z, parts, injection)


g = torch.autograd.grad(loss_surrogate(theta), theta, create_graph=True)[0]
direct = torch.autograd.grad(g.square().sum(), phi, retain_graph=True)[0]
directional = directional_variance_objective(lambda ts:loss_surrogate(ts[0]), (theta,), (g,))
via_jvp = torch.autograd.grad(directional, phi)[0]
meta_error = float((direct-via_jvp).abs().max())
assert meta_error < 1e-9

# A model response even in z disappears from the old global-antithetic feature.
old_feature = lambda t:.5*(f(t, z.flip(0))-f(t, -z.flip(0)))
old_even_response = float(jacrev(old_feature)(theta)[0, 1])
assert old_even_response == 0

# Independent noise/trace draws; the fitted correction never sees a reward.
gen = torch.Generator().manual_seed(9772)
cal_z = torch.randn(16384, 2, generator=gen)
cal_chi = (torch.randint(0, 2, (16384,), generator=gen)*2-1).double()
train_phi = torch.tensor([-.2, 1.3], requires_grad=True)
opt = torch.optim.Adam([train_phi], lr=.025)


def scores(noise, chi, parameters):
    x, y = noise.T
    old = torch.stack((x, x*y.square()-2*y*chi), 1)
    scale = parameters[0]*(2*parameters[1]).sin()
    control = torch.stack((torch.zeros_like(x), scale*(x*(y.square()-1)-y*chi)), 1)
    return old, old+control


for _ in range(250):
    opt.zero_grad()
    _, corrected = scores(cal_z, cal_chi, train_phi)
    loss = corrected.square().sum(1).mean()
    loss.backward()
    opt.step()

test_z = torch.randn(131072, 2, generator=gen)
test_chi = (torch.randint(0, 2, (131072,), generator=gen)*2-1).double()
old, corrected = scores(test_z, test_chi, train_phi.detach())
scale = float(train_phi[0]*(2*train_phi[1]).sin())
assert abs(scale+4/3) < .08
test_results=[]
for name, reward in [('positive_action', (test_z[:, 0]>0).double()),
                     ('sine_action', test_z[:, 0].sin())]:
    a, b = reward[:, None]*old, reward[:, None]*corrected
    va, vb = float(a.var(0).sum()), float(b.var(0).sum())
    assert vb < va*.7
    test_results.append(dict(reward=name, old_variance=va, corrected_variance=vb,
                             old_mean=a.mean(0).tolist(), corrected_mean=b.mean(0).tolist()))

result=dict(status='learned_contrast_small_checks_passed',
            meta_gradient_error=meta_error, old_even_parameter_response=old_even_response,
            learned_amplitude=float(train_phi[0]), learned_angle=float(train_phi[1]),
            learned_effective_scale=scale, expected_optimal_scale=-4/3,
            heldout_results=test_results,
            scope='Analytic two-dimensional policy only. Proves a blind spot of the old contrast and checks learning derivatives; does not explain the native cosine or establish native gains.',
            new_vla_calls=0, new_simulator_episodes=0)
(Path(__file__).parent/'learned_contrast_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Verify algebra and parameter gradients against explicit dense autodiff."""
from pathlib import Path
import json
import torch
from torch.func import jacrev
from implicit_geometry import geometry, divergence_probe, local_surrogate


torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(20261021)


def trial(m, d):
    a = torch.randn(d, m)
    a[:, :d] += 2 * torch.eye(d)
    c = torch.randn(d, m) * .2
    theta = torch.randn(d, requires_grad=True) * .1
    theta.retain_grad()
    ref = theta.detach().clone()
    z = torch.randn(m, requires_grad=True)

    def mapping(x, p):
        return (a @ x) * p.exp() + c @ torch.sin(x) + p + .05 * (a @ x).square()

    f0 = lambda x: mapping(x, ref)
    j = jacrev(f0)(z).detach()
    b, inv, sv = geometry(j)
    def bfun(x):
        jj = jacrev(f0)(x)
        return jj.T @ torch.linalg.inv(jj @ jj.T)
    db = jacrev(bfun)(z)
    exact_div = torch.einsum('idi->d', db)
    eye = torch.eye(m) * m**.5
    probes = [divergence_probe(f0, z, j, b, inv, e) for e in eye]
    estimated_div = torch.stack(probes).mean(0)

    delta = mapping(z, theta) - mapping(z, ref)
    hh = bfun(z) @ delta
    exact_surrogate = z @ hh - torch.trace(jacrev(lambda x: bfun(x) @ (mapping(x, theta)-mapping(x, ref)))(z))
    exact_gradient = torch.autograd.grad(exact_surrogate, theta, retain_graph=True)[0]
    implicit = torch.stack([local_surrogate(lambda x: mapping(x, theta), z, b, div, e)
                            for e, div in zip(eye, probes)]).mean()
    implicit_gradient = torch.autograd.grad(implicit, theta)[0]
    result = {'noise_dim':m,'action_dim':d,'condition_number':float(sv[0]/sv[-1]),
              'right_inverse_error':float((j@b-torch.eye(d)).abs().max()),
              'divergence_error':float((estimated_div-exact_div).abs().max()),
              'parameter_gradient_error':float((implicit_gradient-exact_gradient).abs().max())}
    assert result['right_inverse_error'] < 1e-9
    assert result['divergence_error'] < 1e-9
    assert result['parameter_gradient_error'] < 1e-9
    return result


records=[trial(m,d) for m,d in [(1,1),(3,2),(8,4),(12,5)]]
report={'status':'implicit_algebra_verified','trials':records,
        'scope':'Small smooth full-row-rank maps. Exact isotropic basis averages verify the trace identity; no native VLA or variance/speed claim.',
        'new_training_or_simulation':False}
Path(__file__).with_name('check_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

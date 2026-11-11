"""CPU mathematical checks; not a VLA experiment or performance claim.

Use the bundled Python with NumPy, SciPy and Torch. The test keeps reference
parameters frozen but differentiates the right inverse with respect to noise.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def scalar_check():
    theta, beta = .45, 1.2
    root = brentq(lambda z: theta + z + beta*np.tanh(z), -5, 5)
    def terms(z):
        t = np.tanh(z)
        sech2 = 1-t*t
        jac = 1+beta*sech2
        w = 1/jac
        dw = 2*beta*sech2*t/(jac*jac)
        return z*w, dw
    correct = quad(lambda z: norm.pdf(z)*(terms(z)[0]-terms(z)[1]), root, 12)[0]
    detached = quad(lambda z: norm.pdf(z)*terms(z)[0], root, 12)[0]
    exact = norm.pdf(root)/(1+beta*(1-np.tanh(root)**2))
    return dict(exact_success_derivative=exact, ibp_derivative=correct,
                missing_divergence_derivative=detached,
                absolute_error=abs(correct-exact),
                note="Known nonlinear scalar policy and discontinuous success reward.")


def vector_check():
    # Tensor Gauss-Hermite quadrature on 3-D standard Gaussian; 2-D output.
    x, weights = hermgauss(18)
    ids = np.array(list(itertools.product(range(len(x)), repeat=3)))
    z = torch.tensor(np.sqrt(2)*x[ids], requires_grad=True)
    q = torch.tensor(np.prod(weights[ids]/np.sqrt(np.pi), axis=1))
    reference = torch.tensor([.2, .1, -.15, -.1])
    theta = reference.clone().requires_grad_(True)

    def generator(p):
        return torch.stack((p[0]+p[1].exp()*z[:, 0]+.25*z[:, 1].tanh(),
                            p[2]+.4*z[:, 0].sin()+p[3].exp()*z[:, 1]
                            +.3*z[:, 2].tanh()), dim=1)

    f = generator(theta)
    f0 = generator(reference)
    reward = f[:, 0].sin()+.2*f[:, 1].square()
    pathwise = torch.autograd.grad((q*reward).sum(), theta, retain_graph=True)[0]
    j = torch.stack((torch.stack((torch.ones_like(z[:, 0])*reference[1].exp(),
                                 .25*(1-z[:, 1].tanh().square()),
                                 torch.zeros_like(z[:, 0])), dim=1),
                     torch.stack((.4*z[:, 0].cos(),
                                  torch.ones_like(z[:, 0])*reference[3].exp(),
                                  .3*(1-z[:, 2].tanh().square())), dim=1)), dim=1)
    eye = torch.eye(2).expand(len(z), 2, 2)
    right_inverse = j.transpose(1, 2) @ torch.linalg.solve(j @ j.transpose(1, 2), eye)
    h = (right_inverse @ (f-f0).unsqueeze(-1)).squeeze(-1)
    divergence = sum(torch.autograd.grad(h[:, k].sum(), z, create_graph=True,
                                         retain_graph=True)[0][:, k] for k in range(3))
    loss = (q*reward.detach()*((z*h).sum(dim=1)-divergence)).sum()
    ibp = torch.autograd.grad(loss, theta, retain_graph=True)[0]

    # Wrong variant illustrates why freezing B's noise dependence is not valid.
    h_wrong = (right_inverse.detach() @ (f-f0).unsqueeze(-1)).squeeze(-1)
    div_wrong = sum(torch.autograd.grad(h_wrong[:, k].sum(), z, create_graph=True,
                                       retain_graph=True)[0][:, k] for k in range(3))
    wrong_loss = (q*reward.detach()*((z*h_wrong).sum(dim=1)-div_wrong)).sum()
    wrong = torch.autograd.grad(wrong_loss, theta)[0]
    residual = j @ right_inverse-eye
    return dict(pathwise_gradient=pathwise.tolist(), ibp_gradient=ibp.tolist(),
                max_absolute_error=(pathwise-ibp).abs().max().item(),
                incorrect_frozen_noise_B_gradient=wrong.tolist(),
                incorrect_max_error=(pathwise-wrong).abs().max().item(),
                maximum_right_inverse_residual=residual.abs().max().item(),
                note="Synthetic smooth 3-noise/2-output map, four trainable parameters; numerical quadrature error remains.")


if __name__ == '__main__':
    report = dict(scalar=scalar_check(), vector=vector_check(),
                  scope="Only mathematical identities and autodiff surrogate; no native VLA feasibility or novelty verification.")
    destination = Path(__file__).with_name('native_flow_ibp_math_check_20261020.json')
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

"""Exact-distribution numerical checks of the proposed paired estimator."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss


def velocity(x, t, mu, sigma):
    variance = t*t + (1-t)**2*sigma*sigma
    return ((t-(1-t)*sigma*sigma)*x-t*mu)/variance


def check(mu0, sigma0, mu1, sigma1, t, order=40):
    nodes, mass = hermgauss(order)
    nodes = nodes*np.sqrt(2)
    mass = mass/np.sqrt(np.pi)
    a = mu0+sigma0*nodes[:, None, None]
    b = mu1+sigma1*nodes[None, :, None]
    noise = nodes[None, None, :]
    probability = mass[:, None, None]*mass[None, :, None]*mass[None, None, :]
    x = (1-t)*(a+b)/2+t*noise/np.sqrt(2)
    overlap = np.exp(-(1-t)**2*(a-b)**2/(4*t*t))
    residual = t*(velocity(x,t,mu0,sigma0)-velocity(x,t,mu1,sigma1))-(b-a)
    jac = np.stack(np.broadcast_arrays(t*x,t*np.ones_like(x),-t*x,-t*np.ones_like(x)),axis=-1)
    gradient = (2*probability[...,None]*overlap[...,None]*residual[...,None]*jac).sum((0,1,2))
    gradient /= (probability*overlap).sum()
    x0, x1 = (1-t)*a+t*noise, (1-t)*b+t*noise
    naive_residual = t*(velocity(x0,t,mu0,sigma0)-velocity(x1,t,mu1,sigma1))-(b-a)
    naive_jac = np.stack(np.broadcast_arrays(t*x0,t*np.ones_like(x0),-t*x1,-t*np.ones_like(x1)),axis=-1)
    naive_gradient = (2*probability[...,None]*naive_residual[...,None]*naive_jac).sum((0,1,2))
    return dict(mu=[mu0,mu1],sigma=[sigma0,sigma1],time=t,
                exact_field_pair_gradient=gradient.tolist(),
                naive_shared_source_gradient=naive_gradient.tolist(),
                expected_overlap=float((probability*overlap).sum()))


def main():
    cases=[check(0,.5,1,2,.7),check(0,.5,0,2,.7),check(.3,1,.3,1,.7)]
    assert max(np.abs(c['exact_field_pair_gradient']).max() for c in cases)<1e-9
    assert np.abs(cases[0]['naive_shared_source_gradient']).max()>.01
    x=np.linspace(-2,2,101);t=.7
    f0=x+.2;f1=x*x-.3
    h=.3*x*x+.4
    gauge_error=float(np.abs(((f0+h)-(f1+h))-(f0-f1)).max())
    assert gauge_error<1e-12
    variance_contrast=float(np.max(np.abs(velocity(x,t,0,.5)-velocity(x,t,0,2))))
    at_noise_one=float(np.max(np.abs(velocity(x,1,0,.5)-velocity(x,1,0,2))))
    assert variance_contrast>.1 and at_noise_one==0
    result=dict(passed=True,observed_at=datetime.now(timezone.utc).isoformat(),
                cases=cases,common_field_cancellation_error=gauge_error,
                variance_only_contrast_at_t07=variance_contrast,
                variance_only_contrast_at_t1=at_noise_one,
                scope='Gaussian population and affine-field checks only. This establishes neither novelty, neural optimization benefit, nor closed-loop improvement.')
    Path(__file__).with_name('mathematical_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""An analytic bounded-reward counterexample to naive inverse damping.

Two equally likely contexts, one shared mean parameter, Gaussian native
samplers. All integrals are deterministic quadrature, not model evaluation.
"""
import json
import math
from pathlib import Path

from scipy.integrate import quad


def expectation(f):
    return quad(lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)*f(z),
                -10., 10., epsabs=1e-12, epsrel=1e-12)[0]


def context(theta, sigma, sign, damping):
    def reward(z):
        return .5+sign*.4*math.tanh(theta+sigma*z)
    derivative = expectation(lambda z: sign*.4*(1-math.tanh(theta+sigma*z)**2))
    exact_ibp = expectation(lambda z: (reward(z)-.5)*z/sigma)
    damped_ibp = expectation(lambda z: (reward(z)-.5)*z*sigma/(sigma*sigma+damping))
    return dict(sigma=sigma, sign=sign, expected_reward=expectation(reward),
                exact_derivative=derivative, exact_ibp=exact_ibp,
                damped_ibp=damped_ibp,
                attenuation=sigma*sigma/(sigma*sigma+damping))


def objective(theta):
    return .5*context(theta, 2., -1, 1.)['expected_reward'] + \
           .5*context(theta, .2, 1, 1.)['expected_reward']


if __name__ == '__main__':
    cases=[context(0., 2., -1, 1.), context(0., .2, 1, 1.)]
    exact=sum(c['exact_derivative'] for c in cases)/2
    damped=sum(c['damped_ibp'] for c in cases)/2
    report=dict(
        definition='U=theta+sigma_c*z; z~N(0,1); P(c=1)=P(c=2)=1/2. R_1=.5-.4*tanh(U); R_2=.5+.4*tanh(U). lambda=1.',
        contexts=cases, true_shared_parameter_gradient=exact,
        inverse_damped_shared_parameter_gradient=damped,
        gradient_product=exact*damped,
        before=objective(0.),
        after_exact_gradient_step=objective(.1*exact),
        after_damped_gradient_step=objective(.1*damped),
        conclusion='Even with smooth bounded rewards and full-rank Gaussian samplers, C+lambda*I inverse damping can reverse a shared-parameter policy update. It is not automatically a positive preconditioning of the true parameter gradient.',
        scope='Synthetic analytical counterexample; no claim that this mechanism occurs in the project checkpoint.')
    assert exact>0 and damped<0
    assert report['after_exact_gradient_step']>report['before']
    assert report['after_damped_gradient_step']<report['before']
    path=Path(__file__).with_name('native_flow_damping_counterexample_20261020.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

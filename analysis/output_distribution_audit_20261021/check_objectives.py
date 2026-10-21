"""Separate expected reconstruction loss from marginal likelihood, analytically.

This is a known latent-variable distinction, not a proposed new algorithm or
evidence about VLA closed-loop competence. The example assumes independent
target/noise pairing; an optimized transport coupling need not collapse.
"""
import json
import math
from pathlib import Path


def logaddexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a-m) + math.exp(b-m))


def marginal_nll(theta, sigma=.2):
    # z is equally likely +/-1; target observations equally likely +/-1.
    normalizer = math.log(sigma*math.sqrt(2*math.pi))
    total = 0.
    for y in (-1., 1.):
        a = -((y-theta)**2)/(2*sigma*sigma)-normalizer
        b = -((y+theta)**2)/(2*sigma*sigma)-normalizer
        total -= .5*(logaddexp(a,b)-math.log(2))
    return total


def posterior_gradient(theta, sigma=.2):
    gradient = 0.
    for y in (-1.,1.):
        logw = [-((y-theta*z)**2)/(2*sigma*sigma) for z in (-1.,1.)]
        lse = logaddexp(*logw)
        gradient -= .5*sum(math.exp(w-lse)*(y-theta*z)*z/(sigma*sigma)
                           for z,w in zip((-1.,1.),logw))
    return gradient


def main():
    eps=1e-6
    errors=[]
    for theta in (.1,.3,.8,1.2):
        numerical=(marginal_nll(theta+eps)-marginal_nll(theta-eps))/(2*eps)
        errors.append(abs(numerical-posterior_gradient(theta)))
    assert max(errors)<1e-7
    grid=[i/1000 for i in range(1501)]
    optimum=min(grid,key=marginal_nll)
    assert abs(optimum-1)<.002
    result={'known_math_only':True,'generator':'a=theta*z; z in {-1,+1}; y in {-1,+1}',
            'expected_independent_pair_squared_error':'1 + theta^2; minimizer theta=0',
            'marginal_gaussian_mixture_nll_minimizer_on_grid':optimum,
            'sigma':.2,'posterior_score_gradient_max_finite_difference_error':max(errors),
            'what_is_not_proved':['CFM itself collapses','Optimized noise-data coupling collapses',
                                  'Marginal likelihood is a novel training principle',
                                  'Any advantage on VLA data or closed-loop tasks'],
            'relevance':'A reconstruction objective and true marginal likelihood must not be called identical; ABP already supplies the posterior-gradient recipe.'}
    Path(__file__).with_name('objective_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

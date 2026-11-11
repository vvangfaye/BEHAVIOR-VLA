"""Independent finite-support check at the actual auxiliary noise range."""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss


def field(x,t,endpoints):
    logp=-np.square(x[:,None]-(1-t)*endpoints[None,:])/(2*t*t)
    probability=np.exp(logp-logp.max(1,keepdims=True))
    probability/=probability.sum(1,keepdims=True)
    return (x-probability@endpoints)/t


def main():
    # Both conditional means are zero. Only their mode locations differ.
    a_set=np.asarray([-1.,1.]);b_set=np.asarray([-2.,2.])
    nodes,probability=hermgauss(128);nodes*=np.sqrt(2);probability/=np.sqrt(np.pi)
    rows=[]
    for t in [.9,.95,.99]:
        correct=np.zeros(4);unweighted=np.zeros(4)
        for a in a_set:
            for b in b_set:
                x=(1-t)*(a+b)/2+t*nodes/np.sqrt(2)
                difference=field(x,t,a_set)-field(x,t,b_set)
                residual=t*difference-(b-a)
                basis=np.stack([np.ones_like(x),x,x*x,np.tanh(x)],1)
                per_pair=(probability[:,None]*residual[:,None]*t*basis).sum(0)/4
                weight=np.exp(-(1-t)**2*(a-b)**2/(4*t*t))
                correct+=weight*per_pair;unweighted+=per_pair
        test_x=np.linspace(-3,3,201)
        contrast=field(test_x,t,a_set)-field(test_x,t,b_set)
        assert np.abs(correct).max()<1e-10
        rows.append(dict(time=t,max_weighted_functional_gradient=float(np.abs(correct).max()),
                         max_unweighted_functional_gradient=float(np.abs(unweighted).max()),
                         zero_mean_distribution_contrast=float(np.abs(contrast).max())))
    assert max(r['max_unweighted_functional_gradient'] for r in rows)>1e-5
    assert all(r['zero_mean_distribution_contrast']>0 for r in rows)
    result=dict(passed=True,rows=rows,means_equal=True,
                statement='At finite noise the target distinguishes equal-mean bimodal distributions; exact weighted gradient vanishes at the true marginal field. Removing overlap weights does not preserve that stationarity.',
                limitation='Distributional construction check, not a VLA learning or success result.')
    Path(__file__).with_name('multimodal_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

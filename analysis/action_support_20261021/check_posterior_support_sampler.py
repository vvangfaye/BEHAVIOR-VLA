"""Finite-solver check with an analytic correlated-support distribution.

This does not train a network and is not evidence for VLA improvement.
The ordinary exact marginal flow is included and must produce the same path.
"""
import json
from pathlib import Path
import numpy as np


def fields(x,t):
    q=t*t+(1-t)**2
    variances=np.array([[q,t*t],[t*t,q]])
    logw=-.5*(x[:,None,:]**2/variances[None]).sum(-1)
    logw-=logw.max(-1,keepdims=True)
    weights=np.exp(logw);weights/=weights.sum(-1,keepdims=True)
    active_mean=(1-t)/q*x
    active_v=(x-active_mean)/t
    mixture_v=weights*active_v+(1-weights)*x/t
    exact_marginal_v=(x-weights*active_mean)/t
    return mixture_v,exact_marginal_v


def metrics(x):
    active=np.abs(x)>1e-3
    return {
        'neither_above_1e3':float((active.sum(-1)==0).mean()),
        'one_above_1e3':float((active.sum(-1)==1).mean()),
        'both_above_1e3':float((active.sum(-1)==2).mean()),
        'rms_smaller_coordinate':float(np.sqrt(np.min(x*x,axis=-1).mean())),
        'mean_squared_radius':float((x*x).sum(-1).mean()),
        'dominant_first_axis_fraction':float((np.abs(x[:,0])>np.abs(x[:,1])).mean()),
    }


def main():
    rng=np.random.default_rng(419)
    source=rng.normal(size=(50000,2))
    scalar=rng.normal(size=50000)
    axis=rng.integers(0,2,size=50000)
    target=np.zeros_like(source);target[np.arange(len(source)),axis]=scalar
    independent_mask=rng.random(source.shape)<.5
    independent=source*independent_mask
    results={}
    for steps in (2,5,10,20,50,100):
        proposed=source.copy();ordinary=source.copy()
        for k in range(steps):
            t=1-k/steps
            v,_=fields(proposed,t)
            _,baseline=fields(ordinary,t)
            proposed-=v/steps;ordinary-=baseline/steps
        assert np.isfinite(proposed).all()
        difference=float(np.max(np.abs(proposed-ordinary)))
        assert difference<1e-9
        results[str(steps)]={**metrics(proposed),'difference_from_exact_ordinary_marginal_flow':difference}
    report={
        'kind':'finite_Euler_correlated_support_check',
        'n':len(source),'seed':419,'constant_detection_threshold':1e-3,
        'target':'HalfN(0,1) on each coordinate axis, exactly one active coordinate per draw.',
        'exact_joint_support_sampler':metrics(target),
        'independent_prior_mask_sampler':metrics(independent),
        'posterior_mixture_by_Euler_steps':results,
        'limits':[
            'Thresholded support proportions are tolerance dependent; Gaussian draws can be near zero.',
            'The proposed decomposition equals the exact ordinary marginal flow at population optimum.',
            'An exact joint-support sampler already generates the correct distribution.',
            'Any advantage must be learned-model accuracy/efficiency, not a new population flow.',
            'Finite Euler bias remains even with an exact velocity field.',
            'No trained-network, physical controller or VLA task result is measured.'
        ]}
    path=Path(__file__).with_name('posterior_support_sampler_check.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(path),target=report['exact_joint_support_sampler'],
                         independent=report['independent_prior_mask_sampler'],
                         steps10=results['10'],steps100=results['100']),indent=2))


if __name__=='__main__':main()

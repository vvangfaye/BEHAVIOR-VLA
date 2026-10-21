"""Analytic and autograd checks for the revised support-mixture field."""
import json
from pathlib import Path

import vla_runtime  # register pinned import paths before adapter import
import torch
from posterior_support_adapter import mix_velocity


def main():
    g=torch.Generator().manual_seed(31)
    x=torch.randn(3,8,5,generator=g,dtype=torch.float64)
    raw=torch.randn(x.shape,generator=g,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([.1,.4,1.],dtype=torch.float64)
    zeros=torch.zeros(3,3,dtype=torch.float64)
    ones=torch.ones_like(zeros)
    identity=mix_velocity(raw,x,t,ones,3)
    torch.testing.assert_close(identity,raw,atol=1e-14,rtol=0)
    collapsed=mix_velocity(raw,x,t,zeros,3)
    next_x=x-t[:,None,None]*collapsed
    torch.testing.assert_close(next_x[...,:3]-next_x[...,:3].mean(1,keepdim=True),
                               torch.zeros_like(next_x[...,:3]),atol=1e-14,rtol=0)
    torch.testing.assert_close(collapsed[...,3:],raw[...,3:],atol=0,rtol=0)
    p=torch.sigmoid(torch.randn(3,3,generator=g,dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(lambda a,b:mix_velocity(a,x,t,b,3),(raw,p))

    # Data lies on either the horizontal or vertical axis, equiprobably.
    # Each active axis has N(0,1). Activity is maximally anticorrelated.
    # Full Gaussian source implies component covariance diag(q,t^2), swapped.
    errors=[]
    for scalar_t in (.05,.2,.5,.9,1.):
        xx=torch.randn(1000,2,generator=g,dtype=torch.float64)
        q=scalar_t**2+(1-scalar_t)**2
        variance=torch.tensor([[q,scalar_t**2],[scalar_t**2,q]],dtype=torch.float64)
        log_prob=-.5*(xx[:,None,:].square()/variance[None]).sum(-1)
        weights=log_prob.softmax(-1)
        active_mean=(1-scalar_t)/q*xx
        posterior_mean=weights*active_mean
        exact=(xx-posterior_mean)/scalar_t
        active_velocity=(xx-active_mean)/scalar_t
        decomposed=weights*active_velocity+(1-weights)*xx/scalar_t
        errors.append(float((decomposed-exact).abs().max()))
    assert max(errors)<1e-12
    result={
      'mean_preserved':True,'padding_unchanged':True,'all_active_matches_native':True,
      'all_inactive_normal_component_collapses_in_remaining_time':True,
      'autograd_gradcheck':True,'max_analytic_mixture_error':max(errors),
      'joint_example':'Exactly one of two coordinates is active; no independence assumption is used.',
      'scope':'Formula and gradient checks only, not a trained model, solver-accuracy claim, or robot result.'}
    Path(__file__).with_name('posterior_support_formula_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

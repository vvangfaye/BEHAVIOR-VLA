"""Check the actual adapter loss against its analytic gradient, without VLA load."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

import vla_runtime
from posterior_support_adapter import PosteriorSupportAdapter


class Probe(nn.Module):
    loss=PosteriorSupportAdapter.loss

    def __init__(self,raw,logits):
        super().__init__()
        self.base=SimpleNamespace(action_dim=5,action_horizon=4)
        self.physical_dim=3;self.aux_weight=.1
        self.raw=nn.Parameter(raw.clone());self.logits=nn.Parameter(logits.clone())

    def prepare_context(self,obs):return obs,None,None,None
    def velocity_and_probability(self,*args):return self.raw,self.logits


def main():
    g=torch.Generator().manual_seed(628)
    shape=(2,4,5)
    raw=torch.randn(shape,generator=g)
    logits=torch.randn(2,3,generator=g)
    noise=torch.randn(shape,generator=g)
    source_actions=torch.randn(shape,generator=g)
    results={}
    for name,mask in [('all_active',torch.ones(2,3)),
                      ('all_inactive',torch.zeros(2,3)),
                      ('mixed',torch.tensor([[1.,0.,1.],[0.,1.,0.]]))]:
        actions=source_actions.clone()
        actions[...,:3]=torch.where(mask[:,None].bool(),actions[...,:3],.3)
        probe=Probe(raw,logits)
        kwargs=dict(noise=noise,time=torch.tensor([.3,.8]),raw_active=mask)
        proposed=probe.loss(None,actions,mode='posterior',**kwargs)
        actual,posterior_gradient=torch.autograd.grad(proposed['loss'],(probe.raw,probe.logits))
        error=raw-(noise-actions)
        mean=error.mean(1,keepdim=True)
        weight=torch.ones(2,5);weight[:,:3]=mask
        expected=2*(mean+(error-mean)*weight[:,None])/raw.numel()
        torch.testing.assert_close(actual,expected,atol=2e-8,rtol=2e-6)
        expected_p=.1*(logits.sigmoid()-mask)/mask.numel()
        torch.testing.assert_close(posterior_gradient,expected_p,atol=1e-8,rtol=1e-6)
        standard=probe.loss(None,actions,mode='ambient_aux',**kwargs)
        standard_gradient=torch.autograd.grad(standard['loss'],probe.raw)[0]
        torch.testing.assert_close(actual[...,3:],standard_gradient[...,3:],atol=2e-8,rtol=2e-6)
        if name=='all_active':
            torch.testing.assert_close(proposed['flow_loss'],standard['flow_loss'],atol=2e-7,rtol=2e-6)
            torch.testing.assert_close(actual,standard_gradient,atol=2e-8,rtol=2e-6)
        if name=='all_inactive':
            residual_grad=actual[...,:3]-actual[...,:3].mean(1,keepdim=True)
            torch.testing.assert_close(residual_grad,torch.zeros_like(residual_grad),atol=2e-8,rtol=0)
        results[name]={'max_gradient_error':float((actual-expected).abs().max()),
                       'flow_loss':float(proposed['flow_loss'].detach()),'passed':True}
    result={'passed':True,'cases':results,'all_active_matches_CFM':True,
            'inactive_temporal_residual_gradient_zero':True,'padding_matches_CFM':True,
            'posterior_gradient_matches_BCE':True,
            'scope':'Actual adapter.loss with stub velocity/context; verifies objective algebra, not learned VLA behavior.'}
    Path(__file__).with_name('posterior_support_loss_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Check the actual loss implementation, including its action gradients."""
import json
from pathlib import Path

import numpy as np
import torch
from adapter import bridge,difference_loss


def main():
    torch.manual_seed(73)
    shape=(3,4,32)
    a=torch.randn(shape);b=torch.randn(shape)
    left=torch.randn(shape,requires_grad=True);right=torch.randn(shape,requires_grad=True)
    t=torch.tensor([.9,.95,.99]);noise=torch.randn(shape)
    actual=difference_loss(left,right,a,b,t)
    gradients=torch.autograd.grad(actual,(left,right))
    aa,bb,ll,rr=[x.detach().numpy().astype(np.float64)[...,:23] for x in (a,b,left,right)]
    tt=t.numpy().astype(np.float64)
    w=np.exp(-(1-tt)**2*np.square(aa-bb).sum((1,2))/(4*tt**2))
    residual=tt[:,None,None]*(ll-rr)-(bb-aa)
    expected=.5*np.mean(w*np.square(residual).mean((1,2)))
    analytical=np.zeros(shape);analytical[...,:23]=w[:,None,None]*tt[:,None,None]*residual/(3*4*23)
    value_error=abs(float(actual)-expected)
    gradient_error=max(np.abs(gradients[0].numpy()-analytical).max(),np.abs(gradients[1].numpy()+analytical).max())
    assert value_error<2e-6 and gradient_error<2e-7
    assert not gradients[0][...,23:].count_nonzero() and not gradients[1][...,23:].count_nonzero()
    same=difference_loss(left,left,a,a,t)
    assert same==0
    common=torch.randn(shape)
    assert abs(float(difference_loss(left+common,right+common,a,b,t)-actual))<2e-6
    expected_bridge=(1-t[:,None,None])*(a+b)/2+t[:,None,None]*noise/np.sqrt(2)
    torch.testing.assert_close(bridge(a,b,t,noise),expected_bridge)
    result=dict(passed=True,value_error=value_error,gradient_error=float(gradient_error),
                padding_gradient_zero=True,equal_pair_zero=True,common_field_cancels=True,
                scope='Actual auxiliary loss and bridge only; not trained-policy performance.')
    Path(__file__).with_name('loss_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

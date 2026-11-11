"""Validate the registered point-control implementation against residual algebra."""
import json
from pathlib import Path
import torch

from train_bridge_control import bridge_point_loss
from adapter import difference_loss


def main():
    torch.manual_seed(311)
    shape=(2,5,32)
    a,b,x,left,right=[torch.randn(shape) for _ in range(5)]
    t=torch.tensor([.93,.97]);left.requires_grad_();right.requires_grad_()
    scalar=bridge_point_loss(left,right,x,a,b,t)
    grad=torch.autograd.grad(scalar,(left,right))
    weight=torch.exp(-(1-t).square()*(a[...,:23]-b[...,:23]).square().sum((1,2))/(4*t.square()))
    rl=t[:,None,None]*left[...,:23]-(x[...,:23]-a[...,:23])
    rr=t[:,None,None]*right[...,:23]-(x[...,:23]-b[...,:23])
    expected=.5*(weight*(rl.square().mean((1,2))+rr.square().mean((1,2)))).mean()
    torch.testing.assert_close(scalar,expected,rtol=1e-6,atol=1e-6)
    for actual,residual in zip(grad,[rl,rr]):
        theory=weight[:,None,None]*t[:,None,None]*residual/(2*5*23)
        torch.testing.assert_close(actual[...,:23],theory,rtol=1e-5,atol=1e-7)
        assert actual[...,23:].count_nonzero()==0
    paired=difference_loss(left,right,a,b,t)
    cross=(weight*(rl*rr).mean((1,2))).mean()
    torch.testing.assert_close(scalar-paired,cross,rtol=1e-5,atol=1e-6)
    result=dict(passed=True,weighted_point_gradient_correct=True,padding_gradient_zero=True,
                point_minus_difference_equals_residual_cross_term=True,
                scope='Control algebra only; no model training or performance assertion.')
    Path(__file__).with_name('bridge_control_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Orthogonal projection for conditional constant-column action distributions.

Projection is an implementation primitive, not a standalone VLA policy.
Constant columns still generate their level; they do not hold measured pose.
"""
import torch


def project(actions, active):
    if actions.ndim != 3 or active.shape != (actions.shape[0], actions.shape[2]):
        raise ValueError('Expected [B,H,D] actions and [B,D] activity')
    mean = actions.mean(dim=1, keepdim=True)
    return mean + active[:, None].to(actions.dtype) * (actions-mean)


def labels(actions, atol=0.):
    return (actions.amax(1)-actions.amin(1)) > atol


def conditional_flow_pair(actions, noise, active, time):
    x0 = project(noise, active)
    target = project(actions, active)
    t = time.reshape(-1,1,1)
    return (1-t)*x0+t*target, target-x0


def projected_velocity(raw_velocity, active):
    return project(raw_velocity, active)


def verify():
    torch.manual_seed(17)
    x = torch.randn(4,32,23,dtype=torch.float64,requires_grad=True)
    active = torch.rand(4,23)>.5
    y = project(x, active)
    torch.testing.assert_close(project(y,active), y, rtol=0,atol=1e-14)
    assert ((y.amax(1)-y.amin(1))[~active].abs()<1e-14).all()
    other = torch.randn_like(x)
    torch.testing.assert_close((project(x,active)*other).sum(),(x*project(other,active)).sum())
    xi, velocity = conditional_flow_pair(y, other, active, torch.tensor([.2,.4,.6,.8]))
    torch.testing.assert_close(project(xi,active),xi)
    torch.testing.assert_close(project(velocity,active),velocity)
    # A constant target is preserved by the generator, including its level.
    constant = torch.full((1,32,23),.37,dtype=torch.float64)
    assert not labels(constant).any()
    torch.testing.assert_close(project(constant,labels(constant)),constant)
    (y.square().sum()).backward();assert x.grad.isfinite().all()
    return {'idempotent':True,'self_adjoint':True,'constant_level_preserved':True,
            'flow_path_support_preserved':True,'gradient_finite':True,
            'conditional_dimensions':(23+31*active.sum(1)).tolist(),
            'scope':'Projection algebra only; no mask predictor, policy improvement or novelty proof'}


if __name__=='__main__':
    import json
    from pathlib import Path
    r=verify();Path(__file__).with_name('projection_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

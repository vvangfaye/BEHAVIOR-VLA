"""Verify optimizer equivalence and meta derivative before changing the runner."""
import json
from pathlib import Path
import torch
from core import inner_adam_first_step


torch.set_default_dtype(torch.float64)
theta=torch.tensor([.1,-.2,0.],requires_grad=True)
matrix=torch.tensor([[1.,.2,0.],[.2,2.,0.],[0.,0.,1.]])
def loss(p):return (p['x']@matrix@p['x'])/2
out=inner_adam_first_step({'x':theta},loss,lr=.001,epsilon=1e-6)['x']
reference=theta.detach().clone().requires_grad_()
optimizer=torch.optim.Adam([reference],lr=.001,eps=1e-6)
optimizer.zero_grad();loss({'x':reference}).backward();optimizer.step()
torch.testing.assert_close(out.detach(),reference.detach(),atol=1e-14,rtol=1e-14)
weights=torch.tensor([.3,-.7,.2])
gradient=torch.autograd.grad(out@weights,theta)[0]
finite=[]
for i in range(3):
    # Resolve the optimizer's small epsilon scale around the zero-gradient input.
    delta=torch.zeros_like(theta);delta[i]=1e-10
    def evaluate(x):
        return inner_adam_first_step({'x':x.requires_grad_()},loss,lr=.001,epsilon=1e-6)['x']@weights
    finite.append((evaluate(theta.detach()+delta)-evaluate(theta.detach()-delta))/(2e-10))
finite=torch.stack(finite)
relative=float(((finite-gradient).abs()/gradient.abs().clamp_min(1)).max())
assert relative<2e-4 and torch.isfinite(gradient).all()
result={'passed':True,'adam_forward_matches_torch':True,'max_derivative_relative_error':relative,
        'scope':'Fresh first Adam step only, including zero-gradient coordinate; not a model benefit test'}
Path(__file__).with_name('adam_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

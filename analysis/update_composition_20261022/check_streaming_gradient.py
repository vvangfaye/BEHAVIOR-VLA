"""Ensure memory-bounded gradient composition equals monolithic meta-backprop."""
import json
from pathlib import Path
import torch
from core import inner_sgd
from streaming_gradient import compose_meta_gradient


torch.set_default_dtype(torch.float64)
initial={'x':torch.tensor([.2,-.3,.4],requires_grad=True)}
matrix=torch.tensor([[1.,.2,0.],[.2,1.3,.1],[0.,.1,.7]])
def inner(p):return (p['x']@matrix@p['x'])/2
updated=inner_sgd(initial,inner,lr=.1,steps=2)
old_terms=[torch.sin(initial['x']+t).sum() for t in (.1,.3)]
new_terms=[torch.cos(updated['x']+t).sum() for t in (.2,.4,.6)]
exact=torch.autograd.grad(sum(old_terms)+sum(new_terms),initial['x'],retain_graph=True)[0]
old_leaf=initial['x'].detach().requires_grad_()
new_leaf=updated['x'].detach().requires_grad_()
g_old=torch.zeros_like(old_leaf)
g_new=torch.zeros_like(new_leaf)
for t in (.1,.3):
    g_old+=torch.autograd.grad(torch.sin(old_leaf+t).sum(),old_leaf)[0]
for t in (.2,.4,.6):
    g_new+=torch.autograd.grad(torch.cos(new_leaf+t).sum(),new_leaf)[0]
streamed=compose_meta_gradient(initial,updated,{'x':g_old},{'x':g_new})['x']
error=float((exact-streamed).abs().max())
assert error<1e-12
result={'passed':True,'max_error':error,'scope':'Algebraic equivalence only, no native-memory measurement'}
Path(__file__).with_name('streaming_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

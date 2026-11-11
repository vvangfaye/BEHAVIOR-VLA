import os
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np, torch, jax, jax.numpy as jnp
torch.manual_seed(3); torch.set_num_threads(2)
x=torch.randn(4,32).bfloat16();w=torch.randn(24,32).bfloat16();b=torch.randn(24).bfloat16()
ref=jax.jit(lambda a,k,c:jnp.dot(a,k.T)+c)(*[jnp.asarray(t.float().numpy(),dtype=jnp.bfloat16) for t in (x,w,b)])
r=torch.from_numpy(np.asarray(ref,dtype=np.float32).copy())
for name,v in [('fused',torch.nn.functional.linear(x,w,b)),('separate',torch.nn.functional.linear(x,w)+b)]:
 e=(v.float()-r).abs();print(name,float(e.mean()),float(e.max()),int((e>0).sum()))

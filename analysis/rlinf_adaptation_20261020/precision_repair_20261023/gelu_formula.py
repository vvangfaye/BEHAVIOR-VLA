import os
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np, torch, jax, jax.numpy as jnp
from flax import linen

torch.manual_seed(31); torch.set_num_threads(2)
x=torch.randn(4096).bfloat16();y=torch.randn_like(x)
jx,jy=[jnp.asarray(t.float().numpy(),dtype=jnp.bfloat16) for t in (x,y)]
ref=torch.from_numpy(np.asarray(jax.jit(lambda a,b:linen.gelu(a)*b)(jx,jy),dtype=np.float32).copy())
for cube in ['pow','mul']:
 for scalar_mode in ['sqrt','all']:
  k=torch.tensor(np.sqrt(2/np.pi),dtype=x.dtype)
  q=torch.tensor(.044715,dtype=x.dtype) if scalar_mode=='all' else .044715
  z=x**3 if cube=='pow' else (x*x)*x
  out=(x*(.5*(1+torch.tanh(k*(x+q*z)))))*y
  err=(out.float()-ref).abs();print(cube,scalar_mode,float(err.mean()),float(err.max()),int((err>0).sum()))

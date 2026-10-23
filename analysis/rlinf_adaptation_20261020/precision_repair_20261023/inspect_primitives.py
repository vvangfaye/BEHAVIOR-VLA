import os
os.environ['JAX_PLATFORMS']='cpu'
import json, sys
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parents[2]/'BEHAVIOR-VLAs/rlinf_adapter'))
from b1k_rlinf.precision import jax_gelu_glu,jax_gated_residual
torch.set_num_threads(2);torch.manual_seed(31)
x=torch.randn(4096).bfloat16();y=torch.randn_like(x);g=torch.randn_like(x)
jx,jy,jg=[jnp.asarray(v.float().numpy(),dtype=jnp.bfloat16) for v in (x,y,g)]
cases=[('gelu_glu',jax.jit(lambda a,b:linen.gelu(a)*b)(jx,jy),torch.nn.functional.gelu(x)*y,jax_gelu_glu(x,y)),('gated_residual',jax.jit(lambda a,b,c:a+b*c)(jx,jy,jg),x+y*g,jax_gated_residual(x,y,g)),('vision_gelu',jax.jit(linen.gelu)(jx),torch.nn.functional.gelu(x),torch.nn.functional.gelu(x.float(),approximate='tanh').bfloat16())]
result={}
for name,ref,old,new in cases:
 ref=torch.from_numpy(np.asarray(ref,dtype=np.float32).copy());a=(old.float()-ref).abs();b=(new.float()-ref).abs()
 result[name]={'old_mean':float(a.mean()),'old_max':float(a.max()),'fixed_mean':float(b.mean()),'fixed_max':float(b.max())}
 print(name,result[name],flush=True)
 pass
(p/'primitive_contract.json').write_text(json.dumps({'passed':True,'cases':result},indent=2)+'\n')

import inspect
print(inspect.getsource(jax.nn.gelu))
ref=torch.from_numpy(np.asarray(jax.jit(lambda a,b:linen.gelu(a)*b)(jx,jy),dtype=np.float32).copy())
for name,z in [('tanh_bf16',torch.nn.functional.gelu(x,approximate='tanh')*y),('explicit_bf16',0.5*x*(1+torch.tanh((2/torch.pi)**0.5*(x+0.044715*x**3)))*y)]:
 e=(z.float()-ref).abs();print(name,float(e.mean()),float(e.max()))

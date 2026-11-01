import json
import os
import time
import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('gloo')
assert dist.get_world_size()==4
device=torch.device('cuda',rank)
value=torch.tensor(float(rank+1),device=device)
temporary=value.cpu();dist.all_reduce(temporary);value.copy_(temporary.to(device))
assert value.item()==10
# Match the actual reduction direction: locally normalize by GLOBAL examples,
# then SUM gradients, including unequal numbers of examples on each rank.
weight=torch.tensor([.4,-.2,.7],device=device,requires_grad=True)
x=torch.arange(51,device=device,dtype=torch.float32).reshape(17,3)/51
y=torch.arange(17,device=device,dtype=torch.float32)/17
loss=((x[rank::4]@weight-y[rank::4])**2).sum()/17
loss.backward();temporary=weight.grad.cpu();dist.all_reduce(temporary);weight.grad.copy_(temporary.to(device))
reference=weight.detach().clone().requires_grad_()
((x@reference-y)**2).mean().backward()
torch.testing.assert_close(weight.grad,reference.grad,atol=1e-6,rtol=1e-6)
gradient=weight.grad.detach().clone()
buffer=torch.ones(16*1024**2,device=device)
dist.barrier();torch.cuda.synchronize();start=time.time()
for _ in range(3):
    temporary=buffer.cpu();dist.all_reduce(temporary);buffer.copy_(temporary.to(device))
torch.cuda.synchronize();elapsed=time.time()-start
detail={'rank':rank,'device':torch.cuda.get_device_name(rank),'memory_bytes':torch.cuda.get_device_properties(rank).total_memory,
        'gradient':gradient.tolist(),'three_64MiB_reductions_seconds':elapsed}
details=[None]*4;dist.all_gather_object(details,detail)
if rank==0:
    print('B1K_COMM_RESULT '+json.dumps({'passed':True,'backend':'gloo_cpu_reduction','world_size':4,'ranks':details}),flush=True)
dist.destroy_process_group()

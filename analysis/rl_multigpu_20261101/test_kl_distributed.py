import tempfile
from pathlib import Path
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parallel_training import Parallel
from kl_batch import full_kl
from test_kl_batch import BatchedToy

def worker(rank,root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rdzv'),rank=rank,world_size=4)
    model=BatchedToy();rows=[]
    for i in range(23):
        rows.append({'forward_inputs':{'x':torch.tensor([[i/10.]])},'prev_logprobs':torch.zeros(1,2,23)})
    p=Parallel('cpu');ref=full_kl(model,rows,p,batch_size=1)
    for size in [4,8]:
        seen=[];actual=full_kl(model,rows,p,batch_size=size,observe=lambda ids,*_:seen.extend(ids))
        assert seen==list(range(rank,len(rows),4))
        assert abs(actual['mean']-ref['mean'])<1e-14 and actual['max_abs_log_ratio']==ref['max_abs_log_ratio']
    dist.destroy_process_group()

class DistributedKL(unittest.TestCase):
    def test_shards_and_remainders(self):
        with tempfile.TemporaryDirectory() as root:mp.spawn(worker,args=(root,),nprocs=4,join=True)

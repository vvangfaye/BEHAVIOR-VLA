import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from kl_batch import full_kl, batches
from parallel_training import Parallel, Checkpoints
import test_parallel

class BatchedToy(test_parallel.Toy):
    def default_forward(self, inputs, compute_values=True):
        x = inputs['x']
        return {'logprobs': (self.actor*x).reshape(len(x),1,1).expand(-1,2,23)/23}

class KLBatchTests(unittest.TestCase):
    def test_full_coverage_and_uneven_tail(self):
        model=BatchedToy(); rows=[]
        for i in range(23):
            x=torch.tensor([[i/10.]])
            rows.append({'forward_inputs':{'x':x}, 'prev_logprobs':torch.zeros(1,2,23)})
        ref=full_kl(model,rows,Parallel('cpu'),batch_size=1)
        for size in [4,8]:
            visited=[]
            actual=full_kl(model,rows,Parallel('cpu'),batch_size=size,
                observe=lambda ids,*_:visited.extend(ids))
            self.assertEqual(visited,list(range(23)))
            self.assertEqual(actual,ref)

    def test_shape_changes_do_not_pad(self):
        rows=[{'forward_inputs':{'x':torch.zeros(1,n)}} for n in [2,2,3,3,2]]
        self.assertEqual(list(batches(rows,range(5),4)),[[0,1],[2,3],[4]])

    def test_migration_requires_exact_snapshot_and_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); directory=root/'resume/step_05_test';directory.mkdir(parents=True)
            old={'variant':'control','runtime_signature':'old','optimizer':{'lr':1e-7}}
            metadata={'identity':old,'relative_path':'resume/step_05_test','state':{'accepted':5}}
            (directory/'resume.json').write_text(json.dumps(metadata));(root/'resume/latest.json').write_text(json.dumps(metadata))
            new={**old,'runtime_signature':'new'}
            manager=Checkpoints(root,new,Parallel('cpu'),resume_predecessor={'metadata':metadata,'files':{}})
            self.assertEqual(manager.latest()[1],metadata)
            with self.assertRaises(AssertionError):
                Checkpoints(root,{**new,'optimizer':{'lr':1e-4}},Parallel('cpu'),resume_predecessor={'metadata':metadata,'files':{}}).latest()
            with self.assertRaises(AssertionError):
                Checkpoints(root,new,Parallel('cpu')).latest()

if __name__=='__main__':unittest.main()

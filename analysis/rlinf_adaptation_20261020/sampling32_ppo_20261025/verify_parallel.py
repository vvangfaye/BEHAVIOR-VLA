"""Verify 32-episode partitioning, policy identity, reuse and evaluation swaps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch
import parallel
from common import write


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))
    @property
    def device(self):
        return self.weight.device
    def default_forward(self, inputs):
        return {'logprobs': inputs['x']*self.weight, 'values': inputs['x']*self.weight}


def episode(index):
    x = torch.tensor([index+.5])
    return {'transitions': [{'forward_inputs': {'x': x}, 'prev_logprobs': x.clone(),
             'prev_values': x.clone(), 'reward': 1., 'terminal': True, 'done': True}],
            'bootstrap': torch.zeros(1), 'instance': index % 4 + 1, 'seed': 2000+index,
            'status': {'steps': 16, 'score': 1., 'success': True}}


def worker(path, rank):
    plan = json.loads(Path(path).read_text())
    def collect(model, plan, round_index, index, publish):
        target = Path(plan['output'])/'training/round_01'/f'episode_{index+1:02d}'
        target.mkdir(parents=True)
        write(target/'called.json', {'rank': rank, 'time': time.time()})
        data = episode(index)
        torch.save(data, target/'rollout.pt');write(target/'result.json', data['status'])
        return data
    parallel.collect = collect
    model = Model()
    parallel.collect_partition(model, plan, rank, 1, parallel.trainable_hash(model), lambda **kw: None)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); model = Model()
        source = root/'prior/episode_01';source.mkdir(parents=True)
        torch.save(episode(0), source/'rollout.pt');write(source/'result.json',episode(0)['status'])
        receipt = {'episode': 1, 'policy_hash': parallel.trainable_hash(model),
                   'rollout_path': str(source/'rollout.pt'), 'sha256': parallel.sha256(source/'rollout.pt')}
        plan = {'output': str(root/'out'), 'compute_timeout_seconds': 30,
                'train_instances': [1,2,3,4]*8, 'train_seeds': [list(range(2000,2032))],
                'protocol': {'episodes_per_round': 32},
                'reuse': {'round1_episode_indices': [0], 'round1_receipts': [receipt]}}
        write(root/'plan.json',plan)
        children=[subprocess.Popen([sys.executable,__file__,'--worker',str(root/'plan.json'),str(rank)]) for rank in (0,1)]
        assert all(p.wait(timeout=30)==0 for p in children)
        data=parallel.gather_episodes(plan,1,parallel.trainable_hash(model))
        assert len(data)==32 and [e['seed'] for e in data]==list(range(2000,2032))
        called=list((root/'out').glob('training/round_01/episode_*/called.json'))
        assert len(called)==31 and not (root/'out/training/round_01/episode_01/rollout.pt').exists()
        assert len({json.loads(p.read_text())['rank'] for p in called})==2
        model.weight.data.fill_(2.)
        try: parallel.recover_one(model,plan,1,0)
        except AssertionError: pass
        else: raise AssertionError('Stale collection policy accepted')
        try: parallel.collect_partition(model,plan,0,1,receipt['policy_hash'],lambda **kw:None)
        except AssertionError: pass
        else: raise AssertionError('Stale worker accepted')
        write(parallel.coordination(plan)/'failure_rank1.json',{'error':'test'})
        try: parallel.wait_json(root/'never.json',plan,timeout=1)
        except RuntimeError: pass
        else: raise AssertionError('Peer failure ignored')

        # Every newly evaluated baseline uses original weights, and every final
        # case uses the final weights. A completed baseline must not be repeated.
        original={'weight':torch.tensor([1.])}; final={'weight':torch.tensor([2.])}
        weights=root/'weights';weights.mkdir();torch.save(final,weights/'trainable.pt')
        output=root/'eval';write(output/'baseline/i301_s0/result.json',{'success':True})
        eplan={'output':str(output),'expected_original_trainable_sha256':parallel.hash_parameters(original.items()),
               'evaluation':{'cases':[[301,0],[302,0],[303,0],[304,0]]}}
        calls=[]
        def evaluate(model,plan,phase,publish,**kwargs):
            calls.append((phase,plan['evaluation']['cases'][0],float(model.weight)))
            return [{'score':1.,'success':True}]
        saved=parallel.evaluate;parallel.evaluate=evaluate
        manifest={'directory':str(weights),'trainable_sha256':parallel.hash_parameters(final.items())}
        parallel.evaluate_partition(model,original,eplan,0,manifest,lambda **kw:None)
        parallel.evaluate=saved
        assert calls==[('final',[301,0],2.),('baseline',[303,0],1.),('final',[303,0],2.)]
        receipt={'passed':True,'two_process_32_episode_partition':True,'completed_data_reused_without_copy':True,
                 'stale_policy_rejected':True,'peer_failure_detected':True,'evaluation_weights_correct':True,
                 'completed_baseline_not_rerun':True}
        Path(__file__).with_name('parallel_test_result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))


if __name__=='__main__':
    if len(sys.argv)>1:
        worker(sys.argv[2],int(sys.argv[3]))
    else:
        main()

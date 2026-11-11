"""Exercise two real tiny-model PPO rounds and exact optimizer resume on CPU."""
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TESTS=ROOT/'BEHAVIOR-VLAs/rlinf_adapter/tests'
sys.path.insert(0,str(TESTS))
spec=importlib.util.spec_from_file_location('adapter_fixture_setup',TESTS/'conftest.py')
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
import torch
from test_model import model as make_model, observation, pack_replay
from rlinf.models.embodiment.openpi.modules.model import Observation
from training import build_optimizer, hash_parameters, load_warm_start, optimizer_to, save_checkpoint, update_policy


class TempFactory:
    def __init__(self,root):self.root=root
    def mktemp(self,name):
        p=self.root/name;p.mkdir();return p


def batch(network, seed):
    random.seed(seed)
    def first(value):
        if isinstance(value,dict):return {k:first(v) for k,v in value.items()}
        return value[:1] if torch.is_tensor(value) else value
    obs=Observation.from_dict(first(observation(network).to_dict()))
    rows=[]
    for i in range(5):
        with torch.no_grad():
            _, result=network._predict_train(obs,noise=None,rng=torch.Generator().manual_seed(seed+i),compute_values=True)
        pack_replay(obs,result)
        row={k:{n:v.detach().cpu().clone() for n,v in value.items()} if isinstance(value,dict) else value.detach().cpu().clone()
             for k,value in result.items() if k!='model_actions'}
        row.update(reward=float(i==4),terminal=(i==4),done=(i==4));rows.append(row)
    return rows


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        norm=fixtures.norm_path.__wrapped__(root)
        tokenizer=fixtures.tokenizer_path.__wrapped__(TempFactory(root))
        network=make_model.__wrapped__(norm,tokenizer)
        plan=json.loads((HERE/'plan.json').read_text())
        opt=build_optimizer(network,plan)
        frozen=lambda model:hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        trainable=lambda model:hash_parameters((n,p) for n,p in model.named_parameters() if p.requires_grad)
        original_frozen=frozen(network)
        first,_,_=update_policy(network,opt,batch(network,20),torch.zeros(1),plan)
        assert first['optimizer_steps']==2 and frozen(network)==original_frozen
        optimizer_to(opt,'cpu')
        saved=save_checkpoint(network,opt,root/'checkpoint')
        resumed=make_model.__wrapped__(norm,tokenizer)
        resumed_opt=build_optimizer(resumed,plan)
        plan['warm_start']={'directory':str(root/'checkpoint'),'expected_trainable_sha256':saved['trainable_sha256']}
        load_warm_start(resumed,resumed_opt,plan)
        assert trainable(resumed)==trainable(network)
        assert all(int(s['step'])==2 for s in resumed_opt.state.values() if 'step' in s)
        second,_,_=update_policy(network,opt,batch(network,40),torch.zeros(1),plan)
        restarted,_,_=update_policy(resumed,resumed_opt,batch(resumed,40),torch.zeros(1),plan)
        assert trainable(resumed)==trainable(network), 'Resume changed the next PPO update'
        assert frozen(network)==frozen(resumed)==original_frozen
        assert all(int(s['step'])==4 for s in resumed_opt.state.values() if 'step' in s)
        from experiment import compare
        evaluation_plan = {**plan, 'output': str(root/'evaluation')}
        left=[]; right=[]
        for index, (instance, seed) in enumerate(plan['evaluation']['cases']):
            a={'instance':instance,'seed':seed,'score':0.,'success':False,'steps':3224}
            b={**a,'score':float(index==0),'success':index==0}
            left.append(a);right.append(b)
            initial={'physical_state':{'object':[[0,0,0],[0,0,0,1]]},'physics_dt':1/120,'rendering_dt':1/30,'observation_dt':1/30}
            for phase in ('baseline','final'):
                target=Path(evaluation_plan['output'])/phase/f'i{instance}_s{seed}'/'initial.json'
                target.parent.mkdir(parents=True);target.write_text(json.dumps(initial))
        comparison=compare(evaluation_plan,left,right)
        assert comparison['comparison_valid'] and comparison['failure_to_success']==1
        bad=Path(evaluation_plan['output'])/'final/i301_s0/initial.json'
        value=json.loads(bad.read_text());value['physical_state']['object'][0][0]=1.;bad.write_text(json.dumps(value))
        assert not compare(evaluation_plan,left,right)['comparison_valid']
        report={'passed':True,'real_tiny_model':True,'real_upstream_ppo':True,
                'two_rounds_exact_resume':True,'frozen_weights_unchanged':True,
                'optimizer_step_count_preserved':True,'mispaired_evaluation_rejected':True,'first':first,'second':second}
        (HERE/'training_test_result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('first','second')}))


if __name__=='__main__':main()

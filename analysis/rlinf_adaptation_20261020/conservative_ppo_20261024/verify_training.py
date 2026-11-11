"""Real small-model checks for the new batch, critic-only, and KL rollback contracts."""
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
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
import torch
from test_model import model as make_model, observation, pack_replay
from rlinf.models.embodiment.openpi.modules.model import Observation
from training import (actor_parameters,batch_advantages,build_optimizer,discounted_returns,
                      fit_critic_features,hash_parameters,prefix_features,update_policy)

class TempFactory:
    def __init__(self,root):self.root=root
    def mktemp(self,name):
        p=self.root/name;p.mkdir();return p

def episode(network,seed,reward):
    random.seed(seed)
    def first(value):
        if isinstance(value,dict):return {k:first(v) for k,v in value.items()}
        return value[:1] if torch.is_tensor(value) else value
    obs=Observation.from_dict(first(observation(network).to_dict()));rows=[]
    for i in range(4):
        with torch.no_grad():
            _,result=network._predict_train(obs,noise=None,rng=torch.Generator().manual_seed(seed+i),compute_values=True)
        pack_replay(obs,result)
        row={k:{n:v.detach().cpu().clone() for n,v in value.items()} if isinstance(value,dict) else value.detach().cpu().clone()
             for k,value in result.items() if k!='model_actions'}
        row.update(reward=reward if i==3 else 0.,terminal=i==3,done=i==3);rows.append(row)
    return {'transitions':rows,'bootstrap':torch.zeros(1)}

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);norm=fixtures.norm_path.__wrapped__(root)
        tokenizer=fixtures.tokenizer_path.__wrapped__(TempFactory(root))
        network=make_model.__wrapped__(norm,tokenizer)
        plan=json.loads((HERE/'plan.json').read_text())
        plan['optimizer'].update(accumulate_chunks=4,kl_target=10.)
        actor_hash=hash_parameters(actor_parameters(network))
        batch=[episode(network,20,1.),episode(network,30,0.)]
        features=torch.cat([prefix_features(network,x['forward_inputs']) for e in batch for x in e['transitions']])
        targets=torch.tensor([2.]*len(features))
        warm=dict(plan['critic_warmup']);warm.update(epochs=60,learning_rate=.003,batch_size=8)
        report=fit_critic_features(network,features,targets,warm)
        assert report['after']['validation_mse']<report['before']['validation_mse']
        assert hash_parameters(actor_parameters(network))==actor_hash
        batch=[episode(network,40,1.),episode(network,50,0.)]
        rows,adv,returns,raw=batch_advantages(batch,plan['optimizer'])
        assert abs(float(adv.mean()))<1e-5
        # Returns for the first episode cannot depend on the second episode.
        altered=[batch[0],episode(network,50,10.)]
        _,_,r2,_=batch_advantages(altered,plan['optimizer'])
        torch.testing.assert_close(returns[:4],r2[:4],rtol=0,atol=0)
        assert discounted_returns([{'reward':1.,'terminal':False},{'reward':2.,'terminal':True}],.5,100)==[2.,2.]
        opt=build_optimizer(network,plan)
        before=hash_parameters(actor_parameters(network))
        result,_,_=update_policy(network,opt,batch,plan)
        assert result['optimizer_steps']==2 and result['order']!=list(range(8))
        assert result['expert_gradient_nonzero'] and result['value_gradient_nonzero']
        assert hash_parameters(actor_parameters(network))!=before
        # A violating first step must restore both parameters and empty Adam state.
        batch=[episode(network,60,1.),episode(network,70,0.)]
        plan['optimizer'].update(actor_lr=.01,value_lr=.01,kl_target=0.)
        opt=build_optimizer(network,plan)
        before=hash_parameters((n,p) for n,p in network.named_parameters() if p.requires_grad)
        result,_,_=update_policy(network,opt,batch,plan)
        assert result['early_stopped'] and result['optimizer_steps']==0
        assert len(opt.state)==0
        assert hash_parameters((n,p) for n,p in network.named_parameters() if p.requires_grad)==before
        assert result['final_batch_kl']['mean']==0.
        from experiment import evaluate,start_simulator
        caseplan={**plan,'output':str(root/'reuse'),'evaluation':{**plan['evaluation'],'cases':[[301,0]]}}
        case=root/'reuse'/'previous_final'/'i301_s0';case.mkdir(parents=True)
        (case/'result.json').write_text(json.dumps({'instance':301,'seed':0,'steps':3,'score':1.,'success':True}))
        import experiment
        original=experiment.Simulator
        def forbidden(*args,**kwargs):raise AssertionError('Completed case rerun')
        experiment.Simulator=forbidden
        assert evaluate(network,caseplan,'previous_final',lambda **kw:None,reuse_existing=True)[0]['success']
        experiment.Simulator=original
        receipt={'passed':True,'real_tiny_model':True,'critic_only_preserves_actor':True,
                 'critic_holdout_loss_improved':True,'episode_boundaries_preserved':True,
                 'global_batch_normalization':True,'shuffled_updates':True,'real_actor_gradient':True,
                 'kl_violation_rolls_back_parameters_and_optimizer':True,'completed_eval_skipped':True}
        (HERE/'training_test_result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))

if __name__=='__main__':main()

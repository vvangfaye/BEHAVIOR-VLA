"""Exercise the two-worker recovery and real Adam state on a tiny CPU model."""
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile

import torch

import parallel
from common import write
from training import build_optimizer, hash_parameters, save_checkpoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))
        self.value_head = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)

    def default_forward(self, inputs):
        value = inputs['x'] * self.weight
        return {'logprobs':value, 'values':value}

    @property
    def device(self):
        return self.weight.device


def advance(model, optimizer, count):
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        loss = model.weight.square().sum() + model.value_head(torch.ones(1, 1)).square().sum()
        loss.backward(); optimizer.step()


def worker(rank, plan, original):
    os.environ['B1K_RANK'] = str(rank)
    parallel.torch_model = lambda plan: copy.deepcopy(original)
    def collect(model, plan, round_index, index, publish):
        assert round_index == 2
        path=Path(plan['output'])/'training/round_02'/f'episode_{index+1:02d}'
        path.mkdir(parents=True)
        data={'instance':plan['train_instances'][index], 'seed':plan['train_seeds'][1][index],
              'transitions':[{}], 'status':{'success':True,'score':1.,'steps':16}}
        torch.save(data,path/'rollout.pt')
        write(path/'called.json',{'rank':rank,'round':round_index})
        return data
    parallel.collect=collect
    def update(model, optimizer, episodes, plan, round_index, publish):
        assert round_index==2 and len(episodes)==32
        assert {int(s['step']) for s in optimizer.state.values()}=={17}
        assert [g['lr'] for g in optimizer.param_groups]==[1e-7,1e-4]
        advance(model,optimizer,13)
        write(Path(plan['output'])/'actual_update.json',{'round':2,'steps':13})
        return {'optimizer_steps':13},None,None
    parallel.update_policy=update
    parallel.compare=lambda *args: {'comparison_valid':True}
    parallel.run(plan,lambda **fields: None)


def main():
    torch.manual_seed(0)
    original=Model()
    original_hash=parallel.trainable_hash(original)
    frozen_hash=hash_parameters((n,p) for n,p in original.named_parameters() if not p.requires_grad)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);source=root/'prior';output=root/'resumed';output.mkdir()
        config={'actor_lr':1e-7,'value_lr':1e-4,'weight_decay':.01,'updates_per_round':[17,13]}
        trained=copy.deepcopy(original); optimizer=build_optimizer(trained,{'optimizer':config})
        advance(trained,optimizer,17)
        saved=save_checkpoint(trained,optimizer,source/'checkpoints/round_01')
        old_result={'update':{'optimizer_steps':17},'checkpoint':saved,'rollouts':[]}
        write(source/'training/round_01/result.json',old_result)
        for index in range(32):
            write(source/'coordination'/f'round_01_episode_{index+1:02d}.json',
                  {'round':1,'episode':index+1,'status':{'success':True},'reused':False})
        source_optimizer=source/'checkpoints/round_01/optimizer.pt'
        optimizer_hash=parallel.sha256(source_optimizer)
        cached_path=source/'training/round_02/episode_02'
        cached_path.mkdir(parents=True)
        value=trained.weight.detach().clone()
        cached={'instance':2,'seed':101,'transitions':[{'forward_inputs':{'x':torch.ones(1)},
                'prev_logprobs':value.clone(),'prev_values':value.clone()}],
                'status':{'success':True,'score':1.,'steps':16}}
        torch.save(cached,cached_path/'rollout.pt');write(cached_path/'result.json',cached['status'])
        cached_receipt={'episode':2,'policy_hash':parallel.trainable_hash(trained),
            'rollout_path':str(cached_path/'rollout.pt'),'sha256':parallel.sha256(cached_path/'rollout.pt')}
        expected=copy.deepcopy(trained)
        expected_optimizer=build_optimizer(expected,{'optimizer':config})
        expected_optimizer.load_state_dict(optimizer.state_dict())
        advance(expected,expected_optimizer,13)
        plan={'output':str(output),'gpu_count':2,'model_seed':0,'compute_timeout_seconds':40,
              'action_interface':{'gripper_open_tolerance':0},'optimizer':config,
              'expected_original_trainable_sha256':original_hash,
              'train_instances':[1,2,3,4]*8,'train_seeds':[list(range(32)),list(range(100,132))],
              'protocol':{'episodes_per_round':32},'eval_policy':'test',
              'runtime_storage':{'max_persistent_bytes':100*1024**2,'max_checkpoint_write_reservation_bytes':0},
              'reuse':{'frozen_hash':frozen_hash,'baseline_cases':[],
                       'baseline_checks':{'weights_unchanged':True,'trainable_sha_before':original_hash}},
              'evaluation':{'cases':[]},
              'resume':{'completed_round':1,'source_output':str(source),'round2_receipts':[cached_receipt],
                        'checkpoint_directory':str(source/'checkpoints/round_01'),
                        'expected_trainable_sha256':saved['trainable_sha256'],
                        'source_result_sha256':parallel.sha256(source/'training/round_01/result.json')},
              'warm_start':{'directory':str(source/'checkpoints/round_01'),
                            'expected_trainable_sha256':saved['trainable_sha256']}}
        ctx=multiprocessing.get_context('fork')
        children=[ctx.Process(target=worker,args=(rank,plan,original)) for rank in (0,1)]
        for p in children:p.start()
        for p in children:
            p.join(45)
            if p.is_alive():p.kill();p.join();raise AssertionError('Recovery test hung')
            assert p.exitcode==0, p.exitcode
        summary=json.loads((output/'training_summary.json').read_text())
        assert summary['total_optimizer_steps']==30 and summary['episodes']==64
        assert not list((output/'training/round_01').glob('episode_*/called.json'))
        calls=list((output/'training/round_02').glob('episode_*/called.json'))
        assert len(calls)==31 and {json.loads(p.read_text())['rank'] for p in calls}=={0,1}
        assert not (output/'training/round_02/episode_02/called.json').exists()
        assert (output/'training/round_02/episode_02/recovery.json').exists()
        assert parallel.sha256(source_optimizer)==optimizer_hash
        final=torch.load(output/'checkpoints/round_02/trainable.pt',weights_only=True)
        for name,param in expected.named_parameters():
            if param.requires_grad:assert torch.equal(final[name],param),name
        assert json.loads((output/'optimizer_restore.json').read_text())['step_counters']==[17]
        assert json.loads((output/'actual_update.json').read_text())=={'round':2,'steps':13}
    result={'passed':True,'first_round_not_repeated':True,'second_round_two_workers_32_episodes':True,'late_completed_episode_reused_with_replay_check':True,
            'exactly_31_new_rollouts':True,
            'exact_adam_continuation_matches_uninterrupted_30_steps':True,
            'source_optimizer_preserved':True,'learning_rates_unchanged':True}
    Path(__file__).with_name('resume_test_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

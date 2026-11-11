"""CPU contract tests with synthetic data, never B1K-success claims."""
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_world_model import (Transform, WorldModel, binary_metrics, load_data,
                              loss_terms, run, tensors, windows)


def synthetic(path, seed=11):
    rng=np.random.default_rng(seed)
    records=[]
    basis=rng.normal(size=(23,576)).astype(np.float32)*.015
    for episode in range(18):
        split=0 if episode<10 else 1 if episode<14 else 2
        instance=1+episode if split==0 else 11+(episode-10)//2 if split==1 else 14+(episode-14)//2
        state=rng.normal(size=23).astype(np.float32)*.2
        for chunk in range(20):
            action=rng.normal(size=(16,23)).astype(np.float32)
            # Each future state strongly depends on a fresh, unpredictable action.
            next_state=.7*state+.7*action.mean(0)
            goals=(state[:3]>0).astype(np.float32)
            next_goals=(next_state[:3]>0).astype(np.float32)
            terminal=chunk==19
            records.append(dict(obs_state=state.copy(),next_obs_state=next_state.copy() if not terminal else np.full(23,np.nan,np.float32),
                next_obs_valid=not terminal,execution_actions=action,action_mask=np.ones(16,bool),
                visual_features=.5+state@basis,visual_valid=True,
                next_visual_features=.5+next_state@basis if not terminal else np.full(576,np.nan,np.float32),
                next_visual_valid=not terminal,goals=goals,next_goals=next_goals,
                goal_valid=np.ones(3,bool),next_goal_valid=np.ones(3,bool),
                reward=np.float32(action[:,0].mean()),split=split,episode_index=episode,
                chunk_index=chunk,instance=instance,done=terminal))
            state=next_state
    d={k:np.asarray([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(path,**d)
    return d


def test_masks_and_train_only_preprocessing(tmp_path):
    torch.set_num_threads(2)
    path=tmp_path/'data.npz'
    d=synthetic(path)
    d=load_data(path)
    transform=Transform(d,8)
    perturbed={k:v.copy() for k,v in d.items()}
    heldout=d['split']!=0
    perturbed['obs_state'][heldout]+=1000
    perturbed['visual_features'][heldout]+=1000
    other=Transform(perturbed,8)
    for key in vars(transform):
        np.testing.assert_array_equal(getattr(transform,key),getattr(other,key))
    table=tensors(d,transform)
    assert all(torch.isfinite(t).all() for t in table.values())
    model=WorldModel(3,8,hidden=32)
    loss=sum(loss_terms(model,table).values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_masked_actions_and_terminal_reward(tmp_path):
    d=synthetic(tmp_path/'data.npz')
    d['action_mask'][0,8:]=False
    transform=Transform(d,8)
    a=tensors(d,transform)
    d['execution_actions'][0,8:]=1e8
    b=tensors(d,transform)
    torch.testing.assert_close(a['actions'],b['actions'])
    model=WorldModel(3,8,hidden=32)
    terminal=np.flatnonzero(~d['next_obs_valid'])
    part={k:v[terminal] for k,v in a.items()}
    terms=loss_terms(model,part)
    assert terms['state']==0 and terms['visual']==0
    assert terms['reward']>0


def test_episode_windows_and_split_rejection(tmp_path):
    path=tmp_path/'data.npz'
    d=synthetic(path)
    for split in range(3):
        w=windows(d,split,4)
        assert np.all(d['split'][w]==split)
        assert np.all(d['episode_index'][w]==d['episode_index'][w[:,:1]])
        assert np.all(np.diff(d['chunk_index'][w],axis=1)==1)
        assert np.all(d['next_obs_valid'][w])
    d['split'][0]=1
    np.savez_compressed(path,**d)
    with pytest.raises(ValueError,match='instance'):
        load_data(path)


def test_binary_metrics_ties():
    perfect=binary_metrics([0,0,1,1],[.1,.2,.8,.9])
    assert perfect['auroc']==1 and perfect['average_precision']==1
    tied=binary_metrics([0,1,0,1],[.3,.3,.3,.3])
    assert tied['auroc']==.5 and tied['average_precision']==.5
    assert binary_metrics([0,0],[.1,.2])['auroc'] is None


def test_matched_architecture_and_action_ablation():
    torch.manual_seed(7)
    full=WorldModel(3,8,use_action=True,hidden=32)
    torch.manual_seed(7)
    no_action=WorldModel(3,8,use_action=False,hidden=32)
    for name,param in full.state_dict().items():
        torch.testing.assert_close(param,no_action.state_dict()[name])
    batch={'state':torch.randn(4,23),'goals':torch.zeros(4,3),'goal_valid':torch.ones(4,3),
        'visual':torch.randn(4,8),'visual_valid':torch.ones(4,1),
        'actions':torch.randn(4,368),'action_mask':torch.ones(4,16)}
    changed={**batch,'actions':batch['actions']+100,'action_mask':torch.zeros(4,16)}
    for key,value in no_action(batch).items():
        torch.testing.assert_close(value,no_action(changed)[key])


def test_partial_terminal_lengths_excluded(tmp_path):
    path=tmp_path/'data.npz'
    d=synthetic(path)
    terminal=np.flatnonzero(d['done'])
    d['action_mask'][terminal,8:]=False
    np.savez_compressed(path,**d)
    loaded=load_data(path)
    assert int((~loaded['_full_chunk']).sum())==18
    for split in range(3):
        for h in (1,2,4):
            seq=windows(loaded,split,h)
            assert loaded['_full_chunk'][seq].all()
    baseline=Transform(loaded,8)
    loaded['obs_state'][terminal]+=1000
    loaded['visual_features'][terminal]+=1000
    changed=Transform(loaded,8)
    for key in vars(baseline):
        np.testing.assert_array_equal(getattr(baseline,key),getattr(changed,key))


def test_complete_training_contract(tmp_path):
    path=tmp_path/'synthetic.npz'
    synthetic(path)
    args=Namespace(data=path,output=tmp_path/'out',epochs=2,threads=2,seed=7,
        batch_size=64,hidden=32,lr=1e-3,visual_dim=8,ensemble=1,include_no_visual=False)
    result=run(args)
    assert set(result['models'])=={'persistence','train_mean_reward','full','no_action'}
    assert result['models']['train_mean_reward']['constant_reward']==result['reward_distribution']['train']['mean']
    for name in ['full','no_action']:
        checkpoint=torch.load(args.output/f'{name}.pt',weights_only=True)
        assert checkpoint['config']['use_action']==(name=='full')
        assert result['models'][name]['diagnostics']['device']=='cpu'
        assert result['models'][name]['evaluation']['test']['one_step']['rows']==80
    assert result['models']['no_action']['diagnostics']['predicted_reward_action_gradient']['members']==0
    assert result['task_reward_distribution']['available'] is False
    assert result['models']['full']['evaluation']['test']['one_step']['task_reward_nonzero_rows'] is None
    assert (args.output/'results.json').is_file()


def test_task_events_are_not_inferred_from_shaping(tmp_path):
    path=tmp_path/'with_task_reward.npz'
    d=synthetic(path)
    d['task_reward']=np.zeros(len(d['reward']),np.float32)
    d['task_reward'][[50,250,330]]=[5,-5,5]
    np.savez_compressed(path,**d)
    args=Namespace(data=path,output=tmp_path/'out',epochs=1,threads=2,seed=7,
        batch_size=64,hidden=32,lr=1e-3,visual_dim=8,ensemble=1,include_no_visual=False)
    result=run(args)
    distribution=result['task_reward_distribution']
    assert distribution['available'] is True
    assert distribution['train']['positive_count']==1
    assert distribution['validation']['negative_count']==1
    assert distribution['test']['positive_fraction']==1/80
    test=result['models']['persistence']['evaluation']['test']['one_step']
    assert test['reward_nonzero_rows']==80  # synthetic continuous total reward
    assert test['task_reward_nonzero_rows']==1
    assert test['total_reward_mse_on_task_reward_events']==pytest.approx(float(d['reward'][330]**2),rel=1e-6)

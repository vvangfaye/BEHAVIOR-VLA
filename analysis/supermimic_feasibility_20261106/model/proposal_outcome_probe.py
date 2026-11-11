#!/usr/bin/env python3
"""Offline proposal-to-outcome probe; not an executed-action dynamics model.

Uses all rows, including success-truncated proposals. Only the pre-action full
proposal, current observations, and pre-action remaining-time/request mask are
inputs. Actual termination lengths and future status are never current inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_world_model import binary_metrics, load_data, mean_scale


def preaction_context(d,max_steps=3620):
    """Current row's length can affect a later row, never itself or a prior row."""
    n=len(d['split'])
    past=np.zeros(n,np.int64)
    lengths=d.get('executed_steps',d['action_mask'].sum(1)).astype(np.int64)
    for episode in np.unique(d['episode_index']):
        ids=np.flatnonzero(d['episode_index']==episode)
        ids=ids[np.argsort(d['chunk_index'][ids],kind='stable')]
        if not np.array_equal(d['chunk_index'][ids],np.arange(len(ids))):
            raise ValueError('Expected complete episodes beginning with chunk zero')
        past[ids]=np.r_[0,np.cumsum(lengths[ids[:-1]])]
    remaining=max_steps-past
    if (remaining<=0).any():
        raise ValueError('A decision was logged after its fixed time budget')
    mask=np.arange(16)[None,:] < np.minimum(16,remaining)[:,None]
    times=np.stack([past/max_steps,remaining/max_steps],axis=1).astype(np.float32)
    return {'past_steps':past,'remaining_steps':remaining,'request_mask':mask,'time_features':times}


class ProposalTransform:
    def __init__(self,d,context,visual_dim=32):
        train=d['split']==0
        self.state_mean,self.state_scale=mean_scale(d['obs_state'][train])
        masked=d['execution_actions']*context['request_mask'][...,None]
        self.action_mean,self.action_scale=mean_scale(masked[train].reshape(train.sum(),-1))
        self.reward_mean,self.reward_scale=mean_scale(d['reward'][train,None])
        images=d['visual_features'][train&d['visual_valid']]
        if len(images)<2 or visual_dim==0:
            self.visual_mean=np.zeros(576,np.float32)
            self.components=np.empty((576,0),np.float32)
            self.visual_scale=np.empty(0,np.float32)
        else:
            self.visual_mean=images.mean(0).astype(np.float32)
            centered=torch.as_tensor(images-self.visual_mean)
            _,vectors=torch.linalg.eigh(centered.T@centered/max(len(images)-1,1))
            dim=min(visual_dim,len(images)-1,images.shape[1])
            self.components=vectors[:,-dim:].flip(1).numpy().copy()
            self.visual_scale=np.maximum(((images-self.visual_mean)@self.components).std(0),1e-3).astype(np.float32)

    def save(self):
        return {key:torch.from_numpy(value.copy()) for key,value in vars(self).items()}


def prepare_inputs(d,context,t):
    masked=d['execution_actions']*context['request_mask'][...,None]
    normalized=(masked.reshape(len(masked),-1)-t.action_mean)/t.action_scale
    normalized*=np.repeat(context['request_mask'],23,axis=1)
    filled=np.where(d['visual_valid'][:,None],d['visual_features'],t.visual_mean)
    visual=(filled-t.visual_mean)@t.components/t.visual_scale
    fields={'state':(d['obs_state']-t.state_mean)/t.state_scale,'actions':normalized,
        'request_mask':context['request_mask'],'time_features':context['time_features'],
        'visual':visual,'visual_valid':d['visual_valid'][:,None],
        'goals':d['goals'],'goal_valid':d['goal_valid']}
    return {key:torch.as_tensor(np.ascontiguousarray(value,dtype=np.float32)) for key,value in fields.items()}


class ProposalOutcome(nn.Module):
    def __init__(self,goals,visual_dim,hidden=192,use_action=True):
        super().__init__()
        self.use_action,self.goals=use_action,goals
        inputs=23+368+16+2+visual_dim+1+2*goals
        self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,goals+2))

    def forward(self,b):
        action=b['actions'] if self.use_action else torch.zeros_like(b['actions'])
        x=torch.cat([b['state'],b['goals'],b['goal_valid'],b['visual'],b['visual_valid'],
                     b['time_features'],b['request_mask'],action],dim=-1)
        out=self.net(x)
        goal=out[:,:self.goals]+torch.logit(b['goals'].clamp(.05,.95))
        return {'goal_logits':goal,'reward':out[:,self.goals],'done_logits':out[:,self.goals+1]}


def cut(d,ids):
    return {key:value[ids] for key,value in d.items()}


def targets(d,t):
    return {'next_goals':torch.as_tensor(d['next_goals'],dtype=torch.float32),
        'next_goal_valid':torch.as_tensor(d['next_goal_valid'],dtype=torch.float32),
        'current_goals':torch.as_tensor(d['goals'],dtype=torch.float32),
        'reward':torch.as_tensor((d['reward']-t.reward_mean[0])/t.reward_scale[0],dtype=torch.float32),
        'done':torch.as_tensor(d['done'],dtype=torch.float32)}


def losses(model,b,y,done_positive_weight):
    p=model(b)
    event=(y['next_goals']!=y['current_goals']).float()
    weights=y['next_goal_valid']*(1+4*event)
    goal=(F.binary_cross_entropy_with_logits(p['goal_logits'],y['next_goals'],reduction='none')*weights).sum()/weights.sum().clamp_min(1)
    reward=F.mse_loss(p['reward'],y['reward'])
    done=F.binary_cross_entropy_with_logits(p['done_logits'],y['done'],pos_weight=done_positive_weight)
    return {'goal':goal,'total_reward':reward,'done':done}


def train_one(d,b,y,args,name,seed,done_weight):
    torch.manual_seed(seed)
    model=ProposalOutcome(d['goals'].shape[1],b['visual'].shape[1],args.hidden,name=='full')
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    tr,va=np.flatnonzero(d['split']==0),np.flatnonzero(d['split']==1)
    rng=np.random.default_rng(seed)
    best=float('inf');chosen=None;history=[]
    start=time.perf_counter()
    for epoch in range(args.epochs):
        order=rng.permutation(tr)
        total=0
        model.train()
        for i in range(0,len(order),args.batch_size):
            ids=order[i:i+args.batch_size]
            loss=sum(losses(model,cut(b,ids),cut(y,ids),done_weight).values())
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            pieces=losses(model,cut(b,va),cut(y,va),done_weight)
            validation={key:float(value) for key,value in pieces.items()}
            val=sum(validation.values())
        history.append({'epoch':epoch+1,'train_loss':total/len(tr),'validation_loss':val,'validation_parts':validation})
        if val<best:
            best=val;chosen=copy.deepcopy(model.state_dict())
    model.load_state_dict(chosen);model.eval()
    return model,{'seed':seed,'seconds':time.perf_counter()-start,
        'selected_epoch':min(history,key=lambda x:x['validation_loss'])['epoch'],'history':history}


def prediction(models,b,t,ids):
    with torch.no_grad():
        preds=[m(cut(b,ids)) for m in models]
        return {'goals':torch.stack([p['goal_logits'].sigmoid() for p in preds]).mean(0).numpy(),
            'reward':torch.stack([p['reward'] for p in preds]).mean(0).numpy()*t.reward_scale[0]+t.reward_mean[0],
            'done':torch.stack([p['done_logits'].sigmoid() for p in preds]).mean(0).numpy()}


def metrics(p,d,ids):
    valid=d['goal_valid'][ids]&d['next_goal_valid'][ids]
    events=d['goals'][ids]!=d['next_goals'][ids]
    scores=np.where(d['goals'][ids]>.5,1-p['goals'],p['goals'])
    event_accuracy=(p['goals']>.5)==(d['next_goals'][ids]>.5)
    task_events=np.abs(d['task_reward'][ids])>1e-8 if 'task_reward' in d else None
    err=(p['reward']-d['reward'][ids])**2
    per_goal={}
    for i,name in enumerate(d['_goal_names']):
        mask=valid[:,i]
        per_goal[name]={'change_event':binary_metrics(events[mask,i],scores[mask,i]),
            'next_bit':binary_metrics(d['next_goals'][ids][mask,i],p['goals'][mask,i]),
            'changed_entries':int((mask&events[:,i]).sum()),
            'changed_next_bit_accuracy':float(event_accuracy[mask&events[:,i],i].mean()) if (mask&events[:,i]).any() else None}
    terminal=d['done'][ids].astype(bool)
    return {'rows':len(ids),'goal_change_event':binary_metrics(events[valid],scores[valid]),
        'per_goal':per_goal,'total_reward_mse':float(err.mean()),
        'task_reward_event_rows':int(task_events.sum()) if task_events is not None else None,
        'total_reward_mse_on_task_reward_events':float(err[task_events].mean()) if task_events is not None and task_events.any() else None,
        'done':binary_metrics(terminal,p['done']),
        'terminal_rows':int(terminal.sum()),'total_reward_mse_on_terminal':float(err[terminal].mean()) if terminal.any() else None}


def evaluate(models,b,t,d):
    return {name:metrics(prediction(models,b,t,ids:=np.flatnonzero(d['split']==i)),d,ids)
        for i,name in enumerate(['train','validation','test'])}


def run(args):
    torch.set_num_threads(args.threads)
    d=load_data(args.data)
    if 'done' not in d:
        raise ValueError('Exporter done label required')
    context=preaction_context(d,args.max_steps)
    transform=ProposalTransform(d,context,args.visual_dim)
    b=prepare_inputs(d,context,transform);y=targets(d,transform)
    tr=d['split']==0
    weight=min(float((~d['done'][tr]).sum()/max(d['done'][tr].sum(),1)),100.)
    positive_weight=torch.tensor(weight,dtype=torch.float32)
    args.output.mkdir(parents=True,exist_ok=True)
    result={'schema':'b1k_preaction_proposal_outcome_probe_v1','device':'cpu','threads':args.threads,
        'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'rows':len(d['split']),
        'goal_names':d['_goal_names'],'max_steps':args.max_steps,'seeds':[args.seed+i for i in range(args.seeds)],
        'training_configuration':{'epochs':args.epochs,'seeds':args.seeds,'first_seed':args.seed,
            'batch_size':args.batch_size,'hidden':args.hidden,'visual_dim':args.visual_dim,'learning_rate':args.lr},
        'inputs':'Current obs/goals/visual, full pre-action proposal (known timeout request mask applied), current time budget; no current actual length/mask/done or future status.',
        'time_context':'prefix sum of executed steps strictly before current chunk; timeout request length=min(16, max_steps-prefix)',
        'targets':'Next goal bits, total reward including shaping, done; all terminal and partial rows retained.',
        'preprocessing':'All training rows only; validation/test excluded from normalization and PCA.',
        'model_selection':'Best validation total loss independently per seed; no test selection.',
        'goal_loss':'5x weight for changed goal entries; scores uncalibrated.',
        'done_positive_weight':weight,'done_score_interpretation':'Class-weighted sigmoid is not a calibrated termination probability.',
        'split_counts':{},'models':{},'baselines':{},
        'limitations':['This predicts proposal outcomes, not transitions under the full proposed action execution.',
            'A success-stopped proposal has an unexecuted suffix, legally known before action but not evidence for suffix dynamics.',
            'No predicted next physical state, open-loop dynamics or counterfactual policy gradient is evaluated.',
            'An observational action/no-action improvement is not a causal intervention result.',
            'Composite all_goals events overlap physical subgoals; inspect each named goal.',
            'Only one logged policy/collection seed is represented; three network seeds are not independent data collections.']}
    for split,name in enumerate(['train','validation','test']):
        ids=np.flatnonzero(d['split']==split)
        result['split_counts'][name]={'rows':len(ids),'terminal':int(d['done'][ids].sum()),
            'actual_partial_for_audit_only':int((~d['_full_chunk'][ids]).sum()),
            'request_partial_known_before_action':int((~context['request_mask'][ids].all(axis=1)).sum()),
            'task_reward_nonzero':int((np.abs(d['task_reward'][ids])>1e-8).sum()) if 'task_reward' in d else None,
            'goal_change_events':{g:int(((d['goals'][ids,i]!=d['next_goals'][ids,i])&d['next_goal_valid'][ids,i]&d['goal_valid'][ids,i]).sum()) for i,g in enumerate(d['_goal_names'])}}
        baseline={'goals':d['goals'][ids],'reward':np.full(len(ids),transform.reward_mean[0]),
            'done':np.full(len(ids),d['done'][tr].mean())}
        result['baselines'][name]=metrics(baseline,d,ids)
    for name in ['full','no_action']:
        models=[];training=[];evaluations=[]
        for seed in result['seeds']:
            model,history=train_one(d,b,y,args,name,seed,positive_weight)
            models.append(model);training.append(history);evaluations.append(evaluate([model],b,transform,d))
        mean_std={}
        for key in ['total_reward_mse','total_reward_mse_on_task_reward_events']:
            values=[r['test'][key] for r in evaluations if r['test'][key] is not None]
            mean_std[key]={'n':len(values),'mean':float(np.mean(values)) if values else None,
                'std':float(np.std(values)) if values else None}
        result['models'][name]={'training':training,'individual_evaluations':evaluations,
            'ensemble_evaluation':evaluate(models,b,transform,d),'test_metric_mean_std':mean_std}
        torch.save({'schema':result['schema'],'name':name,'model_states':[m.state_dict() for m in models],
            'transform':transform.save(),'config':{'goals':d['goals'].shape[1],'visual_dim':b['visual'].shape[1],
                'hidden':args.hidden,'use_action':name=='full'},'max_steps':args.max_steps,'data_sha256':result['data_sha256']},args.output/f'{name}.pt')
        (args.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        e=result['models'][name]['ensemble_evaluation']['test']
        print(json.dumps({'model':name,'training_seconds':sum(h['seconds'] for h in training),
            'test_total_reward_mse':e['total_reward_mse'],'test_goal_event_ap':e['goal_change_event']['average_precision'],
            'test_done_ap':e['done']['average_precision'],'test_task_event_reward_mse':e['total_reward_mse_on_task_reward_events']},ensure_ascii=False),flush=True)
    rows=[]
    for name,item in result['models'].items():
        e=item['ensemble_evaluation']['test']
        fmt=lambda value:'unavailable' if value is None else f'{value:.4f}'
        rows.append(f"| {name} | {fmt(e['goal_change_event']['average_precision'])} | {fmt(e['goal_change_event']['recall_at_0_5'])} | {fmt(e['total_reward_mse'])} | {fmt(e['total_reward_mse_on_task_reward_events'])} | {fmt(e['done']['average_precision'])} |")
    text=f'''# 动作提议到结果的离线诊断

完整使用 {len(d['split'])} 行，包括提前成功终止及超时末行。三个网络初始化 seed，每个30 epoch（实际配置 epochs={args.epochs}），全部CPU。训练集拟合归一化和PCA，validation选择epoch。

输入仅是当前观察、完整预先生成动作提议、动作前剩余时间及已知timeout request mask。当前行实际执行长度/actual mask/done不输入；时间只使用过去已执行步的累计。两组共享时间和request mask，no_action仅将动作值置零，保持同架构同初始化。

| 模型（3成员均值） | goal变化AP | goal变化recall@0.5 | 总reward MSE | task事件行的总reward MSE | done AP |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

test事件数：{json.dumps(result['split_counts']['test'],ensure_ascii=False)}。

逐goal、逐seed结果与选择epoch见 results.json。加权BCE输出未校准；AP不等同成功率。该模型预测一次提议的最终结果，不预测完整动作序列被执行后的物理状态；提前成功后未执行的suffix不是动作动力学证据。此诊断不能替代世界模型多步误差或真实动作干预验证。
'''
    (args.output/'report.md').write_text(text)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--hidden',type=int,default=192);p.add_argument('--visual-dim',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--max-steps',type=int,default=3620)
    return p


if __name__=='__main__':
    run(parser().parse_args())

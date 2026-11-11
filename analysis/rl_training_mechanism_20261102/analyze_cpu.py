"""Read-only audit of the frozen rollout batch and saved PPO buffers."""
import collections
import gc
import json
import math
from pathlib import Path
import time
import urllib.request
import numpy as np
import torch


def put(url,value):
 body=json.dumps(value,allow_nan=False).encode();request=urllib.request.Request(url,data=body,method='PUT')
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=90) as r:assert r.status in (200,201,204)

def stats(x):
 x=np.asarray(x,dtype=np.float64)
 return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'p10':float(np.quantile(x,.1)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(x.max())} if len(x) else {}

def mc(reward,gamma):
 out=np.empty(len(reward));v=0.
 for i in range(len(reward)-1,-1,-1):v=float(reward[i])+gamma*v;out[i]=v
 return out

def gae(reward,value,gamma,lam):
 out=np.empty(len(reward));a=0.
 for i in range(len(reward)-1,-1,-1):
  next_value=value[i+1] if i+1<len(value) else 0.
  a=reward[i]+gamma*next_value-value[i]+gamma*lam*a;out[i]=a
 return out

def ev(y,p):
 return float(1-np.var(y-p)/np.var(y)) if np.var(y)>1e-12 else None

def run(config):
 torch.set_num_threads(1)
 started=time.time();buffers={}
 for v,p in config['buffers'].items():
  b=torch.load(p,map_location='cpu',weights_only=True)
  buffers[v]={k:t.detach().float().reshape(-1).numpy() for k,t in b.items() if torch.is_tensor(t)}
  buffers[v]['replay_error']=b.get('replay_error')
  buffers[v]['group_statistics']=b.get('group_statistics')
 rewards={v:[] for v in buffers};returns={v:[] for v in buffers};raw={v:[] for v in buffers}
 phases=[];flags=[];episodes=[];offset=0
 for receipt in config['receipts']:
  e=torch.load(receipt['path'],map_location='cpu',weights_only=True);rows=e['transitions'];n=len(rows)
  task=np.array([float(r['task_reward']) for r in rows]);phi=np.array([sum(r['goals_before']['goals'].values()) for r in rows],dtype=float)
  following=np.array([0. if r['terminal'] else sum(r['goals_after']['goals'].values()) for r in rows],dtype=float)
  r99=task+.99*following-phi;r1=task+following-phi
  stored=np.array([float(r['reward']) for r in rows]);assert np.max(np.abs(r99-stored))<1e-5
  event=[];state=[];ever=[False]*3;last_new=0;no_progress=0
  keys=['bottle_1_on_table','bottle_2_on_table','fridge_closed']
  for i,r in enumerate(rows):
   before=r['goals_before']['goals'];after=r['goals_after']['goals'];code=f"b{int(before[keys[0]])+int(before[keys[1]])}_fridge{int(before[keys[2]])}"
   state.append(code)
   if task[i] or before!=after:
    event.append({'chunk':i,'task_reward':float(task[i]),'shaped_reward99':float(r99[i]),'shaped_reward1':float(r1[i]),'before':before,'after':after,'terminal':bool(r['terminal'])})
  for k in keys[:2]:
   hit=[i for i,r in enumerate(rows) if r['goals_after']['goals'][k]]
   if hit:last_new=max(last_new,hit[0])
  after_last=n-last_new-1
  report={'episode':receipt['episode'],'instance':receipt['instance'],'seed':receipt['seed'],'success':bool(e['status']['success']),'score':float(e['status']['score']),'chunks':n,'steps':e['status']['steps'],'task_reward_sum':float(task.sum()),'shaping99_sum':float((r99-task).sum()),'shaping1_sum':float((r1-task).sum()),'event_count':len(event),'events':event,'goal_phase_counts':dict(collections.Counter(state)),'after_last_first_bottle_goal_chunks':after_last,'variants':{}}
  for v,b in buffers.items():
   gamma=1. if v=='gamma1_ppo' else .99;r=r1 if gamma==1 else r99
   discounted=mc(r,gamma);task_discounted=mc(task,gamma)
   adv=b['advantages'][offset:offset+n];w=b['weights'][offset:offset+n];value=b['prev_values'][offset:offset+n]
   report['variants'][v]={'start_return':float(discounted[0]),'start_task_return':float(task_discounted[0]),'start_value':float(value[0]),'mean_value':float(value.mean()),'value_mc_mse':float(np.mean((value-discounted)**2)),'mean_adv':float(adv.mean()),'positive_adv_fraction':float(np.mean(adv>0)),'weighted_adv_sum':float(np.sum(w*adv)),'weighted_abs_adv_sum':float(np.sum(w*np.abs(adv))),'weighted_positive_sum':float(np.sum(w*np.maximum(adv,0))),'weighted_negative_sum':float(np.sum(w*np.maximum(-adv,0))),'weight':float(w[0]),'head10_adv':float(adv[:max(1,n//10)].mean()),'tail10_adv':float(adv[-max(1,n//10):].mean()),'return_quantiles':[float(discounted[i]) for i in [0,n//4,n//2,3*n//4,n-1]],'value_quantiles':[float(value[i]) for i in [0,n//4,n//2,3*n//4,n-1]]}
   rewards[v].append(r);returns[v].append(discounted);raw[v].append(gae(r,value,gamma,.95))
   assert abs(discounted[0]-(task_discounted[0]-phi[0]))<1e-8
  episodes.append(report);phases.extend(state);flags.extend([report['success']]*n);offset+=n
  put(config['status_put'],{'stage':'reading_episodes','completed':len(episodes),'total':len(config['receipts']),'elapsed':time.time()-started})
  del rows,e;gc.collect()
 assert offset==23908
 outcome=np.array(flags,dtype=bool);phases=np.array(phases);variants={}
 for v,b in buffers.items():
  a=b['advantages'];w=b['weights'];V=b['prev_values'];G=np.concatenate(returns[v]);A=np.concatenate(raw[v])
  normalized=(A-A.mean())/(A.std(ddof=1)+1e-5)
  entry={'replay_error':b['replay_error'],'group_statistics':b['group_statistics'],'advantage_reconstruction_max_error':None if v=='group_outcome' else float(np.max(np.abs(normalized-a))), 'value_mc_mse':float(np.mean((G-V)**2)),'value_mc_explained_variance':ev(G,V),'saved_target_mse':float(np.mean((b['returns']-V)**2)) if 'returns' in b else None,'outcomes':{},'phases':{}}
  for label,mask in [('success',outcome),('failure',~outcome)]:
   entry['outcomes'][label]={'chunks':int(mask.sum()),'sample_weight_fraction':float(w[mask].sum()/w.sum()),'positive_adv_fraction':float((a[mask]>0).mean()),'mean_adv':float(a[mask].mean()),'abs_adv_mass_fraction':float((w[mask]*np.abs(a[mask])).sum()/(w*np.abs(a)).sum()),'positive_mass':float((w[mask]*np.maximum(a[mask],0)).sum()),'negative_mass':float((w[mask]*np.maximum(-a[mask],0)).sum()),'start_returns':stats([e['variants'][v]['start_return'] for e in episodes if e['success']==(label=='success')])}
  for phase in sorted(set(phases)):
   mask=phases==phase
   entry['phases'][phase]={'chunks':int(mask.sum()),'weight_fraction':float(w[mask].sum()/w.sum()),'mean_adv':float(a[mask].mean()),'weighted_abs_adv_mass_fraction':float((w[mask]*np.abs(a[mask])).sum()/(w*np.abs(a)).sum())}
  variants[v]=entry
 pair_ranking={}
 for v in buffers:
  rows=[]
  for instance in range(1,17):
   group=[e for e in episodes if e['instance']==instance]
   if len({x['success'] for x in group})<2:continue
   good=next(x for x in group if x['success']);bad=next(x for x in group if not x['success'])
   g=good['variants'][v]['start_return'];b=bad['variants'][v]['start_return']
   rows.append({'instance':instance,'success_episode':good['episode'],'failure_episode':bad['episode'],'success_return':g,'failure_return':b,'correct_return_ranking':g>b,'success_weighted_mean_adv':good['variants'][v]['mean_adv'],'failure_mean_adv':bad['variants'][v]['mean_adv']})
  pair_ranking[v]={'mixed_pairs':len(rows),'correct_return_rankings':sum(x['correct_return_ranking'] for x in rows),'pairs':rows}
 comparisons={}
 for k in ['advantages','returns','prev_values','weights']:
  comparisons[k]={'max_abs_error':float(np.max(np.abs(buffers['control'][k]-buffers['balanced_ppo'][k]))),'exact_equal':bool(np.array_equal(buffers['control'][k],buffers['balanced_ppo'][k]))}
 report={'episodes':episodes,'variant_statistics':variants,'initial_return_pair_ranking':pair_ranking,'control_vs_balanced_buffers':comparisons,'chunks':offset,'elapsed_seconds':time.time()-started,'read_only':True,'model_updates':0}
 put(config['result_put'],report);put(config['status_put'],{'stage':'complete','episodes':len(episodes),'elapsed':time.time()-started,'result_ready':True})
 print('READ_ONLY_MECHANISM_AUDIT_COMPLETE',flush=True)

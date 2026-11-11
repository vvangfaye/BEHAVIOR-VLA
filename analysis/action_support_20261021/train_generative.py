"""Small generative comparison using fixed proprio/history context, not a VLA.

Compare support projection against identical mask-conditioned continuous flow.
The test uses sampled masks, never test target masks. Energy score evaluates
the distribution rather than whether one draw matches the recorded choice.
"""
import hashlib
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_activity import data
from support_flow import project

HERE=Path(__file__).resolve().parent
METHODS=('full_flow','mask_conditioned_flow','support_flow')


class Flow(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.mask=nn.Linear(128,23)
        self.flow=nn.Sequential(nn.Linear(736+128+23+32,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,736))
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),16)))

    def velocity(self,xt,t,h,mask,method):
        angle=t[:,None]*self.frequencies[None]
        te=torch.cat([angle.sin(),angle.cos()],-1)
        condition=torch.zeros_like(mask) if method=='full_flow' else mask
        v=self.flow(torch.cat([xt.flatten(1),h,condition,te],-1)).view(-1,32,23)
        return project(v,mask) if method=='support_flow' else v


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=17);args=parser.parse_args()
    seed=args.seed
    torch.set_num_threads(4);torch.manual_seed(seed)
    rows=data();tasks=sorted(set(r['task'] for r in rows));fold={}
    for task in tasks:
        eps=sorted({r['episode'] for r in rows if r['task']==task},key=lambda e:hashlib.sha256(str(e).encode()).hexdigest())
        fold.update({e:i%5 for i,e in enumerate(eps)})
    train=np.array([fold[r['episode']] in (2,3,4) for r in rows]);test=np.array([fold[r['episode']]==0 for r in rows])
    raw_x=np.stack([r['x'] for r in rows]).astype(np.float32)
    task_ids=np.array([tasks.index(r['task']) for r in rows]);raw_x=np.concatenate([raw_x,np.eye(len(tasks),dtype=np.float32)[task_ids]],-1)
    x=(raw_x-raw_x[train].mean(0))/np.maximum(raw_x[train].std(0),.01)
    x=torch.from_numpy(np.clip(x,-10,10))
    y=np.stack([r['future']-r['x'][61:84] for r in rows]).astype(np.float32)
    ym=y[train].mean((0,1));ys=np.maximum(y[train].std((0,1)),.01)
    y=torch.from_numpy((y-ym)/ys)
    active=torch.from_numpy(np.stack([r['y'] for r in rows]).astype(np.float32))
    train_ids=torch.tensor(np.flatnonzero(train));test_ids=torch.tensor(np.flatnonzero(test))
    protocol={'methods':METHODS,'seed':seed,'steps':3000,'batch':256,'lr':.0003,'weight_decay':.0001,
              'aux_mask_bce_weight':.1,'inference_euler_steps':10,'test_samples_per_context':8,
              'split':'same episode hash folds: train 2,3,4; test 0; fold 1 unused',
              'test_chunks':int(test.sum()),'train_chunks':int(train.sum()),
              'primary':'paired held-out conditional energy score: support_flow vs mask_conditioned_flow',
              'scope':'proprio/history prototype; no image encoder or VLA checkpoint updates; one seed',
              'feature_source':'same exact past-only features as train_activity; per-channel normalization fitted on train only'}
    suffix='' if seed==17 else f'_seed{seed}'
    pp=HERE/f'generative_protocol{suffix}.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(protocol))
    else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
    output=HERE/f'generative_runs{suffix}';output.mkdir(exist_ok=True)
    summaries={};start=time.monotonic()
    for method in METHODS:
        path=output/f'{method}.npz'
        if path.exists():continue
        torch.manual_seed(seed);model=Flow(x.shape[-1])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        for step in range(3000):
            ids=train_ids[torch.randint(len(train_ids),(256,))];h=model.context(x[ids]);logits=model.mask(h)
            mask=active[ids];noise=torch.randn_like(y[ids]);t=torch.rand(len(ids))
            if method=='support_flow':noise=project(noise,mask)
            xt=(1-t[:,None,None])*noise+t[:,None,None]*y[ids]
            target=y[ids]-noise
            pred=model.velocity(xt,t,h,mask,method)
            loss=F.mse_loss(pred,target)+.1*F.binary_cross_entropy_with_logits(logits,mask)
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            if step%500==0:print(json.dumps({'method':method,'step':step,'loss':float(loss.detach()),'seconds':round(time.monotonic()-start,1)}),flush=True)
        model.eval();scores=[];mse=[];samples=[]
        torch.manual_seed(761)
        with torch.inference_mode():
            for ids in test_ids.split(64):
                h=model.context(x[ids]);probs=model.mask(h).sigmoid()
                h=h.repeat_interleave(8,0);probs=probs.repeat_interleave(8,0)
                mask=torch.bernoulli(probs);z=torch.randn(len(h),32,23)
                if method=='support_flow':z=project(z,mask)
                for k in range(10):z=z+.1*model.velocity(z,torch.full((len(h),),k/10),h,mask,method)
                z=z.reshape(len(ids),8,736);truth=y[ids].flatten(1)
                d=(z-truth[:,None]).norm(dim=-1).mean(-1)/math.sqrt(736)
                pair=torch.cdist(z,z).sum((1,2))/(8*7)/math.sqrt(736)
                scores.append(d-.5*pair);mse.append((z.mean(1)-truth).square().mean(-1));samples.append(z)
        es=torch.cat(scores).numpy();meanmse=torch.cat(mse).numpy()
        np.savez_compressed(path,energy_score=es,mean_mse=meanmse,
                            episode=np.array([rows[i]['episode'] for i in test_ids]),
                            frame=np.array([rows[i]['frame'] for i in test_ids]),
                            samples=torch.cat(samples).numpy(),target=y[test_ids].flatten(1).numpy())
        torch.save(model.state_dict(),output/f'{method}.pt')
        print(json.dumps({'method':method,'energy_score':float(es.mean()),'mean_mse':float(meanmse.mean()),'seconds':round(time.monotonic()-start,1)}),flush=True)
    arrays={name:np.load(output/f'{name}.npz') for name in METHODS}
    for name,a in arrays.items():summaries[name]={'energy_score':float(a['energy_score'].mean()),'mean_mse':float(a['mean_mse'].mean())}
    a,b=arrays['support_flow'],arrays['mask_conditioned_flow'];assert np.array_equal(a['episode'],b['episode'])
    eps=np.unique(a['episode']);delta=np.array([(a['energy_score']-b['energy_score'])[a['episode']==e].mean() for e in eps])
    rng=np.random.default_rng(20261021);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    result={'metrics':summaries,'support_minus_conditioned_energy_episode_mean':float(delta.mean()),
            'episode_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'test_episodes':len(eps),
            'elapsed_seconds':time.monotonic()-start,'scope':protocol['scope']}
    (HERE/f'generative_result{suffix}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

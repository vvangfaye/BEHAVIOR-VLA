"""Same-data SFT and parameter-update-norm-matched SFT controls; no simulator."""
import argparse
from contextlib import nullcontext
import dataclasses
import json
from pathlib import Path

import torch

from lora import install_action_lora
from native import NativeObjective
from vla_data import VlaPilotData
from vla_runtime import load_base


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--match-history',type=Path)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    targets=None
    if args.match_history:
        history=json.loads(args.match_history.read_text())
        if history['method']!='maml':raise ValueError('Norm control must use the predeclared MAML comparator')
        targets=[x['adapter_delta_norm'] for x in history['completed']]
        if len(targets)!=len(plan['iterations']):raise ValueError('MAML comparator is incomplete')
    torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    base=load_base(plan['checkpoint'],plan['tokenizer'],precision=plan['precision'])
    base.rl_cfg=dataclasses.replace(base.rl_cfg,joint_logprob=True)
    install_action_lora(base,rank=plan['rank']);objective=NativeObjective(base)
    params={n:p for n,p in objective.named_parameters() if p.requires_grad}
    torch.save({n:p.detach().cpu() for n,p in params.items()},args.output/'initial_adapter.pt')
    data=VlaPilotData(plan['data_directory'],base)
    optimizer=torch.optim.Adam(params.values(),lr=plan['inner_lr'],eps=plan['inner_epsilon'])
    records=[]
    for index,row in enumerate(plan['iterations']):
        ids=row['support_indices']
        if any(data.plan['samples'][i]['task_name']!=row['task'] for i in ids):
            raise ValueError('Support and planned task differ')
        obs,target,_=data.batch(ids)
        rng=torch.Generator(device=target.device).manual_seed(row['fm_seed'])
        noise=torch.randn(target.shape,device=target.device,generator=rng)
        times=torch.tensor(row['fm_times'],device=target.device,dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if plan['precision']=='bf16' else nullcontext()
        with context:loss=objective('fm',obs,actions=target,noise=noise,time=times)
        if not torch.isfinite(loss):raise ValueError('Nonfinite SFT loss')
        loss.backward()
        before={n:p.detach().clone() for n,p in params.items()}
        optimizer.step()
        proposed_norm=torch.sqrt(sum((p.detach()-before[n]).square().sum() for n,p in params.items()))
        scale=1.
        if targets is not None:
            if proposed_norm==0 and targets[index]>0:
                raise ValueError('Cannot match a nonzero norm with zero SFT direction')
            scale=targets[index]/float(proposed_norm) if proposed_norm>0 else 0.
            with torch.no_grad():
                for n,p in params.items():p.copy_(before[n]+scale*(p-before[n]))
        actual_norm=float(torch.sqrt(sum((p.detach()-before[n]).square().sum() for n,p in params.items())))
        records.append(dict(iteration=index,loss=float(loss.detach()),support_indices=ids,
                            proposed_delta_norm=float(proposed_norm),applied_delta_norm=actual_norm,scale=scale))
    torch.save({n:p.detach().cpu() for n,p in params.items()},args.output/'adapter.pt')
    (args.output/'result.json').write_text(json.dumps(dict(complete=True,
        mode='norm_matched_sft' if targets is not None else 'sft',records=records,
        environment_episodes=0,scope='Offline controls only; no evaluation outcomes used'),indent=2)+'\n')


if __name__=='__main__':main()

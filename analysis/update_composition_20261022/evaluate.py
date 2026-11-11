"""Independent single-policy evaluation after ordinary demonstration Adam.

No old/new policy switching is used for the reported evaluation outcome.
Each task adapts a fresh copy of the same trained initialization.
"""
import argparse
from contextlib import nullcontext
import dataclasses
import json
from pathlib import Path

import torch

from closed_loop import collect,native_policy_step
from lora import install_action_lora
from native import NativeObjective
from simulator_backend import simulator
from vla_data import VlaPilotData
from vla_runtime import load_base


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    base=load_base(plan['checkpoint'],plan['tokenizer'],precision=plan['precision'])
    base.rl_cfg=dataclasses.replace(base.rl_cfg,joint_logprob=True)
    install_action_lora(base,rank=plan['rank']);objective=NativeObjective(base)
    params={n:p for n,p in objective.named_parameters() if p.requires_grad}
    overlay=torch.load(args.adapter,map_location='cpu',weights_only=True)
    if set(overlay)!=set(params):raise ValueError('Adapter keys differ from the declared parameter set')
    data=VlaPilotData(plan['data_directory'],base)
    records=[]
    for task in plan['tasks']:
        name=task['task_name']
        if name in plan['meta_training_tasks']:
            raise ValueError('Expected a task held out from outer-loop training')
        with torch.no_grad():
            for n,p in params.items():p.copy_(overlay[n].to(p.device,p.dtype))
        optimizer=torch.optim.Adam(params.values(),lr=plan['inner_lr'],eps=plan['inner_epsilon'])
        for batch in task['adaptation_batches']:
            ids=batch['indices']
            if any(data.plan['samples'][i]['task_name']!=name for i in ids):
                raise ValueError('Adaptation examples belong to a different task')
            support_instances={int(data.plan['samples'][i]['task_instance_id']) for i in ids}
            if support_instances & {case['instance'] for case in task['cases']}:
                raise ValueError('Adaptation demonstration shares an evaluation instance')
            obs,target,_=data.batch(ids)
            rng=torch.Generator(device=target.device).manual_seed(batch['seed'])
            noise=torch.randn(target.shape,device=target.device,generator=rng)
            times=torch.tensor(batch['times'],device=target.device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if plan['precision']=='bf16' else nullcontext()
            with ctx:loss=objective('fm',obs,actions=target,noise=noise,time=times)
            if not torch.isfinite(loss):raise ValueError('Nonfinite ordinary adaptation loss')
            loss.backward();optimizer.step()
        adapted={n:p.detach().clone() for n,p in params.items()}
        for case in task['cases']:
            generator=torch.Generator(device=next(iter(params.values())).device).manual_seed(case['policy_seed'])
            step=native_policy_step(base,params,adapted,generator,bf16=plan['precision']=='bf16')
            envplan=dict(plan['environment'],task=name,seed=case['environment_seed'])
            output=args.output/f"{name}_{case['instance']}_{case['environment_seed']}"
            with simulator(envplan,output,case['instance'],'public_test',case['max_steps']) as env:
                initial=env.current['initial']
                episode=collect(env,step,max_steps=case['max_steps'],cut_step=0,
                                before_is_updated=False,record_replay=False)
            records.append(dict(task=name,**case,success=episode.success,score=episode.score,
                actual_steps=episode.control_steps,physical_sha256=initial['physical_sha256']))
            write(args.output/'progress.json',dict(records=records))
    write(args.output/'result.json',dict(complete=True,records=records,
        success_count=sum(r['success'] for r in records),episode_count=len(records),
        scope='Single adapted policy on tasks held out from the meta-loop; base pretraining may already include these tasks'))


if __name__=='__main__':main()

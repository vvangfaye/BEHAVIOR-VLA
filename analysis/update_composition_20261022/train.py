"""Bounded on-policy matched-arm trainer; requires a verified real-model update.

No environment is mocked, no reward is inferred from demonstration error, and
no auxiliary data-collection loop is started. All episodes are in the plan.
"""
import argparse
from contextlib import nullcontext
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.func import functional_call

from closed_loop import collect,native_policy_step,batch_meta_gradient
from core import inner_adam_first_step,training_episode_weights
from lora import install_action_lora
from native import NativeObjective
from simulator_backend import simulator,shutdown_renderer
from vla_data import VlaPilotData
from vla_runtime import load_base
from admission import validate_trial_configuration


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    admission=json.loads(Path(plan['admission_result']).read_text())
    validate_trial_configuration(plan,admission)
    if plan['method'] not in ('meta','maml','fixed_crossplay','rl_bc','before_after'):
        raise ValueError('Unknown training arm')
    rows=plan['iterations']
    if not rows or len(rows)*2>plan['max_training_episodes']:
        raise ValueError('Planned episodes exceed fixed budget')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'progress.json',dict(stage='loading',method=plan['method'],completed=[]))
    torch.manual_seed(plan['seed']);torch.set_num_threads(4)
    started=time.monotonic()
    base=load_base(plan['checkpoint'],plan['tokenizer'],precision=plan['precision'])
    base.rl_cfg=dataclasses.replace(base.rl_cfg,joint_logprob=True)
    install_action_lora(base,rank=plan['rank'])
    objective=NativeObjective(base)
    params={name:p for name,p in objective.named_parameters() if p.requires_grad}
    frozen_partner={name:p.detach().clone() for name,p in params.items()}
    data=VlaPilotData(plan['data_directory'],base)
    optimizer=torch.optim.Adam(params.values(),lr=plan['outer_lr'],eps=plan['outer_epsilon'])
    prior_returns={};completed=[]
    bf16=plan['precision']=='bf16'
    def context():return torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext()
    def update(q,loss):
        if plan['method']=='rl_bc':return dict(q)
        return inner_adam_first_step(q,loss,lr=plan['inner_lr'],epsilon=plan['inner_epsilon'])
    for index,row in enumerate(rows):
        if time.monotonic()-started>=plan['max_seconds']:
            raise TimeoutError('Fixed training wall-time bound reached')
        support=row['support_indices']
        if any(i not in set(data.train_ids.tolist()) for i in support):
            raise ValueError('Support contains a non-training example')
        if any(data.plan['samples'][i]['task_name']!=row['task'] for i in support):
            raise ValueError('Support and rollout task differ')
        obs,target,_=data.batch(support)
        generator=torch.Generator(device=target.device).manual_seed(row['fm_seed'])
        noise=torch.randn(target.shape,device=target.device,generator=generator)
        # Explicit values are frozen in the plan, including the FM time draw.
        times=torch.tensor(row['fm_times'],device=target.device,dtype=torch.float32)
        if len(times)!=len(support) or not torch.all((times>0)&(times<1)):
            raise ValueError('Invalid precommitted FM times')
        def demo_loss(q):
            with context():
                return functional_call(objective,q,('fm',obs),dict(actions=target,noise=noise,time=times))
        adapted=update(params,demo_loss)
        snapshot={n:p.detach().clone() for n,p in adapted.items()}
        del adapted
        episodes=[];summaries=[]
        # Use only earlier iterations to form a task baseline.
        history=prior_returns.get(row['task'],[])
        baseline=sum(history)/len(history) if history else 0.
        for slot,instance in enumerate(row['instances']):
            if len(row['instances'])!=2:
                raise ValueError('Exactly two planned episodes per iteration required')
            mixed=plan['method'] in ('meta','fixed_crossplay') and slot==1
            cut=row['cut_step'] if mixed else 0
            before=row['before_is_updated'] if mixed else False
            if plan['method']=='before_after' and slot==1:
                cut=row['max_steps']+1
            gen=torch.Generator(device=target.device).manual_seed(row['rollout_seeds'][slot])
            partner=frozen_partner if plan['method']=='fixed_crossplay' else params
            step=native_policy_step(base,partner,snapshot,gen,bf16=bf16)
            envplan=dict(plan['environment'],task=row['task'],seed=row['environment_seeds'][slot])
            episode_root=Path(plan['environment'].get('renderer_mailbox',{}).get('episode_output_root',args.output))
            write(args.output/'progress.json',dict(stage='episode',method=plan['method'],
                iteration=index,slot=slot,task=row['task'],instance=instance,completed=completed))
            with simulator(envplan,episode_root/f'iteration_{index:03d}_episode_{slot}',
                           instance,'train',row['max_steps']) as env:
                episode=collect(env,step,max_steps=row['max_steps'],cut_step=cut,before_is_updated=before)
            episodes.append(episode)
            torch.save(dict(decisions=episode.decisions),args.output/f'replay_{index:03d}_{slot}.pt')
            summaries.append(dict(task=row['task'],instance=instance,mixed=mixed,
                success=episode.success,score=episode.score,steps=episode.control_steps,
                requested_cut=episode.cut_requested,executed_cut=episode.cut_executed))
            write(args.output/f'episode_{index:03d}_{slot}.json',
                  dict(iteration=index,slot=slot,**summaries[-1]))
            write(args.output/'progress.json',dict(stage='episode_finished',method=plan['method'],
                iteration=index,slot=slot,episode=summaries[-1],completed=completed))
        weights=training_episode_weights(plan['method'],plan['compatibility_weight'])
        # The historical partner is independent of trainable parameters. Keep
        # its state distribution and terminal reward, but exclude its scores.
        gradient_episodes=episodes
        if plan['method']=='fixed_crossplay':
            gradient_episodes=[dataclasses.replace(ep,decisions=[d for d in ep.decisions if d['updated']])
                               for ep in episodes]
        gradients,info=batch_meta_gradient(base,params,demo_loss,gradient_episodes,weights,[baseline]*2,
            adapted_snapshot=snapshot,inner_lr=plan['inner_lr'],bf16=bf16,inner_update=update)
        if plan['method']=='rl_bc':
            bc_gradients=torch.autograd.grad(demo_loss(params),tuple(params.values()))
            for (name,_),gradient in zip(params.items(),bc_gradients):
                gradients[name]=gradients[name]+plan['bc_weight']*gradient
        optimizer.zero_grad(set_to_none=True)
        for name,p in params.items():p.grad=gradients[name]
        norm=torch.nn.utils.clip_grad_norm_(tuple(params.values()),plan['max_grad_norm'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite meta gradient')
        before_step={name:p.detach().clone() for name,p in params.items()}
        optimizer.step()
        delta_norm=float(torch.sqrt(sum((p.detach()-before_step[name]).square().sum()
                                        for name,p in params.items())))
        prior_returns.setdefault(row['task'],[]).extend(float(ep.success) for ep in episodes)
        completed.append(dict(iteration=index,episodes=summaries,gradient_norm=float(norm),
                              adapter_delta_norm=delta_norm,support_indices=support,
                              gradient_diagnostics=info))
        state_path=args.output/f'training_state_{index:03d}.pt'
        temporary=state_path.with_suffix('.tmp')
        torch.save(dict(parameters={n:p.detach().cpu() for n,p in params.items()},
                        optimizer=optimizer.state_dict(),completed=completed,
                        prior_returns=prior_returns,next_iteration=index+1),temporary)
        temporary.replace(state_path)
        torch.save({n:p.detach().cpu() for n,p in params.items()},args.output/f'adapter_{index:03d}.pt')
        write(args.output/'progress.json',dict(stage='updated',method=plan['method'],completed=completed,
            elapsed_seconds=time.monotonic()-started))
    shutdown_renderer(plan['environment'])
    write(args.output/'result.json',dict(complete=True,method=plan['method'],
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),iterations=len(completed),
        training_episodes=2*len(completed),elapsed_seconds=time.monotonic()-started,
        scope='Training outcomes only; independent single-policy post-adaptation evaluation required'))


if __name__=='__main__':main()

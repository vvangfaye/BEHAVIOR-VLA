"""Registered bridge-point control, derived from the frozen four-arm trainer."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from adapter import NativeAdapter,bridge,difference_loss
from vla_runtime import load_base
from vla_data import VlaPilotData,digest
from train_posterior_support import choose_eval,atomic_json


def evaluate(adapter,data,indices,path,draws,steps):
    generator=torch.Generator(device='cpu').manual_seed(761)
    predictions=[];targets=[];hashes=[];rows=[]
    with torch.inference_mode():
        for index in indices:
            obs,target,_=data.batch([index]);samples=[]
            for _ in range(draws):
                noise=torch.randn(target.shape,generator=generator)
                hashes.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=adapter.sample(obs,noise.to(target.device),steps)
                samples.append(generated[...,:23].float().cpu())
            sample=torch.stack(samples,1);truth=target[...,:23].float().cpu()
            flat=sample.flatten(2);truth_flat=truth.flatten(1)
            distance=(flat-truth_flat[:,None]).norm(dim=-1).mean(1)
            between=torch.cdist(flat,flat).sum((1,2))/(draws*(draws-1))
            es=(distance-.5*between)/truth_flat.shape[-1]**.5
            mse=(sample.mean(1)-truth).square().mean((1,2))
            if not torch.isfinite(sample).all():raise FloatingPointError('Nonfinite evaluation')
            rows.append(dict(index=int(index),episode=int(data.episode[index]),frame=int(data.frame[index]),
                             energy_score=float(es[0]),mean_action_mse=float(mse[0])))
            predictions.append(sample.numpy());targets.append(truth.numpy())
    np.savez_compressed(path.with_suffix('.npz'),samples=np.concatenate(predictions),
                        target=np.concatenate(targets),index=indices,episode=data.episode[indices],
                        frame=data.frame[indices],noise_sha256=np.asarray(hashes))
    result=dict(samples=len(indices),draws=draws,records=rows,
                means={k:float(np.mean([r[k] for r in rows])) for k in ('energy_score','mean_action_mse')},
                noise_sha256=hashes,scope='Expert-subtask offline evaluation; not autonomous task success')
    atomic_json(path,result)
    return result['means']


def bridge_point_loss(left,right,noisy,a,b,t):
    distance=(a[...,:23].float()-b[...,:23].float()).square().sum((1,2))
    weight=torch.exp(-(1-t).square()*distance/(4*t.square()))
    left_residual=t[:,None,None]*left[...,:23].float()-(noisy[...,:23].float()-a[...,:23].float())
    right_residual=t[:,None,None]*right[...,:23].float()-(noisy[...,:23].float()-b[...,:23].float())
    return .5*(weight*(left_residual.square().mean((1,2))+right_residual.square().mean((1,2)))).mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert args.method in plan['methods']
    for name,expected in plan['source_sha256'].items():
        if digest(name)!=expected:raise ValueError('Frozen source changed: '+name)
    args.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.manual_seed(plan['seed'])
    rng=np.random.default_rng(plan['seed'])
    noise_rng=torch.Generator(device='cpu').manual_seed(plan['seed']+3401)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    started=time.monotonic()
    base=load_base(plan['checkpoint'],plan['tokenizer'])
    adapter=NativeAdapter(base).cuda().eval()
    data=VlaPilotData(plan['data_directory'],base)
    assert digest(data.directory/'data_plan.json')==plan['data_plan_sha256']
    parameters=[p for p in adapter.parameters() if p.requires_grad]
    assert all(p.dtype==torch.float32 for p in parameters)
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    initial=base.llm.layers[0].attn.q_proj[1].weight.detach().clone()
    initial_hash=hashlib.sha256(initial.float().cpu().numpy().tobytes()).hexdigest()
    task_ids={}
    for i in data.train_ids:
        task_ids.setdefault(data.plan['samples'][int(i)]['task_index'],[]).append(int(i))
    tasks=sorted(task_ids)
    data_trace=hashlib.sha256();noise_trace=hashlib.sha256()
    plan_hash=digest(args.plan)
    atomic_json(args.output/'run.json',dict(plan_sha256=plan_hash,method=args.method,seed=plan['seed'],
                trainable_parameters=sum(p.numel() for p in parameters),initial_attention_sha256=initial_hash,
                source_data_receipt_sha256=digest(data.directory/'image_receipt.json')))
    print(json.dumps(dict(event='start',method=args.method,steps=plan['steps'])),flush=True)
    step=0;expert_forwards=0;prefix_forwards=0
    for step_index in range(plan['steps']):
        if stop[0] or time.monotonic()-started>plan['max_seconds']:break
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr']=plan['learning_rate']*min(1.,(step_index+1)/plan['warmup_steps'])
        total=0.
        for _ in range(plan['groups_per_update']):
            task=int(rng.choice(tasks));group=task_ids[task]
            a,b=int(rng.choice(group)),int(rng.choice(group))
            c=int(rng.choice(group))
            while data.episode[c]==data.episode[b]:c=int(rng.choice(group))
            indices=[a,b,c];data_trace.update(np.asarray(indices,dtype=np.int64).tobytes())
            prepared=[];target=[]
            for index in indices:
                obs,y,_=data.batch([index]);prepared.append(adapter.prepare(obs));target.append(y)
                prefix_forwards+=1
            # Draw every arm's possible inputs in a common fixed order.
            noises=torch.randn((6,*target[0].shape),generator=noise_rng)
            uniform=torch.rand(5,generator=noise_rng)
            times=(uniform[:4].pow(2/3)*.999+.001)
            pair_t=plan['pair_time_min']+(plan['pair_time_max']-plan['pair_time_min'])*uniform[4]
            noise_trace.update(noises.numpy().tobytes());noise_trace.update(uniform.numpy().tobytes())
            noises=noises.cuda();times=times.cuda();pair_t=pair_t.reshape(1).cuda()
            # Identical three-context ordinary flow loss for all methods.
            for k in range(3):
                t=times[k:k+1];noisy=t[:,None,None]*noises[k]+(1-t[:,None,None])*target[k]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    v=adapter.velocity(prepared[k],noisy,t)
                    loss=(v-(noises[k]-target[k])).square().mean()/(3*plan['groups_per_update'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite base loss')
                loss.backward();total+=float(loss.detach());expert_forwards+=1
            # Two more expert evaluations per arm, with the same B/C contexts.
            if args.method in ('relative_difference','mean_difference','bridge_single'):
                t=pair_t if args.method in ('relative_difference','bridge_single') else torch.ones_like(pair_t)
                noisy=bridge(target[1],target[2],t,noises[5])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    left=adapter.velocity(prepared[1],noisy,t)
                    right=adapter.velocity(prepared[2],noisy,t)
                    auxiliary=(bridge_point_loss(left,right,noisy,target[1],target[2],t) if args.method=='bridge_single' else difference_loss(left,right,target[1],target[2],t))
            else:
                t=times[3:4]
                xleft=t[:,None,None]*noises[3]+(1-t[:,None,None])*target[1]
                xright=t[:,None,None]*noises[4]+(1-t[:,None,None])*target[2]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    left=adapter.velocity(prepared[1],xleft,t)[...,:23]
                    right=adapter.velocity(prepared[2],xright,t)[...,:23]
                    uleft=(noises[3]-target[1])[...,:23]
                    uright=(noises[4]-target[2])[...,:23]
                    auxiliary=.5*((left-uleft).square().mean()+(right-uright).square().mean())
                    if args.method=='deltafm_aux':
                        auxiliary-=plan['deltafm_lambda']*.5*((left-uright).square().mean()+(right-uleft).square().mean())
            loss=plan['auxiliary_weight']*auxiliary/plan['groups_per_update']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite auxiliary loss')
            loss.backward();total+=float(loss.detach());expert_forwards+=2
        norm=torch.nn.utils.clip_grad_norm_(parameters,plan['clip_grad'])
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        if step_index==0:
            assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
            assert base.llm.layers[0].attn.q_proj[1].weight.grad.norm()>0
        optimizer.step();step=step_index+1
        if step==1 or step%plan['log_every']==0:
            print(json.dumps(dict(event='update',method=args.method,step=step,loss=total,
                 gradient_norm=float(norm),step_seconds=time.monotonic()-tick,
                 elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())),flush=True)
    changed=float((base.llm.layers[0].attn.q_proj[1].weight.detach()-initial).abs().max())
    complete=step==plan['steps']
    assert not complete or changed>0
    weights={n:p.detach().cpu() for n,p in adapter.named_parameters() if p.requires_grad}
    torch.save(dict(complete=complete,step=step,method=args.method,seed=plan['seed'],plan_sha256=plan_hash,
                    base_checkpoint=plan['checkpoint'],weights=weights,optimizer=optimizer.state_dict(),
                    numpy_rng=rng.bit_generator.state,noise_rng=noise_rng.get_state()),args.output/'checkpoint.pt')
    if not complete:
        atomic_json(args.output/'result.json',dict(complete=False,completed_steps=step))
        raise SystemExit(2)
    results={}
    for split,ids in [('validation',data.validation_ids),('test',data.test_ids)]:
        selected=choose_eval(data,ids,plan['eval_per_task'])
        results[split]=evaluate(adapter,data,selected,args.output/(split+'.json'),plan['eval_draws'],plan['euler_steps'])
    result=dict(complete=True,completed_steps=step,method=args.method,seed=plan['seed'],plan_sha256=plan_hash,
                elapsed_seconds=time.monotonic()-started,first_attention_max_change=changed,
                training_expert_forwards=expert_forwards,training_prefix_forwards=prefix_forwards,
                data_trace_sha256=data_trace.hexdigest(),noise_trace_sha256=noise_trace.hexdigest(),
                offline=results,scope='Exploratory fixed-compute offline pilot; no native task-success measurement')
    atomic_json(args.output/'result.json',result)
    print(json.dumps(dict(event='complete',**result)),flush=True)


if __name__=='__main__':main()

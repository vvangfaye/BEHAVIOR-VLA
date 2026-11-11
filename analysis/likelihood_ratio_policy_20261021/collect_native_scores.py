"""Collect model-only samples and readout tangents for native score fitting."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'analysis/relative_flow_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=HERE/'native_score_ada01';pp=root/'plan.json';plan=json.loads(pp.read_text())
    if (root/'collection_complete.json').exists():raise RuntimeError('Existing collection; refusing rerun')
    for path,value in plan['source_sha256'].items():
        if sha(path)!=value:raise RuntimeError('Frozen source mismatch: '+path)
    prerequisite=json.loads((HERE/'strict_fp32_probe/native_tangent_result.json').read_text())
    if not prerequisite['complete'] or not prerequisite['by_precision']['fp32']:
        raise RuntimeError('Strict native tangent prerequisite failed')
    lock=root/'collection.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.monotonic();calls=0;draws=0;receipts=[]

    def state(**kw):
        (root/'collection_state.json').write_text(json.dumps(dict(
            observed_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            elapsed_seconds=time.monotonic()-start,sampler_calls=calls,generated_draws=draws,**kw),indent=2)+'\n')

    try:
        torch.set_num_threads(4);state(stage='loading')
        base=load_base(plan['checkpoint'],plan['tokenizer'],precision='fp32')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=NativeAdapter(base).cuda().eval();data=VlaPilotData(plan['data_directory'],base)
        weight,bias=base.action_out_proj.weight,base.action_out_proj.bias
        w0,b0=weight.detach().clone(),bias.detach().clone()
        rng=torch.Generator().manual_seed(plan['direction_seed'])
        rank1=torch.randn((23,1),generator=rng)@torch.randn((1,weight.shape[1]),generator=rng)
        rank1*=w0[:23].cpu().norm()/rank1.norm()
        d_bias=torch.zeros_like(b0);d_bias[0]=.1
        d_rank=torch.zeros_like(w0);d_rank[:23]=rank1.to(w0)
        d_scale=torch.zeros_like(w0);d_scale[:23]=w0[:23]
        directions=[(bias,d_bias),(weight,d_rank),(weight,d_scale)]

        def sample(prepared,noise):
            nonlocal calls,draws
            if time.monotonic()-start>plan['max_collection_seconds']:raise TimeoutError('Collection wall cap')
            with torch.inference_mode():
                x=noise.clone()
                for k in range(plan['euler_steps']):
                    t=torch.full((len(x),),1-k/plan['euler_steps'],device=x.device)
                    x=x-adapter.velocity(prepared,x,t)/plan['euler_steps']
                output=base.decode_actions(x,prepared[0].state).float().cpu().numpy().reshape(len(x),-1)
            calls+=1;draws+=len(noise)
            if output.shape[1]!=368 or not np.isfinite(output).all():raise ValueError('Invalid decoded prefixes')
            return output

        batch=plan['batch_size'];total=plan['train_samples_per_context']+plan['test_samples_per_context']
        for index in plan['context_indices']:
            obs,_,_=data.batch([index]*batch)
            with torch.inference_mode():prepared=adapter.prepare(obs)
            generator=torch.Generator().manual_seed(plan['noise_seed']+index*17)
            xs=[];hs=[];noise_trace=hashlib.sha256();first_batch_seconds=None;batch_error=None;spot=[]
            for offset in range(0,total,batch):
                tick=time.monotonic();noise_cpu=torch.randn((batch,32,32),generator=generator)
                noise_trace.update(noise_cpu.numpy().tobytes());noise=noise_cpu.cuda()
                x=sample(prepared,noise)
                if offset==0:
                    obs1,_,_=data.batch([index])
                    with torch.inference_mode():prepared1=adapter.prepare(obs1)
                    single=sample(prepared1,noise[:1]);batch_error=float(np.max(np.abs(single[0]-x[0])))
                    if batch_error>plan['batch_parity_atol']:raise ValueError('Single versus batch output mismatch')
                    del prepared1,obs1
                tangents=[]
                for j,(parameter,direction) in enumerate(directions):
                    saved=parameter.detach().clone()
                    try:
                        epsilon=plan['half_step']
                        with torch.no_grad():parameter.copy_(saved+epsilon*direction)
                        plus=sample(prepared,noise)
                        with torch.no_grad():parameter.copy_(saved-epsilon*direction)
                        minus=sample(prepared,noise)
                        h=(plus.astype(np.float64)-minus.astype(np.float64))/(2*epsilon)
                        tangents.append(h)
                        if offset==0:
                            epsilon2=plan['spot_half_step']
                            with torch.no_grad():parameter.copy_(saved+epsilon2*direction)
                            plus2=sample(prepared,noise)
                            with torch.no_grad():parameter.copy_(saved-epsilon2*direction)
                            minus2=sample(prepared,noise)
                            h2=(plus2.astype(np.float64)-minus2.astype(np.float64))/(2*epsilon2)
                            spot.append(dict(direction=j,relative_error=float(np.linalg.norm(h-h2)/max(np.linalg.norm(h),1e-12))))
                    finally:
                        with torch.no_grad():parameter.copy_(saved)
                xs.append(x);hs.append(np.stack(tangents,1).astype(np.float32))
                assert torch.equal(weight,w0) and torch.equal(bias,b0)
                if offset==batch:first_batch_seconds=time.monotonic()-tick
                if offset%(batch*16)==0:
                    state(stage='collecting',context=index,completed=offset+batch,total=total,
                          regular_batch_seconds=first_batch_seconds)
                    print(json.dumps(dict(context=index,completed=offset+batch,total=total,
                                          elapsed_seconds=time.monotonic()-start)),flush=True)
            path=root/f'context_{index}.npz'
            np.savez_compressed(path,actions=np.concatenate(xs),tangents=np.concatenate(hs))
            receipt=dict(context=index,samples=total,train_samples=plan['train_samples_per_context'],
                         batch_parity_max_error=batch_error,spot_tangent_checks=spot,
                         regular_batch_seconds=first_batch_seconds,noise_sha256=noise_trace.hexdigest(),
                         file=path.name,file_sha256=sha(path))
            receipts.append(receipt)
            (root/f'context_{index}.json').write_text(json.dumps(receipt,indent=2)+'\n')
            del xs,hs,prepared,obs
        result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
                    plan_sha256=sha(pp),elapsed_seconds=time.monotonic()-start,sampler_calls=calls,
                    generated_draws=draws,contexts=receipts,precision='strict_fp32',
                    new_vla_training=0,new_simulator_episodes=0,goal_complete=False)
        (root/'collection_complete.json').write_text(json.dumps(result,indent=2)+'\n');state(stage='complete')
    except Exception as exc:
        state(stage='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()

"""Bounded read-only checkpoint probe; modifies only private in-memory heads."""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'analysis/relative_flow_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan_path=ROOT/'native_tangent_plan.json'
    plan=json.loads(plan_path.read_text())
    confirmation=json.loads((ROOT/'confirmation_result.json').read_text())
    if not confirmation['complete'] or not confirmation['confirmation_passed']:
        raise RuntimeError('Independent confirmation did not pass')
    for path,value in plan['source_sha256'].items():
        if sha(path)!=value:raise RuntimeError('Frozen source changed: '+path)
    output=ROOT/'native_tangent_result.json'
    if output.exists():raise RuntimeError('Existing native result; no rerun')
    lock=ROOT/'native_tangent.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.monotonic();torch.set_num_threads(4)
    rows=[];arrays={};calls=0

    def state(**kwargs):
        (ROOT/'native_tangent_state.json').write_text(json.dumps(dict(
            observed_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            elapsed_seconds=time.monotonic()-start,sampler_calls=calls,**kwargs),indent=2)+'\n')

    try:
        for precision in plan['precisions']:
            state(stage='loading',precision=precision)
            base=load_base(plan['checkpoint'],plan['tokenizer'],precision=precision)
            adapter=NativeAdapter(base).cuda().eval()
            data=VlaPilotData(plan['data_directory'],base)
            weight=base.action_out_proj.weight;bias=base.action_out_proj.bias
            weight0=weight.detach().clone();bias0=bias.detach().clone()
            rng=torch.Generator().manual_seed(plan['direction_seed'])
            rank1=torch.randn((23,1),generator=rng)@torch.randn((1,weight.shape[1]),generator=rng)
            rank1*=weight0[:23].float().cpu().norm()/rank1.norm()
            directions={}
            d=torch.zeros_like(bias0);d[0]=.1;directions['bias0']=(bias,d)
            d=torch.zeros_like(weight0);d[:23]=rank1.to(d);directions['readout_rank1']=(weight,d)
            d=torch.zeros_like(weight0);d[:23]=weight0[:23];directions['readout_scale']=(weight,d)
            for index in plan['context_indices']:
                observation,_,_=data.batch([index])
                autocast=(lambda:torch.autocast('cuda',dtype=torch.bfloat16)) if precision=='bf16' else nullcontext
                with torch.inference_mode(),autocast():prepared=adapter.prepare(observation)
                for draw in range(plan['draws']):
                    noise=torch.randn((1,32,32),generator=torch.Generator().manual_seed(plan['noise_seed']+index*17+draw)).cuda()

                    def sample():
                        nonlocal calls
                        if time.monotonic()-start>plan['max_seconds']:raise TimeoutError('Probe wall cap')
                        with torch.inference_mode(),autocast():
                            x=noise.clone()
                            for step in range(plan['euler_steps']):
                                t=torch.full((1,),1-step/plan['euler_steps'],device=x.device)
                                x=x-adapter.velocity(prepared,x,t)/plan['euler_steps']
                            decoded=base.decode_actions(x,observation.state)
                            result=decoded[0].float().cpu().numpy()
                        calls+=1
                        if result.shape!=(16,23) or not np.isfinite(result).all():raise ValueError('Invalid physical prefix')
                        return result

                    before=sample();repeat=sample()
                    repeated_error=float(np.max(np.abs(before-repeat)))
                    for name,(parameter,direction) in directions.items():
                        saved=parameter.detach().clone();tangents=[]
                        try:
                            for epsilon in plan['half_steps']:
                                with torch.no_grad():parameter.copy_(saved+epsilon*direction)
                                plus=sample()
                                with torch.no_grad():parameter.copy_(saved-epsilon*direction)
                                minus=sample()
                                tangents.append((plus.astype(np.float64)-minus.astype(np.float64))/(2*epsilon))
                        finally:
                            with torch.no_grad():parameter.copy_(saved)
                        comparisons=[]
                        for left,right in zip(tangents[:-1],tangents[1:]):
                            norm=float(np.linalg.norm(right))
                            relative=float(np.linalg.norm(left-right)/max(norm,1e-12))
                            cosine=float(np.vdot(left,right)/max(np.linalg.norm(left)*norm,1e-24))
                            comparisons.append(dict(relative_error=relative,cosine=cosine,reference_norm=norm))
                        row=dict(precision=precision,context=index,draw=draw,direction=name,
                                 repeat_max_error=repeated_error,comparisons=comparisons,
                                 smallest_scales_pass=bool(repeated_error<=1e-5 and comparisons[-1]['reference_norm']>1e-8
                                     and comparisons[-1]['relative_error']<.1 and comparisons[-1]['cosine']>.99))
                        rows.append(row)
                        key=f'{precision}_i{index}_n{draw}_{name}'
                        arrays[key]=np.stack(tangents)
                        state(stage='probing',precision=precision,context=index,draw=draw,direction=name)
                        print(json.dumps(row),flush=True)
                    assert torch.equal(weight,weight0) and torch.equal(bias,bias0)
                del prepared,observation
            del adapter,base,data,weight,bias,weight0,bias0,directions
            torch.cuda.empty_cache()
        np.savez_compressed(ROOT/'native_tangents.npz',**arrays)
        result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
                    plan_sha256=sha(plan_path),confirmation_sha256=sha(ROOT/'confirmation_result.json'),
                    elapsed_seconds=time.monotonic()-start,sampler_calls=calls,rows=rows,
                    by_precision={p:all(r['smallest_scales_pass'] for r in rows if r['precision']==p) for p in plan['precisions']},
                    scope='Static cached expert contexts; physical decoded prefix; only in-memory readout parameters perturbed. Not task success or full score estimation.',
                    new_simulator_episodes=0,new_policy_training=0,goal_complete=False)
        output.write_text(json.dumps(result,indent=2)+'\n');state(stage='complete')
    except Exception as exc:
        state(stage='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()

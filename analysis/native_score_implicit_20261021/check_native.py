"""One bounded CPU feasibility check of a full native local gradient.

No parameter updates, rollout, synthetic reward claims, or cloud resources.
"""
from pathlib import Path
import argparse, gc, hashlib, json, resource, sys, time
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/relative_flow_20261021'))
sys.path.insert(0,str(ROOT/'analysis/action_support_20261021'))
from adapter import NativeAdapter
from vla_runtime import load_base
from vla_data import VlaPilotData
from implicit_geometry import geometry, divergence_probe, local_surrogate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261021)
    start=time.monotonic();report={'complete':False,'training':False,'simulation':False,'cloud':False,
                                  'device':'cpu','precision':'float32 model; float64 geometry',
                                  'euler_steps':10,'executed_shape':[16,23]}
    def record(stage,**kw):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-start,
                      peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,**kw)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'stage':stage,'elapsed_seconds':report['elapsed_seconds'],**kw}),flush=True)
    try:
        plan=json.loads((ROOT/'analysis/action_support_20261021/local_vla_pilot_ada01/plan.json').read_text())
        record('loading')
        base=load_base(plan['checkpoint'],plan['tokenizer'],device='cpu',precision='fp32')
        adapter=NativeAdapter(base).eval()
        trainable=[(n,p) for n,p in adapter.named_parameters() if p.requires_grad]
        for p in base.parameters():p.requires_grad_(False)
        data=VlaPilotData(plan['data_directory'],base)
        index=int(data.train_ids[0]);obs,_,_=data.batch([index]);prepared=adapter.prepare(obs)
        record('prepared',index=index,episode=int(data.episode[index]),frame=int(data.frame[index]),
               intended_trainable_parameters=sum(p.numel() for _,p in trainable))
        def function(z):
            x=z.reshape(1,32,32)
            for k in range(10):
                t=torch.full((1,),1-k/10,dtype=x.dtype,device=x.device)
                x=x-adapter.velocity(prepared,x,t)/10
            return x[0,:16,:23].reshape(-1)
        z=torch.randn(1024)
        with torch.no_grad():
            reference=adapter.sample(obs,z.reshape(1,32,32),10)[0,:16,:23].reshape(-1)
            direct=function(z)
        err=float((reference-direct).abs().max())
        assert err==0,err
        record('sampler_parity',maximum_sampler_difference=err)
        zgrad=z.clone().requires_grad_(True)
        output=function(zgrad);rows=[];tick=time.monotonic()
        for first in range(0,368,8):
            last=min(368,first+8)
            eye=torch.eye(last-first,dtype=output.dtype)
            block=torch.autograd.grad(output[first:last],zgrad,grad_outputs=eye,
                                      is_grads_batched=True,retain_graph=True,create_graph=False)[0]
            rows.append(block.detach().double())
            if first==0 or last%64==0 or last==368:
                record('jacobian',rows_completed=last,jacobian_seconds=time.monotonic()-tick)
        j=torch.cat(rows);del output,rows,zgrad;gc.collect()
        b,m,s=geometry(j)
        error=float((j@b-torch.eye(368,dtype=j.dtype)).abs().max())
        assert error<1e-5,error
        record('geometry',jacobian_shape=list(j.shape),right_inverse_error=error,
               singular_max=float(s[0]),singular_min=float(s[-1]),condition_number=float(s[0]/s[-1]))
        probe=(torch.randint(0,2,(1024,))*2-1).double()
        tick=time.monotonic()
        div=divergence_probe(function,z,j,b,m,probe).detach()
        assert torch.isfinite(div).all()
        record('divergence',divergence_seconds=time.monotonic()-tick,divergence_norm=float(div.norm()))
        gc.collect()
        for _,p in trainable:p.requires_grad_(True)
        tick=time.monotonic()
        surrogate=local_surrogate(function,z,b,div,probe)
        gradients=torch.autograd.grad(surrogate,[p for _,p in trainable],allow_unused=True)
        finite=all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        assert finite
        norm=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
        nonzero=sum(int(g.count_nonzero()) for g in gradients if g is not None)
        attention=[float(g.norm()) for (n,_),g in zip(trainable,gradients)
                   if n=='base.llm.layers.0.attn.q_proj.1.weight' and g is not None]
        assert norm>0 and attention and attention[0]>0
        record('complete',complete=True,parameter_gradient_seconds=time.monotonic()-tick,
               gradient_norm=norm,nonzero_gradient_elements=nonzero,
               first_action_attention_gradient_norm=attention[0],
               scope='One native local IBP surrogate derivative, unit coefficient, one trace probe. No reward learning, variance assessment or closed-loop claim.')
    except Exception as exc:
        record('failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()

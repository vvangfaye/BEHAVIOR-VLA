"""Read-only same-observation, same-noise checkpoint audit; no simulator."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import urllib.request

import numpy as np
import torch

from common import torch_model, write
from training import hash_parameters, load_overlay


def upload(url, data):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, data=data, method='PUT'), timeout=90):pass


def brief(values):
    x = np.asarray(values, dtype=np.float64)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'max':float(x.max()),'p95':float(np.quantile(x,.95))}


def main():
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    root = Path(plan['output']);root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',plan)
    started=time.time()
    status={'state':'running','stage':'loading_data','started_at':started}
    def publish(**fields):
        status.update(fields,updated_at=time.time());write(root/'status.json',status)
        upload(os.environ['B1K_STATUS_PUT'],json.dumps(status).encode())
    def timeout(*args):raise TimeoutError('Read-only audit exceeded 50-minute compute budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['compute_timeout_seconds'])
    report={}
    try:
        samples=[]
        for r in (1,2):
            for e in range(1,5):
                directory=Path(plan['audit']['source_results'])/'training'/f'round_{r:02d}'/f'episode_{e:02d}'
                path=directory/'rollout.pt'
                if not path.exists():path=Path(json.loads((directory/'recovery.json').read_text())['rollout_path'])
                data=torch.load(path,map_location='cpu',weights_only=True)
                rows=data['transitions']
                for position in (0,len(rows)//2,len(rows)-1):
                    samples.append({'id':f'r{r}_e{e}_t{position}','source_round':r,'episode':e,
                                    'position':position,'row':rows[position]})
        assert len(samples)==24
        publish(stage='loading_model',samples=len(samples))
        model=torch_model(plan)
        names=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        base={n:p.detach().cpu().clone() for n,p in names}
        frozen_before=hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        assert hash_parameters(names)==plan['expected_original_trainable_sha256']
        outputs={};statistics={};raw_arrays={}
        for ck in plan['audit']['checkpoints']:
            if ck['directory']:load_overlay(model,ck['directory'],ck['hash'])
            assert hash_parameters(names)==ck['hash']
            assert hash_parameters((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_before
            delta_sq=0.;base_sq=0.;max_abs=0.;changed=[];groups={}
            for name,parameter in names:
                original=base[name].float();current=parameter.detach().cpu().float();diff=current-original
                ds=float(diff.double().square().sum());bs=float(original.double().square().sum());ma=float(diff.abs().max())
                delta_sq+=ds;base_sq+=bs;max_abs=max(max_abs,ma)
                if ma:changed.append(name)
                group=name.split('.')[0];g=groups.setdefault(group,{'delta_sq':0.,'base_sq':0.,'max_abs':0.,'changed':0})
                g['delta_sq']+=ds;g['base_sq']+=bs;g['max_abs']=max(g['max_abs'],ma);g['changed']+=int(ma>0)
                assert torch.isfinite(current).all()
            for g in groups.values():g['relative_l2']=(g['delta_sq']/max(g['base_sq'],1e-30))**.5
            statistics[ck['name']]={'relative_l2':(delta_sq/max(base_sq,1e-30))**.5,'max_abs':max_abs,
                                    'changed_tensors':len(changed),'groups':groups,'trainable_sha256':ck['hash']}
            records=[]
            for i,sample in enumerate(samples):
                row=sample['row'];inputs={k:v.to(model.device) for k,v in row['forward_inputs'].items()}
                with torch.no_grad():
                    actual=model.default_forward(inputs,compute_values=False)['logprobs'].float()
                    old=row['prev_logprobs'].to(model.device).float()
                    logratio=float((actual-old).sum())
                    replay_error=float((actual-old).abs().max())
                    obs=model._observation_from_forward_inputs(inputs)
                    noise=torch.from_numpy(np.random.default_rng(6000+i).standard_normal((1,32,32)).astype(np.float32)).to(model.device)
                    ode,_=model._predict_eval(obs,noise=noise.clone(),rng=None)
                    random.seed(8000+i)
                    rng=torch.Generator(device=model.device).manual_seed(9000+i)
                    sde,_=model._predict_train(obs,noise=noise.clone(),rng=rng,compute_values=False)
                ode=ode.detach().cpu().numpy()[0];sde=sde.detach().cpu().numpy()[0]
                raw_arrays[f"{ck['name']}_{sample['id']}_ode"]=ode
                raw_arrays[f"{ck['name']}_{sample['id']}_sde"]=sde
                assert np.isfinite(ode).all() and np.isfinite(sde).all()
                records.append({'id':sample['id'],'source_round':sample['source_round'],'episode':sample['episode'],
                                'replay_error':replay_error,'sampled_conditional_log_ratio':logratio,
                                'ode_sde_mae':float(abs(ode-sde).mean())})
                publish(stage='same_observation_audit',checkpoint=ck['name'],completed_samples=i+1,total_samples=len(samples))
            outputs[ck['name']]=records
        contrasts={}
        for left,right in [('base','round_1'),('base','round_2'),('round_1','round_2')]:
            for sampler in ('ode','sde'):
                ma=[];mx=[];bydim=[]
                for sample in samples:
                    d=abs(raw_arrays[f"{left}_{sample['id']}_{sampler}"]-raw_arrays[f"{right}_{sample['id']}_{sampler}"])
                    ma.append(float(d.mean()));mx.append(float(d.max()));bydim.append(d.mean(0))
                contrasts[f'{left}_to_{right}_{sampler}']={'action_mae':brief(ma),'sample_max_delta':brief(mx),
                    'mean_absolute_delta_by_action_dimension':np.stack(bydim).mean(0).tolist()}
        replay_checks={}
        for ck,r in [('base',1),('round_1',2)]:
            values=[x['replay_error'] for x in outputs[ck] if x['source_round']==r]
            replay_checks[ck]=brief(values)
            assert max(values)<=1e-4, 'Saved rollout is not from its claimed policy'
        report={'passed':True,'samples':len(samples),'frozen_parameters_unchanged':True,'frozen_hash':frozen_before,
                'weights':statistics,'sample_results':outputs,'contrasts':contrasts,'on_policy_replay_checks':replay_checks,
                'interpretation_limits':['Conditional actions on saved training observations and labels, not a new success-rate test.',
                  'ODE and SDE probes share initial noise; SDE additionally uses a fixed selected denoise step and step noise.',
                  'This audit does not identify unseen-state or long-horizon failure causes by itself.']}
        np.savez_compressed(root/'actions.npz',**raw_arrays)
        write(root/'audit.json',report)
        upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        status.update(state='complete',stage='complete',elapsed_seconds=time.time()-started)
        publish(**status)
    except BaseException as error:
        report={'passed':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        write(root/'failure.json',report)
        upload(os.environ['B1K_RESULT_PUT'],json.dumps(report).encode())
        publish(state='failed',stage='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()

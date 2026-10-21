"""Fit and check native prefix parameter scores after model-only collection."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ROOT=Path(__file__).resolve().parent/'native_score_ada01'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def witnesses(x,h,projection,center):
    p=x@projection.T
    tangent=np.einsum('njd,kd->njk',h,projection)
    values=[p,p*p];derivatives=[np.ones_like(p),2*p]
    for frequency in [.5,1.,2.]:
        values.extend([np.sin(frequency*p),np.cos(frequency*p)])
        derivatives.extend([frequency*np.cos(frequency*p),-frequency*np.sin(frequency*p)])
    f=np.concatenate(values,1)
    d=np.concatenate([tangent*q[:,None,:] for q in derivatives],2).transpose(0,2,1)
    return f-center,d


def fit(index,plan):
    result_path=ROOT/f'estimator_{index}.json'
    if result_path.exists():raise RuntimeError('Existing estimator result; refusing rerun')
    path=ROOT/f'context_{index}.npz';receipt=json.loads((ROOT/f'context_{index}.json').read_text())
    if sha(path)!=receipt['file_sha256']:raise ValueError('Collected sample checksum mismatch')
    if receipt['batch_parity_max_error']>plan['batch_parity_atol']:raise ValueError('Batch parity failed')
    if max(x['relative_error'] for x in receipt['spot_tangent_checks'])>.1:
        raise ValueError('Tangent spot checks exceed numerical tolerance')
    with np.load(path,allow_pickle=False) as z:x=z['actions'].astype(np.float64);h=z['tangents'].astype(np.float64)
    n=plan['train_samples_per_context'];p=plan['estimator'];started=time.monotonic()
    mean=x[:n].mean(0);std=x[:n].std(0);positive=std[std>1e-8]
    if not len(positive):raise ValueError('No varying action coordinates')
    floor=.02*np.median(positive);std=np.maximum(std,floor)
    x=(x-mean)/std;h=h/std[None,None,:]
    train=torch.from_numpy(x[:n]).float();ht=torch.from_numpy(h[:n]).float()
    torch.manual_seed(p['seed']+index);rng=torch.Generator().manual_seed(p['seed']+index+1000)
    widths=p['hidden_layers'];model=nn.Sequential(nn.Linear(x.shape[1],widths[0]),nn.SiLU(),
        nn.Linear(widths[0],widths[1]),nn.SiLU(),nn.Linear(widths[1],3))
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    for step in range(p['updates']):
        ids=torch.randint(n,(p['batch_size'],),generator=rng)
        q=train[ids].clone().requires_grad_(True)
        v=(2*torch.randint(2,(len(ids),3),generator=rng)-1).float()
        score=(model(q)*v).sum(1)
        derivative=torch.autograd.grad(score.sum(),q,create_graph=True)[0]
        response=(ht[ids]*v[:,:,None]).sum(1)
        loss=(.5*score.square()-(derivative*response).sum(1)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite score fitting loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    features=model[:-1].double().eval();fsize=widths[-1]+1
    gram=torch.zeros((fsize,fsize),dtype=torch.float64);rhs=torch.zeros((fsize,3),dtype=torch.float64)
    # Only the small score network is differentiated here. Exact first-order
    # JVPs avoid cancellation when a standardized native tangent is large.
    with torch.no_grad():
        for begin in range(0,n,128):
            q=torch.from_numpy(x[begin:min(begin+128,n)]);hs=torch.from_numpy(h[begin:min(begin+128,n)])
            f=torch.cat([features(q),torch.ones((len(q),1),dtype=torch.float64)],1)
            gram+=f.T@f/n
            for j in range(3):
                _,df=torch.func.jvp(features,(q,),(hs[:,j],))
                rhs[:-1,j]+=df.sum(0)/n
        ridge=p['head_ridge_trace_multiplier']*float(gram.trace())/fsize
        penalty=torch.eye(fsize,dtype=torch.float64)*ridge;penalty[-1,-1]=0
        weights=torch.linalg.solve(gram+penalty,rhs)
        test=torch.from_numpy(x[n:])
        f=torch.cat([features(test),torch.ones((len(test),1),dtype=torch.float64)],1)
        estimated=(f@weights).numpy()
    vr=plan['verification'];wrng=np.random.default_rng(vr['witness_seed'])
    projection=wrng.normal(size=(vr['random_projection_count'],x.shape[1]));projection/=np.linalg.norm(projection,axis=1,keepdims=True)
    ftrain,_=witnesses(x[:n],h[:n],projection,0.)
    ftest,dtest=witnesses(x[n:],h[n:],projection,ftrain.mean(0))
    predicted=ftest[:,:,None]*estimated[:,None,:]
    residual=predicted-dtest;delta=residual.mean(0);reference=dtest.mean(0)
    residual_se=residual.std(0,ddof=1)/len(test)**.5
    reference_se=dtest.std(0,ddof=1)/len(test)**.5
    informative=np.abs(reference)>2*reference_se
    count=int(informative.sum())
    relative=float(np.linalg.norm(delta[informative])/max(np.linalg.norm(reference[informative]),1e-12))
    coverage=float((np.abs(delta)<=3*residual_se+1e-12).mean())
    passed=count>=12 and relative<.25 and coverage>=.9
    np.savez_compressed(ROOT/f'estimator_{index}_moments.npz',reference=reference,predicted=predicted.mean(0),
        residual_se=residual_se,reference_se=reference_se,informative=informative,projection=projection)
    torch.save(dict(features=features.state_dict(),weights=weights,mean=mean,std=std,
                    context=index,source_sha256=sha(path)),ROOT/f'estimator_{index}.pt')
    result=dict(context=index,completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        training_samples=n,test_samples=len(test),informative_moments=count,
        relative_informative_moment_error=relative,three_se_coverage=coverage,passed=bool(passed),
        ridge=ridge,normalization_std_floor=float(floor),source_sha256=sha(path),
        script_sha256=sha(Path(__file__)),scope='Held-out model moment identities only, not task success or density accuracy everywhere.')
    result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result


def main():
    plan=json.loads((ROOT/'plan.json').read_text());torch.set_num_threads(4)
    protocol=json.loads((ROOT/'estimator_protocol.json').read_text())
    if protocol['script_sha256']!=sha(Path(__file__)):raise ValueError('Estimator source changed')
    collection=json.loads((ROOT/'collection_complete.json').read_text())
    if not collection['complete'] or collection['plan_sha256']!=sha(ROOT/'plan.json'):
        raise ValueError('Incomplete or mismatched collection')
    rows=[fit(i,plan) for i in plan['context_indices']]
    result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),contexts=rows,
                native_moment_gate_passed=all(r['passed'] for r in rows),
                scope='Frozen three-context model-only score estimation; no VLA update or simulator reward measured.',
                new_vla_training=0,new_simulator_episodes=0,goal_complete=False)
    (ROOT/'estimator_result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

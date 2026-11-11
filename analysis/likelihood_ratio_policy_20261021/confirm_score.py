"""Independent synthetic geometry for the fixed weak-score / head-solve recipe."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def generate(n, d, rng, rotation, plan):
    z = torch.randn((n,d+1),generator=rng)
    mode = torch.where(torch.rand(n,generator=rng)<plan['mode_probability_positive'],1.,-1.)
    raw = z[:,0]+plan['extra_noise_coefficient']*z[:,-1]
    x = z[:,:d].clone()
    x[:,0]=plan['mode_means'][0]*mode+raw
    x[:,1]+=plan['mode_means'][1]*mode
    y=(x+.15*torch.sin(x))@rotation.T
    # Every parameter's tangent is a scalar times this observation-space basis.
    basis=(1+.15*torch.cos(x[:,0,None]))*rotation[:,0][None,:]
    coefficients=torch.stack([torch.ones(n),raw,z[:,1]],1)
    return x,y,basis,coefficients


def truth(x,plan):
    variance=1+plan['extra_noise_coefficient']**2
    mean=np.zeros(x.shape[1]);mean[:2]=plan['mode_means']
    residual=np.stack([x-mean,x+mean],1)
    energy=residual[:,:,0]**2/variance+residual[:,:,1]**2
    probs=np.asarray([plan['mode_probability_positive'],1-plan['mode_probability_positive']])
    logits=-.5*energy+np.log(probs)[None,:];logits-=logits.max(1,keepdims=True)
    weights=np.exp(logits);weights/=weights.sum(1,keepdims=True)
    r0,r1=residual[:,:,0],residual[:,:,1]
    scores=(weights[:,:,None]*np.stack([r0/variance,r0*r0/variance-1,r0*r1/variance],-1)).sum(1)
    directions=np.asarray([[1.,0.],[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]])
    advantages,gradients=[],[]
    for w,threshold in zip(directions,plan['event_thresholds']):
        var=variance*w[0]**2+w[1]**2;std=math.sqrt(var)
        dmean=np.asarray([w[0],0.,0.]);dvar=np.asarray([0.,2*variance*w[0]**2,2*w[0]*w[1]])
        expected,gradient=0.,np.zeros(3)
        for mode,p in zip([1,-1],probs):
            z=(threshold-mode*(w@mean[:2]))/std
            expected+=p*.5*math.erfc(z/math.sqrt(2))
            gradient+=p*math.exp(-z*z/2)/math.sqrt(2*math.pi)*(dmean/std+z*dvar/(2*var))
        advantages.append((x[:,:2]@w>threshold).astype(np.float64)-expected)
        gradients.append(gradient)
    return scores,np.stack(advantages,1),np.stack(gradients)


def main():
    pp=ROOT/'confirmation_plan.json';plan=json.loads(pp.read_text())
    output=ROOT/'confirmation_result.json'
    if output.exists():raise RuntimeError('Existing result; no rerun')
    torch.set_num_threads(4);start=time.monotonic();rows=[]
    for d in plan['dimensions']:
        rotation=torch.linalg.qr(torch.randn((d,d),generator=torch.Generator().manual_seed(plan['rotation_seed']+d)))[0]
        for seed in plan['seeds']:
            torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+1000+d)
            _,train,basis,coeff=generate(plan['training_samples'],d,rng,rotation,plan)
            latent_test,test,_,_=generate(plan['test_samples'],d,rng,rotation,plan)
            width=plan['hidden_width']
            model=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3))
            opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
            for update in range(plan['updates']):
                if time.monotonic()-start>plan['max_cpu_seconds']:raise TimeoutError('Fixed CPU cap')
                index=torch.randint(len(train),(plan['batch_size'],),generator=rng)
                x=train[index].clone().requires_grad_(True)
                v=(2*torch.randint(2,(len(index),3),generator=rng)-1).float()
                projected=(model(x)*v).sum(1)
                derivative=torch.autograd.grad(projected.sum(),x,create_graph=True)[0]
                response=basis[index]*(coeff[index]*v).sum(1)[:,None]
                loss=(.5*projected.square()-(derivative*response).sum(1)).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            features=model[:-1].double().eval()
            gram=torch.zeros((width+1,width+1),dtype=torch.float64);rhs=torch.zeros((width+1,3),dtype=torch.float64)
            eps=plan['feature_derivative_half_step']
            with torch.inference_mode():
                for begin in range(0,len(train),2000):
                    x=train[begin:begin+2000].double();b=basis[begin:begin+2000].double();h=coeff[begin:begin+2000].double()
                    f=torch.cat([features(x),torch.ones((len(x),1),dtype=torch.float64)],1)
                    df=torch.cat([(features(x+eps*b)-features(x-eps*b))/(2*eps),torch.zeros((len(x),1),dtype=torch.float64)],1)
                    gram+=f.T@f/len(train);rhs+=df.T@h/len(train)
                ridge=plan['ridge_trace_multiplier']*float(gram.trace())/(width+1)
                penalty=torch.eye(width+1,dtype=torch.float64)*ridge;penalty[-1,-1]=0
                weights=torch.linalg.solve(gram+penalty,rhs)
                prediction=[]
                for x in test.double().split(10000):
                    f=torch.cat([features(x),torch.ones((len(x),1),dtype=torch.float64)],1)
                    prediction.append(f@weights)
                prediction=torch.cat(prediction).numpy()
            expected,advantage,gradient=truth(latent_test.numpy().astype(np.float64),plan)
            estimate=advantage.T@prediction/len(prediction)
            error=float(np.linalg.norm(estimate-gradient)/np.linalg.norm(gradient))
            cosine=(estimate*gradient).sum(1)/(np.linalg.norm(estimate,axis=1)*np.linalg.norm(gradient,axis=1))
            mse=float(np.square(prediction-expected).mean()/np.square(expected).mean())
            oracle=advantage.T@expected/len(prediction)
            row=dict(dimension=d,seed=seed,relative_gradient_error=error,event_cosines=cosine.tolist(),
                     relative_score_mse=mse,oracle_mc_relative_error=float(np.linalg.norm(oracle-gradient)/np.linalg.norm(gradient)),
                     passed=bool(error<.25 and np.all(cosine>.9) and mse<.25))
            rows.append(row);print(json.dumps(row),flush=True)
            # Persist each fitted estimator for inspection; no checkpoint selection.
            torch.save(dict(features=features.state_dict(),weights=weights,rotation=rotation,
                            dimension=d,seed=seed),ROOT/f'confirmation_d{d}_s{seed}.pt')
    result=dict(complete=True,observed_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                elapsed_seconds=time.monotonic()-start,results=rows,
                confirmation_passed=all(r['passed'] for r in rows),
                scope='Changed synthetic geometry only; not native VLA performance or proof of novelty.',
                new_gpu_jobs=0,new_simulator_episodes=0,goal_complete=False)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

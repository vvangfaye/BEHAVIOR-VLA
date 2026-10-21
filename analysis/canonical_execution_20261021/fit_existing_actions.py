"""Bounded CPU head feasibility; no visual VLA or closed-loop claim."""
from pathlib import Path
import sys, json, hashlib, time, copy
import numpy as np
import torch
from torch import nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'action_support_20261021'))
from train_activity import data
from head import CanonicalExecutionHead, two_sample_energy


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    rows = data()
    tasks = sorted({r['task'] for r in rows})
    fold = {}
    for task in tasks:
        eps = sorted({r['episode'] for r in rows if r['task'] == task},
                     key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())
        fold.update({e:i % 5 for i,e in enumerate(eps)})
    inds = {name:np.array([i for i,r in enumerate(rows) if pred(fold[r['episode']])])
            for name,pred in [('train',lambda f:f>=2),('validation',lambda f:f==1),('test',lambda f:f==0)]}
    x = np.stack([r['x'] for r in rows]).astype('float32')
    previous = x[:,61:84].copy()
    y = (np.stack([r['future'][:16] for r in rows])-previous[:,None,:]).reshape(len(rows),-1)
    x = np.concatenate([x,np.eye(len(tasks),dtype='float32')[[tasks.index(r['task']) for r in rows]]],-1)
    tr=inds['train']; xm=x[tr].mean(0); xs=np.maximum(x[tr].std(0),.01)
    ym=y[tr].mean(0); ys=np.maximum(y[tr].std(0),.01)
    xt=torch.from_numpy(np.clip((x-xm)/xs,-10,10))
    yt=torch.from_numpy(((y-ym)/ys).astype('float32'))
    lo=yt[tr].amin(0); hi=yt[tr].amax(0)
    center=(lo+hi)/2; radius=(hi-lo)/2+1e-4
    models={
        'canonical':CanonicalExecutionHead(x.shape[-1],368,512,center=center,radius=radius,temperature=3.),
        'deterministic':nn.Sequential(nn.Linear(x.shape[-1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,368))}
    optimizers={k:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4) for k,m in models.items()}
    protocol={'seed':20261021,'updates':800,'batch_size':32,'learning_rate':.001,
              'normalization':'train-only feature z-score clipped 10, residual action coordinate z-score floor .01',
              'action':'next 16 x 23 commands minus most recent command',
              'split':'within-task sha256 episode ordering; train folds 2,3,4; validation 1; test 0',
              'features':'proprioception and past commands, task one-hot; no image, no future input',
              'counts':{k:len(v) for k,v in inds.items()},
              'selection':'minimum fixed-noise validation energy score per model; test once',
              'scope':'action-head feasibility only; deterministic model is fitting sanity comparator, not a strong generative baseline'}
    (HERE/'fit_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    val=inds['validation']; generator=torch.Generator().manual_seed(6301)
    fixed=torch.randn((len(val),2,368),generator=generator)

    def evaluate(name, ix, noises):
        m=models[name]
        with torch.no_grad():
            terms=[]; squared=[]
            for offset in range(0,len(ix),32):
                ii=ix[offset:offset+32]; xx=xt[ii]; yy=yt[ii]
                if name=='canonical':
                    a=m(xx,noises[offset:offset+len(ii),0]); b=m(xx,noises[offset:offset+len(ii),1])
                else: a=b=m(xx)
                terms.append(float(two_sample_energy(a,b,yy))*len(ii))
                squared.append(float(((a+b)/2-yy).square().mean())*len(ii))
        return {'energy':sum(terms)/len(ix),'two_draw_mean_mse':sum(squared)/len(ix)}

    best={k:float('inf') for k in models}; weights={}; logs=[]; start=time.monotonic()
    for step in range(801):
        if step%100==0:
            result={name:evaluate(name,val,fixed) for name in models}
            for name in models:
                if result[name]['energy']<best[name]:
                    best[name]=result[name]['energy']; weights[name]=copy.deepcopy(models[name].state_dict())
            row={'step':step,'validation':result,'seconds':time.monotonic()-start}
            logs.append(row); print(json.dumps(row),flush=True)
        if step==800:break
        ii=tr[torch.randint(len(tr),(32,)).numpy()]; xx=xt[ii]; yy=yt[ii]
        for name,m in models.items():
            if name=='canonical':a=m(xx,torch.randn(len(ii),368)); b=m(xx,torch.randn(len(ii),368))
            else:a=b=m(xx)
            loss=two_sample_energy(a,b,yy)
            optimizers[name].zero_grad();loss.backward();optimizers[name].step()
    testnoise=torch.randn((len(inds['test']),2,368),generator=torch.Generator().manual_seed(6302))
    test={}
    for name in models:
        models[name].load_state_dict(weights[name]);test[name]=evaluate(name,inds['test'],testnoise)
    outside={k:float(((yt[ix]<lo)|(yt[ix]>hi)).float().mean()) for k,ix in inds.items()}
    report={'protocol':protocol,'validation_curve':logs,'test':test,
            'target_coordinate_fraction_outside_training_bounds':outside,
            'scope':protocol['scope'],'vla_fidelity_verified':False,'closed_loop_verified':False}
    (HERE/'fit_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test':test,'outside':outside,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()

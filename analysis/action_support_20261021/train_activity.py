"""Episode-held-out activity predictability using proprio/history, not vision."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent


def data():
    receipt=json.loads((HERE/'training_column_receipt.json').read_text())
    examples=[]
    for record in receipt['records']:
        path=HERE/record['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        with np.load(path) as f:
            a,s,ep,frame,task=[f[k] for k in ['action','observation.state','episode_index','frame_index','task_index']]
        for episode in np.unique(ep):
            ix=np.flatnonzero(ep==episode);ix=ix[np.argsort(frame[ix])]
            for k in range(32,len(ix)-31,128):
                window=ix[k-32:k+32]
                if not np.array_equal(np.diff(frame[window]),np.ones(63)):continue
                past=a[ix[k-32:k]];future=a[ix[k:k+32]]
                feature=np.concatenate([s[ix[k]],past[-1],past[-1]-past[-2],np.ptp(past,axis=0)])
                examples.append({'episode':int(episode),'task':int(task[ix[k]]),'frame':int(frame[ix[k]]),
                                 'x':feature,'y':np.ptp(future,axis=0)>0,'persistent':np.ptp(past,axis=0)>0,
                                 'future':future.copy()})
    return examples


def summarize(y,p,threshold,variable):
    pred=p>=threshold
    active=y.astype(bool)
    return {'brier':float(np.mean((p-y)**2)),
            'active_recall':float(pred[active].mean()) if active.any() else None,
            'flat_recall':float((~pred[~active]).mean()) if (~active).any() else None,
            'flat_precision':float((~active[~pred]).mean()) if (~pred).any() else None,
            'predicted_flat_fraction':float((~pred).mean()),
            'variable_channel_brier':float(np.mean((p[:,variable]-y[:,variable])**2))}


def main():
    torch.set_num_threads(2)
    rows=data();tasks=sorted(set(r['task'] for r in rows));episodes=sorted(set(r['episode'] for r in rows))
    fold={}
    for task in tasks:
        eps=sorted(set(r['episode'] for r in rows if r['task']==task),key=lambda x:hashlib.sha256(str(x).encode()).hexdigest())
        for i,e in enumerate(eps):fold[e]=i%5
    x=np.stack([r['x'] for r in rows]).astype(np.float32);y=np.stack([r['y'] for r in rows]).astype(np.float32)
    t=np.array([tasks.index(r['task']) for r in rows]);x=np.concatenate([x,np.eye(len(tasks),dtype=np.float32)[t]],-1)
    protocol={'seeds':[17,29,43],'epochs':30,'batch':256,'lr':.001,'weight_decay':.001,'stride':128,'horizon':32,
              'split':'episode hash order, 5 folds within task; test fold f, validation f+1, training rest',
              'features':'current proprio and strictly past commands only; no vision or labels as inputs',
              'threshold':'validation fifth percentile among active targets, targeting 95% recall',
              'episodes':episodes,'fold':fold}
    path=HERE/'activity_protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(protocol))
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    records=[];start=time.monotonic()
    exports=HERE/'activity_predictions';exports.mkdir(exist_ok=True)
    for f in range(5):
        tr=np.array([fold[r['episode']] not in [f,(f+1)%5] for r in rows]);va=np.array([fold[r['episode']]==(f+1)%5 for r in rows]);te=np.array([fold[r['episode']]==f for r in rows])
        assert all(mask.any() for mask in [tr,va,te])
        mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),.01);xt=torch.from_numpy(np.clip((x-mean)/std,-10,10));yt=torch.from_numpy(y)
        ti=torch.tensor(np.flatnonzero(tr));vi=torch.tensor(np.flatnonzero(va));ei=torch.tensor(np.flatnonzero(te))
        variable=(y[tr].sum(0)>=10)&((1-y[tr]).sum(0)>=10)
        for seed in protocol['seeds']:
            torch.manual_seed(seed)
            model=nn.Sequential(nn.Linear(x.shape[-1],128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,23))
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(30):
                for inds in ti[torch.randperm(len(ti))].split(256):
                    loss=F.binary_cross_entropy_with_logits(model(xt[inds]),yt[inds]);opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():pv=model(xt[vi]).sigmoid().numpy();pe=model(xt[ei]).sigmoid().numpy()
            threshold=float(np.quantile(pv[y[va].astype(bool)],.05))
            prior=np.stack([y[tr&(t==task)].mean(0) for task in range(len(tasks))])
            prior_test=prior[t[te]];prior_val=prior[t[va]]
            prior_threshold=float(np.quantile(prior_val[y[va].astype(bool)],.05))
            persistence=np.stack([r['persistent'] for r in rows])[te].astype(float)
            record={'fold':f,'seed':seed,'train_episodes':len(set(r['episode'] for r,ok in zip(rows,tr) if ok)),
                    'test_episodes':len(set(r['episode'] for r,ok in zip(rows,te) if ok)), 'test_chunks':int(te.sum()),
                    'model':summarize(y[te],pe,threshold,variable), 'model_threshold':threshold,
                    'task_prior':summarize(y[te],prior_test,prior_threshold,variable),
                    'persistence':summarize(y[te],persistence,.5,variable)}
            np.savez_compressed(exports/f'fold{f}_seed{seed}.npz', probabilities=pe, threshold=np.array(threshold),
                                index=np.flatnonzero(te),future=np.stack([rows[i]['future'] for i in np.flatnonzero(te)]),
                                episode=np.array([rows[i]['episode'] for i in np.flatnonzero(te)]))
            records.append(record);print(json.dumps(record),flush=True)
    result={'examples':len(rows),'episodes':len(episodes),'tasks':tasks,'seconds':time.monotonic()-start,'fold_results':records,
            'mean_metrics':{method:{metric:float(np.mean([r[method][metric] for r in records if r[method][metric] is not None]))
                                   for metric in records[0][method]} for method in ['model','task_prior','persistence']},
            'scope':'proprio/past-command mask predictability only; no trained full VLA or closed-loop benefit'}
    (HERE/'activity_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='fold_results'},indent=2))


if __name__=='__main__':main()

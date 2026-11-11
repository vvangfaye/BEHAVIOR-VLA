"""Fixed-protocol observation-based critic screen, with held-out instances."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from outcome_head import OutcomeHead

HERE = Path(__file__).resolve().parent
METHODS = ('scalar', 'scalar_aux', 'independent_product', 'joint')
SEEDS = (17, 29, 43)


class Critic(nn.Module):
    def __init__(self, input_dim, clause_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.scalar = nn.Linear(64, 1)
        self.outcome = OutcomeHead(64, clause_dim, width=32, layers=1, heads=2)

    def evaluate(self, x, clauses, y, valid, method):
        h = self.trunk(x)
        scalar = self.scalar(h).squeeze(-1)
        if method == 'scalar':
            return F.binary_cross_entropy_with_logits(scalar, y.bool().logical_or(~valid).all(-1).float()), scalar.sigmoid()
        targets = y if method == 'joint' else torch.zeros_like(y)
        logits = self.outcome(h, clauses, targets, valid)
        bits_loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none')*valid).sum(-1)/valid.sum(-1)
        if method == 'scalar_aux':
            success = y.bool().logical_or(~valid).all(-1).float()
            return F.binary_cross_entropy_with_logits(scalar, success)+bits_loss.mean(), scalar.sigmoid()
        if method == 'joint':
            prob = self.outcome.success_log_probability(h, clauses, valid).exp()
        else:
            prob = (F.logsigmoid(logits)*valid).sum(-1).exp()
        return bits_loss.mean(), prob


def load_data():
    manifest = json.loads((HERE/'feature_manifest.json').read_text())
    assert manifest['complete']
    labels = {(x['task'],x['instance']):x for x in json.loads((HERE/'labels.json').read_text())}
    rows = []
    for r in sorted(manifest['records'], key=lambda r:(r['task'],r['instance'])):
        p = HERE/r['features']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['features_sha256']
        with np.load(p, allow_pickle=False) as f:
            assert f['frames'].tolist() == [0,1500,3000]
            rows.append({**labels[r['task'],r['instance']], 'vision':f['vision'].reshape(-1), 'state':f['state'].reshape(-1)})
    return rows


def prepare(rows, train, test, task_names):
    # Fit every feature transformation on training instances only.
    v = torch.tensor(np.stack([r['vision'] for r in rows]))
    s = torch.tensor(np.stack([r['state'] for r in rows]))
    vm, vs = v[train].mean(0), v[train].std(0).clamp_min(.01)
    normalized = ((v-vm)/vs).clamp(-10,10)
    _, _, vh = torch.linalg.svd(normalized[train], full_matrices=False)
    projected = normalized @ vh[:16].T
    projected /= projected[train].std(0).clamp_min(.01)
    s = ((s-s[train].mean(0))/s[train].std(0).clamp_min(.01)).clamp(-10,10)
    task_ids = torch.tensor([task_names.index(r['task']) for r in rows])
    task_vec = F.one_hot(task_ids,len(task_names)).float()
    x = torch.cat((projected,s,task_vec),-1)
    m = max(len(r['bits']) for r in rows)
    valid = torch.tensor([[j<len(r['bits']) for j in range(m)] for r in rows])
    y = torch.tensor([r['bits']+[0]*(m-len(r['bits'])) for r in rows]).float()
    cid = task_ids[:,None]*m + torch.arange(m)[None]
    clauses = F.one_hot(cid,len(task_names)*m).float()
    return x, clauses, y, valid


def metrics(rows, predictions, select=None):
    inds = [i for i,r in enumerate(rows) if select is None or select(r)]
    y = np.array([rows[i]['success'] for i in inds])
    p = np.clip(np.array(predictions)[inds],1e-7,1-1e-7)
    return {'n':len(y),'successes':int(y.sum()),'brier':float(np.mean((p-y)**2)),
            'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))}


def main():
    torch.set_num_threads(2)
    rows = load_data(); tasks=sorted(set(r['task'] for r in rows))
    folds = {}
    for task in tasks:
        inds=sorted([i for i,r in enumerate(rows) if r['task']==task],
                    key=lambda i:hashlib.sha256(f"{task}:{rows[i]['instance']}".encode()).hexdigest())
        for rank,i in enumerate(inds): folds[i]=rank%5
    protocol={'methods':METHODS,'seeds':SEEDS,'folds':5,'epochs':200,'lr':.001,'weight_decay':.001,
              'visual_pca_dimensions':16,'early_stopping':False,'primary':'multi-clause held-out-instance Brier; joint versus scalar_aux',
              'split':[{ 'task':r['task'],'instance':r['instance'],'fold':folds[i]} for i,r in enumerate(rows)],
              'features':'frozen original SigLIP; 3 camera x 3 fixed times; robot state; task identity',
              'clause_embedding':'learned task/clause-ID representation; no cross-task semantic generalization claimed',
              'no_terminal_features':True}
    pp=HERE/'training_protocol.json'
    if pp.exists(): assert json.loads(pp.read_text())==json.loads(json.dumps(protocol))
    else: pp.write_text(json.dumps(protocol,indent=2)+'\n')
    predictions={method:[] for method in METHODS}
    started=time.monotonic()
    for seed in SEEDS:
        pred={method:[None]*len(rows) for method in METHODS}
        for fold in range(5):
            train=[i for i in range(len(rows)) if folds[i]!=fold]; test=[i for i in range(len(rows)) if folds[i]==fold]
            x,c,y,v=prepare(rows,train,test,tasks)
            torch.manual_seed(seed)
            initial=Critic(x.shape[1],c.shape[-1])
            for method in METHODS:
                model=copy.deepcopy(initial)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                model.train()
                for epoch in range(200):
                    opt.zero_grad();loss,_=model.evaluate(x[train],c[train],y[train],v[train],method)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                model.eval()
                with torch.no_grad(): _,p=model.evaluate(x[test],c[test],y[test],v[test],method)
                assert p.isfinite().all()
                for i,value in zip(test,p.tolist()): pred[method][i]=value
                print(json.dumps({'seed':seed,'fold':fold,'method':method,'seconds':round(time.monotonic()-started,1)}),flush=True)
        for method in METHODS: predictions[method].append(pred[method])
    scores={method:{'seeds':[], 'all':metrics(rows,np.mean(ps,axis=0)),
                    'multi_clause':metrics(rows,np.mean(ps,axis=0),lambda r:len(r['bits'])>1)} for method,ps in predictions.items()}
    for method,ps in predictions.items():
        scores[method]['seeds']=[metrics(rows,p,lambda r:len(r['bits'])>1) for p in ps]
    # Same instance resampled together across methods; seeds averaged before bootstrap.
    joint=np.mean(predictions['joint'],axis=0); aux=np.mean(predictions['scalar_aux'],axis=0)
    diffs=np.array([(joint[i]-r['success'])**2-(aux[i]-r['success'])**2 for i,r in enumerate(rows) if len(r['bits'])>1])
    rng=np.random.default_rng(761); bootstrap=diffs[rng.integers(0,len(diffs),(10000,len(diffs)))].mean(1)
    result={'scores':scores,'joint_minus_scalar_aux_multi_clause_brier':float(diffs.mean()),
            'instance_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'rows':[{k:r[k] for k in ('task','instance','success','bits')} for r in rows],
            'predictions':predictions,'seconds':time.monotonic()-started,
            'limits':['Exploratory 90-instance study; no policy training','Bootstrap does not cover unseen tasks or training populations',
                      'Frozen spatial-mean vision may miss geometric details','Historical mixed instance outcomes do not prove a mechanism for SFT instability']}
    (HERE/'visual_pilot_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('predictions','rows')},indent=2))


if __name__=='__main__': main()

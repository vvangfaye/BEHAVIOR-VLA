"""Matched-supervision CPU learning test of persistent-fact gradient relay.

Four methods share model initialization, batches, labels, losses, optimizer
steps, and truncation length. The comparison changes only an extra backward
signal at the original observation of x. Synthetic diagnostic, not robotics.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn


class MemoryPolicy(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.memory = nn.GRUCell(6, width)
        self.probe = nn.Linear(width, 2)
        self.actor = nn.Linear(width, 2)
        with torch.no_grad():
            # GRU has reset, update, candidate chunks. Positive update bias
            # gives an ordinary persistent initialization, not a zero saddle.
            self.memory.bias_ih[width:2*width].fill_(2.)
            self.memory.bias_hh[width:2*width].zero_()
            self.probe.weight.copy_(torch.tensor([[1., .5], [.2, 1.]]) @ self.probe.weight)


def batch(seed, length, size):
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(size, generator=gen)*2-1
    y0 = torch.rand(size, generator=gen)*2-1
    y1 = torch.rand(size, generator=gen)*2-1
    # Avoid an accidental unchanged y. This transform is fixed for all groups.
    y1 = torch.where((y1-y0).abs()<.25, -y0, y1)
    cue = torch.randint(0, 2, (size,), generator=gen)
    obs = torch.zeros(length, size, 6)
    obs[:, :, 5] = .3*torch.randn(length, size, generator=gen)
    obs[0, :, 0] = x; obs[0, :, 1] = y0
    obs[0, :, 2:4] = 1.
    mid = length//2
    obs[mid, :, 1] = y1; obs[mid, :, 3] = 1.
    obs[-1, :, 4] = cue.float()*2-1
    facts = torch.stack([x, y0], -1).unsqueeze(0).repeat(length, 1, 1)
    facts[mid:, :, 1] = y1
    target = torch.where(cue==0, x, y1)
    return dict(obs=obs, facts=facts, cue=cue, target=target)


def run_sequence(model, data, cut=None, collect=True, chosen_blocks=None):
    obs, facts = data['obs'], data['facts']
    h = torch.zeros(obs.shape[1], model.width)
    aux = []
    source = None
    block_ends=[]
    for t in range(len(obs)):
        h = model.memory(obs[t], h)
        if t==0: source=h
        if collect and (t in (0, len(obs)//2) or (t+1)%8==0 or t==len(obs)-1):
            aux.append((model.probe(h)-facts[t]).square().mean())
        if cut and (t+1)%cut==0 and t!=len(obs)-1:
            if chosen_blocks is not None:
                block_ends.append(h)
            h=h.detach()
    if chosen_blocks is not None:
        source=torch.stack(block_ends,0)[chosen_blocks,torch.arange(obs.shape[1])]
    outputs = model.actor(h)
    pred = outputs.gather(1, data['cue'][:,None]).squeeze(1)
    return h, source, pred, torch.stack(aux).mean() if aux else None


@torch.no_grad()
def evaluate(model, data):
    h, _, pred, _ = run_sequence(model, data, collect=False)
    err = (pred-data['target']).square()
    facterr = (model.probe(h)-data['facts'][-1]).square()
    return dict(action_mse=err.mean().item(),
                x_action_mse=err[data['cue']==0].mean().item(),
                changed_y_action_mse=err[data['cue']==1].mean().item(),
                x_fact_mse=facterr[:,0].mean().item(),
                changed_y_fact_mse=facterr[:,1].mean().item())


def train(model, method, train_data, evaluations, args):
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    curves=[]; conds=[]; relays=[]
    for step, data in enumerate(train_data):
        optimizer.zero_grad(set_to_none=True)
        chosen=None
        if method=='block_joint':
            number_blocks=len(data['obs'])//args.cut-1
            chosen=torch.randint(number_blocks,(len(data['target']),),generator=torch.Generator().manual_seed(700000+step))
        h, source, pred, aux=run_sequence(model, data, cut=args.cut,chosen_blocks=chosen)
        action=(pred-data['target']).square().mean()
        # The endpoint action gradient, and the probe chart, are frozen within
        # the relay. No auxiliary/native target enters this covector.
        g=torch.autograd.grad(action,h,retain_graph=True)[0].detach()
        j=model.probe.weight.detach()
        gram=j@j.T
        lam=g@j.T@torch.linalg.pinv(gram, rtol=1e-6)
        if method=='block_joint':
            # Before the overwrite only x persists to the endpoint. Afterwards
            # both x and the new y persist. Terminal block is never sampled.
            lam[:,1]*=((chosen+1)*args.cut-1>=len(data['obs'])//2).float()
        else:
            lam[:,1]=0.  # x persists; y changed since the early source.
        joint=(lam@j).detach()
        signal=torch.zeros_like(joint)
        if method=='joint': signal=joint
        elif method=='block_joint': signal=number_blocks*joint
        elif method=='identity':
            signal=g*(joint.norm()/(g.norm()+1e-12))
        elif method=='extra_fact':
            source_fact=(model.probe(source)-data['facts'][0]).square().mean()
            ag=torch.autograd.grad(source_fact,source,retain_graph=True)[0].detach()
            # Same norm and one extra source backward signal, but its direction
            # comes solely from the already-shared factual auxiliary target.
            signal=ag*(joint.norm()/(ag.norm()+1e-12))
        loss=action+args.fact_weight*aux+(signal.detach()*source).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        conds.append(float(torch.linalg.cond(gram)))
        relays.append(float(signal.norm()))
        if (step+1)%40==0 or step+1==len(train_data):
            curves.append(dict(step=step+1, train_action=action.item(), train_fact=aux.item(),
                               eval_train_delay=evaluate(model,evaluations[args.delay])))
    return dict(curve=curves,
                probe_gram_condition_mean=sum(conds)/len(conds),
                relay_norm_mean=sum(relays)/len(relays),
                evaluation={str(k):evaluate(model,v) for k,v in evaluations.items()})


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start=time.time(); rows=[]
    methods=args.methods
    for seed in range(3):
        torch.manual_seed(1000+seed)
        original=MemoryPolicy()
        initial=copy.deepcopy(original.state_dict())
        train_data=[batch(100000+10000*seed+i,args.delay,args.batch) for i in range(args.steps)]
        evals={n:batch(900000+10000*seed+n,n,512) for n in [args.delay,2*args.delay,4*args.delay]}
        for method in methods:
            model=MemoryPolicy(); model.load_state_dict(initial)
            t=time.time()
            result=train(model,method,train_data,evals,args)
            rows.append(dict(seed=seed,method=method,seconds=time.time()-t,**result))
            print(f'seed={seed} method={method} time={time.time()-t:.2f}s '+
                  ' '.join(f'T{k}:action={v["action_mse"]:.4f},x={v["x_action_mse"]:.4f},y={v["changed_y_action_mse"]:.4f}' for k,v in result['evaluation'].items()),flush=True)
    summary={}
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        summary[method]={}
        for length in [args.delay,2*args.delay,4*args.delay]:
            summary[method][str(length)]={key:sum(r['evaluation'][str(length)][key] for r in rr)/3
                                          for key in rr[0]['evaluation'][str(length)]}
    output=dict(description='Synthetic end-to-end fact-lifetime relay diagnostic; all initializations, batches, labels, optimizer steps shared.',
                settings=vars(args)|{'output':str(args.output),'methods':methods,'seeds':3,'memory_width':8,'optimizer':'Adam','gradient_clip':5.,'initial_update_gate_bias':2.,'fact_labels':'source, y-update, each 8-step endpoint, final','relay_strength':1.,'matching':'identity and extra-fact source signal norms match joint signal within each method'},
                caveats=['No hyperparameter selection after viewing results.','A norm-matched extra-fact control is additional use of existing labels, not new information.','This is a biased semantic backward estimator except under local differential invariance.','Linear probe means J_source=J_endpoint; nonlinear coordinate transport is not tested.','block_joint is a separately authorized follow-up, samples one nonterminal block per example and reweights by the number of blocks; it lacks its own matched block-auxiliary control.','No robotics or full-VLA result.'],
                seconds=time.time()-start,summary=summary,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Total CPU wall seconds={output["seconds"]:.2f}; results={args.output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--delay',type=int,default=48)
    parser.add_argument('--cut',type=int,default=8)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--fact-weight',type=float,default=.2)
    parser.add_argument('--methods',nargs='+',default=['tbptt','joint','identity','extra_fact'],choices=['tbptt','joint','identity','extra_fact','block_joint'])
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('fact_relay_training_results.json'))
    main(parser.parse_args())

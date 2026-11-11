"""Independently recompute per-episode generative scores, including body only."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
METHODS=['full_flow','mask_conditioned_flow','support_flow']


def energy(samples,target,coordinates):
    x=samples.reshape(-1,8,32,23)[...,coordinates].reshape(len(samples),8,-1).astype(np.float64)
    y=target.reshape(-1,32,23)[...,coordinates].reshape(len(samples),-1).astype(np.float64)
    n=np.sqrt(y.shape[-1])
    first=np.linalg.norm(x-y[:,None],axis=-1).mean(1)/n
    second=np.zeros(len(y))
    for i in range(8):
        for j in range(i):second+=np.linalg.norm(x[:,i]-x[:,j],axis=-1)
    return first-second/(8*7*n)


def main():
    out={}
    for seed in [17,29,43]:
        folder=HERE/('generative_runs' if seed==17 else f'generative_runs_seed{seed}')
        if not all((folder/f'{m}.npz').exists() for m in METHODS):continue
        sources={m:np.load(folder/f'{m}.npz') for m in METHODS}
        ep=sources[METHODS[0]]['episode'];episodes=np.unique(ep)
        seed_result={}
        for name,coords in [('all',list(range(23))),('body_without_constant_and_grippers',[j for j in range(23) if j not in [6,14,22]])]:
            scores={m:energy(x['samples'],x['target'],coords) for m,x in sources.items()}
            if name=='all':
                for m in METHODS:np.testing.assert_allclose(scores[m],sources[m]['energy_score'],rtol=2e-5,atol=1e-6)
            delta=np.array([(scores['support_flow']-scores['mask_conditioned_flow'])[ep==e].mean() for e in episodes])
            rng=np.random.default_rng(20261021);boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
            seed_result[name]={'mean_energy_scores':{m:float(v.mean()) for m,v in scores.items()},
                               'episode_mean_delta':float(delta.mean()),'episode_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
                               'episode_deltas':dict(zip(map(str,episodes),map(float,delta)))}
        out[str(seed)]=seed_result
    result={'seeds':out,'scope':'Distributional proprio/history prototype; same 14 held-out episodes across seeds; no full VLA or closed-loop claim',
            'group_selection':'Remove coordinate 6 (constant in source) and grippers 14,22. Follow-up check after first seed, not a new confirmatory endpoint.'}
    (HERE/'generative_reanalysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:{g:{k:v for k,v in values.items() if k!='episode_deltas'} for g,values in groups.items()} for s,groups in out.items()},indent=2))


if __name__=='__main__':main()

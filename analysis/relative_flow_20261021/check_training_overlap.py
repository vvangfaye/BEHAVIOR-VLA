"""Training-fold-only feasibility of Gaussian-product context pairing."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=PROJECT/'analysis/action_support_20261021/vla_pilot'
    plan=json.loads((data/'data_plan.json').read_text())
    source=data/plan['raw_targets']['path']
    assert digest(source)==plan['raw_targets']['sha256']
    norm=PROJECT/'analysis/rlinf_adaptation_20261020/checkpoint1688977_openpi/norm_stats.json'
    stats=json.loads(norm.read_text());stats=stats.get('norm_stats',stats)
    with np.load(source,allow_pickle=False) as z:
        raw=z['state'].astype(np.float32)
        state=np.concatenate((raw[:,:3],raw[:,53:57],raw[:,3:10],
                              raw[:,24:26].sum(1,keepdims=True),raw[:,28:35],
                              raw[:,49:51].sum(1,keepdims=True)),axis=1).astype(np.float64)
        delta=np.zeros(23);delta[list(range(3,14))+list(range(15,22))]=1
        actions=(z['actions'].astype(np.float64)-state[:,None]*delta
                 -np.asarray(stats['actions']['mean'])[:23])/(np.asarray(stats['actions']['std'])[:23]+1e-6)
        folds=z['fold'].copy();episodes=z['episode'].copy()
    train=np.flatnonzero(np.isin(folds,[2,3,4]))
    rng=np.random.default_rng(907)
    groups={}
    for i in train:groups.setdefault(plan['samples'][int(i)]['task_name'],[]).append(int(i))
    output=[]
    for task,group in sorted(groups.items()):
        pairs=[]
        while len(pairs)<4096:
            a,b=rng.choice(group,size=2)
            if episodes[a]!=episodes[b]:pairs.append((a,b))
        pair=np.asarray(pairs);difference=actions[pair[:,0]]-actions[pair[:,1]]
        distance=np.square(difference).sum((1,2))
        rows=[]
        for t in [.5,.7,.9,.95,.99]:
            log_weight=-(1-t)**2*distance/(4*t*t)
            weight=np.exp(log_weight)
            rows.append(dict(time=t,mean_weight=float(weight.mean()),
                             median_weight=float(np.median(weight)),
                             fraction_above_001=float((weight>.01).mean()),
                             fraction_above_01=float((weight>.1).mean()),
                             effective_weight_fraction=float(weight.sum()**2/(len(weight)*np.square(weight).sum())),
                             weighted_target_difference_mse=float((weight*distance/actions.shape[1]/23).sum()/weight.sum())))
        output.append(dict(task=task,training_anchors=len(group),independent_episode_pairs=len(pairs),times=rows))
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),seed=907,
                split='training folds 2,3,4 only; different-episode pairs within task',
                input_sha256=dict(data_plan=digest(data/'data_plan.json'),raw_targets=digest(source),normalization=digest(norm)),
                tasks=output,bridge_conditional_noise_rms_ratio=float(1/np.sqrt(2)),
                limitations=['Weight concentration is not a policy-quality measurement.',
                             'Gaussian-product queries have half native conditional noise covariance; deployment-noise coverage remains a risk.',
                             'No test-set selection, policy training, or simulator interaction.'])
    (HERE/'training_overlap.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

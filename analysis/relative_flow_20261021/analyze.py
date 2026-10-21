"""Prespecified paired analysis of the four-arm relative-flow pilot."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'action_support_20261021'))
from analyze_vla_pilot import score

ROOT=HERE/'pilot_ada01'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def comparison(left,right,episode):
    delta=left-right;ids=np.unique(episode)
    means=np.asarray([delta[episode==i].mean() for i in ids])
    counts=np.asarray([(episode==i).sum() for i in ids])
    selection=np.random.default_rng(673).integers(len(ids),size=(10000,len(ids)))
    resampled=(means[selection]*counts[selection]).sum(1)/counts[selection].sum(1)
    return dict(mean_delta=float(delta.mean()),
                simultaneous_interval=np.quantile(resampled,[.05/6,1-.05/6]).tolist(),
                episodes_better=int((means<0).sum()),episodes_worse=int((means>0).sum()),episodes=len(ids))


def main():
    plan=read(ROOT/'plan.json');methods=plan['methods'];plan_sha=sha(ROOT/'plan.json')
    missing=[m for m in methods if not (ROOT/('formal_'+m)/'completion.json').exists()]
    if missing:
        print(json.dumps(dict(ready=False,waiting_for=missing)));return
    assert read(ROOT/'formal_matched_protocol.json')['passed']
    arrays={};metrics={};sources={}
    for method in methods:
        folder=ROOT/('formal_'+method);output=folder/'output'
        receipt=read(folder/'completion.json');result=read(output/'result.json');run=read(output/'run.json')
        assert receipt['accepted'] and receipt['returncode']==0
        assert receipt['result_sha256']==sha(output/'result.json')
        assert result['complete'] and result['completed_steps']==plan['steps']==1000
        assert result['plan_sha256']==run['plan_sha256']==receipt['plan_sha256']==plan_sha
        assert result['seed']==run['seed']==plan['seed']
        assert result['method']==run['method']==method
        with np.load(output/'test.npz',allow_pickle=False) as z:
            arrays[method]={k:z[k].copy() for k in ['samples','target','index','episode','frame','noise_sha256']}
        metrics[method]=score(arrays[method]['samples'],arrays[method]['target'])
        reported=read(output/'test.json')['means']
        for key,value in metrics[method].items():assert abs(value.mean()-reported[key])<2e-5
        sources[method]={str(p.relative_to(ROOT)):sha(p) for p in
                         [folder/'completion.json',output/'result.json',output/'test.npz',output/'test.json']}
    ref=arrays['extra_fm']
    for method in methods:
        for key in ['target','index','episode','frame','noise_sha256']:
            np.testing.assert_array_equal(arrays[method][key],ref[key])
    means={m:{k:float(v.mean()) for k,v in values.items()} for m,values in metrics.items()}
    comparisons={m:{k:comparison(metrics['relative_difference'][k],metrics[m][k],ref['episode'])
                    for k in metrics[m]} for m in methods if m!='relative_difference'}
    passed=all(c['energy_score']['mean_delta']<0 and c['energy_score']['simultaneous_interval'][1]<0
               and means['relative_difference']['mean_action_mse']<=1.05*means[m]['mean_action_mse']
               for m,c in comparisons.items())
    data=read(Path(plan['data_directory'])/'data_plan.json')
    tasks=np.asarray([data['samples'][int(i)]['task_name'] for i in ref['index']])
    task_results={task:{m:{k:float(v[tasks==task].mean()) for k,v in metrics[m].items()}
                        for m in methods} for task in sorted(set(tasks))}
    result=dict(ready=True,observed_at=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_sha,
                analysis_plan_sha256=sha(ROOT/'analysis_plan.json'),source_sha256=sources,
                means=means,comparisons=comparisons,by_task=task_results,screen_passed=passed,
                anchors=len(ref['index']),episodes=len(np.unique(ref['episode'])),
                matched_evaluation_noise_draws=len(ref['noise_sha256']),
                decision=('eligible_for_fresh_native_pilot_not_a_paper_claim' if passed else
                          'candidate_fails_prespecified_four_arm_screen'),
                limitations=['Exploratory split already used for earlier method development.',
                             'Three tasks and one training seed; intervals cover episodes only.',
                             'Expert subtask supplied; this does not measure autonomous task success.'],
                goal_complete=False)
    (ROOT/'reanalysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['means','comparisons','screen_passed','decision']},indent=2))


if __name__=='__main__':main()

"""Independent scoring of the frozen two-arm posterior-support pilot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from analyze_vla_pilot import score, compare, sha

HERE=Path(__file__).resolve().parent
ROOT=HERE/'posterior_support_pilot_ada01'


def read(p):return json.loads(p.read_text())


def main():
    methods=('ambient_aux','posterior')
    missing=[m for m in methods if not (ROOT/('formal_'+m)/'completion.json').exists()]
    if missing:
        print(json.dumps(dict(ready=False,waiting_for=missing)))
        return
    plan=read(ROOT/'plan.json');plan_hash=sha(ROOT/'plan.json')
    results={};arrays={};sources={}
    for method in methods:
        folder=ROOT/('formal_'+method)
        completion,run,result=[read(folder/n) for n in ('completion.json','run.json','result.json')]
        assert completion['accepted'] and completion['returncode']==0
        assert completion['result_sha256']==sha(folder/'result.json')
        assert run['plan_sha256']==plan_hash and run['method']==method and run['seed']==17
        assert result['complete'] and result['completed_steps']==plan['steps']==1000
        assert result['first_expert_attention_max_change']>0
        with np.load(folder/'test.npz',allow_pickle=False) as z:
            arrays[method]={k:z[k].copy() for k in ('samples','target','index','episode','frame')}
        results[method]=score(arrays[method]['samples'],arrays[method]['target'])
        reported=read(folder/'test.json')['means']
        for key,value in results[method].items():
            assert abs(value.mean()-reported[key])<2e-5,(method,key)
        sources[method]={n:sha(folder/n) for n in ('test.npz','test.json','result.json','completion.json')}
    ref=arrays['ambient_aux']
    for key in ('index','episode','frame','target'):
        np.testing.assert_array_equal(ref[key],arrays['posterior'][key])
    means={m:{k:float(v.mean()) for k,v in values.items()} for m,values in results.items()}
    differences={k:compare(results['posterior'][k],results['ambient_aux'][k],ref['episode'])
                 for k in results['posterior']}
    es=differences['energy_score']
    mse_ratio=means['posterior']['mean_action_mse']/means['ambient_aux']['mean_action_mse']
    advance=(es['anchor_mean_delta']<0 and es['anchor_weighted_cluster_bootstrap95'][1]<0
             and mse_ratio<=1.05)
    report=dict(ready=True,observed_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=plan_hash,analysis_plan_sha256=sha(ROOT/'analysis_plan.json'),
                means=means,posterior_minus_ambient_aux=differences,source_hashes=sources,
                anchors=len(ref['index']),episodes=len(np.unique(ref['episode'])),
                screen_passed=advance,
                decision=('eligible_for_separate_frozen_native_pilot_not_proven_task_gain' if advance
                          else 'current_recipe_does_not_pass_prespecified_offline_screen'),
                limits=[
                    'SFT-heldout episodes, but the split was already used in earlier exploratory method development.',
                    'One seed, three tasks, four draws; bootstrap covers episodes, not model seeds or novel tasks.',
                    'Expert subtask conditioning; no autonomous task success is measured.',
                    'Final-step support Brier is not posterior calibration because the generated sample need not match the recorded future.',
                    'No early checkpoint or alternate hyperparameter selection from this test set.'
                ],goal_complete=False)
    (ROOT/'reanalysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('ready','means','posterior_minus_ambient_aux','screen_passed','decision')},indent=2))


if __name__=='__main__':main()

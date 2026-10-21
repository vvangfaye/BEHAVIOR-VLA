"""Compare proposed training against the same transform without that training."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from analyze_vla_pilot import score,compare,sha
from analyze_posterior_support import main as analyze_primary

HERE=Path(__file__).resolve().parent
PILOT=HERE/'posterior_support_pilot_ada01'
POSTHOC=HERE/'posterior_posthoc_control_ada01'


def read(p):return json.loads(p.read_text())


def main():
    required=[PILOT/'formal_ambient_aux/completion.json',PILOT/'formal_posterior/completion.json',
              POSTHOC/'completion.json']
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        print(json.dumps(dict(ready=False,missing=missing)));return
    analyze_primary()
    primary=read(PILOT/'reanalysis.json')
    gate=read(PILOT/'training_contribution_gate_plan.json')
    assert sha(POSTHOC/'plan.json')==gate['source_posthoc_plan_sha256']
    completion=read(POSTHOC/'completion.json');result=read(POSTHOC/'result.json')
    assert completion['accepted'] and completion['returncode']==0
    assert completion['result_sha256']==sha(POSTHOC/'result.json')
    assert result['complete'] and result['noise_streams_equal']
    assert result['replay_max_abs_difference']<=2e-5
    hashes=read(POSTHOC/'noise_hashes.json')
    assert hashes['replay']==hashes['posthoc'] and len(hashes['replay'])==96*4
    arrays={}
    for name,path in [('trained',PILOT/'formal_posterior/test.npz'),
                      ('control',PILOT/'formal_ambient_aux/test.npz'),
                      ('replay',POSTHOC/'replay.npz'),('posthoc',POSTHOC/'posthoc.npz')]:
        with np.load(path,allow_pickle=False) as z:
            arrays[name]={k:z[k].copy() for k in ['index','episode','frame','target','samples']}
    for name in ['control','replay','posthoc']:
        for key in ['index','episode','frame','target']:
            np.testing.assert_array_equal(arrays[name][key],arrays['trained'][key])
    difference=float(np.abs(arrays['control']['samples']-arrays['replay']['samples']).max())
    assert difference<=2e-5
    metrics={n:score(v['samples'],v['target']) for n,v in arrays.items()}
    comparison={k:compare(metrics['trained'][k],metrics['posthoc'][k],arrays['trained']['episode'])
                for k in metrics['trained']}
    means={n:{k:float(v.mean()) for k,v in vals.items()} for n,vals in metrics.items()}
    es=comparison['energy_score']
    posthoc_pass=(es['anchor_mean_delta']<0 and es['anchor_weighted_cluster_bootstrap95'][1]<0
                 and means['trained']['mean_action_mse']<=1.05*means['posthoc']['mean_action_mse'])
    passed=bool(primary['screen_passed'] and posthoc_pass)
    report=dict(ready=True,observed_at=datetime.now(timezone.utc).isoformat(),
                training_contribution_gate_passed=passed,primary_gate_passed=primary['screen_passed'],
                posthoc_gate_passed=posthoc_pass,means=means,trained_minus_posthoc=comparison,
                replay_max_difference=difference,source_primary_report_sha256=sha(PILOT/'reanalysis.json'),
                gate_plan_sha256=sha(PILOT/'training_contribution_gate_plan.json'),
                source_posthoc_result_sha256=sha(POSTHOC/'result.json'),
                decision=('eligible_for_separate_native_pilot_not_a_paper_result' if passed else
                          'training_contribution_not_established_by_prespecified_screen'),
                limits=['Single seed and three tasks; no closed-loop outcome measured.',
                        'Same transformed control is an inference-only ablation, not a calibrated conditional-active expert.',
                        'A positive result still needs further ablations, fresh instances and task success comparisons.'],
                goal_complete=False)
    (PILOT/'training_contribution_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

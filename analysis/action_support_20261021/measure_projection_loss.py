"""Compare lost demonstrated variation at exactly matched continuous dimensions.

This measures representation error only, not prediction or closed-loop gains.
"""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    h,d=32,23
    time=np.arange(h)[None,:]; freq=np.arange(h)[:,None]
    basis=np.cos(np.pi*(time+.5)*freq/h)*np.sqrt(2/h);basis[0]/=np.sqrt(2)
    np.testing.assert_allclose(basis@basis.T,np.eye(h),atol=1e-14)
    results=[]
    for seed in [17,29,43]:
        total=np.zeros(d);loss=np.zeros(d);dct_loss=np.zeros(d);budgets=[];miss=[];n=0
        for fold in range(5):
            with np.load(HERE/'activity_predictions'/f'fold{fold}_seed{seed}.npz') as f:
                y=f['future'].astype(np.float64);mask=f['probabilities']>=float(f['threshold']);index=f['index']
            centered=y-y.mean(1,keepdims=True)
            total+=(centered**2).sum((0,1))
            loss+=(centered**2*(~mask[:,None,:])).sum((0,1))
            coef=np.einsum('kt,ntd->nkd',basis,y)
            budget=d+(h-1)*mask.sum(1);budgets.extend(budget.tolist())
            # Identical per-chunk dimension budget. Residual slots are assigned
            # in a cyclic coordinate order independent of target values.
            q,r=np.divmod(budget,d)
            order=(np.arange(d)[None,:]-index[:,None])%d
            kept=q[:,None]+(order<r[:,None])
            excluded=np.arange(h)[None,:,None]>=kept[:,None,:]
            dct_loss+=(coef**2*excluded).sum((0,1))
            n+=len(y)
        valid=total>1e-12
        result={'seed':seed,'chunks':n,'mean_continuous_dimensions':float(np.mean(budgets)),
                'full_dimensions':h*d,'dimension_fraction':float(np.mean(budgets)/(h*d)),
                'mask_fraction_of_temporal_energy_lost_by_coordinate':np.divide(loss,total,out=np.zeros(d),where=valid).tolist(),
                'dct_fraction_of_temporal_energy_lost_by_coordinate':np.divide(dct_loss,total,out=np.zeros(d),where=valid).tolist(),
                'variable_coordinates':np.flatnonzero(valid).tolist(),
                'mask_mean_fraction_lost':float(np.mean(loss[valid]/total[valid])),
                'dct_mean_fraction_lost':float(np.mean(dct_loss[valid]/total[valid]))}
        results.append(result)
    initial=json.loads((HERE/'activity_results_initial.json').read_text());rerun=json.loads((HERE/'activity_results.json').read_text())
    assert initial['mean_metrics']==rerun['mean_metrics'], 'Prediction-export repeat changed metrics'
    out={'results':results,'prediction_export_repeat_identical':True,
         'scope':'GT encoding error caused by predicted support, versus a fixed low-frequency DCT representation with matched dimension budget; no action decoder trained',
         'limits':['Column-average ratios preserve physical-unit separation but are not task-success costs',
                   'This is a post-hoc feasibility comparison motivated by observed false-flat predictions',
                   'Does not establish full VLA or closed-loop performance']}
    (HERE/'projection_loss_result.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({**out,'results':[{k:v for k,v in x.items() if 'coordinate' not in k} for x in results]},indent=2))


if __name__=='__main__':main()

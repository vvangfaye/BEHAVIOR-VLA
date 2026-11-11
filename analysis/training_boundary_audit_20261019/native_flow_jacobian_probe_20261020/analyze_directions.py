"""Analyze measured numerical action directions, without new model calls.

The directions are hypothetical additive output translations, not identified
LoRA updates. Inverse norms alone are not policy-gradient variance estimates.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
path=HERE/'reverse_jacobian.npy'
j=np.load(path).astype(np.float64)
assert j.shape==(368,1024)
u,s,vt=np.linalg.svd(j,full_matrices=False)
records=[]
for name,times in [('constant',np.ones(16)),('linear_ramp',np.arange(16)-7.5),
                   ('alternating',(-1.)**np.arange(16))]:
    times=times/np.linalg.norm(times)
    for coordinate in range(23):
        target=np.zeros((16,23));target[:,coordinate]=times
        b=target.reshape(-1)
        coefficients=u.T@b
        thresholds={}
        for relative in [1e-2,1e-3,1e-4,1e-5,1e-6]:
            keep=s>relative*s[0]
            thresholds[str(relative)]={
                'target_energy_retained':float(np.sum(coefficients[keep]**2)),
                'truncated_lift_norm':float(np.linalg.norm(coefficients[keep]/s[keep]))}
        records.append(dict(pattern=name,coordinate=coordinate,thresholds=thresholds))
result=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,
    summaries={name:{threshold:{
        'median_target_energy_retained':float(np.median([r['thresholds'][threshold]['target_energy_retained'] for r in records if r['pattern']==name])),
        'minimum_target_energy_retained':float(np.min([r['thresholds'][threshold]['target_energy_retained'] for r in records if r['pattern']==name])),
        'median_truncated_lift_norm':float(np.median([r['thresholds'][threshold]['truncated_lift_norm'] for r in records if r['pattern']==name]))}
        for threshold in ['0.01','0.001','0.0001','1e-05','1e-06']}
        for name in ['constant','linear_ramp','alternating']},
    caveats=['No truncated inverse is proposed as an unbiased training update.',
        'Targets are normalized output-space probes, not real policy-parameter update directions.',
        'The norm excludes the divergence term and therefore is not a gradient variance.',
        'Finite-precision Jacobian from one observation only.'])
(HERE/'direction_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summaries'],indent=2))

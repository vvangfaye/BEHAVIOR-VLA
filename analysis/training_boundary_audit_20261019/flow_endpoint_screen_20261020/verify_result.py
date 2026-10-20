"""Independent NumPy recomputation of the completed pairing's numeric results."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from eval_cached_probe import eval_openloop as ev


def array(record):
    result=np.asarray(record['values'],dtype=record['dtype'])
    assert list(result.shape)==record['shape'] and np.isfinite(result).all()
    return result


def main():
    documents=[json.loads((HERE/name).read_text()) for name in ['baseline1688977.json','expert1000.json']]
    saved=json.loads((HERE/'paired_comparison.json').read_text())
    checks=0; maximum=0.0
    for doc in documents:
        for row in doc['samples']:
            for numeric,expected in [(row['endpoint']['actions'],row['endpoint']['actions_sha256']),
                    (row['training_target'],row['training_target']['sha256']),
                    (row['rng']['time'],row['rng']['time_sha256'])]:
                assert ev.input_content_sha256(array(numeric))==expected
                checks+=1
    for anchor in saved['anchors']:
        groups=[[r for r in doc['samples'] if (r['episode_index'],r['frame_index'])==
                 (anchor['episode_index'],anchor['frame_index'])] for doc in documents]
        for rows in groups:rows.sort(key=lambda r:r['draw_index'])
        for name in ('action_loss','subtask_loss'):
            direct=np.array([[r['cfm']['metrics'][name]['values'] for r in group] for group in groups],dtype=np.float64)
            delta=direct[1]-direct[0]
            computed=[direct[0].mean(),direct[1].mean(),delta.mean(),delta.std(ddof=1),delta.std(ddof=1)/np.sqrt(8)]
            stored=[anchor['native_metrics'][name][k] for k in ['baseline_mean','candidate_mean','candidate_minus_baseline_mean','paired_delta_sd','paired_delta_monte_carlo_sem']]
            assert np.allclose(computed,stored,rtol=1e-10,atol=1e-12)
        truth=array(groups[0][0]['endpoint']['ground_truth']).astype(np.float64)
        valid=np.asarray(groups[0][0]['endpoint']['valid_mask'],bool)
        for mask_name, mask in [('valid_prefix16',valid & (np.arange(32)<16)),('valid_horizon',valid)]:
            gt=truth[mask]
            for index,rows in enumerate(groups):
                predictions=np.stack([array(r['endpoint']['actions']).astype(np.float64)[mask] for r in rows])
                # Ordered pairs with the diagonal explicitly removed, independent
                # of compare_screens.energy's unordered-pair implementation.
                pairwise=predictions[:,None,:,:]-predictions[None,:,:,:]
                distances=np.sqrt(np.mean(pairwise**2,axis=2))
                distances=distances[~np.eye(8,dtype=bool)].reshape(8,7,23)
                energy=np.sqrt(np.mean((predictions-gt)**2,axis=1)).mean(axis=0)-distances.mean(axis=(0,1))/2
                mse=((predictions-gt)**2).mean(axis=(0,1))
                field='baseline_energy' if index==0 else 'candidate_energy'
                mse_field='baseline_mean' if index==0 else 'candidate_mean'
                for j,coordinate in enumerate(anchor['endpoint'][mask_name]['coordinates']):
                    errors=[abs(energy[j]-coordinate[field]['score']),abs(mse[j]-coordinate['sample_mse'][mse_field])]
                    maximum=max(maximum,*errors)
                    assert max(errors)<1e-12
    receipt={'status':'passed','numeric_content_hashes_checked':checks,'energy_and_mse_max_abs_difference':maximum,
             'independent_method':'NumPy ordered off-diagonal energy score; float64 direct MSE and paired mean/SD/SEM.',
             'scientific_scope_unchanged':True,
             'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'numeric_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()

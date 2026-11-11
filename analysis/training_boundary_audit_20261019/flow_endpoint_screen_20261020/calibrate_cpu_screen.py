"""Apply the frozen 42-parameter comparator to the completed 3x8 CPU screen.

Old exports supply only state-based hold actions, validated against identical GT,
input hashes and cache identity. No prediction from another RNG protocol enters
the regression. Eight draws are stacked within an anchor, so fold bookkeeping
never confuses repeated draws with independent states or overwrites their keys.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import action_change_calibration as calibration
from compare_screens import compare


def main():
    base_path=HERE/'baseline1688977.json'
    candidate_path=HERE/'expert1000.json'
    old_path=HERE.parent/'cpu_raw_feasibility_20261020/baseline36_raw.json'
    base=json.loads(base_path.read_text())
    candidate=json.loads(candidate_path.read_text())
    compare(base,candidate)
    old=json.loads(old_path.read_text())
    assert old['raw_cache']['manifest_sha256']==base['raw_cache']['manifest_sha256']
    old_rows={(r['episode_index'],r['frame_index']):r for r in old['sample_keys']}
    cand={(r['episode_index'],r['frame_index'],r['draw_index']):r for r in candidate['samples']}
    grouped={}
    for row in base['samples']:
        grouped.setdefault((row['episode_index'],row['frame_index']),[]).append(row)
    samples=[]
    holds=[]
    for key,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda r:r['draw_index'])
        assert [r['draw_index'] for r in rows]==list(range(8))
        old_row=old_rows[key]
        arrays=calibration._raw_arrays(old_row,32,f'hold_source/{key}')
        assert rows[0]['policy_input_sha256']==old_row['policy_input_sha256']
        assert np.array_equal(np.asarray(rows[0]['endpoint']['ground_truth']['values']),arrays['ground_truth'])
        assert np.array_equal(rows[0]['endpoint']['valid_mask'],arrays['valid_mask'])
        selected=arrays['valid_mask'] & (np.arange(32)<16)
        hold=arrays['hold_action'][selected].astype(np.float64)[:,calibration.DIMENSIONS]
        features=[]; deltas=[]
        for row in rows:
            changed=cand[(*key,row['draw_index'])]
            prediction=np.asarray(row['endpoint']['actions']['values'])[selected][:,calibration.DIMENSIONS]
            after=np.asarray(changed['endpoint']['actions']['values'])[selected][:,calibration.DIMENSIONS]
            features.append(prediction-hold)
            deltas.append(after-prediction)
        samples.append(calibration.Sample(old_row['task'],*key,'8_draws_per_anchor',
                        np.concatenate(features),np.concatenate(deltas)))
        holds.append({'episode_index':key[0],'frame_index':key[1],
                      'old_raw_actions_sha256':old_row['raw_actions']['sha256'],
                      'new_and_old_gt_identical':True,'policy_input_sha256':row['policy_input_sha256']})
    folds,predictions=calibration.cross_validated_predictions(samples)
    result={'kind':'three_anchor_cpu_calibration_screen','status':'complete',
            'scope':'42 parameters; no change and constant offset comparators; leave one complete task/anchor out. '
                    'Only three physical observations with eight noise draws each. No closed-loop or cross-task-population claim.',
            'coupling_limit':'Same-noise regression diagnoses this sampling implementation, not a coupling-invariant policy distribution.',
            'old_export_use':'Only validated GT/mask/input and state-based hold. Old predictions and old PCG64 noise are never used.',
            'prediction_key_policy':'One stacked Sample per anchor: all eight draws retained, no duplicate-key overwrite.',
            'physical_anchor_count':3,'draws_per_anchor':8,'holds':holds,
            'out_of_task_scores':calibration.score_predictions(samples,predictions),'folds':folds,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base_path,candidate_path,old_path]}}
    with (HERE/'cpu_calibration.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    for group in result['out_of_task_scores']['groups']:
        print(group['action_group'],group['dimensionless_macro'])


if __name__=='__main__':
    main()

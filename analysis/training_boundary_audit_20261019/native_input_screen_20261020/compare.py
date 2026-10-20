"""Pair the native-observation screen without treating parent actions as labels."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def difference(left,right):
    for value in (left,right):
        assert value['shape']==[32,23]
        assert len(value['values'])==32
        assert all(len(row)==23 for row in value['values'])
        assert all(math.isfinite(x) for row in value['values'] for x in row)
    delta=[[b-a for a,b in zip(x,y)] for x,y in zip(left['values'],right['values'])]
    return {'prefix16_rms_per_coordinate':[math.sqrt(sum(row[j]**2 for row in delta[:16])/16)
                                             for j in range(23)],
            'horizon32_rms_per_coordinate':[math.sqrt(sum(row[j]**2 for row in delta)/32)
                                              for j in range(23)],
            'max_absolute_difference':max(abs(x) for row in delta for x in row)}


def compare(base,candidate):
    assert base['status']==candidate['status']=='complete'
    for field in ['kind','backend','model_config','normalization_sha256','parameter_dtype_counts']:
        assert base[field]==candidate[field],field
    for doc in (base,candidate):
        assert doc['same_model_repeat']['label_equal']
        assert doc['same_model_repeat']['max_action_difference']<1e-6
    def indexed(doc):
        rows={}
        for row in doc['samples']:
            key=(row['id'],row['frame'],row['mode'])
            assert key not in rows
            rows[key]=row
        assert len(rows)==12
        return rows
    a,b=indexed(base),indexed(candidate)
    assert a.keys()==b.keys()
    modes=('recorded_label','fresh_generated_label')
    fields=('task','instance','recorded_label','input_sha256','npz_sha256','noise_sha256')
    for key in a:
        for field in fields:assert a[key][field]==b[key][field],(key,field)
    result=[]
    for identity,frame in sorted({key[:2] for key in a}):
        base_rows=[a[(identity,frame,mode)] for mode in modes]
        candidate_rows=[b[(identity,frame,mode)] for mode in modes]
        for group in [base_rows,candidate_rows]:
            for field in fields:assert group[0][field]==group[1][field]
        row={'id':identity,'frame':frame,'recorded_label':base_rows[0]['recorded_label'],
             'baseline_generated_label':base_rows[1]['model_label'],
             'candidate_generated_label':candidate_rows[1]['model_label'],
             'cross_checkpoint':{},'within_checkpoint':{}}
        for i,mode in enumerate(modes):
            row['cross_checkpoint'][mode]=difference(base_rows[i]['actions'],candidate_rows[i]['actions'])
        for label,group in [('baseline',base_rows),('candidate',candidate_rows)]:
            same=group[0]['model_label']==group[1]['model_label']
            delta=difference(group[0]['actions'],group[1]['actions'])
            row['within_checkpoint'][label]={**delta,'same_decoded_label':same,
                'same_label_action_parity':not same or delta['max_absolute_difference']<1e-6}
        result.append(row)
    return {'kind':'paired_native_observation_label_action_comparison','status':'complete',
            'rows':result,
            'scope':'Six selected states from two parent-policy episodes of one task. Differences are descriptive. '
              'Recorded labels are historical decisions, not correct subtask annotations. No historical RNG replay, '
              'ground-truth action error, recovery success, or physical stability is inferred.',
            'changed_generated_labels':sum(r['baseline_generated_label']!=r['candidate_generated_label'] for r in result),
            'same_label_action_parity_all_passed':all(v['same_label_action_parity'] for r in result
                                                     for v in r['within_checkpoint'].values())}


if __name__=='__main__':
    result=compare(json.loads((HERE/'baseline1688977.json').read_text()),
                   json.loads((HERE/'expert1000.json').read_text()))
    with (HERE/'comparison.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'changed_generated_labels':result['changed_generated_labels'],
                      'same_label_action_parity_all_passed':result['same_label_action_parity_all_passed']}))

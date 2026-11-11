"""Complete label coverage while retaining original timeout provenance."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    partial = json.loads((root/'partial_result.json').read_text())
    plan = json.loads((root/'continuation_v1/plan.json').read_text())
    names = ('baseline1688977', 'expert1000')
    ref = json.loads((root.parent/'calibration_screen_20261020/retrieved/baseline1688977.json').read_text())
    expected = {(x['episode_index'],x['frame_index']):x for x in ref['sample_keys']}
    original = {}; resumed = {}; merged = {}
    for name in names:
        log = root/f'{name}.log'
        assert hashlib.sha256(log.read_bytes()).hexdigest() == partial['source_log_sha256'][name]
        assert plan['original_log_sha256'][name] == partial['source_log_sha256'][name]
        original[name] = [json.loads(line) for line in log.read_text().splitlines()
                          if line.startswith('{') and 'sample' in json.loads(line)]
        assert len(original[name]) == partial['completed_counts'][name]
        path = root/'continuation_v1'/f'{name}.json'
        r = json.loads(path.read_text()); resumed[name] = r
        assert r['status'] == 'complete' and r['backend'] == 'cpu'
        assert r['parameter_dtypes'] == ['bfloat16']
        assert r['action_sampling_calls'] == 0 and all(r['same_model_repeat'].values())
        assert r['original_sample_indices'] == plan['subset_original_indices'][name]
        assert len(r['samples']) == len(r['original_sample_indices'])
        assert r['generation_calls'] == len(r['samples'])+1
        by_index = {x['sample']:dict(x,source='original_log') for x in original[name]}
        for index, row in zip(r['original_sample_indices'],r['samples']):
            source = expected[row['episode_index'],row['frame_index']]
            assert row['provided_policy_input_sha256'] == source['policy_input_sha256']
            converted = {'sample':index,'task':row['task'],'frame':row['frame_index'],
                         'reference':row['annotation_label'],'generated':row['generated_label'],
                         'source':'continuation_json','valid_generation':row['valid_generation']}
            if index in by_index:
                assert index == 1
                assert all(converted[k] == by_index[index][k]
                           for k in ('sample','task','frame','reference','generated'))
            else:
                by_index[index] = converted
        assert set(by_index) == set(range(1,37))
        merged[name] = by_index
    for field in ('model_config','normalization_sha256','cache_manifest_sha256'):
        assert resumed[names[0]][field] == resumed[names[1]][field]
    anchors = [resumed[n]['samples'][0] for n in names]
    for field in ('provided_policy_input_sha256','free_generation_input_sha256','processed_prefix_sha256'):
        assert anchors[0][field] == anchors[1][field]
    by_frame = {(x['task'],x['frame_index']):x for x in ref['sample_keys']}
    paired=[];groups=defaultdict(lambda:dict(n=0,changed=0,baseline_annotation_agreement=0,
                                           expert_annotation_agreement=0))
    for index in range(1,37):
        b,c=(merged[n][index] for n in names)
        assert all(b[k] == c[k] for k in ('sample','task','frame','reference'))
        source = by_frame[b['task'],b['frame']]
        distance = source['segment_end']-b['frame']
        row={'sample':index,'task':b['task'],'episode':source['episode_index'],'frame':b['frame'],
             'frames_to_boundary':distance,'annotation':b['reference'],
             'annotated_next':source['probe_metadata']['next_subtask'],
             'baseline':b['generated'],'expert':c['generated'],
             'baseline_source':b['source'],'expert_source':c['source'],
             'changed':b['generated'] != c['generated']}
        paired.append(row)
        for group in ('all',b['task'],f'boundary_distance_{distance}'):
            g=groups[group];g['n']+=1;g['changed']+=row['changed']
            g['baseline_annotation_agreement']+=b['generated']==b['reference']
            g['expert_annotation_agreement']+=c['generated']==c['reference']
    result={'status':'coverage_complete_with_original_timeout_deviation',
            'kind':'merged_original_log_and_continuation_label_comparison',
            'groups':dict(groups),'changed_rows':[x for x in paired if x['changed']],
            'paired_rows':paired,'cross_run_first_anchor_text_equal':True,
            'continuation_first_anchor_prefix_hash_equal':True,
            'original_counts':partial['completed_counts'],
            'continuation_samples':{n:resumed[n]['original_sample_indices'] for n in names},
            'limits':['Original jobs timed out; this is not the originally planned full one-process comparison.',
                      'Original generated token arrays and processed-prefix hashes were not saved; only completed stdout labels survived.',
                      'Continuation results have raw tokens, prefix hashes and repeat checks. Cross-run anchor checks establish text equality, not original token-array equality.',
                      'Training-seen expert observations from 3 tasks and 9 episodes; no autonomous task outcomes.',
                      'Exact annotation agreement is not semantic correctness; switching early at a boundary can be valid.']}
    with (root/'comparison_with_continuation.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'groups':result['groups'],
                      'changed_rows':result['changed_rows']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

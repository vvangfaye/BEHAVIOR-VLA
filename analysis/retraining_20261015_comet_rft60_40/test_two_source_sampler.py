import collections
import copy
import hashlib
import json
from pathlib import Path

import pytest
from openpi.training.subtask_mixture import MixturePlan
from openpi.training.subtask_mixture import consumed_contract,compare_consumed_contract


def write_index(root,kind,episodes,intervals):
    data=dict(schema_version=1,kind='b1k_valid_intervals',source_kind=kind,root_id=root,
              repo_id='fixture',dataset_root='/prepared/'+root,action_horizon=4,held_out_instances=[],
              episodes=episodes,intervals=intervals,provenance={})
    if kind=='full_episode':
        data['provenance']=dict(label_semantics='self_generated_weak_subtask',language_loss_enabled=False)
        for e in episodes:e['content_sha256']=hashlib.sha256(f'{root}:{e["episode_index"]}'.encode()).hexdigest()
    data['fingerprint']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    return data


def setup(tmp_path):
    sources=[]
    for kind,weight in [('replay',.6),('full_episode',.4)]:
        episodes=[dict(episode_index=i,length=100,task_name='A' if i<2 else 'B',task_instance_id=1)
                  for i in range(3)]
        intervals=[]
        for e in episodes:
            i=e['episode_index']
            intervals += [dict(episode_index=i,start=0,end=10,subtask='short'),
                          dict(episode_index=i,start=10,end=100,subtask='long')]
        index=write_index(kind,kind,episodes,intervals);path=tmp_path/(kind+'.json')
        path.write_text(json.dumps(index));sources.append(dict(kind=kind,weight=weight,
            indexes=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())]))
    manifest=dict(schema_version=3,kind='b1k_two_source_mixture',seed=42,global_batch_size=16,action_horizon=4,
                  phases=[dict(start_step=0,end_step=5000,sources=sources)])
    return manifest,MixturePlan(manifest,base_dir=tmp_path)


def test_two_source_only_and_uniform_task_episode_not_skill(tmp_path):
    manifest,plan=setup(tmp_path)
    counts=collections.Counter();tasks=collections.Counter();segments=collections.Counter();episodes=collections.Counter()
    for i in range(50000):
        r=plan.reference(i);counts[r['source_kind']]+=1;tasks[r['task_name']]+=1
        segments[r['subtask']]+=1
        if r['task_name']=='A':episodes[r['episode_index']]+=1
        assert r['segment_start']<=r['frame'] and r['frame']+4<=r['segment_end']
    assert set(counts)=={'replay','full_episode'}
    assert .39<counts['full_episode']/50000<.41
    assert .49<tasks['A']/50000<.51
    assert .48<episodes[0]/sum(episodes.values())<.52
    # Seven short legal anchors vs87 long anchors. Label splitting must not oversample short skills.
    assert .065<segments['short']/50000<.085


def test_two_source_rejects_extra_or_missing_source(tmp_path):
    manifest,_=setup(tmp_path)
    bad=copy.deepcopy(manifest);bad['phases'][0]['sources'].pop()
    with pytest.raises(ValueError,match='exactly positive'):MixturePlan(bad,base_dir=tmp_path)
    bad=copy.deepcopy(manifest);bad['phases'][0]['sources'][0]['weight']=0
    with pytest.raises(ValueError,match='exactly positive'):MixturePlan(bad,base_dir=tmp_path)


def test_restart_and_distributed_stream_same_samples(tmp_path):
    manifest,plan=setup(tmp_path);restored=MixturePlan(manifest,base_dir=tmp_path)
    saved=consumed_contract(plan,step=200)
    compare_consumed_contract(saved,restored,step=200)
    expected=[plan.reference(i) for i in range(3200,3392)]
    rank0=[restored.reference(i) for i in range(3200,3296)]
    rank1=[restored.reference(i) for i in range(3296,3392)]
    assert rank0+rank1==expected
    changed=copy.deepcopy(manifest);changed['phases'][0]['sources'][0]['weight']=.5
    different=MixturePlan(changed,base_dir=tmp_path)
    with pytest.raises(ValueError):compare_consumed_contract(saved,different,step=200)


def test_prepared_bundle_uses_new_lr_and_no_formal_authorization():
    here=Path(__file__).resolve().parent/'launch_bundle'
    argv=json.loads((here/'inputs/compiled/training_argv.json').read_text())
    args=argv['argv']
    assert args[args.index('--lr-schedule.peak-lr')+1]=='1e-06'
    assert args[args.index('--lr-schedule.decay-steps')+1]=='20000'
    assert args[args.index('--num-train-steps')+1]=='5000'
    assert not argv['launch_authorized']
    assert not (here/'inputs/formal_authorization.json').exists()
    assert not (here/'inputs/gpu_smoke_success.json').exists()

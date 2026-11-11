"""Prepare an isolated two-source launch bundle; deliberately no launch authorization."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
BUNDLE=HERE/'launch_bundle'
OLD=WORKSPACE/'analysis/retraining_20261013'
sys.path.insert(0,str(BUNDLE/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False)+'\n')


def main():
    compiled=BUNDLE/'inputs/compiled'
    refs={'replay':[],'full_episode':[]}; roots={};counts=Counter()
    indexes=[read(OLD/'training_integration_v1/replay_index.json')]
    indexes += [read(p) for p in sorted((OLD/'training_integration_v1/compiled_v2/indexes').glob('*_full_episode.json'))]
    for p in sorted((HERE/'evidence/new_full_indexes').glob('*.json')):
        idx=read(p)['index'];idx['source_kind']='full_episode';idx['provenance']['language_loss_enabled']=False
        idx['fingerprint']=hashlib.sha256(json.dumps({k:v for k,v in idx.items() if k!='fingerprint'},
                sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        indexes.append(idx)
    assert len(indexes)==271
    for i,idx in enumerate(indexes):
        kind=idx['source_kind'];assert kind in refs
        path=compiled/'indexes'/f'{i:04d}_{kind}.json';dump(path,idx)
        refs[kind].append(dict(path=path.relative_to(compiled).as_posix(),sha256=sha(path)))
        roots[idx['root_id']]=idx['dataset_root'];counts[kind]+=len(idx['episodes'])
    manifest=dict(schema_version=3,kind='b1k_two_source_mixture',seed=42,global_batch_size=192,action_horizon=32,
        phases=[dict(start_step=0,end_step=20000,sources=[dict(kind=k,weight=w,indexes=refs[k],group_sampling='task_balanced')
            for k,w in [('replay',0.6),('full_episode',0.4)]])])
    dump(compiled/'mixture.json',manifest);dump(compiled/'roots.json',roots)
    contract=read(HERE/'learning_rate_contract.json');dump(compiled/'learning_rate_contract.json',contract)
    plan=MixturePlan.load(compiled/'mixture.json')
    sampled=Counter()
    for i in range(100000):
        r=plan.reference(i);sampled[r['source_kind']]+=1
        assert r['segment_start']<=r['frame'] and r['frame']+32<=r['segment_end']
    assert set(sampled)=={'replay','full_episode'} and abs(sampled['full_episode']/100000-0.4)<0.01
    for i in [0,1,191,192,383,384,959999,960000,3839999]:assert plan.reference(i)==MixturePlan.reference(plan,i)
    old=OLD/'nebula_formal_20261014_080449_20k/bundle'
    argv=read(old/'inputs/compiled/training_argv.json')['argv']
    changes={'--exp-name':'two_source_rft_1688977','--num-train-steps':'5000',
        '--lr-schedule.peak-lr':'1e-06','--lr-schedule.decay-lr':'2e-07',
        '--data.subtask-mixture-path':'inputs/compiled/mixture.json',
        '--data.subtask-mixture-roots-path':'inputs/compiled/roots.json',
        '--learning-rate-contract-path':'inputs/compiled/learning_rate_contract.json'}
    for k,v in changes.items():argv[argv.index(k)+1]=v
    dump(compiled/'training_argv.json',dict(argv=argv,launch_authorized=False,
        root_mapping_must_be_resolved_on_training_host=True,checkpoint_1688977_required=True))
    job=read(old/'nebula/worker_job.json')
    job.update(mode='preparation_only',formal_training_authorized=False,
        run_id='b1k-two-source-rft-1688977-20261015',attempt_id='stage5000-g0',stage_end_step=5000,
        max_steps=20000,resume=False,validation_gate_passed=False,mixture_fingerprint=plan.fingerprint,
        output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/two_source_rft/b1k-two-source-rft-1688977-20261015')
    for k in ['gpu_smoke_prefix','data_manifest_key','data_manifest_sha256']:
        job[k]='PENDING_NEW_TWO_SOURCE_ACCEPTANCE'
    dump(BUNDLE/'nebula/worker_job.json',job)
    dump(compiled/'preparation.json',dict(prepared_at=time.time(),mixture_fingerprint=plan.fingerprint,
        selected_episode_counts=dict(counts),source_kinds=['replay','full_episode'],roots=len(roots),
        sampling_checked_100000=dict(sampled),excluded_sources_never_sampled=True,
        original_102_holdouts_retained=True,metadata_ready=True,training_host_data_ready=False,
        real_decoded_CPU_batch_verified=False,GPU_smoke_verified=False,training_started=False,
        sampling_policy='uniform task, then instance, then distinct episode; uniform legal anchor within episode',
        sampler_version=3,realized_source_ratio_is_expectation=True))
    assert not (BUNDLE/'inputs/formal_authorization.json').exists()
    assert not (BUNDLE/'inputs/gpu_smoke_success.json').exists()
    assert counts==dict(replay=15949,full_episode=270)
    print(json.dumps(dict(episodes=dict(counts),sampled_100000=dict(sampled),roots=len(roots),
        actual_decoded_batch_verified=False,formal_training_authorized=False)))


if __name__=='__main__':main()

"""Prepare the 1000-update paired control; require a verified smoke before submission."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DEST = ROOT/'formal_stage/bundle'
RUN = 'b1k-expert-control-1688977-20261016'


def load(p):
    return json.loads(p.read_text())


def save(p,value):
    p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite a prepared experiment')
    source = ROOT/'smoke/bundle'
    for name,digest in load(source/'bundle_manifest.json').items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    shutil.copytree(source,DEST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    original = PARENT/'formal_stage/bundle'
    for name in ('nebula/train_four_source.py','nebula/training_cluster.json','inputs/formal_run_contract.json'):
        shutil.copy2(original/name,DEST/name)
    smoke = load(source/'nebula/smoke_job.json')
    job = load(original/'nebula/worker_job.json')
    job.update(run_id=RUN,attempt_id='stage1000-g0',max_steps=1000,stage_end_step=1000,
        output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/expert_control/'+RUN,
        gpu_smoke_prefix=smoke['output'],mixture_fingerprint=smoke['mixture_fingerprint'],
        experiment='paired_expert_only_control',formal_training_authorized=True)
    save(DEST/'nebula/worker_job.json',job)
    # This records the current goal and finite experiment decision, not old trial authorization.
    auth={'kind':'expert_control_under_active_improvement_goal',
        'human_instruction':'请不断尝试各种方法，使模型能力提升',
        'formal_training_allowed':True,'workers':2,'gpus_per_worker':8,
        'max_completed_updates':1000,'global_batch_size':192,'initial_checkpoint':1688977,
        'fresh_optimizer':True,'peak_learning_rate':1e-6,'fixed_decay_end':20000,
        'automatic_extension':False,'requires_verified_expert_smoke':True,
        'rationale':'Compare identical 1000-update checkpoints; replace only full-episode windows.'}
    save(DEST/'inputs/formal_authorization.json',auth)
    contract=load(DEST/'inputs/formal_run_contract.json')
    contract.update(run_id=RUN,max_authorized_updates=1000,
        mixture_fingerprint=smoke['mixture_fingerprint'],
        quality_sampling_policy=load(DEST/'inputs/compiled/mixture.json')['quality_sampling'])
    save(DEST/'inputs/formal_run_contract.json',contract)
    # The full finite sampling schedule stays at 5000; stopping earlier must not reshuffle it.
    assert load(DEST/'inputs/compiled/mixture.json')['quality_sampling']['finite_budget_updates']==5000
    assert not (DEST/'inputs/gpu_smoke_success.json').exists()
    assert (DEST/'inputs/compiled/learning_rate_contract.json').read_bytes()==(original/'inputs/compiled/learning_rate_contract.json').read_bytes()
    save(DEST/'bundle_manifest.json',{str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts})
    save(ROOT/'formal_stage/preparation.json',{'run_id':RUN,'updates':1000,
        'mixture_fingerprint':job['mixture_fingerprint'],'sampling_schedule_updates':5000,
        'learning_rate_schedule_unchanged':True,'cloud_jobs_created':0,
        'submission_ready':False,'missing_gate':'verified expert-only two-update GPU smoke'})
    print(json.dumps(load(ROOT/'formal_stage/preparation.json')))


if __name__=='__main__':
    main()

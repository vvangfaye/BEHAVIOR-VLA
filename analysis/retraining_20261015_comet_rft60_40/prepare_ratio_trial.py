"""Prepare a ratio-only overlay; never uploads, creates claims, or submits jobs."""
from __future__ import annotations
import argparse,collections,copy,hashlib,json,math,shutil
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE/'formal_stage/bundle'
sys.path.insert(0,str(BASE/'openpi/src'))
from openpi.training.subtask_mixture import MixturePlan

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')

def build(replay,full):
    if any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in (replay,full)) or not math.isclose(replay+full,1.,abs_tol=1e-12):
        raise ValueError('Both sources must have positive weights summing to one')
    compiled=BASE/'inputs/compiled';original=load(compiled/'mixture.json');new=copy.deepcopy(original)
    assert original['kind']=='b1k_two_source_mixture' and len(new['phases'])==1
    sources=new['phases'][0]['sources'];assert {s['kind'] for s in sources}=={'replay','full_episode'}
    for s in sources:s['weight']={'replay':replay,'full_episode':full}[s['kind']]
    # No source identity, episode, task, seed, curriculum boundary, or sampler change.
    restored=copy.deepcopy(new)
    for s in restored['phases'][0]['sources']:s['weight']=next(x['weight'] for x in original['phases'][0]['sources'] if x['kind']==s['kind'])
    assert restored==original
    plan=MixturePlan(new,base_dir=compiled)
    argv=load(compiled/'training_argv.json')['argv']
    assert argv[argv.index('--num-train-steps')+1]=='5000' and '--resume' not in argv
    lr=load(compiled/'learning_rate_contract.json')
    assert lr['schedule']=={'warmup_steps':200,'peak_lr':1e-6,'decay_steps':20000,'decay_lr':2e-7}
    return new,plan

def prepare(replay,full,name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):raise ValueError('Unsafe trial name')
    manifest,plan=build(replay,full)
    out=HERE/'ratio_trials'/name;out.mkdir(parents=True,exist_ok=False)
    counts=collections.Counter()
    for n in range(100000):
        r=plan.reference(n);counts[r['source_kind']]+=1
        assert r['segment_start']<=r['frame'] and r['frame']+32<=r['segment_end']
    assert set(counts)=={'replay','full_episode'} and abs(counts['full_episode']/100000-full)<0.01
    job=load(BASE/'nebula/worker_job.json')
    job.update(mode='preparation_only',formal_training_authorized=False,run_id='b1k-ratio-'+name+'-1688977',attempt_id='stage5000-g0',
        output='/data/oss_bucket_0/wangfaye/result/behavior-challenge/two_source_rft/b1k-ratio-'+name+'-1688977',
        gpu_smoke_prefix='PENDING_NEW_RATIO_ACCEPTANCE',mixture_fingerprint=plan.fingerprint,resume=False,max_steps=5000,stage_end_step=5000)
    contract=load(BASE/'inputs/formal_run_contract.json');contract.update(run_id=job['run_id'],mixture_fingerprint=plan.fingerprint)
    save(out/'mixture.json',manifest);save(out/'worker_job.prepared.json',job);save(out/'formal_run_contract.prepared.json',contract)
    receipt={'kind':'ratio_only_trial_preparation','trial':name,'weights':{'replay':replay,'full_episode':full},
        'base_compiled_directory':str((BASE/'inputs/compiled').resolve()),'base_bundle_manifest_sha256':sha(BASE/'bundle_manifest.json'),
        'base_mixture_sha256':sha(BASE/'inputs/compiled/mixture.json'),'data_manifest_sha256':job['data_manifest_sha256'],
        'learning_rate_contract_sha256':sha(BASE/'inputs/compiled/learning_rate_contract.json'),
        'mixture_fingerprint':plan.fingerprint,'sampler_version':3,'selected_episodes':{'replay':15949,'full_episode':270},
        'empirical_100000_draws':dict(counts),'experimental_change':['source sampling weights only'],
        'fresh_from_checkpoint':1688977,'new_optimizer':True,'budget_updates':5000,'global_batch':192,
        'cpu_real_batch_verified':False,'gpu_smoke_verified':False,'full_bundle_assembled':False,
        'needs_future_bundle_assembly':'Copy unchanged base inputs/indexes with original relative paths; overlay mixture and new job/contract; never reuse old proof files.',
        'training_launch_authorized_by_this_file':False,'cloud_jobs_created':0}
    save(out/'preparation.json',receipt)
    return receipt

def assemble(name):
    """Assemble an inert bundle without borrowing any old authorization/proof."""
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Unsafe trial name')
    out=HERE/'ratio_trials'/name;receipt=load(out/'preparation.json')
    if receipt['base_bundle_manifest_sha256']!=sha(BASE/'bundle_manifest.json'):
        raise ValueError('Base bundle manifest changed')
    expected,plan=build(receipt['weights']['replay'],receipt['weights']['full_episode'])
    if load(out/'mixture.json')!=expected or plan.fingerprint!=receipt['mixture_fingerprint']:
        raise ValueError('Prepared ratio no longer matches the declared experiment')
    job=load(out/'worker_job.prepared.json');contract=load(out/'formal_run_contract.prepared.json')
    if (job['mode']!='preparation_only' or job['formal_training_authorized'] is not False
        or job['mixture_fingerprint']!=plan.fingerprint or contract['mixture_fingerprint']!=plan.fingerprint
        or contract['run_id']!=job['run_id'] or job['resume'] is not False):
        raise ValueError('Prepared job identity or launch guard changed')
    manifest=load(BASE/'bundle_manifest.json')
    for rel,digest in manifest.items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or (BASE/path).is_symlink():
            raise ValueError('Unsafe base file path')
        if sha(BASE/path)!=digest:raise ValueError('Base file changed: '+rel)
    bundle=out/'bundle';bundle.mkdir(exist_ok=False)
    replacements={'inputs/compiled/mixture.json':expected,'nebula/worker_job.json':job,
        'inputs/formal_run_contract.json':contract,
        'inputs/formal_authorization.json':dict(formal_training_allowed=False,
            scope='Prepared ratio experiment only; not a cloud submission authorization',
            source_weights=receipt['weights'],run_id=job['run_id'],automatic_retry=False,
            require_new_two_source_cpu_proof=True,require_new_two_source_gpu_smoke=True)}
    for rel in manifest:
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if rel in replacements:save(dest,replacements[rel])
        else:shutil.copy2(BASE/rel,dest)
    result={rel:sha(bundle/rel) for rel in manifest}
    if {rel for rel in manifest if manifest[rel]!=result[rel]}!=set(replacements):
        raise ValueError('Unexpected change outside the ratio and experiment identity')
    assembled=MixturePlan.load(bundle/'inputs/compiled/mixture.json')
    if assembled.fingerprint!=plan.fingerprint:raise ValueError('Relocated mixture identity differs')
    save(bundle/'bundle_manifest.json',result)
    receipt.update(full_bundle_assembled=True,bundle_manifest_sha256=sha(bundle/'bundle_manifest.json'),
        copied_files=len(result),changed_files=sorted(replacements),cpu_real_batch_verified=False,
        gpu_smoke_verified=False,training_launch_authorized_by_this_file=False,cloud_jobs_created=0)
    receipt.pop('needs_future_bundle_assembly',None)
    save(out/'preparation.json',receipt)
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',type=float);parser.add_argument('--full',type=float);parser.add_argument('--name',required=True);parser.add_argument('--assemble',action='store_true');a=parser.parse_args()
    if not a.assemble and (a.replay is None or a.full is None):parser.error('--replay and --full required for preparation')
    print(json.dumps(assemble(a.name) if a.assemble else prepare(a.replay,a.full,a.name),ensure_ascii=False))

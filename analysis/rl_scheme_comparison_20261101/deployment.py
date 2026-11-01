"""Submit immutable, isolated jobs for the bounded RL comparison."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import zlib

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ADAPT=PROJECT/'analysis/rlinf_adaptation_20261020'
sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
sys.path.insert(0,str(PROJECT/'analysis/serf_multitask_20261029'))
sys.path.insert(0,str(ADAPT/'multitask_validation_20261029/capacity_repair_20261030'))
from video_eval_adapter import aoss_client
from cluster_helpers import query_standard_capacity
from read_quota_exact import query_quota

CLI='/home/faye/.local/bin/sco'
ENV={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
TERMINAL={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def describe(name):
    p=subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],
        env=ENV,capture_output=True,text=True,check=True,timeout=40)
    job=json.loads(p.stdout);assert job['ownership']['user_name']=='260010165'
    return job


def storage_proof(worker_plan):
    cache=HERE/'quota.json'
    quota=json.loads(cache.read_text()) if cache.exists() else None
    if quota is None or time.time()-quota['observed_at']>180:
        quota=query_quota();save(cache,quota)
    campaign=json.loads((HERE/'plan.json').read_text())
    bounds=campaign['storage_bounds']
    completed=sum(json.loads(f.read_text()).get('model_bytes',0)+json.loads(f.read_text()).get('optimizer_bytes',0)
                  for f in (HERE/'checkpoints').glob('*.json'))
    reserve=bounds['comparison_total_budget_bytes']-completed
    assert reserve>=bounds['training_worker_limit_bytes']
    water=ADAPT/'multitask_validation_20261029/water'
    original=json.loads((water/'plan.json').read_text())
    other=0
    credited_original=0
    if not (water/'outcome.json').exists() and not campaign.get('combined_training_storage',False):
        other=original['runtime_storage']['max_persistent_bytes']
        owned_path=water/'owned_storage_latest.json'
        owned=json.loads(owned_path.read_text()) if owned_path.exists() else {}
        if (owned.get('experiment')==original['experiment'] and owned.get('output')==original['output']
            and owned.get('method')=='committed_regular_files_no_external_hardlinks'
            and 0<=time.time()-owned.get('observed_at',0)<=180
            and owned.get('observed_at',0)<=quota['observed_at']):
            credited_original=int(owned['committed_bytes'])
            assert 0<=credited_original<=other
            other-=credited_original
    row=quota['rows'][0];margin=20*1024**3;required=reserve+other+margin
    proof={'complete':True,'passed':row['quota_bytes']-row['used_bytes']>=required,
        'experiment':worker_plan['experiment'],'account_root':'/mnt/afs/260010165',
        'account_allocated_bytes':row['used_bytes'],'quota_lower_bound_bytes':row['quota_bytes'],
        'new_task_reserve_bytes':reserve,'other_remaining_commitments_bytes':other,
        'comparison_total_budget_bytes':bounds['comparison_total_budget_bytes'],
        'comparison_completed_checkpoint_bytes':completed,'original_committed_bytes':credited_original,
        'headroom_margin_bytes':margin,'required_remaining_bytes':required,
        'observed_at':quota['observed_at'],'quota_source':quota['method']}
    assert proof['passed'],'Waiting for the fixed comparison storage budget'
    return proof


def worker_plan(spec):
    campaign=json.loads((HERE/'plan.json').read_text())
    plan=json.loads((HERE/'source_plan.json').read_text())
    role=spec['role'];variant=spec['variant']
    plan['experiment']='rlinf_multitask_rlcompare_'+variant+'_20261101'
    plan['worker_prefix']='cmp_';plan['parallel_workers']=1
    plan['baseline_sources']={};plan['rounds']=1
    plan['phase_limits_seconds']={'baseline':campaign['eval_timeout_seconds'],
        'final':campaign['eval_timeout_seconds'],'train1':campaign['train_timeout_seconds']}
    plan['phase_order']=['baseline','train1','final']
    plan['comparison_role']=role;plan['comparison_variant']=variant
    plan['source_rollout_root']=campaign['source_output']
    plan['source_rollout_receipts']=[{k:r[k] for k in ['episode','instance','seed','path','sha256','policy_hash','contract','chunks']}
        for r in json.loads((HERE/'source_training_receipts.json').read_text())]
    plan['source_reward_contract_sha256']=hashlib.sha256(json.dumps(
        {'goals':plan['goal_contract'],'shaping':plan['shaping']},sort_keys=True).encode()).hexdigest()
    if variant=='gamma1_ppo':plan['optimizer']['gamma']=plan['shaping']['gamma']=1.0
    if role=='train':
        plan['output']=campaign['remote_root']+'/variants/'+variant+'/results'
        plan['eval_instances']=[]
    else:
        case=spec['case']
        # torch_model uses plan.seed to initialize the auxiliary value head.
        # Keep model initialization identical; evaluate() supplies the case seed
        # to both the simulator and action-noise generator independently.
        assert plan['seed']==plan['model_seed']
        plan['evaluation']['cases']=[case];plan['evaluation_runtime_order']=[case]
        plan['eval_instances']=[case[0]]
        plan['output']=campaign['remote_root']+f'/evaluations/{variant}/i{case[0]}_s{case[1]}/results'
        if variant!='original':
            checkpoint=(campaign['source_control_checkpoint'] if variant=='control' else
                campaign['remote_root']+'/variants/'+variant+'/results/checkpoints/round_01')
            plan['comparison_checkpoint']=checkpoint
    limits=campaign['storage_bounds']
    limit=limits['training_worker_limit_bytes'] if role=='train' else limits['evaluation_worker_limit_bytes']
    plan['runtime_storage']={**plan['runtime_storage'],'max_persistent_bytes':limit,'shared_budget_roots':[plan['output']]}
    return plan


def submit(spec):
    campaign=json.loads((HERE/'plan.json').read_text());folder=HERE/'jobs'/spec['id'];receipt_path=folder/'submission.json'
    assert not receipt_path.exists(),'Reconcile exact existing job instead of submitting twice'
    marker=json.loads((HERE/'implementation_verified.json').read_text())
    assert hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest()==marker['plan_sha256']
    for name,digest in marker['runtime_sha256'].items():
        assert hashlib.sha256((HERE/'runtime_source'/name).read_bytes()).hexdigest()==digest, 'Frozen comparison implementation changed'
    plan=worker_plan(spec);proof=storage_proof(plan)
    policy_path=HERE/'resource_policy.json'
    if policy_path.exists():
        policy=json.loads(policy_path.read_text());assert policy['quota_type']=='spot'
        quota='spot'
        capacity={'selection':'explicit_user_spot','policy':policy,'fresh_query':False}
    else:
        capacity=query_standard_capacity(CLI,ENV)
        quota='reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
    folder.mkdir(parents=True,exist_ok=True);save(folder/'capacity.json',capacity);save(folder/'worker_plan.json',plan)
    template=describe('b1k-repeat1688-1019-batch-00')
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    names=['common.py','training.py','experiment.py','simulator.py','helpers.py','eval_utils.py',
           'resource_watch.py','goals.py','storage_guard.py','capacity_admission.py']
    names+=['b1k_rlinf/'+n+'.py' for n in ['__init__','action_interface','factory','io','model','precision','tokenizer']]
    contents={n:(HERE/'runtime_source'/n).read_bytes() for n in names}
    if spec['variant']=='original':contents['driver.py']=(HERE/'runtime_source/baseline_driver.py').read_bytes()
    else:
        contents['driver.py']=(HERE/'runtime_source/comparison_driver.py').read_bytes()
        contents['variant_update.py']=(HERE/'runtime_source/variant_update.py').read_bytes()
    contents['plan.json']=(json.dumps(plan,sort_keys=True)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=9) as archive:
        for name,data in contents.items():
            info=tarfile.TarInfo('run/'+name);info.size=len(data);info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    bundle=stream.getvalue();assert len(bundle)<1024**2
    prefix=campaign['aoss_prefix']+spec['id']+'/'
    name='b1k-rlexp-1101-'+spec['id'].replace('_','-')
    client=aoss_client()
    try:
        client.put_object(Bucket='260010165',Key=prefix+'bundle.tar.gz',Body=bundle)
        assert client.get_object(Bucket='260010165',Key=prefix+'bundle.tar.gz')['Body'].read()==bundle
        def signed(key):return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+key},ExpiresIn=5*86400)
        before=json.loads((ADAPT/'cloud_smoke/local_preflight.json').read_text());artifacts=json.loads((ADAPT/'cloud_smoke/artifacts.json').read_text())
        reuse={'checkpoint/'+k:v['sha256'] for k,v in artifacts['checkpoint'].items()}
        reuse['bundle.tar.gz']=artifacts['bundle']['sha256']
        reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
        for n in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
            reuse['upstream/'+n]=hashlib.sha256((ADAPT/'upstream'/n).read_bytes()).hexdigest()
        phase='train1' if spec['role']=='train' else 'baseline' if spec['variant']=='original' else 'final'
        config={'root':campaign['remote_root']+'/jobs/'+spec['id'],
            'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
            'task_phase':phase,'rank':0,'attempt':spec['attempt'],'job_name':name,
            'bundle_inline':base64.b64encode(bundle).decode(),
            'bundle':{'bytes':len(bundle),'sha256':hashlib.sha256(bundle).hexdigest()},
            'reuse_sha256':reuse,'status_put':signed('status.json'),'result_put':signed('result.tar.gz'),
            'artifact_puts':{},'storage_proof':proof}
        code=(HERE/'runtime_source/bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(code.encode())).decode()
        limit=plan['phase_limits_seconds'][phase]
        command=f'timeout --signal=TERM --kill-after=60s {limit-60}s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000
        receipt={**spec,'job':name,'quota':quota,'aoss_prefix':prefix,'output':plan['output'],
            'max_seconds':limit,'state':'submitting','submitted_at':time.time(),
            'source_sha256':{n:hashlib.sha256(d).hexdigest() for n,d in contents.items()},
            'startup_sha256':hashlib.sha256(command.encode()).hexdigest()}
        with receipt_path.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
        try:
            result=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00','--name='+name,'--job-name='+name,
                '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
                '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template['roles'][0]['image_path'],'--command='+command],
                env=ENV,capture_output=True,text=True,timeout=90)
            receipt['submit_returncode']=result.returncode
        except subprocess.TimeoutExpired:receipt['submit_returncode']='unknown_timeout'
        save(receipt_path,receipt)
        actual=describe(name)
        assert actual['scheduling']['quota_type']==quota.upper()
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=actual['uid'],state=actual['state']);save(receipt_path,receipt)
        return receipt
    finally:
        client.close()

import base64
import copy
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

sys.path.insert(0,str(Path(__file__).parent))
import submit_codex_pilot_evaluation as module


def ref(path,raw):return dict(path=str(path),size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def dump(path,value):path.write_text(json.dumps(value));return path


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'INTENTS',tmp_path/'canonical')
    afs=module.cloud.AFS;run='fixture-run';checkpoint=afs+'/inference/'+run
    manifest=('a'*64+'  params/_METADATA\n'+'b'*64+'  assets/subtask_conditioning.json\n'+'c'*64+'  assets/b1k_codex_'+run+'/norm_stats.json\n').encode()
    manifest_ref=ref(checkpoint+'/SHA256SUMS.inference',manifest)
    plan=ref(afs+'/training/'+run+'/training_plan.json',b'not opened locally')
    train=dict(kind='codex_pilot_trained_checkpoint_v1',checkpoint_id='pi05_b1k_codex_'+run+'_256',
        parent_checkpoint_step=1688977,optimizer_steps=256,ready_for_evaluation=True,real_lerobot_readback_verified=True,
        base_normalization_preserved=True,inference_staging_complete=True,training_plan=plan,
        repo_id='b1k_codex_'+run,staged_checkpoint=checkpoint,checkpoint_manifest=manifest_ref,
        output_checkpoint_manifest_sha256=manifest_ref['sha256'],base_norm_sha256='c'*64)
    train_raw=json.dumps(train).encode();train_ref=ref(afs+'/training/'+run+'/training_receipt.json',train_raw)
    phases={key:{'state':'verified'} for key in ('data_preflight','jax_preflight','smoke','full','stage')}
    phases['stage']['evidence']=train_ref
    report=dict(kind=module.common.worker.KIND,state='complete',ready_for_evaluation=True,model_teacher_enabled=False,
                phases=phases,training_plan=plan,training_receipt=train_ref,checkpoint=checkpoint)
    paths=[tmp_path/'training-report.json',tmp_path/'training-receipt.json',tmp_path/'SHA256SUMS.inference',tmp_path/'submission.json']
    dump(paths[0],report);paths[1].write_bytes(train_raw);paths[2].write_bytes(manifest)
    return paths,report,train


class Store:
    def __init__(self):self.data={};self.corrupt=False;self.puts=0
    def put_object(self,*,Bucket,Key,Body):self.data[Key]=Body;self.puts+=1
    def get_object(self,*,Bucket,Key):return {'Body':io.BytesIO(b'changed' if self.corrupt else self.data[Key])}
    def generate_presigned_url(self,*args,**kwargs):return 'https://invalid.example/?signature=PRIVATE'


@pytest.fixture
def cloud(monkeypatch):
    template=json.loads((module.common.ROOT/'eval_results/sco_subtask_collection_20261010/template.json').read_text())
    store=Store();calls=[];jobs={};describes=[]
    def describe(name):
        describes.append(name)
        return copy.deepcopy(template if name==module.TEMPLATE else jobs[name])
    def run(args,**kwargs):
        calls.append((args,kwargs));name=next(x.split('=',1)[1] for x in args if x.startswith('--name='))
        jobs[name]={**copy.deepcopy(template),'name':name,'uid':'uid-'+name,'state':'RUNNING'}
        jobs[name]['roles'][0]['image_path']=module.common.IMAGE
        jobs[name]['scheduling']['quota_type']='RESERVED'
        return subprocess.CompletedProcess(args,0)
    def capacity(*args):
        active=[dict(name=job['name'],uid=job['uid'],state=job['state'],gpu_count=1) for job in jobs.values()
                if job['state'] in module.admission.ACTIVE_STATES]
        return dict(owner=module.cloud.BUCKET,quota_type='reserved',fresh_query=True,observed_at=module.common.now().isoformat(),
            standard_limit=4,used_gpus=len(active),active_reserved=active,
            queried_states=sorted(module.admission.ACTIVE_STATES),pages_read=10)
    monkeypatch.setattr(module.cloud,'describe',describe);monkeypatch.setattr(module.cloud,'storage',lambda:store)
    monkeypatch.setattr(module.subprocess,'run',run);monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    return store,calls,jobs,describes


def submit(inputs,shard='radio-pilot',suffix='first'):
    paths=inputs[0]
    return module.submit(*paths,shard=shard,name='behavior-codex-native-'+suffix)


def test_single_fixed_native_shard_and_real_resources(inputs,cloud):
    value=submit(inputs);args,kw=cloud[1][0];p=value['payload']
    assert p['shard_ids']==['radio-pilot'] and p['code']==module.CODE and p['driver_sha256']==module.DRIVER_SHA
    assert p['shard_timeout_seconds']==13200 and value['worker_seconds']==13800 and value['max_seconds']==14400
    assert '--worker-spec=n12lp.nn.i10a.1' in args and '--quota-type=reserved' in args
    assert '--container-image-url='+module.common.IMAGE in args and '--storage-mount='+module.common.MOUNT in args
    command=next(x for x in args if x.startswith('--command='))
    assert command.startswith('--command=timeout --signal=TERM --kill-after=30 14400 '+module.common.PYTHON)
    assert kw['timeout']==90 and len(cloud[1])==1 and 'signature' not in json.dumps(value)
    assert json.loads((module.INTENTS/'panel.json').read_text())['admitted']==['radio-pilot']


@pytest.mark.parametrize('shard',module.worker.SHARD_IDS)
def test_each_shard_payload_keeps_original_protocol(inputs,shard):
    p,*_=module.prepare_inputs(*inputs[0][:3],shard=shard,name='behavior-codex-native-fixture')
    assert p['shard_ids']==[shard] and p['source_manifest_sha256']==module.worker.SOURCE_SHA256
    assert not {'task','max_steps','seed','teacher','instance_ids','prompt_mode'} & p.keys()


@pytest.mark.parametrize('field,value',[('state','failed'),('ready_for_evaluation',False),('model_teacher_enabled',True),('kind','other')])
def test_not_ready_never_contacts_cloud(inputs,cloud,field,value):
    inputs[1][field]=value;dump(inputs[0][0],inputs[1])
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[3]==[] and cloud[1]==[] and not module.INTENTS.exists()


@pytest.mark.parametrize('fault',['phase','receipt_bytes','manifest_bytes','phase_binding','plan_binding','norm','steps','destination','manifest_path'])
def test_training_evidence_mutation_rejected(inputs,cloud,fault):
    paths,report,train=inputs
    if fault=='phase':report['phases']['full']['state']='failed'
    elif fault=='receipt_bytes':paths[1].write_text('{}')
    elif fault=='manifest_bytes':paths[2].write_text('changed')
    elif fault=='phase_binding':report['phases']['stage']['evidence']={}
    else:
        if fault=='plan_binding':train['training_plan']={}
        elif fault=='norm':train['base_norm_sha256']='d'*64
        elif fault=='steps':train['optimizer_steps']=2
        elif fault=='destination':train['staged_checkpoint']='/elsewhere'
        elif fault=='manifest_path':train['checkpoint_manifest']['path']=module.cloud.AFS+'/wrong'
        raw=json.dumps(train).encode();paths[1].write_bytes(raw)
        newref=ref(report['training_receipt']['path'],raw);report['training_receipt']=newref;report['phases']['stage']['evidence']=newref
    dump(paths[0],report)
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[1]==[] and cloud[3]==[]


@pytest.mark.parametrize('state',sorted(module.admission.ACTIVE_STATES | module.TERMINAL))
def test_exact_active_or_terminal_predecessor_allows_next_shard(inputs,cloud,state):
    first=submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    cloud[2][first['name']]['state']=state
    result=submit(inputs,'a000','second')
    assert result['state']=='submitted' and len(cloud[1])==2
    assert len(result['panel_active_before']) == int(state in module.admission.ACTIVE_STATES)


def test_four_active_jobs_block_fifth_without_intent_until_one_exact_job_ends(inputs,cloud):
    receipts=[]
    for index,shard in enumerate(module.worker.SHARD_IDS[:4]):
        inputs[0][3]=inputs[0][3].with_name(f'submission-{index}.json')
        receipts.append(submit(inputs,shard,f'ordered-{index}'))
    assert len(cloud[1])==cloud[0].puts==4
    inputs[0][3]=inputs[0][3].with_name('fifth.json')
    with pytest.raises(ValueError,match='Four exact panel jobs'):
        submit(inputs,'a089','fifth')
    assert not inputs[0][3].exists() and not (module.INTENTS/'intent-a089.json').exists()
    assert len(cloud[1])==cloud[0].puts==4
    cloud[2][receipts[0]['name']]['state']='FAILED'
    fifth=submit(inputs,'a089','fifth')
    assert fifth['state']=='submitted' and len(fifth['panel_active_before'])==3
    assert len(cloud[1])==5
    assert json.loads((module.INTENTS/'panel.json').read_text())['admitted']==list(module.worker.SHARD_IDS)


def test_panel_active_limit_does_not_depend_on_capacity_list_visibility(inputs,cloud,monkeypatch):
    original=module.admission.query_standard_capacity
    def empty_capacity(*args):
        value=original();value.update(used_gpus=0,active_reserved=[]);return value
    monkeypatch.setattr(module.admission,'query_standard_capacity',empty_capacity)
    for index,shard in enumerate(module.worker.SHARD_IDS[:4]):
        inputs[0][3]=inputs[0][3].with_name(f'submission-{index}.json')
        value=submit(inputs,shard,f'ordered-{index}')
        assert value['capacity_preflight']['effective_used_gpus']==index
        assert value['admission']['effective_used_gpus']==index
    inputs[0][3]=inputs[0][3].with_name('fifth.json')
    with pytest.raises(ValueError,match='Four exact panel jobs'):submit(inputs,'a089','fifth')
    assert len(cloud[1])==4 and not (module.INTENTS/'intent-a089.json').exists()


@pytest.mark.parametrize('when',['preflight','final'])
def test_hidden_panel_gpu_plus_three_unrelated_gpus_blocks_both_capacity_gates(inputs,cloud,monkeypatch,when):
    submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    original=module.admission.query_standard_capacity;queries=[]
    def capacity(*args):
        queries.append(True);value=original()
        if when=='preflight' or len(queries)==2:
            value.update(used_gpus=3,active_reserved=[dict(name=f'other-{i}',uid=f'other-uid-{i}',
                state='RUNNING',gpu_count=1) for i in range(3)])
        return value
    monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    with pytest.raises(ValueError):submit(inputs,'a000','second')
    assert len(cloud[1])==1
    intent=module.INTENTS/'intent-a000.json'
    if when=='preflight':
        assert not intent.exists() and not inputs[0][3].exists() and cloud[0].puts==1
    else:
        record=json.loads(intent.read_text())
        assert record['state']=='not_submitted' and record['submission_attempted'] is False
        assert record['not_submitted_stage']=='final_capacity_check' and cloud[0].puts==2


def test_same_uid_is_counted_once_across_owner_and_panel_observations(inputs,cloud,monkeypatch):
    first=submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    original=module.admission.query_standard_capacity
    def capacity(*args):
        value=original()
        # A lifecycle transition does not change this exact UID's allocation.
        value['active_reserved'][0]['state']='STARTING'
        value['active_reserved'] += [dict(name=f'other-{i}',uid=f'other-uid-{i}',state='RUNNING',gpu_count=1) for i in range(2)]
        value['used_gpus']=3
        return value
    monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    second=submit(inputs,'a000','second')
    for key in ('capacity_preflight','admission'):
        assert second[key]['effective_used_gpus']==3
        assert len(second[key]['effective_active_reserved'])==3
        assert sum(row['uid']==first['uid'] for row in second[key]['effective_active_reserved'])==1
    assert len(cloud[1])==2


@pytest.mark.parametrize('fault',['missing_uid','empty_name','unknown_state','missing_gpu','zero_gpu','float_gpu','bool_gpu',
                                'uid_name_conflict','uid_gpu_conflict','name_uid_conflict','wrong_owner_total'])
def test_capacity_union_rejects_unknown_or_conflicting_fields(inputs,cloud,fault):
    first=submit(inputs)
    value=module.admission.query_standard_capacity()
    panel=[copy.deepcopy(value['active_reserved'][0])]
    row=value['active_reserved'][0]
    if fault=='missing_uid':row.pop('uid')
    elif fault=='empty_name':row['name']=''
    elif fault=='unknown_state':row['state']='UNKNOWN'
    elif fault=='missing_gpu':row.pop('gpu_count')
    elif fault=='zero_gpu':row['gpu_count']=0;value['used_gpus']=0
    elif fault=='float_gpu':row['gpu_count']=1.0
    elif fault=='bool_gpu':row['gpu_count']=True
    elif fault=='uid_name_conflict':row['name']='different-name'
    elif fault=='uid_gpu_conflict':row['gpu_count']=2;value['used_gpus']=2
    elif fault=='name_uid_conflict':row['uid']='different-uid'
    else:value['used_gpus']=2
    with pytest.raises(ValueError):module.validate_capacity(value,panel)


def test_identical_duplicate_uid_rows_are_not_double_counted(inputs,cloud):
    submit(inputs)
    value=module.admission.query_standard_capacity()
    value['active_reserved']*=2
    merged=module.validate_capacity(value,[value['active_reserved'][0]])
    assert merged['effective_used_gpus']==1 and len(merged['effective_active_reserved'])==1


@pytest.mark.parametrize('first', [True,False])
def test_fixed_submission_order_rejects_skipped_shard_before_new_intent(inputs,cloud,first):
    if not first:
        submit(inputs);inputs[0][3]=inputs[0][3].with_name('later.json')
    with pytest.raises(ValueError,match='declared order'):submit(inputs,'a001','skipped')
    assert len(cloud[1])==int(not first) and not (module.INTENTS/'intent-a001.json').exists()


def test_unknown_live_state_blocks_new_intent(inputs,cloud):
    first=submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    cloud[2][first['name']]['state']='UNRECOGNIZED'
    with pytest.raises(ValueError,match='unknown state'):submit(inputs,'a000','second')
    assert len(cloud[1])==1 and not (module.INTENTS/'intent-a000.json').exists()


def test_same_job_name_cannot_be_shared_by_concurrent_shards(inputs,cloud):
    submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    with pytest.raises(ValueError,match='unique job name'):submit(inputs,'a000','first')
    assert len(cloud[1])==1 and not (module.INTENTS/'intent-a000.json').exists()


def test_two_local_submitters_share_one_panel_lock_and_only_copy_once(inputs,cloud):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier=Barrier(2)
    def start(index):
        barrier.wait(timeout=5)
        try:
            return module.submit(*inputs[0][:3],inputs[0][3].with_name(f'parallel-{index}.json'),
                                 shard='radio-pilot',name=f'behavior-codex-native-parallel-{index}')
        except ValueError as error:return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes=list(pool.map(start,range(2)))
    assert sum(isinstance(value,dict) and value['state']=='submitted' for value in outcomes)==1
    assert sum(isinstance(value,str) and 'already has an intent' in value for value in outcomes)==1
    assert len(cloud[1])==cloud[0].puts==1 and len(list(module.INTENTS.glob('intent-*.json')))==1


def test_existing_panel_prefix_cannot_be_reordered(inputs,cloud):
    submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json');submit(inputs,'a000','second')
    path=module.INTENTS/'panel.json';panel=json.loads(path.read_text());panel['admitted'].reverse();dump(path,panel)
    inputs[0][3]=inputs[0][3].with_name('third.json')
    with pytest.raises(ValueError,match='declared order'):submit(inputs,'a001','third')
    assert len(cloud[1])==2 and not (module.INTENTS/'intent-a001.json').exists()


@pytest.mark.parametrize('fault',['panel_checkpoint','record_checkpoint','resource_shape'])
def test_concurrent_predecessor_must_stay_on_the_exact_checkpoint_and_one_gpu(inputs,cloud,fault):
    first=submit(inputs)
    if fault=='resource_shape':cloud[2][first['name']]['roles'][0]['resource_spec'][0]['limits']['device']='4'
    else:
        path=module.INTENTS/('panel.json' if fault=='panel_checkpoint' else 'intent-radio-pilot.json')
        record=json.loads(path.read_text());record['identity']['checkpoint_id']='other';dump(path,record)
    inputs[0][3]=inputs[0][3].with_name('second.json')
    with pytest.raises(ValueError):submit(inputs,'a000','second')
    assert len(cloud[1])==1 and not (module.INTENTS/'intent-a000.json').exists()


def test_changed_uid_does_not_settle_panel_lease(inputs,cloud):
    first=submit(inputs);inputs[0][3]=inputs[0][3].with_name('second.json')
    cloud[2][first['name']].update(uid='different',state='SUCCEEDED')
    with pytest.raises(ValueError,match='exact evaluation'):submit(inputs,'a000','second')
    assert len(cloud[1])==1


def test_same_shard_cannot_replay_with_new_name(inputs,cloud):
    first=submit(inputs);cloud[2][first['name']]['state']='SUCCEEDED'
    inputs[0][3]=inputs[0][3].with_name('new-name.json')
    with pytest.raises(ValueError,match='already has an intent'):submit(inputs,suffix='new')
    assert len(cloud[1])==1


def test_missing_intent_cannot_refund_slot(inputs,cloud):
    submit(inputs);(module.INTENTS/'intent-radio-pilot.json').unlink()
    inputs[0][3]=inputs[0][3].with_name('other.json')
    with pytest.raises(ValueError,match='intent is missing'):submit(inputs,'a000','second')
    assert len(cloud[1])==1


@pytest.mark.parametrize('outcome',['timeout','nonzero','readback'])
def test_unknown_outcome_blocks_whole_experiment(inputs,cloud,monkeypatch,outcome):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        assert load_intent()['state']=='submission_unknown'
        if outcome=='timeout':raise subprocess.TimeoutExpired(args,90)
        return subprocess.CompletedProcess(args,2 if outcome=='nonzero' else 0)
    def load_intent():return json.loads((module.INTENTS/'intent-radio-pilot.json').read_text())
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(RuntimeError):submit(inputs)
    inputs[0][3]=inputs[0][3].with_name('next.json')
    with pytest.raises(ValueError,match='unresolved'):submit(inputs,'a000','next')
    assert len(calls)==1


@pytest.mark.parametrize('used',[0,1,2,3,4])
def test_capacity_requires_free_standard_gpu(inputs,cloud,monkeypatch,used):
    original=module.admission.query_standard_capacity
    def capacity(*args):
        result=original();result['used_gpus']=used
        result['active_reserved']=[] if used==0 else [dict(name='other-job',uid='other-uid',state='RUNNING',gpu_count=used)]
        return result
    monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    if used<4:assert submit(inputs)['state']=='submitted'
    else:
        with pytest.raises(ValueError,match='free standard GPU'):submit(inputs)
        assert len(cloud[1])==0
        assert cloud[0].puts==0 and not inputs[0][3].exists()
        assert not (module.INTENTS/'panel.json').exists() and not list(module.INTENTS.glob('intent-*.json'))


@pytest.mark.parametrize('fault',['full','stale','query_error'])
def test_final_capacity_conflict_is_durable_not_submitted_and_never_retried(inputs,cloud,monkeypatch,fault):
    original=module.admission.query_standard_capacity;queries=[]
    def capacity(*args):
        queries.append(True);value=original()
        if len(queries)==2:
            if fault=='query_error':raise RuntimeError('query unavailable')
            if fault=='full':value.update(used_gpus=4,active_reserved=[dict(name='other-job',uid='other-uid',state='RUNNING',gpu_count=4)])
            if fault=='stale':value['observed_at']=(module.common.now()-timedelta(seconds=70)).isoformat()
        return value
    monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    with pytest.raises(ValueError,match='no submission attempted'):submit(inputs)
    record=json.loads((module.INTENTS/'intent-radio-pilot.json').read_text())
    assert record['state']=='not_submitted' and record['submission_attempted'] is False
    assert record['not_submitted_stage']=='final_capacity_check' and record['capacity_preflight']['used_gpus']==0
    assert json.loads(inputs[0][3].read_text())==record
    assert len(cloud[1])==0 and cloud[0].puts==1
    inputs[0][3]=inputs[0][3].with_name('retry.json')
    with pytest.raises(ValueError,match='already has an intent'):submit(inputs,suffix='new-name')
    with pytest.raises(ValueError,match='unresolved'):submit(inputs,'a000','next')
    assert len(cloud[1])==0 and len(queries)==2


@pytest.mark.parametrize('fault',['stale','incomplete','owner','readback'])
def test_admission_and_source_checks_fail_closed(inputs,cloud,monkeypatch,fault):
    original=module.admission.query_standard_capacity
    def capacity(*args):
        result=original()
        if fault=='stale':result['observed_at']=(module.common.now()-timedelta(seconds=70)).isoformat()
        elif fault=='incomplete':result['queried_states']=['RUNNING']
        elif fault=='owner':result['owner']='other'
        return result
    monkeypatch.setattr(module.admission,'query_standard_capacity',capacity)
    if fault=='readback':cloud[0].corrupt=True
    with pytest.raises(ValueError):submit(inputs)
    assert cloud[1]==[]


def test_bootstrap_real_file_hash_and_finite_signal(tmp_path):
    source=b"from pathlib import Path\nimport hashlib\ndef run(payload):\n assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==payload['driver_sha256']\n return {'state':'complete','external_agent':False}\n"
    digest=hashlib.sha256(source).hexdigest();root=tmp_path/'bootstrap'
    config=dict(bootstrap_root=str(root),source_inline=base64.b64encode(zlib.compress(source)).decode(),
                source_sha256=digest,payload={'driver_sha256':digest},report_put='unused')
    code=('CONFIG='+repr(config)+'\n'+module.BOOTSTRAP).replace('try:upload()','try:pass')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
    assert result.returncode==0 and (root/'evaluation_worker.py').read_bytes()==source
    assert json.loads((root/'report.json').read_text())['external_agent'] is False
    assert 'signal.alarm(13800)' in code and 'signal.signal(signal.SIGTERM,expire)' in code
    assert (root.stat().st_mode & 0o777)==0o700


def test_bootstrap_deadline_remains_failure(tmp_path):
    source=b"import signal\ndef run(payload):\n signal.raise_signal(signal.SIGALRM)\n return {'state':'complete','external_agent':False}\n"
    config=dict(bootstrap_root=str(tmp_path/'bootstrap'),source_inline=base64.b64encode(zlib.compress(source)).decode(),
                source_sha256=hashlib.sha256(source).hexdigest(),payload={},report_put='unused')
    code=('CONFIG='+repr(config)+'\n'+module.BOOTSTRAP).replace('try:upload()','try:pass')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=10)
    report=json.loads((tmp_path/'bootstrap/report.json').read_text())
    assert result.returncode==1 and report['state']=='bootstrap_failed' and report['error_type']=='EvalDeadline'

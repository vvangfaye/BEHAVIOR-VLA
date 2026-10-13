"""Batch admission, crash recovery and unchanged per-episode artifact boundaries."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_video_eval_queue import setup, accept_pilot, FakeAdapter
from video_eval_queue import Controller, pending_gpu_count, accepted_execution
from test_video_eval_worker import payload, result

SCO = Path(__file__).parent
BASE = SCO / '../eval_results/sco_submission1000_2000000_20261013/frozen/video_eval_worker.py'

def namespace():
    ns={'__name__':'batch_test'}
    raw=BASE.read_bytes()+b'\n'+(SCO/'video_eval_batch_worker.py').read_bytes()
    exec(compile(raw, 'batch_test', 'exec'),ns)
    ns['_EXECUTED_WORKER_SOURCE']=raw
    return ns

class BatchAdapter(FakeAdapter):
    batch_worker_sha256='f'*64
    def __init__(self):
        super().__init__();self.groups=[];self.fail=False
    def create_group(self, records):
        self.groups.append([r['index'] for r in records])
        for r in records:
            self.jobs[r['name']]=dict(uid=records[0]['batch_id'],name=r['name'],state=self.create_state)
        if self.fail:raise TimeoutError('unknown create outcome')
        return dict(uid=records[0]['batch_id'],state=self.create_state)


def batch_setup(setup):
    root,_,_=setup
    adapter=BatchAdapter();ctl=Controller(root,adapter)
    accept_pilot(adapter,ctl)
    (root/'batch_policy.json').write_text(json.dumps(dict(enabled=True,pilot_indices=[1,2],
        max_episodes_per_worker=5,plan_fingerprint=ctl.fingerprint,batch_worker_sha256='f'*64)))
    return root,adapter,ctl


def test_batch_pilot_gate_then_five_workers_each_at_most_five_episodes(setup):
    root,a,c=batch_setup(setup)
    first=c.run_once()
    assert first['submitted_this_pass']==[1,2]
    assert first['running_gpus']==1 and first['submitted_workers']==2
    assert c.run_once()['submitted_this_pass']==[]
    for i in [1,2]:
        a.jobs[c._name(i)]['state']='SUCCEEDED'
        a.inspections[i]=dict(verified=True,batch_completed_original=True)
    s=c.run_once()
    assert len(a.groups)==6 and all(2<=len(g)<=5 for g in a.groups)
    assert all(len({c.episodes[i]['task'] for i in g})==1 for g in a.groups)
    assert s['reserved_this_pass']==5 and s['running_gpus']==5
    assert len(s['submitted_this_pass'])==22  # task0 remainder7 then 3 complete5s
    assert len(set(sum(a.groups,[])))==sum(map(len,a.groups))


def test_unknown_batch_is_reconciled_once_and_reserves_one_card(setup):
    root,a,c=batch_setup(setup);a.fail=True
    s=c.run_once()
    assert s['query_complete'] is False and s['pending_gpus']==1
    # Simulate crash after durable group intent, before one individual record write.
    (root/'submissions/0002.json').unlink()
    s=Controller(root,a).run_once()
    assert len(a.groups)==1 and (root/'submissions/0002.json').exists()
    assert s['submitted_this_pass']==[]
    assert s['running_gpus']==1


def test_mixed_batch_state_counts_conservatively_and_complete_child_protected():
    rows=[dict(name='a',physical_name='group',state='RUNNING'),dict(name='b',physical_name='group',state='UNKNOWN')]
    assert pending_gpu_count(rows)==1
    assert accepted_execution(dict(batch_id='group',state='FAILED',inspection=dict(verified=True,batch_completed_original=True)))
    assert not accepted_execution(dict(batch_id='group',state='RUNNING',inspection=dict(verified=True,batch_completed_original=True)))


def test_hook_same_evaluator_independent_resets_and_monotonic_watchdog(tmp_path):
    ns=namespace();hook=tmp_path/'hook';hook.mkdir()
    (hook/'sitecustomize.py').write_text(ns['BATCH_PROGRESS_HOOK'])
    pkg=tmp_path/'omnigibson/eval';pkg.mkdir(parents=True)
    (pkg/'evaluator.py').write_text('''class Evaluator:
 def reset(self, **kwargs): self._rollout_steps=0
 def load_task_instance(self, instance_id): self.instance=instance_id
 def step(self, done):
  self._rollout_steps+=1
  return (done,False)
''')
    runner=tmp_path/'eval_b1k_queue.py'
    runner.write_text('''from omnigibson.eval.evaluator import Evaluator
import json, os
from pathlib import Path
p=Path(os.environ['VIDEO_EVAL_PROGRESS']);e=Evaluator();totals=[]
for instance in [301,302]:
 e.reset(prepare_for_task_instance=True);e.load_task_instance(instance);e.reset()
 e.step(False);e.step(True)
 totals.append(json.loads(p.read_text())['steps'])
assert totals==[2,4],totals
''')
    entries={str(i):dict(run_id='e'+str(i),path=str(tmp_path/('%d.json'%i))) for i in [301,302]}
    env=dict(os.environ,PYTHONPATH=str(hook)+os.pathsep+str(tmp_path),VIDEO_EVAL_PROGRESS=str(tmp_path/'all.json'),VIDEO_EVAL_RUN_ID='group',VIDEO_EVAL_EPISODES=json.dumps(entries))
    subprocess.run([sys.executable,str(runner)],env=env,check=True)
    events=json.loads((tmp_path/'reuse_evidence.json').read_text())
    assert len({e['pid'] for e in events})==1
    for i in [301,302]:
        assert json.loads((tmp_path/('%d.json'%i)).read_text())==dict(state='progress',steps=2,terminal=True,run_id='e'+str(i),pid=events[0]['pid'],observed_at=json.loads((tmp_path/('%d.json'%i)).read_text())['observed_at'],instance_id=i)
        assert any(e['event']=='reset' and e['instance_id']==i and not e['prepare_for_task_instance'] for e in events)


def test_batch_validation_only_same_task_new_production(tmp_path):
    ns=namespace();children=[]
    for i in [301,302]:
        child=payload(tmp_path);child.update(pilot=False,instance_id=i,run_id='e'+str(i),assignment_root=str(tmp_path/'episodes'/str(i)))
        children.append(child)
    p=dict(episodes=children,batch_name='group',assignment_root=str(tmp_path/'batches/group'))
    assert ns['validate_batch'](p)==children
    for changes in [dict(task='another'),dict(instance_id=301),dict(pilot=True),dict(recovery_chain=[{}]),dict(assignment_root='/elsewhere')]:
        bad=copy.deepcopy(p);bad['episodes'][1].update(changes)
        with pytest.raises(ValueError):ns['validate_batch'](bad)


def test_finalize_one_child_real_queue_unchanged_bytes(tmp_path,monkeypatch):
    sys.path.insert(0,str(SCO.parent/'scripts'))
    from pi05_eval_queue import EvaluationQueue
    ns=namespace();group=tmp_path/'group';group.mkdir();root=tmp_path/'child';root.mkdir()
    p=payload(tmp_path);p.update(pilot=False,assignment_root=str(root),batch_context=dict(batch_name='test',root=str(group)))
    code=Path(p['code']);code.mkdir();Path(p['checkpoint']).mkdir()
    (Path(p['checkpoint'])/'_READY').write_text('{}');(code/'_PUBLIC_TEST_SOURCE_READY').write_text('{}')
    for rel in ns['REQUIRED_SOURCE_FILES']:
        f=code/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('#source')
    q=EvaluationQueue(group/'queue.sqlite3');q.initialize([(p['task'],301),(p['task'],302)],max_attempts=1,metadata={})
    # Exercise genuine queue claim -> validated originals -> completed provenance.
    lease=q.claim(p['task'],'gpu0',group/'attempts')
    attempt=lease.attempt_dir
    value=result();value.update(steps=123)
    for d in [attempt,group/'results/pi05'/p['task']]:
        (d/'json').mkdir(parents=True);(d/'videos').mkdir(parents=True)
        (d/'json'/f'{p["task"]}_301_0.json').write_text(json.dumps(value))
        (d/'videos'/f'{p["task"]}_301_0.mp4').write_bytes(b'original-video')
    # complete() uses lease-bound attempt path; register it through the public API.
    q.publish(lease,group/'results/pi05'/p['task'],num_rollouts=1,require_videos=True)
    (group/'logs').mkdir();(group/'logs/server.log').write_text('server log')
    (group/'driver.log').write_text('policy server pid=123 task=x')
    (group/'reuse_evidence.json').write_text(json.dumps([dict(event='load',instance_id=301,pid=42),dict(event='reset',instance_id=301,pid=42)]))
    (group/'batch_worker.py').write_bytes(ns['_EXECUTED_WORKER_SOURCE'])
    (root/'native_progress.json').write_text(json.dumps(dict(run_id=p['run_id'],steps=123,terminal=True)))
    (root/'payload.json').write_text(json.dumps(ns['validate_payload'](p)))
    hook=tmp_path/'hook';hook.mkdir();(hook/'sitecustomize.py').write_text(ns['BATCH_PROGRESS_HOOK'])
    # Codec and external storage are separate tested boundaries; retain real queue and SHA packaging here.
    ns['subprocess']=SimpleNamespace(check_output=lambda *a,**k:json.dumps(dict(decoded_frames=123,advertised_frames=123,frames_equal_simulator_steps=True,full_sequential_decode=True)))
    uploaded=[];ns['put_file']=lambda path,url:uploaded.append(path)
    ns['verify_remote_archive']=lambda *a:True
    receipt=ns['_batch_finalize'](p,group,{}, {},hook,q,q.snapshot(readonly=True),['5090'],{})
    assert receipt['state']=='complete' and receipt['verified']
    assert q.snapshot(readonly=True)['jobs'][1]['state']=='pending'
    assert (root/'bundle/videos'/f'{p["task"]}_301_0.mp4').read_bytes()==b'original-video'
    assert len(uploaded)==2 and receipt['batch_reuse']['single_evaluator_process']
    # A later failure cannot overwrite this child.
    before=(root/'receipt.json').read_bytes();ns['_batch_failure'](p,RuntimeError('later sibling failed'))
    assert (root/'receipt.json').read_bytes()==before


def test_recovery_probe_sees_only_child_attempt_and_protects_shared_completed_json(tmp_path,monkeypatch):
    import urllib.request
    from video_eval_recovery_probe import run
    from pi05_eval_queue import EvaluationQueue
    root=tmp_path/'episodes/1';root.mkdir(parents=True)
    group=tmp_path/'batches/group';group.mkdir(parents=True)
    p=payload(tmp_path);p.update(assignment_root=str(root),old_writer_stopped=True,old_state='FAILED',old_uid='uid',old_name='group',afs_output_root=str(tmp_path),report_put='https://example.test')
    safe=dict(p,batch_context=dict(root=str(group),batch_name='group'))
    (root/'payload.json').write_text(json.dumps(safe))
    q=EvaluationQueue(group/'queue.sqlite3');q.initialize([(p['task'],301),(p['task'],302)],max_attempts=1)
    q.claim(p['task'],'worker',group/'attempts',instance_id=301)
    q.claim(p['task'],'worker',group/'attempts',instance_id=302)
    d=group/'results';d.mkdir();(d/(p['task']+'_301_0.json')).write_text(json.dumps(dict(terminated=True)))
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return b''
    monkeypatch.setattr(urllib.request,'build_opener',lambda *a:SimpleNamespace(open=lambda *a,**kw:Response()))
    report=run(p)
    assert not report['eligible'] and report['attempt_count']==1
    assert report['protected_completed_outputs']
    assert report['attempts'][0]['job_id']==1


def test_final_provenance_checks_batch_identity_and_requires_raw_evidence(setup):
    from video_eval_finalize import validate_batch_provenance
    root,a,c=batch_setup(setup);c.run_once()
    plan=dict(c.plan,afs_output_root='/mnt/run')
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    policy=dict(plan_fingerprint=fingerprint,batch_worker_sha256='f'*64)
    name=plan['run_id']+'-b0001-0002'
    launch=dict(batch_indices=[1,2],physical_name=name,batch_id=name,batch_root='/mnt/run/batches/'+name,batch_worker_sha256='f'*64)
    report=dict(batch_context=dict(batch_name=name,root=launch['batch_root'],batch_worker_sha256='f'*64,instance_ids=[302,303]),
        batch_reuse=dict(single_model_server=True,single_evaluator_process=True,server_pid=1,evaluator_pid=2,reset_instance_ids=[302,303],loaded_instance_ids=[302,303]),
        files=[dict(path='code/batch_worker.py',sha256='f'*64),dict(path='metadata/batch_reuse.json'),dict(path='metadata/batch_queue_snapshot.json')])
    validate_batch_provenance(plan,c.episodes[1],report,launch,policy)
    for field,changes in [('batch_reuse',dict(reset_instance_ids=[303])),('batch_context',dict(instance_ids=[303,302]))]:
        bad=copy.deepcopy(report);bad[field].update(changes)
        with pytest.raises(ValueError):validate_batch_provenance(plan,c.episodes[1],bad,launch,policy)
    bad=copy.deepcopy(report);bad['files']=[]
    with pytest.raises(ValueError):validate_batch_provenance(plan,c.episodes[1],bad,launch,policy)


def test_real_adapter_creates_one_physical_worker_with_two_separate_artifact_destinations(tmp_path,monkeypatch):
    import video_eval_adapter as m
    a=object.__new__(m.Adapter);a.root=tmp_path;a.prefix='isolated/run';a.driver_sha='d'*64
    a.batch_source=b'# frozen batch';a.batch_worker_sha256=hashlib.sha256(a.batch_source).hexdigest()
    a.plan=dict(afs_output_root='/mnt/run')
    a.ready=dict(checkpoint='/mnt/checkpoint',code='/mnt/code',runtime='/mnt/runtime',source_manifest_sha256='a'*64,checkpoint_ready_sha256='b'*64)
    (tmp_path/'launches').mkdir()
    bodies=[];a.client=SimpleNamespace(put_object=lambda **kw:bodies.append(kw))
    a.signed=lambda op,key:'https://example.test/'+key
    seen=[]
    monkeypatch.setattr(m,'environment',lambda:{})
    monkeypatch.setattr(m.subprocess,'run',lambda args,**kw:(seen.append(args) or SimpleNamespace(returncode=0)))
    def lookup(args):
        if m.TEMPLATE in args:return dict(uid=m.TEMPLATE_UID,ownership={'user_name':m.BUCKET},resource_pool={'name':m.POOL})
        return dict(uid='one-uid',name='batch-name',state='STARTING',ownership={'user_name':m.BUCKET},resource_pool={'name':m.POOL})
    monkeypatch.setattr(m,'cli_json',lookup)
    records=[dict(index=i,task='task',instance_id=301+i,name='episode'+str(i),batch_id='batch-name',physical_name='batch-name') for i in [1,2]]
    reply=a.create_group(records)
    assert reply['uid']=='one-uid' and len(seen)==1
    assert '--worker-nodes=1' in seen[0] and '--quota-type=spot' in seen[0]
    assert '--worker-spec=n12lp.nn.i10a.1' in seen[0]
    body=json.loads(bodies[0]['Body'])
    assert [c['instance_id'] for c in body['episodes']]==[302,303]
    assert body['episodes'][0]['artifact_puts']['archive']!=body['episodes'][1]['artifact_puts']['archive']
    for i in [1,2]:
        launch=json.loads((tmp_path/'launches'/('%04d.json'%i)).read_text())
        assert launch['physical_name']=='batch-name' and launch['batch_worker_sha256']==a.batch_worker_sha256


def test_solo_recovery_of_batch_pilot_cannot_establish_reuse_gate(setup):
    root,a,c=batch_setup(setup);c.run_once()
    for i in [1,2]:
        a.jobs[c._name(i)]['state']='SUCCEEDED'
        a.inspections[i]=dict(verified=True,batch_completed_original=True)
    # A recovered single episode proves its result, not reuse across two episodes.
    p=root/'submissions/0002.json';r=json.loads(p.read_text());r.pop('batch_id');p.write_text(json.dumps(r))
    assert c.run_once()['submitted_this_pass']==[]


def test_user_waiver_continues_five_episode_groups_without_recreating_live_pilot(setup):
    root,a,c=batch_setup(setup);c.run_once()
    p=root/'batch_policy.json';policy=json.loads(p.read_text())
    policy.update(require_batch_pilot=False,pilot_gate_waiver=dict(authorized_by_user=True,instruction='No additional batch pilot'))
    p.write_text(json.dumps(policy))
    status=c.run_once()
    assert status['reserved_this_pass']==5
    assert a.groups[1]==[3,4,5,6,7]
    assert all(i not in [1,2] for i in status['submitted_this_pass'])
    assert a.groups.count([1,2])==1
    assert len(a.groups)==6


def test_batch_pilot_not_silently_disabled_without_user_waiver(setup):
    root,a,c=batch_setup(setup);c.run_once()
    p=root/'batch_policy.json';policy=json.loads(p.read_text());policy['require_batch_pilot']=False
    p.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError,match='explicit user authorization'):c.run_once()
    assert len(a.groups)==1


def test_ten_episode_admission_keeps_five_physical_pending_budget(setup):
    root,a,c=batch_setup(setup);c.run_once()
    p=root/'batch_policy.json';policy=json.loads(p.read_text())
    policy.update(max_episodes_per_worker=10,require_batch_pilot=False,pilot_gate_waiver=dict(authorized_by_user=True,instruction='10 per worker'))
    p.write_text(json.dumps(policy));a.create_state='QUEUEING'
    s=c.run_once()
    assert s['pending_gpus']==5 and s['reserved_this_pass']==5
    assert [len(g) for g in a.groups[1:]]==[7,10,10,10,10]
    assert len(s['submitted_this_pass'])==47
    assert len(set(sum(a.groups,[])))==49
    assert c.run_once()['submitted_this_pass']==[]


def test_worker_ten_unique_same_task_episodes_and_rejects_eleventh(tmp_path):
    ns=namespace();children=[]
    for i in range(301,311):
        child=payload(tmp_path);child.update(pilot=False,instance_id=i,run_id='e'+str(i),assignment_root=str(tmp_path/'episodes'/str(i)))
        children.append(child)
    p=dict(episodes=children,batch_name='group',assignment_root=str(tmp_path/'batches/group'))
    assert len(ns['validate_batch'](p))==10
    with pytest.raises(ValueError):ns['validate_batch'](dict(p,episodes=children+[children[0]]))


def test_version_registry_keeps_old_five_and_new_ten_provenance_separate(setup):
    from video_eval_finalize import batch_version,validate_batch_provenance
    root,a,c=batch_setup(setup)
    plan=dict(c.plan,afs_output_root='/mnt/run')
    fp=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    old,new='e'*64,'f'*64
    policy=dict(plan_fingerprint=fp,batch_worker_sha256=new,max_episodes_per_worker=10,worker_versions={
        old:dict(source_file='frozen/old.py',max_episodes=5),new:dict(source_file='frozen/ten.py',max_episodes=10)})
    def pair(ids,digest):
        name=plan['run_id']+'-b%04d-%04d'%(min(ids),max(ids))
        launch=dict(batch_indices=ids,physical_name=name,batch_id=name,batch_root='/mnt/run/batches/'+name,batch_worker_sha256=digest)
        instances=[c.episodes[i]['instance_id'] for i in ids]
        report=dict(batch_context=dict(batch_name=name,root=launch['batch_root'],batch_worker_sha256=digest,instance_ids=instances),
            batch_reuse=dict(single_model_server=True,single_evaluator_process=True,server_pid=1,evaluator_pid=2,reset_instance_ids=instances,loaded_instance_ids=instances),
            files=[dict(path='code/batch_worker.py',sha256=digest),dict(path='metadata/batch_reuse.json'),dict(path='metadata/batch_queue_snapshot.json')])
        return launch,report
    for ids,digest in [(list(range(1,6)),old),(list(range(10,20)),new)]:
        launch,report=pair(ids,digest);validate_batch_provenance(plan,c.episodes[ids[0]],report,launch,policy)
    launch,report=pair(list(range(1,7)),old)
    with pytest.raises(ValueError):validate_batch_provenance(plan,c.episodes[1],report,launch,policy)
    with pytest.raises(ValueError):batch_version(policy,dict(batch_worker_sha256='a'*64))


def test_adapter_loads_old_and_new_frozen_sources(tmp_path,monkeypatch):
    import video_eval_adapter as m
    root=tmp_path;frozen=root/'frozen';frozen.mkdir();base=b'#base\n';old=b'#five\n';new=b'#ten\n'
    (frozen/'video_eval_worker.py').write_bytes(base);(frozen/'old.py').write_bytes(old);(frozen/'ten.py').write_bytes(new)
    sha=lambda b:hashlib.sha256(b).hexdigest();old_hash,new_hash=sha(base+b'\n'+old),sha(base+b'\n'+new)
    ready=dict(state='ready',checkpoint_step=2000000,source_manifest_sha256='a'*64,checkpoint_ready_sha256='b'*64,worker_sha256=sha(base))
    (root/'preparation_ready.json').write_text(json.dumps(ready))
    (root/'batch_policy.json').write_text(json.dumps(dict(batch_worker_sha256=new_hash,worker_versions={old_hash:dict(source_file='frozen/old.py',max_episodes=5),new_hash:dict(source_file='frozen/ten.py',max_episodes=10)})))
    monkeypatch.setattr(m,'aoss_client',lambda:object())
    adapter=m.Adapter(root,dict(object_prefix='x',immutable_inputs={k:ready[k] for k in ['source_manifest_sha256','checkpoint_ready_sha256','worker_sha256']}))
    assert adapter.max_batch_episodes==10 and adapter.batch_worker_sha256==new_hash
    assert adapter.batch_source_versions[old_hash]==base+b'\n'+old
    (frozen/'old.py').write_bytes(b'changed old')
    with pytest.raises(AssertionError):m.Adapter(root,adapter.plan)


def test_ten_pending_workers_still_run_only_five_episodes_each(setup):
    root,a,c=batch_setup(setup);c.run_once()
    p=root/'batch_policy.json';policy=json.loads(p.read_text())
    policy.update(max_pending_gpus=10,pending_limit_user_authorization='10 pending workers,5 episodes each',require_batch_pilot=False,
        pilot_gate_waiver=dict(authorized_by_user=True,instruction='No additional pilot'))
    p.write_text(json.dumps(policy));a.create_state='QUEUEING'
    c=Controller(root,a);s=c.run_once()
    assert s['max_pending_gpus']==10 and s['pending_gpus']==10
    assert s['reserved_this_pass']==10
    assert all(1<=len(g)<=5 for g in a.groups)
    assert s['running_gpus']==1  # original live two-episode worker, not two GPU requests
    assert c.run_once()['submitted_this_pass']==[]


def test_changed_pending_budget_needs_persisted_user_instruction(setup):
    root,a,c=batch_setup(setup)
    p=root/'batch_policy.json';policy=json.loads(p.read_text());policy['max_pending_gpus']=10;p.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match='user authorization'):Controller(root,a)

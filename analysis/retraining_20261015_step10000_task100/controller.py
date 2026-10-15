"""Finite 100-task regression: reuse verified20 and baseline100, submit only80."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import submit_helpers as helper

ROOT=Path(__file__).resolve().parent
TERMINAL=helper.TERMINAL

def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
save=helper.save

def read_optional(client,key):
    try:
        with client.get_object(Bucket='260010165',Key=key)['Body'] as body:return json.loads(body.read(16*1024**2))
    except client.exceptions.ClientError as error:
        if error.response['ResponseMetadata']['HTTPStatusCode']==404:return None
        raise

def reconcile(client,plan):
    def one(path):
        intent=load(path);id=intent['id'];report=helper.read_report(client,plan,id)
        if report:
            assert report['worker_bundle_sha256']==intent['worker_bundle_sha256']
            save(ROOT/'reports'/(id+'.json'),report)
        prior=ROOT/'observations'/(id+'.json')
        if prior.exists() and load(prior)['state'] in TERMINAL:observation=load(prior)
        else:
            observation=helper.observe(intent);save(prior,observation)
        if not intent.get('uid'):
            intent.update(uid=observation['uid'],state=observation['state']);save(path,intent)
        if report and report.get('state')=='accepted' and intent['gpu']:
            assert report['checkpoint_manifest_sha256']==plan['candidate_manifest_sha256']
            assert report['video_validation']['full_sequential_decode']
            record=report['record'];assert record['policy_seed_acknowledged'] and record['artifact_accepted']
            assert [record['task'],record['instance_id']] in plan['pairs']
            assert all(r['full_destination_readback'] for r in report['object_artifacts'].values())
        return id,dict(intent=intent,observation=observation,report=report)
    paths=list((ROOT/'intents').glob('*.json'))
    with ThreadPoolExecutor(max_workers=8) as pool:return dict(pool.map(one,paths))

def counts(rows):
    gpu=[r for r in rows.values() if r['intent']['gpu']]
    pending=sum(r['observation']['state'] not in TERMINAL|{'RUNNING'} for r in gpu)
    active=sum(r['observation']['state'] not in TERMINAL for r in gpu)
    running=sum(r['observation']['state']=='RUNNING' for r in gpu)
    return pending,active,running

def candidates(rows):
    result={}
    preflight=rows.get('preflight',{}).get('report')
    if preflight and preflight['state']=='accepted':
        for row in preflight['reused_candidate_results']:
            r=row['record'];result[(r['task'],r['instance_id'])]=dict(record=r,reused=True,evidence=row)
    for id,row in rows.items():
        report=row['report']
        if row['intent']['gpu'] and report and report['state']=='accepted':
            r=report['record'];key=(r['task'],r['instance_id'])
            assert key not in result,'Repeated candidate result'
            result[key]=dict(record=r,reused=False,report_key=row['intent']['report_key'],uid=row['intent']['uid'],evidence=report)
    return result

def summarize(plan,rows):
    records=candidates(rows);baseline=load(ROOT/'baseline_accepted.json')['records']
    bykey={(r['task'],r['instance_id']):r for r in baseline}
    pairs=[]
    for task,instance in plan['pairs']:
        key=(task,instance)
        if key not in records:continue
        before=bykey[key];after=records[key]['record']
        pairs.append(dict(task=task,instance_id=instance,baseline_success=before['model_success'],
            candidate_success=after['model_success'],baseline_q=before['q_score'],candidate_q=after['q_score'],
            candidate_reused=records[key]['reused'],candidate_result_sha256=after['result_sha256']))
    n=len(pairs);full=n==100
    result=dict(kind='task_balanced_public301_training_seen_regression',complete=full,expected_pairs=100,completed_pairs=n,
        baseline_checkpoint=1688977,candidate_completed_updates=10000,training_seen=True,heldout_generalization_claim=False,
        baseline_full100=dict(successes=sum(r['model_success'] for r in baseline),success_rate=sum(r['model_success'] for r in baseline)/100,
                              mean_q=sum(r['q_score'] for r in baseline)/100),
        candidate_reused=sum(r['candidate_reused'] for r in pairs),new_candidate_accepted=sum(not r['candidate_reused'] for r in pairs),
        completed_pair_metrics=dict(baseline_success_rate=sum(r['baseline_success'] for r in pairs)/n if n else None,
            candidate_successes=sum(r['candidate_success'] for r in pairs),candidate_success_rate=sum(r['candidate_success'] for r in pairs)/n if n else None,
            baseline_mean_q=sum(r['baseline_q'] for r in pairs)/n if n else None,
            candidate_mean_q=sum(r['candidate_q'] for r in pairs)/n if n else None),
        per_instance=pairs,observed_at=time.time(),protocol_sha256=plan['protocol_sha256'])
    save(ROOT/('comparison.json' if full else 'partial_comparison.json'),result)
    if full:
        save(ROOT/'candidate_accepted.json',dict(completed_updates=10000,records=[records[tuple(p)]['record'] for p in plan['pairs']],
            protocol_sha256=plan['protocol_sha256'],checkpoint_manifest_sha256=plan['candidate_manifest_sha256']))
        index=[]
        for key,row in records.items():
            index.append(dict(task=key[0],instance_id=key[1],reused=row['reused'],objects=row['evidence']['object_artifacts']))
        save(ROOT/'video_and_json_index.json',dict(checkpoint_step=10000,episodes=index))
    return result

def tick(client,plan):
    rows=reconcile(client,plan)
    if 'preflight' not in rows:
        helper.submit(client,plan,'preflight','preflight')
        return False
    pending,active,running=counts(rows)
    gate=read_optional(client,plan['object_prefix']+'/checkpoint_gate.json')
    ready=False
    if gate:
        assert gate['state']=='verified' and gate['completed_updates']==10000
        assert gate['protocol_sha256']==plan['protocol_sha256']
        assert gate['worker_bundle_sha256']==rows['preflight']['intent']['worker_bundle_sha256']
        assert gate['checkpoint_manifest_sha256']==plan['candidate_manifest_sha256']
        assert gate['checkpoint_full_sha256_readback']
        save(ROOT/'checkpoint_verified.json',gate);ready=True
    preflight=rows['preflight']['report']
    status=dict(observed_at=time.time(),state='evaluating' if ready else 'checkpoint_preflight',
        expected_episodes=100,baseline_reused=100,created_new_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values()),
        pending_gpu_jobs=pending,active_gpu_jobs=active,running_gpu_jobs=running,checkpoint_verified=ready,
        reused_candidate_audit_complete=bool(preflight and preflight['state']=='accepted'),
        failures=[id for id,r in rows.items() if r['report'] and r['report']['state']=='failed'])
    save(ROOT/'status.json',status)
    summary=summarize(plan,rows);status['accepted_episodes']=summary['completed_pairs']
    status['reused_candidate_accepted']=summary['candidate_reused'];status['new_candidate_accepted']=summary['new_candidate_accepted']
    save(ROOT/'status.json',status)
    if summary['complete']:
        save(ROOT/'status.json',dict(status,state='comparison_complete'));return True
    if preflight and preflight['state']!='accepted':
        save(ROOT/'status.json',dict(status,state='preflight_needs_diagnostic'));return True
    if (ROOT/'STOP').exists() or time.time()>=plan['deadline_epoch']:
        save(ROOT/'status.json',dict(status,state='admission_stopped'));return True
    if not ready:return False
    reused={tuple(item['report']['record'][k] for k in ['task','instance_id']) for item in plan['reuse_candidates']}
    expected=[]
    for index,pair in enumerate(plan['pairs']):
        if tuple(pair) in reused:continue
        id=f'candidate-{index:03d}';expected.append(id)
        if id in rows:continue
        pending,active,running=counts(rows)
        if pending>=plan['max_pending'] or active>=plan['max_active_gpu_jobs']:break
        helper.submit(client,plan,id,'evaluate',role='candidate',pair=pair)
        rows=reconcile(client,plan) # Includes each just-created UID before assigning another slot.
    pending,active,running=counts(rows)
    status.update(observed_at=time.time(),created_new_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values()),
        pending_gpu_jobs=pending,active_gpu_jobs=active,running_gpu_jobs=running)
    save(ROOT/'status.json',status)
    all_ids=[f'candidate-{i:03d}' for i,p in enumerate(plan['pairs']) if tuple(p) not in reused]
    if all(id in rows and rows[id]['observation']['state'] in TERMINAL for id in all_ids):
        path=ROOT/'terminal_grace.json'
        if not path.exists():save(path,dict(started_at=time.time()))
        if time.time()-load(path)['started_at']>180:
            save(ROOT/'status.json',dict(status,state='finished_with_missing_or_failed_artifacts'));return True
    return False

def main():
    for n in ['private','intents','observations','reports']:(ROOT/n).mkdir(exist_ok=True)
    (ROOT/'private').chmod(0o700)
    with (ROOT/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=(ROOT/'plan.json').read_bytes();plan=json.loads(raw)
        assert len(plan['pairs'])==len({tuple(p) for p in plan['pairs']})==100
        assert len({p[0] for p in plan['pairs']})==100 and all(p[1]==301 for p in plan['pairs'])
        assert plan['candidate_step']==10000 and plan['new_candidate_rollouts']==80 and len(plan['reuse_candidates'])==20
        assert plan['maximum_attempts_per_new_episode']==1 and plan['max_pending']==5
        save(ROOT/'controller_launch.json',dict(pid=os.getpid(),started_at=time.time(),controller_sha256=sha(__file__),
            helper_sha256=sha(ROOT/'submit_helpers.py'),plan_sha256=hashlib.sha256(raw).hexdigest(),recurring_automation=False))
        client=helper.aoss_client()
        while time.time()<plan['deadline_epoch']:
            assert (ROOT/'plan.json').read_bytes()==raw
            try:
                if tick(client,plan):return
            except Exception as error:
                message=re.sub(r'https?://[^\s]+','<URL>',str(error))
                save(ROOT/'last_error.json',dict(observed_at=time.time(),error_type=type(error).__name__,message=message[:700]))
            time.sleep(plan['poll_seconds'])

if __name__=='__main__':main()

"""Bounded infrastructure retries and conservative measured GPU accounting."""
from datetime import datetime,timezone
import json

TERMINAL={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}


def attempt_cap(plan,phase,root):
    path=root/'auto_recovery_policy.json'
    if path.is_file():
        policy=json.loads(path.read_text())
        if phase in policy['eligible_phases']:
            return int(policy['max_attempts_per_shard'])
    return plan.get('attempt_caps',{}).get(phase,plan['max_attempts_per_phase'])


def gpu_seconds_upper(job,receipt,now=None):
    """Terminal update time bounds a stopped job even without complete_time."""
    limit=receipt['max_seconds']*receipt.get('gpu_count',1)
    def parsed(x):return datetime.fromisoformat(x.replace('Z','+00:00'))
    start=job.get('start_time')
    if job['state'] not in TERMINAL:
        if not start:return float(limit)
        now=now or datetime.now(timezone.utc)
        return max(float(limit),(now-parsed(start)).total_seconds()*receipt.get('gpu_count',1))
    if not start:return 0.0
    end=job.get('complete_time') or job.get('update_time')
    if not end:return float(limit)
    seconds=(parsed(end)-parsed(start)).total_seconds()
    if seconds<0:raise ValueError('Terminal timestamp predates GPU start')
    return seconds*receipt.get('gpu_count',1)


def other_rl_jobs(root,describe):
    """Account for the already-started task suite while modem gets priority."""
    campaign=root.parent/'multitask_validation_20261029'
    result=[]
    for slug in ('trash','water'):
        for path in (campaign/slug/'jobs').glob('*/submission.json'):
            r=json.loads(path.read_text())
            d=describe(r['job'])
            assert d['uid']==r['uid'] and d['ownership']['user_name']=='260010165'
            if d['state'] not in TERMINAL:result.append({'job':r['job'],'uid':r['uid'],'state':d['state']})
    return result

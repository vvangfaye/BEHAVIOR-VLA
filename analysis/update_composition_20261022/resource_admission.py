"""Count this user's requested standard GPUs before a new reserved role."""
from datetime import datetime,timezone
import json
import subprocess
import time

from submit import cli,write


def read_jobs(*args):
    for attempt in range(3):
        try:
            response=cli(*args)
            if response.returncode==0:return response
            detail=f'CLI return code {response.returncode}'
        except (OSError,subprocess.TimeoutExpired) as error:
            detail=type(error).__name__
        if attempt<2:time.sleep(2**attempt)
    raise RuntimeError('Cannot verify complete standard-resource occupancy: '+detail)


def check_standard_capacity(output,additional=1,limit=4):
    active={}
    states=('RUNNING','STARTING','QUEUEING','INIT','CREATING','WAITING','RESTARTING','RECOVERING','SUSPENDING')
    pages={}
    for state in states:
        page=1
        while True:
            response=read_jobs('acp','jobs','list','--workspace-name=share-space','--user-name=260010165',
                         '--state='+state,'--page-size=5','--page-token='+str(page),'-o','json')
            if response.stdout.strip()=='No jobs found':items=[]
            else:
                value=json.loads(response.stdout)
                items=value if isinstance(value,list) else next((x for x in value.values() if isinstance(x,list)),[])
            for job in items:active[job['uid']]=job
            if len(items)<5:break
            page+=1
            if page>100:raise ValueError('Occupancy pagination exceeded bound')
        pages[state]=page
    jobs=[];used=0
    for uid,job in active.items():
        name=job['name']
        # The list response already carries quota, ownership and requested
        # resources. Avoid a second, racing lookup for every active job.
        if not all(key in job for key in ('ownership','scheduling','roles')):
            response=read_jobs('acp','jobs','describe',name,'--workspace-name=share-space','-o','json')
            job=json.loads(response.stdout)
        if job['uid']!=uid or job['ownership']['user_name']!='260010165':
            raise ValueError('Occupancy identity mismatch')
        if job.get('scheduling',{}).get('quota_type')!='RESERVED':continue
        count=0
        for role in job.get('roles',[]):
            specs=role.get('resource_spec',[])
            if len(specs)!=1:raise ValueError('Ambiguous standard-resource shape')
            count+=int(specs[0]['limits'].get('nvidia.com/gpu',0))*int(role['total_replicas'])
        used+=count;jobs.append(dict(name=name,uid=uid,state=job['state'],gpus=count))
    evidence=dict(observed_at=datetime.now(timezone.utc).isoformat(),workspace='share-space',
        rule_limit=limit,rule_source='Published project rules snapshot 2026-09-11',
        standard_requested_gpus=used,additional_gpus=additional,standard_jobs=jobs,pages=pages,
        passed=used+additional<=limit)
    write(output,evidence)
    if not evidence['passed']:raise RuntimeError('No unused standard quota for this role')
    return evidence

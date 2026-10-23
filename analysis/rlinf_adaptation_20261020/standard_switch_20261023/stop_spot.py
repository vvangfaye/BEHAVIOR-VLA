"""Stop only the identified, not-yet-started Spot predecessor."""
import json
import subprocess
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
CLI='/home/faye/.local/bin/sco'
NAME='b1k-ppo5-1023-p01'
UID='93fd4f3df3fe45d8bc7d0674e3dc4777'


def describe():
    r=subprocess.run([CLI,'acp','jobs','describe',NAME,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40)
    d=json.loads(r.stdout)
    assert d['name']==NAME and d['uid']==UID and d['ownership']['user_name']=='260010165'
    assert d['scheduling']['quota_type']=='SPOT'
    return d


def safe(d):
    return {k:d.get(k) for k in ('name','uid','state','start_time','complete_time')}


d=describe()
assert d.get('start_time') is None,'Worker execution may have begun; preserve its outputs before migrating'
with (HERE/'stop_intent.json').open('x') as f:
    json.dump({'job':NAME,'uid':UID,'before':safe(d),'reason':'User explicitly requested moving this one-GPU PPO experiment to reserved quota without duplicate execution'},f,indent=2)
r=subprocess.run([CLI,'acp','jobs','stop',NAME,'--workspace-name=share-space'],capture_output=True,text=True,timeout=45)
(HERE/'stop_request.json').write_text(json.dumps({'returncode':r.returncode,'at':time.time()},indent=2)+'\n')
deadline=time.monotonic()+60
while time.monotonic()<deadline:
    d=describe()
    if d['state'] in {'SUSPENDED','STOPPED','TERMINATED','CANCELED','FAILED','SUCCEEDED'}:
        assert d.get('start_time') is None,'Execution began during stop; reconcile results before submitting'
        result={'passed':True,'old_job':safe(d),'never_started':True,'new_job_submitted':False}
        (HERE/'stop_verified.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));break
    time.sleep(1)
else:
    raise TimeoutError('Stop is pending; do not create the replacement yet')

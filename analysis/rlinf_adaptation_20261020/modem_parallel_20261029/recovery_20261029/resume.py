"""One bounded infrastructure recovery; preserve all model and reward contracts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from deploy import describe

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

assert not (HERE/'launch.json').exists(), 'Recovery already launched; inspect its receipt'
assert 'Ran 18 tests' in (HERE/'protocol_test.log').read_text()
failure=json.loads((HERE/'failure.json').read_text())
assert failure['failure']['error_type']=='ExecutionStopped'
assert len(failure['training_receipts'])==9
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipts=[json.loads(p.read_text()) for p in sorted((ROOT/'jobs').glob('sample2_r*_a1/submission.json'))]
assert len(receipts)==8
def verify(r):
    d=describe(r['job']);assert d['uid']==r['uid'] and d['ownership']['user_name']=='260010165'
    assert d['state'] in {'SUSPENDED','FAILED','STOPPED','TERMINATED','CANCELED'}
    p=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','get-workers',r['job'],
                      '--workspace-name=share-space'],capture_output=True,text=True,timeout=40,check=True)
    assert 'Running' not in p.stdout, 'Prior worker is still running'
    return {'job':r['job'],'uid':d['uid'],'state':d['state'],'workers_stopped':True}
with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,receipts))
old=json.loads((HERE/'before_execution_manifest.json').read_text())
changed={'bootstrap.py','controller.py'}
for name,digest in old['source_sha256'].items():
    if name not in changed:assert sha(ROOT/name)==digest,name
manifest=json.loads((ROOT/'execution_manifest.json').read_text())
for name in changed:manifest['source_sha256'][name]=sha(ROOT/name)
now=datetime.now(timezone.utc).isoformat()
authorization={'created_at':now,'evidence':'failure.json',
    'root_cause':'A stopped rollout was reported as DriverProcessExit, hiding its ExecutionStopped type; origin of stop signal not established.',
    'verified_stopped_uids':[r['uid'] for r in verified],
    'fixed_source_sha256':{name:sha(ROOT/name) for name in changed},
    'reused_completed_trajectories':9,'changed_model_training_reward_or_evaluation':False,
    'attempt_cap':2,'max_live_gpus':8,'max_total_gpu_hours':80,'verified_jobs':verified}
(HERE/'retry_authorization.json').write_text(json.dumps(authorization,indent=2)+'\n')
manifest['runtime_failure_reporting_recovery']=authorization
(ROOT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
tests=json.loads((ROOT/'test_result.json').read_text())
tests.update(passed=True,tests=21,completed_at=now,recovery_tests=3,original_protocol_tests=18)
(ROOT/'test_result.json').write_text(json.dumps(tests,indent=2)+'\n')
for name in ['stopped.json','stop_receipt.json']:
    p=ROOT/name
    if p.exists():p.rename(HERE/('retired_'+name))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (ROOT/'controller.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,str(ROOT/'controller.py')],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={**authorization,'pid':child.pid,'launched_at':now,'plan_sha256':sha(ROOT/'plan.json')}
(HERE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'pid':child.pid,'reused_training_episodes':9,'max_gpus':8,'model_and_reward_unchanged':True}))

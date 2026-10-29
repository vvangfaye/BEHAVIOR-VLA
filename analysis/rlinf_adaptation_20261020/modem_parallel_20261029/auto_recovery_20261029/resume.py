"""Resume bounded automatic recovery without resetting attempts or training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from deploy import describe

assert not (HERE/'launch.json').exists(),'Inspect existing recovery launch'
for log,number in [('protocol_test.log',18),('reporting_test.log',3),('recovery_test.log',5)]:
    text=(HERE/log).read_text();assert f'Ran {number} tests' in text and text.strip().endswith('OK'),log
lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((HERE/'before_execution_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in old['source_sha256'].items():
    if name not in ['controller.py','deploy.py']:assert sha(ROOT/name)==digest,name
receipts=[json.loads(p.read_text()) for p in sorted((ROOT/'jobs').glob('sample2_r*_a2/submission.json'))]
assert len(receipts)==8
def verify(r):
    d=describe(r['job']);assert d['uid']==r['uid'] and d['ownership']['user_name']=='260010165'
    assert d['state'] in {'SUSPENDED','FAILED','STOPPED','TERMINATED','CANCELED'}
    p=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','get-workers',r['job'],'--workspace-name=share-space'],capture_output=True,text=True,timeout=40,check=True)
    assert 'Running' not in p.stdout
    return {'job':r['job'],'uid':d['uid'],'state':d['state'],'worker_stopped':True}
with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,receipts))
manifest=json.loads((ROOT/'execution_manifest.json').read_text())
for name in ['controller.py','deploy.py','recovery_policy.py','auto_recovery_policy.json']:
    manifest['source_sha256'][name]=sha(ROOT/name)
now=datetime.now(timezone.utc).isoformat()
manifest['bounded_auto_recovery']={'at':now,'tests_passed':26,'training_and_evaluation_unchanged':True,
                                 'attempt_limit':8,'gpu_hour_limit':80,'old_jobs':verified}
(ROOT/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
t=json.loads((ROOT/'test_result.json').read_text());t.update(passed=True,tests=26,completed_at=now)
(ROOT/'test_result.json').write_text(json.dumps(t,indent=2)+'\n')
for name in ['stopped.json','stop_receipt.json']:
    p=ROOT/name
    if p.exists():p.rename(HERE/('retired_'+name))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (ROOT/'controller.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,str(ROOT/'controller.py')],cwd=ROOT,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
multi=ROOT.parent/'multitask_validation_20261029'
with (multi/'coordinator.log').open('ab',buffering=0) as log:
    coordinator=subprocess.Popen([sys.executable,str(multi/'coordinator.py')],cwd=multi,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={'launched_at':now,'modem_controller_pid':child.pid,'task_coordinator_pid':coordinator.pid,
         'max_rl_gpus_including_existing_baseline':8,'max_modem_gpu_hours':80,
         'max_attempts_per_shard':8,'preserved_existing_attempts':True,
         'preserved_new_task_baseline':True,'verified_old_workers':verified}
(HERE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='verified_old_workers'}))

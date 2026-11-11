"""Resume the explicitly requested full two-round experiment once."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from deploy import describe

assert not (root/'resume_full_launch_intent.json').exists(),'Inspect existing resume intent before retrying'
assert not list((root/'jobs').glob('*/submission.json')),'Existing full-run receipts must be reconciled'
manifest=json.loads((root/'execution_manifest.json').read_text())
for name,digest in manifest['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name in ('test_result.json','preflight_result.json','baseline_reuse_audit.json','resume_equivalence.json'):
    assert json.loads((root/name).read_text())['passed']
noon=root.with_name('modem_noon_20261029')
assert json.loads((noon/'latest_status.json').read_text())['finished']
receipts=[json.loads(path.read_text()) for path in (noon/'jobs').glob('*/submission.json')]
def verify(receipt):
    actual=describe(receipt['job'])
    assert actual['uid']==receipt['uid'] and actual['ownership']['user_name']=='260010165'
    assert actual['state']=='SUCCEEDED','Noon validation still occupies a slot: '+receipt['job']
    return {'job':receipt['job'],'uid':actual['uid'],'state':actual['state']}
with ThreadPoolExecutor(max_workers=4) as pool:
    terminal=list(pool.map(verify,receipts))
(root/'resume_full_20261029/noon_jobs_verified.json').write_text(json.dumps(terminal,indent=2)+'\n')
lock=(root/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
intent={'requested_at':datetime.now(timezone.utc).isoformat(),'max_live_gpus':8,
        'max_total_gpu_hours':80,'baseline_reused':4,'new_baseline_episodes':42,
        'rounds':2,'training_episodes_per_round':32,'optimizer_steps_per_round':17,
        'heldout_pairs':30,'training_pairs':16,'checkpoint_start':'original1688977',
        'previous_noon_jobs_all_terminal':True}
with (root/'resume_full_launch_intent.json').open('x') as output:
    json.dump(intent,output,ensure_ascii=False,indent=2)
for name in ('CANCELLED_BY_USER','DEFERRED_FOR_NOON_PILOT.json'):
    path=root/name
    if path.exists():path.rename(root/'resume_full_20261029'/('retired_'+name))
with (root/'controller.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,str(root/'controller.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={**intent,'pid':child.pid,'started_at':datetime.now(timezone.utc).isoformat(),
         'experiment':'modem_ppo_20261028','generation':'parallel8_20261029',
         'plan_sha256':manifest['source_sha256']['plan.json'],'automatic_phase_progression':True}
(root/'controller_launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))

from pathlib import Path
import sys,os,json,fcntl,subprocess,hashlib
from datetime import datetime,timezone
root=Path(__file__).resolve().parent.parent
intent=root/'startup_recovery_20261029/relaunch_intent.json'
assert not intent.exists(),'Reconcile existing launch before retry'
assert not (root/'CANCELLED_BY_USER').exists()
lock=(root/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=json.loads((root/'execution_manifest.json').read_text())
for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert json.loads((root/'test_result.json').read_text())['passed']
assert json.loads((root/'startup_recovery_20261029/retry_authorization.json').read_text())['training_and_evaluation_settings_unchanged']
receipt={'requested_at':datetime.now(timezone.utc).isoformat(),'reason':'Repair concurrent storage JSON temporary file collision','max_live_gpus':8,'max_total_gpu_hours':80,'two_rounds_unchanged':True}
with intent.open('x') as h:json.dump(receipt,h,indent=2)
if (root/'stopped.json').exists():(root/'stopped.json').rename(root/'startup_recovery_20261029/retired_stopped.json')
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (root/'controller.log').open('ab',buffering=0) as log:
 child=subprocess.Popen([sys.executable,str(root/'controller.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt.update(pid=child.pid,started_at=datetime.now(timezone.utc).isoformat())
(root/'startup_recovery_20261029/relaunch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

"""Start one local coordinator for the explicitly requested finite task suite."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
assert not (root/'queue_launch.json').exists(),'Inspect the existing coordinator before relaunching'
for task in ['trash','water']:
    folder=root/task
    assert json.loads((folder/'test_result.json').read_text())['passed']
    manifest=json.loads((folder/'execution_manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,(task,name)
lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (root/'coordinator.log').open('ab',buffering=0) as log:
    process=subprocess.Popen([sys.executable,str(root/'coordinator.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={'pid':process.pid,'launched_at':datetime.now(timezone.utc).isoformat(),
         'scope':'Two task replications, each 32 train trajectories, 17 PPO updates, 24 fixed eval pairs',
         'tasks':['picking_up_trash','bringing_water'],'max_new_campaign_gpus':8,
         'max_new_campaign_gpu_hours':256,'max_new_persistent_gib':112,
         'dispatch':'After modem completion or confirmed stopped workers; then one new task at a time',
         'plan_sha256':hashlib.sha256((root/'campaign.json').read_bytes()).hexdigest(),
         'user_request':'再用多个任务说明下目前的RL是有效的'}
(root/'queue_launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))

"""Run the existing two baseline panels, with no PPO until storage is cleared."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
receipt=ROOT/'baseline_unblock_20261030/launch.json'
assert not receipt.exists(),'Reconcile existing baseline coordinator PIDs first'
for slug in ['trash','water']:
    folder=ROOT/slug
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    marker={'scope':'baseline only; no training or optimizer updates',
            'max_live_gpus':4,'max_shared_baseline_bytes':8*1024**3}
    (folder/'BASELINE_ONLY.json').write_text(json.dumps(marker,indent=2)+'\n')
    shutil.copyfile(ROOT/'task_controller.py',folder/'controller.py')
    manifest=json.loads((folder/'execution_manifest.json').read_text())
    for name in ['controller.py','BASELINE_ONLY.json']:
        manifest['source_sha256'][name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    (folder/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,(slug,name)
out={'launched_at':datetime.now(timezone.utc).isoformat(),'scope':'two baseline panels only','max_total_gpus':8,'tasks':[]}
for slug in ['trash','water']:
    folder=ROOT/slug
    with (folder/'controller.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,str(folder/'controller.py')],cwd=folder,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
    out['tasks'].append({'slug':slug,'pid':child.pid,'max_live_gpus':4})
receipt.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

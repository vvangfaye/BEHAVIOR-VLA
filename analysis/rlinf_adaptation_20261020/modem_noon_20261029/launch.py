"""Launch the exact user-authorized noon pilot once, with bounded cloud work."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root=Path(__file__).resolve().parent
assert not (root/'controller_launch.json').exists(),'A controller receipt already exists; inspect it'
assert json.loads((root/'test_result.json').read_text())['passed']
assert json.loads((root/'preflight_result.json').read_text())['passed']
manifest=json.loads((root/'execution_manifest.json').read_text())
for name,digest in manifest['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert (root.with_name('modem_parallel_20261029')/'CANCELLED_BY_USER').is_file()
with (root/'controller.log').open('ab',buffering=0) as log:
    child=subprocess.Popen([sys.executable,str(root/'controller.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
receipt={'pid':child.pid,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'experiment':'modem_noon_20261029','max_live_gpus':8,'max_total_gpu_hours':12,
         'max_wall_hours':3,'training_episodes':4,'ppo_optimizer_steps':2,'evaluation_pairs':4,
         'deadline_shanghai':'2026-10-29T12:00:00+08:00','original_baseline_included_in_gpu_limit':True,
         'full_experiment_deferred':True}
(root/'controller_launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))

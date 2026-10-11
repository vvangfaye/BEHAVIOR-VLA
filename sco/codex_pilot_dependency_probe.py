"""CPU-only dependency audit; reads the restored environments without changing them."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request

runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
code=Path('/mnt/afs/260010165/behavior_1k/codex_pilot_20261011/code-v1')
probe='''import importlib,json,sys
result={'python':sys.version,'modules':{}}
for name in ['numpy','h5py','pyarrow','PIL','av','torch','lerobot','yaml','omegaconf','stage_subtask_replay','codex_collection_demonstrations']:
 try:
  m=importlib.import_module(name);result['modules'][name]={'ok':True,'version':str(getattr(m,'__version__','unknown'))}
 except Exception as e:result['modules'][name]={'ok':False,'type':type(e).__name__,'missing_module':getattr(e,'name',None),'message':str(e)[:300]}
print('DEPENDENCY_RESULT='+json.dumps(result))
'''
result=dict(state='dependency_probe_complete',environments={})
for name in ['b1k-pi05','b1k-eval']:
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=str(code/'scripts'),OMP_NUM_THREADS='2')
 p=subprocess.run([str(runtime/'envs'/name/'bin/python'),'-c',probe],env=env,capture_output=True,text=True,timeout=180)
 lines=[l for l in p.stdout.splitlines() if l.startswith('DEPENDENCY_RESULT=')]
 result['environments'][name]=json.loads(lines[-1].split('=',1)[1]) if lines else dict(exit_code=p.returncode)
request=urllib.request.Request(os.environ['CODEX_PILOT_REPORT_PUT'],data=json.dumps(result).encode(),method='PUT')
with urllib.request.urlopen(request,timeout=30):pass
print(json.dumps(result),flush=True)

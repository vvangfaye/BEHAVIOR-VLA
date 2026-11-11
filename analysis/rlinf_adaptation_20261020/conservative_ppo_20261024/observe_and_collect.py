"""Observe this existing allocation and audit its returned artifacts; never submit."""
import json
import fcntl
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
lock=(HERE/'observer.lock').open('w')
try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('An observer already holds this run lock')
ROOT=HERE.parents[2]
PYTHON=str(HERE.parent/'venv/bin/python')
deadline=time.monotonic()+12*3600
paths=list(HERE.glob('live_[0-9]*.json'))
index=max([int(p.stem.split('_')[1]) for p in paths]+[0])+1
previous=None
queue_state=None
while time.monotonic()<deadline:
    status=subprocess.run([PYTHON,str(HERE/'status.py')],cwd=ROOT,capture_output=True,text=True,timeout=75)
    job=json.loads((HERE/'latest_job_status.json').read_text()) if (HERE/'latest_job_status.json').exists() else {}
    if job.get('state') in {'FAILED','TERMINATED','CANCELED','SUCCEEDED','STOPPED'}:
        result=subprocess.run([PYTHON,str(HERE/'monitor.py')],cwd=ROOT,capture_output=True,text=True,timeout=1200)
        print(json.dumps({'terminal':job,'retrieval_returncode':result.returncode}),flush=True)
        receipt=json.loads((HERE/'monitor_state.json').read_text()) if (HERE/'monitor_state.json').exists() else {}
        if result.returncode or receipt.get('state')!='results_retrieved':
            (HERE/'retrieval_error.log').write_text(result.stderr)
            print('Retrieval did not complete; no acceptance audit will run on missing files.',flush=True)
            break
        audit=subprocess.run([PYTHON,str(HERE/'audit_result.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
        print(audit.stdout,flush=True)
        break
    if job.get('state')!='RUNNING':
        if queue_state!=job.get('state'):
            print(json.dumps({'job_state':job.get('state'),'job':job.get('name')}),flush=True)
            queue_state=job.get('state')
        time.sleep(30)
        continue
    output=HERE/f'live_{index:02d}.json'
    with (HERE/f'live_{index:02d}_transport.log').open('w') as log:
        result=subprocess.run([PYTHON,str(HERE.parent/'forensics_20261023/exec_probe.py'),
            '--job','b1k-ppoc2-1024-a01','--uid',json.loads((HERE/'submission_intent.json').read_text())['uid'],
            '--script',str(HERE/'live_probe.py'),'--output',str(output),'--timeout','60'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=110)
    if output.exists():
        data=json.loads(output.read_text());files=data.get('files',{})
        concise={'status':files.get('status.json'),'gpu':data.get('gpu')}
        for name,value in files.items():
            if name.endswith(('/result.json','/progress.json','/initialization.json')) or name in ('failure.json','latest_checkpoint.json','comparison.json'):
                concise[name]=value
        text=json.dumps(concise,ensure_ascii=False,sort_keys=True)
        if text!=previous:print(text,flush=True);previous=text
    else:print(json.dumps({'observation_error':result.returncode,'index':index}),flush=True)
    index+=1
    time.sleep(30)

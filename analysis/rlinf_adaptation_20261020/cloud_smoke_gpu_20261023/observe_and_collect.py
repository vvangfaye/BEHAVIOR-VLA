"""Observe this existing allocation and audit its returned artifacts; never submit."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PYTHON=sys.executable
deadline=time.monotonic()+6*3600
paths=list(HERE.glob('live_[0-9]*.json'))
index=max([int(p.stem.split('_')[1]) for p in paths]+[0])+1
previous=None
while time.monotonic()<deadline:
    status=subprocess.run([PYTHON,str(HERE/'status.py')],cwd=ROOT,capture_output=True,text=True,timeout=75)
    job=json.loads((HERE/'latest_job_status.json').read_text()) if (HERE/'latest_job_status.json').exists() else {}
    if job.get('state') in {'FAILED','TERMINATED','CANCELED','SUCCEEDED','STOPPED'}:
        result=subprocess.run([PYTHON,str(HERE/'monitor.py')],cwd=ROOT,capture_output=True,text=True,timeout=1200)
        print(json.dumps({'terminal':job,'retrieval_returncode':result.returncode}),flush=True)
        if result.returncode:
            (HERE/'retrieval_error.log').write_text(result.stderr)
            print('Retrieval did not complete; no acceptance audit will run on missing files.',flush=True)
            break
        audit=subprocess.run([PYTHON,str(HERE/'audit_result.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
        print(audit.stdout,flush=True)
        break
    output=HERE/f'live_{index:02d}.json'
    with (HERE/f'live_{index:02d}_transport.log').open('w') as log:
        result=subprocess.run([PYTHON,str(HERE.parent/'forensics_20261023/exec_probe.py'),
            '--job','b1k-rlinf-1023-r07','--uid','cc2919cc6b5045d3ad07009a842762d7',
            '--script',str(HERE/'live_probe.py'),'--output',str(output),'--timeout','60'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=110)
    if output.exists():
        data=json.loads(output.read_text());files=data.get('files',{})
        concise={'status':files.get('results/status.json'),'gpu':data.get('gpu')}
        for backend in ('jax','torch'):
            for instance in (301,302):
                for field in ('initialization','progress','result','failure'):
                    key=f'results/{backend}/{instance}/{field}.json'
                    if key in files:concise[f'{backend}/{instance}/{field}']=files[key]
        for field in ('results/gate.json','results/ppo/progress.json','results/ppo/result.json','takeover_state.json'):
            if field in files:concise[field]=files[field]
        text=json.dumps(concise,ensure_ascii=False,sort_keys=True)
        if text!=previous:print(text,flush=True);previous=text
    else:print(json.dumps({'observation_error':result.returncode,'index':index}),flush=True)
    index+=1
    time.sleep(30)

import json,base64,subprocess
from pathlib import Path
root=Path('/tmp/b1k-serf20-eval01-worker0')
r={}
f=root/'status.json'
if f.exists():
 s=json.loads(f.read_text());r={k:s[k] for k in ['phase','complete','downloaded_files','maps_verified','error'] if k in s}
 for dest in root.iterdir():
  if not dest.is_dir() or not (dest/'server.log').exists():continue
  item={}
  f=dest/'progress.json'
  if f.exists():item['progress']=json.loads(f.read_text())
  lines=(dest/'server.log').read_text(errors='replace').splitlines()
  item['alignment']=[x for x in lines if 'POLICY_STEP_ALIGNMENT_PASSED' in x][-3:]
  item['server_tail']='\n'.join(lines[-8:])
  r[dest.name]=item
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)

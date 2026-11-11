import base64,json,subprocess
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_eval17_20261027/baseline_first_attempt01')
report={'workers':{}}
for rank in (0,1):
 r=root/f'worker_{rank}';row={'root_exists':r.exists()}
 for name in ['runtime_receipt.json','checkpoint_verified.json','bootstrap_error.json','results/status.json','results/model_identity.json','results/failure.json']:
  path=r/name
  if path.is_file():row[name]=json.loads(path.read_text())
 progress=[]
 for path in (r/'results').glob('*/i*_s*/initialization.json'):
  progress.append({'case':str(path.parent.relative_to(r/'results')),**json.loads(path.read_text())})
 row['initializations']=progress[-5:]
 report['workers'][str(rank)]=row
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report['processes']=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=path.read_bytes().replace(b'\x00',b' ')
  for label in ['run/driver.py','run/simulator.py']:
   if label.encode() in cmd and b'base64' not in cmd:report['processes'].append({'pid':int(path.parent.name),'program':label})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')

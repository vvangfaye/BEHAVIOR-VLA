import base64,json,subprocess,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/init_diagnosis_20261026/pilot01/results')
report={'time':time.time(),'exists':root.exists(),'files':{},'processes':[]}
for rel in ['status.json','python.json','trials.json','result.json','failure.json','original_240s/initialization.json','original_240s/pre_abort_evidence.json','stress_0p05s/initialization.json','stress_0p05s/pre_abort_evidence.json','disabled/initialization.json']:
 p=root/rel
 if p.is_file():report['files'][rel]=json.loads(p.read_text())
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  if not any(str(root.parent).encode() in x and (x.endswith(b'/driver.py') or x.endswith(b'/simulator.py')) for x in args):continue
  report['processes'].append({'pid':int(p.parent.name),'status':[x for x in (p.parent/'status').read_text().splitlines() if x.split(':')[0] in ['State','VmRSS','Threads']],'fd_count':len(list((p.parent/'fd').iterdir()))})
 except (FileNotFoundError,PermissionError):pass
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True).stdout
for rel in ['original_240s/simulator.log','stress_0p05s/simulator.log','disabled/simulator.log']:
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-1600));report['files'][rel+'_tail']=f.read().decode(errors='replace')
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')

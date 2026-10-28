import base64,json,re,subprocess
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_midreward_20261028/pilot01')
out={'workers':{},'completed_training_episodes':len(list((root/'results/training_receipts').glob('*.json')))}
for rank in [0,1]:
 job=root/'jobs'/f'sample_r{rank}_a3';row={}
 for name in ['runtime_receipt.json','bootstrap_error.json']:
  if (job/name).is_file():row[name]=json.loads((job/name).read_text())
 log=job/'driver.log'
 if log.exists():row['log_tail']=re.sub(r'https?://\S+','[URL omitted]',log.read_text(errors='replace')[-1400:])
 status=root/'results/worker_status'/f'sample_r{rank}_a3.json'
 if status.exists():row['status']=json.loads(status.read_text())
 out['workers'][str(rank)]=row
out['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')

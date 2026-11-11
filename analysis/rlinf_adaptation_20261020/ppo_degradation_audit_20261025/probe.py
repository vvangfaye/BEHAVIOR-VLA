import json,pathlib,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/diagnostic_20261025a')
r={'exists':p.exists(),'files':[x.name for x in p.iterdir()] if p.exists() else []}
for rel in ['runtime_receipt.json','bootstrap_error.json','results/status.json','results/failure.json']:
 f=p/rel
 if f.exists():r[rel]=json.loads(f.read_text())
r['model_reads']=[]
for process in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  for fd in (process/'fd').iterdir():
   target=fd.resolve()
   if str(target).startswith('/mnt/afs/260010165/behavior_1k/') and target.suffix in ('.safetensors','.pt'):
    info=(process/'fdinfo'/fd.name).read_text()
    pos=next((x for x in info.splitlines() if x.startswith('pos:')),None)
    r['model_reads'].append({'pid':int(process.name),'path':str(target),'position':pos})
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)

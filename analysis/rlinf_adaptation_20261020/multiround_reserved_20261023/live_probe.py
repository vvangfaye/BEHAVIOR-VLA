import base64,json,pathlib,re,subprocess,time
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_multiround_20261023/pilot01_reserved')
out={'at':time.time(),'root_exists':root.exists(),'files':{}}
results=root/'results'
for name in ['status.json','failure.json','comparison.json','latest_checkpoint.json','warm_start.json']:
 p=results/name
 if p.exists():out['files'][name]=json.loads(p.read_text())
for phase in ('baseline','final'):
 for p in sorted((results/phase).glob('*/result.json')):out['files'][str(p.relative_to(results))]=json.loads(p.read_text())
for p in sorted((results/'training').glob('round_*/result.json')):
 d=json.loads(p.read_text());d['update'].pop('metrics',None);out['files'][str(p.relative_to(results))]=d
status=out['files'].get('status.json',{})
if status.get('stage') in ('baseline','final') and 'instance' in status:
 active=results/status['stage']/f"i{status['instance']}_s{status['seed']}"
elif 'round' in status:
 active=results/'training'/f"round_{status['round']:02d}"/'rollout'
else:active=None
if active is not None:
 for name in ['initialization.json','progress.json','result.json','failure.json']:
  p=active/name
  if p.exists():out['files'][str(p.relative_to(results))]=json.loads(p.read_text())
 for name in ['simulator.log']:
  p=active/name
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-600));tail=f.read().decode(errors='replace')
   out['files'][str(p.relative_to(results))]={'tail':re.sub(r'https?://\S+','[URL omitted]',tail)}
try:out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
except Exception as e:out['gpu_error']=type(e).__name__
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END',flush=True)

import base64, json, pathlib, subprocess, time
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry07')
out={'at':time.time(), 'files':{}}
for rel in ['results/gpu_fp32_probe/progress.json','results/gpu_fp32_probe/result.json','results/jax/302/progress.json','results/jax/302/result.json']:
 p=root/rel
 if p.exists():
  d=json.loads(p.read_text())
  d.pop('memory_samples',None)
  if 'comparisons' in d:
   comparisons=d.pop('comparisons')
   d.update(input_count=len(comparisons),max_mean_error=max(x['mean_error'] for x in comparisons),max_action_error=max(x['max_error'] for x in comparisons),mean_warm_inference_seconds=sum(x['seconds'] for x in comparisons[1:])/max(1,len(comparisons)-1))
  out['files'][rel]=d
p=root/'results/gpu_fp32_probe/run.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-1800));out['log_tail']=f.read().decode(errors='replace')
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
try:
 launch=json.loads((root/'results/gpu_fp32_probe/launch.json').read_text()); timeout_pid=launch['pid']; rows=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   stat=(p/'stat').read_text().split()
   if int(p.name)==timeout_pid or int(stat[3])==timeout_pid:
    status=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
    rows.append({'pid':int(p.name),'comm':stat[1], 'cpu_ticks':int(stat[13])+int(stat[14]),'threads':status.get('Threads','').strip(),'rss':status.get('VmRSS','').strip(),'allowed_cpus':status.get('Cpus_allowed_list','').strip(),'wchan':(p/'wchan').read_text()})
  except (FileNotFoundError,PermissionError):pass
 out['probe_processes']=rows
except Exception as e:out['process_query_error']=str(e)
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END',flush=True)

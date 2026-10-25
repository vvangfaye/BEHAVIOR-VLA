import base64,json,subprocess
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/init02')
r={}
for n in ['status.json','image.log','serf.log','image/initial.json','image/initialize_result.json','serf/initialize_result.json']:
 f=p/n
 if f.is_file():r[n]=f.read_text()[-3500:]
r['processes']=subprocess.check_output(['ps','-eo','pid,ppid,comm,etime,pcpu,pmem,wchan'],text=True)[-6500:]
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')

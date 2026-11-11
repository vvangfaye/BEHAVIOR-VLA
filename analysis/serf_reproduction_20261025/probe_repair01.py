import base64,json,os
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01')
r={}
for n in ['iteration_1.json','repair_01.log','sim_1.log','policy_1.log']:
 f=p/n
 if f.is_file():r[n]=f.read_text()[-6000:]
try:os.kill(409,0);r['pid_alive']=True
except ProcessLookupError:r['pid_alive']=False
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')

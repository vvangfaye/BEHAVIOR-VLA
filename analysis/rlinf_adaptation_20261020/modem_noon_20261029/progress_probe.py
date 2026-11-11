import base64,json,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/noon01')
out={'observed_at':time.time(),'baseline':{},'training':{},'resource_memory':{}}
for key in ('i2_s0','i301_s0','i302_s0'):
    path=root/'results/baseline'/key
    out['baseline'][key]={name:json.loads((path/name).read_text()) for name in
                          ('progress.json','initialization.json','result.json','process_failure.json','python_startup_failure.json')
                          if (path/name).is_file()}
for i in range(1,5):
    path=root/'results/training/round_01'/f'episode_{i:02d}'
    row={}
    for name in ('progress.json','result.json','rollout/initialization.json','rollout/process_failure.json','rollout/python_startup_failure.json'):
        file=path/name
        if file.is_file():row[name]=json.loads(file.read_text())
    out['training'][str(i)]=row
for folder in (root/'jobs').glob('*'):
    file=folder/'resource_latest.json'
    if file.is_file():out['resource_memory'][folder.name]=json.loads(file.read_text())['cgroup']
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')

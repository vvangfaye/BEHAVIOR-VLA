import base64,json,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01')
out={'observed_at':time.time(),'jobs':{},'baseline_complete':[]}
for rank in range(8):
 folder=root/'jobs'/f'p8_baseline_r{rank}_a1'
 row={}
 for name in ('status.json','driver_exit.json','resource_before.json','resource_latest.json'):
  file=folder/name
  if file.is_file():row[name]=json.loads(file.read_text())
 file=folder/'driver.log'
 if file.is_file():
  with file.open('rb') as h:
   h.seek(max(0,file.stat().st_size-10000));row['driver_tail']=h.read(10000).decode(errors='replace')
 out['jobs'][str(rank)]=row
for folder in (root/'results/baseline').glob('i*_s*'):
 if all((folder/n).is_file() for n in ('result.json','actions.json','goal_summary.json')):
  d=json.loads((folder/'result.json').read_text());out['baseline_complete'].append({'case':folder.name,'result':d})
noon=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/noon01')
folder=noon/'jobs/noon_train1_r0_a1'
out['noon_train_timing']={}
for name in ('driver_exit.json','resource_before.json','resource_latest.json'):
 file=folder/name
 if file.is_file():out['noon_train_timing'][name]=json.loads(file.read_text())
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')

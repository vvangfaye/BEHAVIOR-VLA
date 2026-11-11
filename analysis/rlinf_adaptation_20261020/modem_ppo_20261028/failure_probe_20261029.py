import base64,json,re
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01')
out={'completed_baseline':[],'jobs':{},'cases':{}}
for p in sorted((root/'results/baseline').glob('*/result.json')):
    out['completed_baseline'].append(json.loads(p.read_text()))
def tail(path,size):
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-size))
        return re.sub(r'https?://\S+','[URL omitted]',f.read().decode(errors='replace'))
for rank in (0,1):
    key=f'baseline_r{rank}_a2';job=root/'jobs'/key;row={}
    for name in ('runtime_receipt.json','bootstrap_error.json'):
        if (job/name).is_file():row[name]=json.loads((job/name).read_text())
    log=job/'driver.log'
    if log.is_file():row['driver_log_tail']=tail(log,5000)
    out['jobs'][key]=row
for i in (1,2):
    folder=root/'results/baseline'/f'i{i}_s0';row={}
    for name in ('progress.json','initialization.json','initial.json','prompt_contract.json',
                 'process_failure.json','python_startup_failure.json','goal_contract.json','goal_summary.json','failure.json'):
        path=folder/name
        if path.is_file() and path.stat().st_size<40000:row[name]=json.loads(path.read_text())
    log=folder/'simulator.log'
    if log.is_file():row['simulator_tail']=tail(log,7000)
    logs=list((folder/'native_logs').rglob('*.log'))
    row['native_log_tails']=[{'path':str(x.relative_to(folder)),'tail':tail(x,1800)} for x in logs[-2:]]
    out['cases'][str(i)]=row
out['failure_files']={str(p.relative_to(root)):json.loads(p.read_text()) for p in (root/'results/failures').glob('*.json')}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')

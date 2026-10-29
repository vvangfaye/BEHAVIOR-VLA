import base64,json,re
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01')
job=root/'jobs/baseline_r0_a3'
out={'job_files':{},'baseline_complete':[],'current':{},'at':__import__('time').time()}
for name in ('runtime_receipt.json','resource_latest.json','driver_exit.json','bootstrap_error.json'):
    p=job/name
    if p.is_file():out['job_files'][name]=json.loads(p.read_text())
p=job/'driver.log'
if p.is_file():
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-1800))
        out['driver_log_tail']=re.sub(r'https?://\S+','[URL omitted]',f.read().decode(errors='replace'))
for p in sorted((root/'results/baseline').glob('*/result.json')):
    out['baseline_complete'].append(json.loads(p.read_text()))
case=root/'results/baseline/i1_s0'
for name in ('initialization.json','progress.json','prompt_contract.json','goal_summary.json','process_failure.json','python_startup_failure.json'):
    p=case/name
    if p.is_file():out['current'][name]=json.loads(p.read_text())
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')

"""Read only this pool's reported usage and our seven exact job workers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent
cli='/home/faye/.local/bin/sco'
def run(args):
    try:
        p=subprocess.run([cli]+args,capture_output=True,text=True,timeout=35)
        clean=lambda s:re.sub(r'https?://\S+','[URL omitted]',s)[-7000:]
        return {'returncode':p.returncode,'stdout':clean(p.stdout),'stderr':clean(p.stderr)}
    except subprocess.TimeoutExpired:
        return {'returncode':'timeout','stdout':'','stderr':'bounded 35-second timeout'}
usage=run(['aec2','clusters','usage','--name=computing-cluster-5090-01g'])
receipts=[json.loads(f.read_text()) for f in sorted((root/'jobs').glob('*/submission.json'))]
def observe(receipt):
    row={'job':receipt['job'],'uid':receipt.get('uid'),'quota':receipt['quota']}
    raw=subprocess.run([cli,'acp','jobs','describe',receipt['job'],'--workspace-name=share-space','-o','json'],
                       capture_output=True,text=True,timeout=35,check=True)
    actual=json.loads(raw.stdout)
    assert actual['uid']==receipt['uid'] and actual['ownership']['user_name']=='260010165'
    row.update(state=actual['state'],start_time=actual.get('start_time'))
    row['workers']=run(['acp','jobs','get-workers',receipt['job'],'--workspace-name=share-space'])
    return row
with ThreadPoolExecutor(max_workers=4) as pool:
    workers=list(pool.map(observe,receipts))
result={'observed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'pool':'computing-cluster-5090-01g','quota_type':'spot','pool_usage':usage,'our_workers':workers}
(root/'spot_capacity_observed_20261029.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))

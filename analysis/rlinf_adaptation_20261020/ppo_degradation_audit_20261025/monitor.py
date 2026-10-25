import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

receipt=json.loads((HERE/'submission.json').read_text())
client=aoss_client();previous=None
try:
    deadline=time.monotonic()+4500
    while time.monotonic()<deadline:
        try:
            data=json.loads(client.get_object(Bucket='260010165',Key='behavior1k/ppo-degradation-audit-20261025/a01/audit.json')['Body'].read())
            (HERE/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps({'finished':True,'passed':data.get('passed'),'error':data.get('error')}),flush=True)
            break
        except client.exceptions.NoSuchKey:pass
        d=json.loads(subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',receipt['job'],'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=45).stdout)
        assert d['uid']==receipt['uid']
        state={'state':d['state']}
        try:
            data=json.loads(client.get_object(Bucket='260010165',Key='behavior1k/ppo-degradation-audit-20261025/a01/status.json')['Body'].read())
            (HERE/'status.json').write_text(json.dumps(data,indent=2)+'\n')
            state.update({k:data.get(k) for k in ['stage','checkpoint','completed_samples','error_type']})
        except client.exceptions.NoSuchKey:pass
        if state!=previous:print(json.dumps(state),flush=True);previous=state
        if d['state'] in ['FAILED','SUSPENDED','TERMINATED','SUCCEEDED']:
            print('Terminal without audit result',flush=True);break
        time.sleep(20)
finally:client.close()

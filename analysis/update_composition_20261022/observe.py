"""One bounded read of this exact job and its private progress/result objects."""
import json
from pathlib import Path
import subprocess
import sys
import argparse
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--attempt',choices=['01','02','03','04','05','06','06h100'],default='01')
args=parser.parse_args()
work=HERE/('full_checkpoint_'+args.attempt)
intent=json.loads((work/'submission.json').read_text())
response=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',intent['job_name'],
    '--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=45)
if response.returncode:
    raise SystemExit('Job observation failed; no terminal state inferred')
job=json.loads(response.stdout)
assert job['uid']==intent['uid']
status={k:job.get(k) for k in ('uid','name','state','start_time','complete_time')}
status['observed_at']=datetime.now(timezone.utc).isoformat()
(work/'job_status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status))
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
from botocore.exceptions import ClientError
client=aoss_client()
try:
    for name in ('progress.json','result.json'):
        try:
            item=client.get_object(Bucket=intent['bucket'],Key=intent['prefix']+name)
        except ClientError as error:
            if error.response['Error']['Code'] in ('NoSuchKey','404','NotFound'):
                continue
            raise
        try:
            raw=item['Body'].read(128*1024+1)
        finally:item['Body'].close()
        assert len(raw)<=128*1024
        data=json.loads(raw)
        (work/name).write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(artifact=name,**{k:v for k,v in data.items()
            if k in ('stage','passed','worker_returncode','elapsed_seconds','peak_gpu_bytes','error_type','error')})))
finally:client.close()

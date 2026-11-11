import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
c=aoss_client();records=[]
for phase in ['archive','release']:
 p=HERE/(phase+'_cpu_submission.json')
 if not p.exists():continue
 r=json.loads(p.read_text())
 try:
  response=c.get_object(Bucket='260010165',Key=r['result_key'])
  try:value=json.loads(response['Body'].read())
  finally:response['Body'].close()
 except c.exceptions.NoSuchKey:records.append({'phase':phase,'state':'not_finished'});continue
 (HERE/(phase+'_cpu_result.json')).write_text(json.dumps(value,indent=2)+'\n');records.append({'phase':phase,'result':value})
print(json.dumps(records));c.close()

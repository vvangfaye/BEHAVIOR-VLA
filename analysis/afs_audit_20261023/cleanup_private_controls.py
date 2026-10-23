import json,sys
from pathlib import Path
from botocore.exceptions import ClientError
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
plan=json.loads((HERE/'storage_plan.json').read_text());prefix='behavior1k/afs-cleanup/'+plan['archive_id']+'/private/';keys=[]
for name in ['archive_cpu_submission_failed_create.json','archive_cpu_submission_attempt2.json','archive_cpu_submission.json','release_cpu_submission.json']:
 p=HERE/name
 if p.exists():
  key=json.loads(p.read_text()).get('private_config_key')
  if key:assert key.startswith(prefix);keys.append(key)
c=aoss_client()
for key in sorted(set(keys)):
 c.delete_object(Bucket='260010165',Key=key)
 try:c.head_object(Bucket='260010165',Key=key)
 except ClientError as error:assert error.response['ResponseMetadata']['HTTPStatusCode']==404
 else:raise RuntimeError('Private execution config still exists')
(HERE/'private_controls_cleanup.json').write_text(json.dumps({'deleted_keys':sorted(set(keys)),'verified_absent':True},indent=2)+'\n');print('Removed and verified',len(set(keys)),'temporary signed execution configs');c.close()

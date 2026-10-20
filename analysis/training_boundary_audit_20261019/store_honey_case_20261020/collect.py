#!/usr/bin/env python3
"""Copy four existing, hash-identified AFS artifacts via one bounded CPU job."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CLI='/home/faye/.local/bin/sco'
NAME='b1k-storehoney-audit-1020'
TEMPLATE='b1k-confirm-video-review-1017'
UID='3ce7c2e990de4d548f1111f547b1bd26'
BUCKET='260010165'
PREFIX='behavior1k/store-honey-audit-20261020/'
IMAGE='registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'
SPEC='n6ls.iu.i40.8c32g'
MOUNT='01a04263-91e5-7603-bc01-c67e503da6b5'


def cli(args, timeout=45):
    result=subprocess.run([CLI,*args],capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('SCO return code '+str(result.returncode))
    return json.loads(result.stdout)


def save(path, value, exclusive=False):
    if exclusive:
        with path.open('x') as stream:
            json.dump(value,stream,indent=2);stream.write('\n')
    else:
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(path)


WORKER='''import hashlib,json,urllib.request
from pathlib import Path
config=__CONFIG__
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
completed=[]
try:
 for row in config['files']:
  path=Path(row['path'])
  assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
  content=path.read_bytes()
  assert hashlib.sha256(content).hexdigest()==row['sha256']
  req=urllib.request.Request(row['url'],data=content,method='PUT')
  with opener.open(req,timeout=60) as response: assert response.status in (200,201,204)
  completed.append({key:row[key] for key in ('relative_path','bytes','sha256')})
 receipt={'complete':True,'source_sha256_verified':True,'files':completed,'simulation':False,'training':False}
 with opener.open(urllib.request.Request(config['receipt_url'],data=json.dumps(receipt).encode(),method='PUT'),timeout=60) as response:
  assert response.status in (200,201,204)
 print('FOUR_EXISTING_ARTIFACTS_COPIED',flush=True)
except Exception as error:
 print(json.dumps({'failed':True,'error_type':type(error).__name__}),flush=True)
 raise SystemExit(1)
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    campaign=PROJECT/'analysis/retraining_20261016_quality_taskbalanced_v1/confirmation_selected_vs_baseline'
    rows=[]
    for role in ('baseline','candidate'):
        report=json.loads((campaign/'reports'/f'{role}-15.json').read_text())
        assert report['record']['task']=='store_honey' and report['record']['instance_id']==140
        assert report['state']=='accepted'
        for row in report['artifacts']:
            if row['path'].endswith(('.json','.mp4')) and '/results/' in row['path']:
                rows.append({**row,'relative_path':role+'/'+Path(row['path']).name})
    assert len(rows)==4 and sum(row['bytes'] for row in rows)==30805309
    plan={'authorization':'Standing project authorization: 以后都批准',
          'job_name':NAME,'template':TEMPLATE,'template_uid':UID,'files':rows,
          'source':'Existing accepted independent confirmation artifacts; selected because this is the sole observed success loss.',
          'resource':{'cpu':8,'memory_gib':32,'gpu':0,'workers':1,'quota':'spot','maximum_seconds':600},
          'training':False,'simulation':False,'destination_bucket':BUCKET,'destination_prefix':PREFIX,
          'interpretation':'One post hoc mechanism case; not an estimate of population regression or a repeatability test.'}
    if not (HERE/'plan.json').exists():save(HERE/'plan.json',plan,exclusive=True)
    else:assert json.loads((HERE/'plan.json').read_text())==plan
    if not args.submit:
        print(json.dumps({'preflight':'local_passed','files':len(rows),'bytes':sum(row['bytes'] for row in rows)}));return
    intent=HERE/'submission.json'
    state={'submitted':False,'job_name':NAME,'created_at_utc':datetime.now(timezone.utc).isoformat()}
    save(intent,state,exclusive=True)
    client=None
    try:
        template=cli(['acp','jobs','describe',TEMPLATE,'--workspace-name=share-space','-o','json'])
        assert template['uid']==UID and template['ownership']['user_name']=='260010165'
        assert template['resource_pool']['name']=='share-cluster' and template['scheduling']['quota_type']=='SPOT'
        assert len(template['roles'])==1 and len(template['roles'][0]['resource_spec'])==1
        role=template['roles'][0]; spec=role['resource_spec'][0]
        assert role['image_path']==IMAGE and role['total_replicas']==1 and spec['name']==SPEC
        assert spec['limits']=={'cpu':'8','memory':'32'} and spec['replicas']==1
        assert any(m['id']==MOUNT and m['mount_path']=='/mnt/afs' for m in template['mount'])
        sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
        from video_eval_adapter import aoss_client
        client=aoss_client()
        config={'files':[{**row,'url':client.generate_presigned_url('put_object',Params={'Bucket':BUCKET,'Key':PREFIX+row['relative_path']},ExpiresIn=3600)} for row in rows],
                'receipt_url':client.generate_presigned_url('put_object',Params={'Bucket':BUCKET,'Key':PREFIX+'receipt.json'},ExpiresIn=3600)}
        command='timeout --signal=TERM --kill-after=30 570 python3 -u -c '+shlex.quote(WORKER.replace('__CONFIG__',repr(config)))
        state.update(submitted=True,startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        save(intent,state)
        try:
            result=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name='+TEMPLATE,'--name='+NAME,'--job-name='+NAME,'--worker-spec='+SPEC,
                '--worker-nodes=1','--quota-type=spot','--retry-times=0','--enable-fault-tolerance=false',
                '--enable-anomaly-detection=false','--storage-mount='+MOUNT+':/mnt/afs',
                '--container-image-url='+IMAGE,'--command='+command],capture_output=True,text=True,timeout=90)
            state['cli_returncode']=result.returncode
        except Exception as error:
            state['submission_observation_error_type']=type(error).__name__
        actual=cli(['acp','jobs','describe',NAME,'--workspace-name=share-space','-o','json'])
        assert actual['name']==NAME and actual['ownership']['user_name']=='260010165'
        actual_role=actual['roles'][0]
        assert len(actual['roles'])==1 and actual_role['total_replicas']==1
        assert len(actual_role['resource_spec'])==1
        observed_spec=actual_role['resource_spec'][0]
        assert {k:v for k,v in observed_spec.items() if k!='replicas'}=={k:v for k,v in spec.items() if k!='replicas'}
        # Platform reports current replicas=0 while the desired worker queues.
        assert observed_spec['replicas'] in (0,1)
        assert actual_role['startup_script']==command and actual_role['image_path']==IMAGE
        state.update(uid=actual['uid'],state=actual['state'],actual_resource_and_command_verified=True)
    except Exception as error:
        state.update(error_type=type(error).__name__)
    finally:
        if client is not None:client.close()
        save(intent,state)
    print(json.dumps(state))


if __name__=='__main__':
    main()

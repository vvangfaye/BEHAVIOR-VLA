"""Prepare one scoped macpro CPU continuation. Cloud export bytes stay immutable."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
PREP=WORKSPACE/'analysis/retraining_20261015_round3_v2'
sys.path.insert(0,str(PREP))
os.environ['BEHAVIOR_PREP_SIGNING_SSH']='macpro'
import transfer_pipeline as old


def main():
    deadline=json.loads((PREP/'pipeline_launch.json').read_text())['deadline_epoch']
    assert time.time()<deadline and not (PREP/'STOP').exists()
    transfer=json.loads((PREP/'transfer_launch.json').read_text())
    obs=old.describe(transfer)
    assert obs['uid']=='b41c7100a14643e6ae244b693cc19596' and obs['state']=='FAILED'
    accepted={p['source_key'] for p in map(lambda p:json.loads(p.read_text()),(PREP/'destination_receipts').glob('*.json'))}
    proofs=[p for p in map(lambda p:json.loads(p.read_text()),(PREP/'cloud_receipts').glob('*.json')) if p['source_key'] not in accepted]
    assert len(accepted)==16 and len(proofs)==66
    public=old.aoss_client();internal=old.internal_client(public)
    with ThreadPoolExecutor(max_workers=4) as pool:
        items=list(pool.map(lambda p:old.unsigned(p,public,internal),proofs))
    # A Mac must use the public AOSS endpoint, never the cloud-only internal DNS.
    for item in items:
        item['source']['signed_get_url']=public.generate_presigned_url('get_object',
            Params={'Bucket':'260010165','Key':item['source']['key']},ExpiresIn=129600)
        item['receipt_put']=public.generate_presigned_url('put_object',
            Params={'Bucket':'260010165','Key':old.PREFIX+'receipts/'+item['identity']+'.json'},ExpiresIn=129600)
    items=old.remote(dict(operation='sign',items=items))
    queue=dict(items=items,deadline_epoch=deadline,training_allowed=False,route='compact_exports_via_macpro_to_training_OSS')
    private=HERE/'private';private.mkdir(exist_ok=True);private.chmod(0o700)
    (private/'macpro_queue.json').write_text(json.dumps(queue));(private/'macpro_queue.json').chmod(0o600)
    files={}
    for name,path in [('direct_oss_worker.py',WORKSPACE/'BEHAVIOR-VLAs/sco/direct_oss_worker.py'),
                      ('prepare_retraining_data.py',WORKSPACE/'BEHAVIOR-VLAs/sco/prepare_retraining_data.py'),
                      ('subtask_trajectory.py',WORKSPACE/'BEHAVIOR-VLAs/scripts/subtask_trajectory.py')]:
        body=path.read_bytes()
        if name=='direct_oss_worker.py':
            body=body.replace(b"'transfer_route':'source_cloud_direct_to_OSS'",b"'transfer_route':'compact_exports_via_macpro_to_training_OSS'")
        files['code/'+name]=base64.b64encode(body).decode()
    files['macpro_compact_transfer.py']=base64.b64encode((HERE/'macpro_compact_transfer.py').read_bytes()).decode()
    files['private_queue.json']=base64.b64encode(json.dumps(queue).encode()).decode()
    deploy='''import base64,json,os,subprocess,sys,time
from pathlib import Path
data=json.load(sys.stdin)
root=Path('/Users/faye/Library/Caches/behavior-comet-rft-20261015/compact-transfer');root.mkdir(parents=True,exist_ok=True);root.chmod(0o700)
launch=root/'launch.json'
if launch.exists():
 previous=json.loads(launch.read_text())
 try:os.kill(previous['pid'],0)
 except ProcessLookupError:pass
 else:print(json.dumps(dict(previous,already_active=True)));sys.exit(0)
for name,value in data['files'].items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));p.chmod(0o600)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
python='/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python'
with (root/'worker.log').open('a') as out:
 child=subprocess.Popen([python,str(root/'macpro_compact_transfer.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=child.pid,root=str(root),started_at=time.time(),expected=66,training_started=False)
launch.write_text(json.dumps(result));print(json.dumps(result))
'''
    args=['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=8','macpro',shlex.join(['python3','-c',deploy])]
    result=subprocess.run(args,input=json.dumps({'files':files}),text=True,capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError('Scoped macpro continuation deployment failed')
    receipt=json.loads(result.stdout)
    old.save(HERE/'compact_transfer_launch.json',dict(receipt,old_cloud_task=obs,deadline_epoch=deadline,
              scope='only66 missing converted archives; keep prior16 receipts;no raw archives or training'))
    print(json.dumps(receipt))


if __name__=='__main__':main()

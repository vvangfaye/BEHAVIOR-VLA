"""Build/submit a single bounded restore probe; signed URLs stay in memory."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
JOB='b1k-restore-probe-1020-p01'
PREFIX='behavior1k/snapshot-restore-audit-20261020/'+JOB+'/'
REMOTE='/mnt/afs/260010165/behavior_1k/training_boundary_audit_20261019/snapshot_probe_p01'
FILES=['plan.json','driver.py','worker.py','helpers.py','runtime_environment.py']


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    plan=json.loads((HERE/'plan.json').read_text())
    assert plan['job_name']==JOB and plan['model_calls']==0 and plan['training'] is False
    assert plan['max_job_seconds']==2700 and plan['worker_timeout_seconds']==2400
    for name,expected in plan['bundle_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w') as archive:
        for name in FILES:
            body=(HERE/name).read_bytes();info=tarfile.TarInfo(name)
            info.size=len(body);info.mode=0o600;info.mtime=0
            archive.addfile(info,io.BytesIO(body))
    body=gzip.compress(raw.getvalue(),mtime=0)
    assert len(body)<65536
    digest=hashlib.sha256(body).hexdigest()
    path=HERE/'bundle.tar.gz'
    if path.exists():assert path.read_bytes()==body
    else:path.write_bytes(body)
    return {'job_name':JOB,'bundle_bytes':len(body),'bundle_sha256':digest,
            'bundle_key':PREFIX+'bundle-'+digest+'.tar.gz','result_key':PREFIX+'result.tar.gz',
            'gpu_count':1,'quota_type':'spot','max_job_seconds':2700,
            'model_calls':0,'training':False,'simulation':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    receipt=prepare()
    if not args.submit:
        print(json.dumps(receipt));return
    intent=HERE/'submission_intent.json'
    with intent.open('x') as stream:
        json.dump({**receipt,'state':'preparing','submission_attempted':False,
                   'created_at_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
    spec=importlib.util.spec_from_file_location('existing_submit',HERE.parent/'calibration_screen_20261020/submit_screen.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    state=json.loads(intent.read_text())
    client=aoss_client()
    try:
        # Inspect the same approved single-worker template freshly; omit startup text.
        current=subprocess.run([module.CLI,'acp','jobs','describe',module.TEMPLATE,
                 '--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40)
        template=json.loads(current.stdout)
        assert template['uid']==module.TEMPLATE_UID
        assert template['ownership']['user_name']=='260010165'
        assert template['resource_pool']['name']==module.POOL
        assert len(template['roles'])==1 and template['roles'][0]['total_replicas']==1
        assert template['roles'][0]['image_path']==module.IMAGE
        assert template['roles'][0]['resource_spec'][0]['name']==module.WORKER_SPEC
        write(HERE/'template_receipt.json',{'uid':template['uid'],'name':template['name'],
            'pool':template['resource_pool']['name'],'image':template['roles'][0]['image_path'],
            'worker_spec':module.WORKER_SPEC,'checked_at_utc':datetime.now(timezone.utc).isoformat()})
        body=(HERE/'bundle.tar.gz').read_bytes()
        client.put_object(Bucket=module.BUCKET,Key=receipt['bundle_key'],Body=body,
                          Metadata={'sha256':receipt['bundle_sha256']})
        stream=client.get_object(Bucket=module.BUCKET,Key=receipt['bundle_key'])['Body']
        try:readback=stream.read(len(body)+1)
        finally:stream.close()
        assert readback==body
        config={'root':REMOTE,'get':client.generate_presigned_url('get_object',Params={'Bucket':module.BUCKET,'Key':receipt['bundle_key']},ExpiresIn=604800),
                'put':client.generate_presigned_url('put_object',Params={'Bucket':module.BUCKET,'Key':receipt['result_key']},ExpiresIn=604800),
                'bytes':len(body),'sha':receipt['bundle_sha256']}
        bootstrap='''import hashlib,io,os,pathlib,subprocess,sys,tarfile,urllib.request
c=CONFIG
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(c['get'],timeout=90) as response: body=response.read(c['bytes']+1)
assert len(body)==c['bytes'] and hashlib.sha256(body).hexdigest()==c['sha']
root=pathlib.Path(c['root']);root.mkdir(parents=True,exist_ok=False)
dest=root/'bundle';dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as archive:
 members=archive.getmembers()
 assert {m.name for m in members}==set(FILES) and len(members)==len(FILES)
 for m in members:
  assert m.isfile() and m.size<1000000
  with archive.extractfile(m) as src: content=src.read(m.size+1)
  assert len(content)==m.size
  (dest/m.name).write_bytes(content)
env=dict(os.environ,RESTORE_PROBE_PUT_URL=c['put'])
raise SystemExit(subprocess.call([sys.executable,str(dest/'driver.py')],env=env))
'''.replace('CONFIG',repr(config),1).replace('FILES',repr(FILES))
        encoded=base64.b64encode(zlib.compress(bootstrap.encode())).decode()
        wire='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
        command='timeout --signal=TERM --kill-after=30s 2670s python3 -c '+shlex.quote(wire)
        state.update(submission_attempted=True,upload_readback_verified=True,
                     startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent,state)
        argv=[module.CLI,'acp','jobs','copy','--workspace-name=share-space',
              '--copy-job-name='+module.TEMPLATE,'--name='+JOB,'--job-name='+JOB,
              '--worker-spec='+module.WORKER_SPEC,'--worker-nodes=1','--quota-type=spot',
              '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
              '--storage-mount='+module.MOUNT_ID+':/mnt/afs','--container-image-url='+module.IMAGE,
              '--env=NVIDIA_DRIVER_CAPABILITIES:all','--command='+command]
        try:
            result=subprocess.run(argv,capture_output=True,text=True,timeout=120)
            state['cli_returncode']=result.returncode
        except Exception as error:state['submission_error_type']=type(error).__name__
        module.JOB=JOB
        module.reconcile(state)
    except Exception as error:
        state.update(state='pre_submission_failed' if not state['submission_attempted'] else 'requires_exact_name_query',
                     error_type=type(error).__name__)
    finally:
        client.close();write(intent,state)
    print(json.dumps({k:state.get(k) for k in ['job_name','name','uid','state','error_type','cli_returncode']}))


if __name__=='__main__':main()

"""Single, finite local checkpoint trigger + independent Spot evaluation queue.

Never invokes AIHub train/stop/edit or any other campaign's creation code.
"""
from concurrent.futures import ThreadPoolExecutor
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / 'BEHAVIOR-VLAs'
sys.path.insert(0, str(REPO / 'sco'))
from video_eval_adapter import aoss_client, cli_json, environment, CLI

TERMINAL = {'SUCCEEDED','FAILED','STOPPED','CANCELLED','DELETED','SUSPENDED'}
ACTIVE = {'RUNNING','QUEUED','QUEUEING','PENDING','CREATING','STARTING','SCHEDULING','SUBMITTED','INITIALIZING','RESTARTING','UNSPECIFIED','UNKNOWN'}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.chmod(tmp,0o600)
    tmp.replace(path)


def new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
    os.chmod(path,0o600)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_job(job, intent):
    expected_pool = 'computing-cluster-5090-01g' if intent['gpu'] else 'share-cluster'
    if (job['name'] != intent['name'] or job['ownership']['user_name'] != '260010165'
            or job['workspace_name'] != 'share-space' or job['scheduling']['quota_type'] != 'SPOT'
            or job['resource_pool']['name'] != expected_pool):
        raise ValueError('Exact own Spot job identity mismatch')
    if intent.get('uid') and intent['uid'] != job['uid']:
        raise ValueError('Job UID changed; never operate it')
    roles = job['roles']
    requested = sum(int(spec['limits'].get('nvidia.com/gpu',0))*(spec['replicas'] or role.get('total_replicas',1))
                    for role in roles for spec in role['resource_spec'])
    if requested != intent['gpu']:
        raise ValueError('Platform GPU request differs from finite intent')
    for role in roles:
        if role['image_path'] != intent['image']:
            raise ValueError('Platform image differs')
    return {'name':job['name'],'uid':job['uid'],'state':job['state'],
            'observed_at':time.time(),'requested_gpus':requested}


def observe(intent):
    job = cli_json(['acp','jobs','describe',intent['name'],'--workspace-name=share-space','-o','json'])
    return verify_job(job,intent)


def read_report(client, plan, id):
    from botocore.exceptions import ClientError
    key = plan['object_prefix'] + '/reports/' + id + '.json'
    try:
        raw = client.get_object(Bucket='260010165',Key=key)['Body'].read(16*1024**2)
        value = json.loads(raw)
        if value.get('id') != id or value.get('protocol_sha256') != plan['protocol_sha256']:
            raise ValueError('Worker report identity differs')
        if id.startswith(('baseline-','candidate-')):
            intent=load(ROOT/'intents'/(id+'.json'))
            if value.get('worker_bundle_sha256') != intent['worker_bundle_sha256']:
                raise ValueError('Evaluation overlay changed between paired conditions')
        return value
    except ClientError as error:
        if error.response['Error']['Code'] in ('NoSuchKey','404'):
            return None
        raise


def bootstrap(plan, payload, report_url):
    files = {p.name:p.read_text() for p in (ROOT/'frozen').glob('*.py') if p.name != 'remote_probe.py'}
    manifest = {k:sha_bytes(v.encode()) for k,v in files.items()}
    pinned = load(ROOT/'frozen_sha256.json')
    if manifest != {k:v for k,v in pinned.items() if k != 'remote_probe.py'}:
        raise ValueError('Worker source changed after authorization freeze')
    packed = base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
    packed_payload = base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()
    dest = plan['afs_root'] + '/workers/' + payload['id']
    source = '''import base64,hashlib,json,os,sys,time,traceback,urllib.request,zlib
from pathlib import Path
root=Path(DEST);root.mkdir(parents=True,exist_ok=False)
fd=os.open(str(root/'worker.log'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
files=json.loads(zlib.decompress(base64.b64decode(FILES)))
expected=HASHES
for name,text in files.items():
 assert hashlib.sha256(text.encode()).hexdigest()==expected[name]
 (root/name).write_text(text)
payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
sys.path.insert(0,str(root))
try:
 import worker
 result=worker.main(payload)
except BaseException as exc:
 result={'state':'failed','error_type':type(exc).__name__}
 traceback.print_exc()
result.update(id=payload['id'],protocol_sha256=payload['plan']['protocol_sha256'],finished_at=time.time(),worker_bundle_sha256=payload['worker_bundle_sha256'])
(root/'report.json').write_text(json.dumps(result,indent=2)+'\\n')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(4):
 try:
  with opener.open(urllib.request.Request(REPORT,data=json.dumps(result).encode(),method='PUT'),timeout=40) as response:pass
  break
 except OSError:
  if attempt==3:raise
  time.sleep(2**attempt)
print(json.dumps({'id':payload['id'],'state':result['state']}),flush=True)
if result['state']!='accepted':raise SystemExit(1)
'''
    values = {'DEST':dest,'FILES':packed,'HASHES':manifest,'PAYLOAD':packed_payload,'REPORT':report_url}
    import re
    native_startup = re.sub(r'\b(DEST|FILES|HASHES|PAYLOAD|REPORT)\b',lambda m:repr(values[m[0]]),source)
    if payload['mode'] != 'evaluate':
        return native_startup
    task,instance=payload['pair']
    config=dict(id=payload['id'],original_uid=None,checkpoint_step=10000,
                python=plan['runtime']+'/envs/b1k-eval/bin/python',
                worker_root=dest,episode_root=plan['afs_root']+'/episodes/'+payload['id'],
                monitor_root=plan['afs_root']+'/supervision/'+payload['id'],
                result_name=f'{task}_{instance}_0',
                child_source=base64.b64encode(zlib.compress(native_startup.encode(),9)).decode(),
                child_sha256=sha_bytes(native_startup.encode()),
                heartbeat_put_url=payload['heartbeat_put_url'],diagnostics_put_url=payload['diagnostics_put_url'])
    return (ROOT/'frozen/supervisor.py').read_text()+'\nraise SystemExit(supervisor('+repr(config)+'))\n'


def submit(client, plan, id, mode, **extra):
    path = ROOT/'intents'/ (id+'.json')
    if path.exists():
        raise ValueError('One creation per exact intent; reconcile existing name')
    if (ROOT/'STOP').exists() or time.time() >= plan['deadline_epoch']:
        raise ValueError('Finite campaign stopped/deadline passed')
    if mode == 'evaluate' and extra.get('role') != 'candidate':
        raise ValueError('Historical baseline is read-only; no new baseline rollout')
    gpu = int(mode == 'evaluate')
    name = plan['run_id'] + '-' + id
    bundle_sha=sha_bytes((ROOT/'frozen_sha256.json').read_bytes())
    payload = dict(id=id,mode=mode,job_name=name,plan=plan,worker_bundle_sha256=bundle_sha,**extra)
    if mode=='evaluate':
        for kind in ('heartbeat','diagnostics'):
            payload[kind+'_put_url']=client.generate_presigned_url('put_object',
                Params={'Bucket':'260010165','Key':plan['object_prefix']+'/telemetry/'+id+'/'+kind+'.json'},ExpiresIn=259200)
    def routes(identity):
        result={}
        for label,filename in [('json','original.json'),('video','rollout.mp4')]:
            key=plan['object_prefix']+'/artifacts/'+identity+'/'+filename
            result[label]=dict(key=key,
                put=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key,'IfNoneMatch':'*'},ExpiresIn=259200),
                get=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=259200))
        return result
    if mode=='evaluate':payload['artifact_routes']=routes(id)
    if mode=='preflight':
        payload['checkpoint_gate_put_url']=client.generate_presigned_url('put_object',
            Params={'Bucket':'260010165','Key':plan['object_prefix']+'/checkpoint_gate.json'},ExpiresIn=259200)
        payload['reuse_artifact_routes']={}
        for item in plan['reuse_candidates']:
            record=item['report']['record'];index=plan['pairs'].index([record['task'],record['instance_id']])
            identity=f'candidate-{index:03d}'
            payload['reuse_artifact_routes'][identity]=routes(identity)
    report_key = plan['object_prefix']+'/reports/'+id+'.json'
    report_url = client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':report_key},ExpiresIn=259200)
    startup = bootstrap(plan,payload,report_url)
    startup_hash = sha_bytes(startup.encode())
    # Preserve a private, immutable source of the exact startup without printing
    # signed URLs or embedding any permanent credentials in remote workers.
    private = ROOT/'private'/ (id+'.startup.py')
    private.write_text(startup);os.chmod(private,0o600)
    intent = dict(id=id,name=name,mode=mode,gpu=gpu,image=plan['image'],created_intent_at=time.time(),
                  startup_sha256=startup_hash,creation_attempted=False,uid=None,attempts=1,
                  protocol_sha256=plan['protocol_sha256'],report_key=report_key)
    intent['worker_bundle_sha256']=bundle_sha
    new(path,intent)
    python = plan['runtime']+'/envs/b1k-eval/bin/python'
    command = 'timeout --signal=TERM --kill-after=30 86400 '+shlex.quote(python)+' -c '+shlex.quote(startup)
    common = ['--workspace-name=share-space','--name='+name,'--job-name='+name,
              '--worker-nodes=1','--quota-type=spot','--retry-times=0',
              '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
              '--container-image-url='+plan['image'],'--command='+command]
    if gpu:
        args = [CLI,'acp','jobs','copy','--copy-job-name='+plan['gpu_copy_template'],'--worker-spec=n12lp.nn.i10a.1',
                '--env=NVIDIA_DRIVER_CAPABILITIES:all']+common
    else:
        args = [CLI,'acp','jobs','create','--aec2-name=share-cluster','--training-framework=pytorch',
                '--worker-spec=n6ls.iu.i40.8c32g']+common
    intent.update(creation_attempted=True,creation_started_at=time.time(),state='creation_unknown')
    save(path,intent)
    try:
        cp = subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=90)
        intent['cli_exit_code'] = cp.returncode
        if cp.returncode:
            save(ROOT/'private'/(id+'.cli_failure.json'),dict(exit_code=cp.returncode,stdout=cp.stdout,stderr=cp.stderr))
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    save(path,intent)
    # Never repeat the create call, including after a timeout/nonzero return.
    obs = observe(intent)
    intent.update(uid=obs['uid'],state=obs['state'],verified_at=obs['observed_at'])
    save(path,intent)
    return intent


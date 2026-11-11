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
REPO = ROOT.parents[2] / 'BEHAVIOR-VLAs'
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
    config=dict(id=payload['id'],original_uid=None,checkpoint_step=1000,
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
    name = 'b1k-qbal1k-1016-' + id
    bundle_sha=sha_bytes((ROOT/'frozen_sha256.json').read_bytes())
    payload = dict(id=id,mode=mode,job_name=name,plan=plan,worker_bundle_sha256=bundle_sha,**extra)
    if mode=='evaluate':
        for kind in ('heartbeat','diagnostics'):
            payload[kind+'_put_url']=client.generate_presigned_url('put_object',
                Params={'Bucket':'260010165','Key':plan['object_prefix']+'/telemetry/'+id+'/'+kind+'.json'},ExpiresIn=172800)
    report_key = plan['object_prefix']+'/reports/'+id+'.json'
    report_url = client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':report_key},ExpiresIn=172800)
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
    command = 'timeout --signal=TERM --kill-after=30 28800 '+shlex.quote(python)+' -c '+shlex.quote(startup)
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
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    save(path,intent)
    # Never repeat the create call, including after a timeout/nonzero return.
    obs = observe(intent)
    intent.update(uid=obs['uid'],state=obs['state'],verified_at=obs['observed_at'])
    save(path,intent)
    return intent


def remote_probe(transfer=False):
    remote = ('source "/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs/nebula/config.sh"; '
              'export OSS_ENDPOINT=https://oss-cn-zhangjiakou.aliyuncs.com; '
              + ('export PAIRED_TRANSFER_PROBE=1; ' if transfer else '') +
              'export PYTHONPATH="/Users/faye/code/Behavior-1k Challenge/BEHAVIOR-VLAs"; '
              '"/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python" -')
    cp = subprocess.run(['ssh','-S','none','-o','ConnectTimeout=8',
         '-o','HostKeyAlias=MBP16.local','-o','HostName=fe80::100e:8e6c:1f72:2de%%en7','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','-o','BatchMode=yes','macpro',remote],
         input=(ROOT/'frozen/remote_probe.py').read_text(),capture_output=True,text=True,timeout=120)
    if cp.returncode:
        category = 'ssh_connection_failed' if cp.returncode == 255 else 'checkpoint_probe_failed'
        if 'Missing, empty, or oversized metadata:' in cp.stderr:
            category = 'checkpoint_metadata_read_limit'
        save(ROOT/'probe_failure.json', dict(observed_at=time.time(), category=category,
             exit_code=cp.returncode, training_modified=False, evaluation_creation=False))
        raise RuntimeError(category + '; no evaluation creation')
    result = json.loads(cp.stdout)
    save(ROOT/('private/transfer_probe.json' if transfer else 'private/last_probe.json'),result)
    public = {k:v for k,v in result.items() if k not in ('files',)}
    save(ROOT/('transfer_observation.json' if transfer else 'checkpoint_observation.json'),public)
    return result


def reconcile(client, plan):
    intents = [load(p) for p in sorted((ROOT/'intents').glob('*.json'))]
    def one(intent):
        previous = ROOT/'observations'/(intent['id']+'.json')
        report = read_report(client,plan,intent['id'])
        if report:
            save(ROOT/'reports'/(intent['id']+'.json'),report)
        if previous.exists() and load(previous)['state'] in TERMINAL:
            obs = load(previous)
        else:
            obs = observe(intent)
            save(previous,obs)
        if intent['gpu'] and obs['state'] in TERMINAL | {'RUNNING'}:
            from botocore.exceptions import ClientError
            for kind in ('heartbeat','diagnostics'):
                try:
                    response=client.get_object(Bucket='260010165',Key=plan['object_prefix']+'/telemetry/'+intent['id']+'/'+kind+'.json')
                    try:
                        telemetry=json.loads(response['Body'].read(8*1024**2))
                    finally:
                        response['Body'].close()
                    (ROOT/'telemetry').mkdir(exist_ok=True)
                    save(ROOT/'telemetry'/(intent['id']+'.'+kind+'.json'),telemetry)
                except ClientError as error:
                    if error.response['Error']['Code'] not in ('404','NoSuchKey'):
                        raise
        if not intent.get('uid'):
            intent.update(uid=obs['uid'],state=obs['state'])
            save(ROOT/'intents'/(intent['id']+'.json'),intent)
        return dict(intent=intent,observation=obs,report=report)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one,intents))
    return {row['intent']['id']:row for row in rows}


def pending_and_active(rows):
    gpu = [r for r in rows.values() if r['intent']['gpu']]
    pending = sum(r['observation']['state'] not in TERMINAL | {'RUNNING'} for r in gpu)
    active = sum(r['observation']['state'] not in TERMINAL for r in gpu)
    return pending,active


def accepted(rows, id):
    return bool(id in rows and rows[id]['report'] and rows[id]['report']['state']=='accepted')


def finalize(plan, rows):
    sys.path.insert(0,str(ROOT/'frozen'))
    import scorer
    files=[('baseline_accepted.json','baseline_accepted_sha256'),('step5000_candidate_accepted.json','previous_candidate_accepted_sha256'),('step10000_candidate_accepted.json','additional_candidate_accepted_sha256'),('step20000_candidate_accepted.json','extra_candidate_accepted_sha256')]
    history=[]
    for file,field in files:
        if sha_bytes((ROOT/file).read_bytes())!=plan[field]:raise ValueError('Frozen historical results changed')
        history.append(load(ROOT/file))
    reports=[rows[f'candidate-{i:02d}']['report'] for i in range(20)]
    identities={r['checkpoint_manifest_sha256'] for r in reports};bundles={r['worker_bundle_sha256'] for r in reports}
    if len(identities)!=1 or len(bundles)!=1:raise ValueError('Candidate identity differs across20')
    candidate=dict(kind='artifact_accepted_historical_reuse_regression',policy_seed=0,environment_seed=0,protocol_sha256=plan['protocol_sha256'],checkpoint_manifest_sha256=identities.pop(),acceptance_manifest_sha256=sha_bytes(json.dumps(reports,sort_keys=True).encode()),records=[r['record'] for r in reports],completed_updates=1000,run_id=plan['run_id'])
    scorer.validate(candidate,plan['pairs'],'candidate');save(ROOT/'candidate_accepted.json',candidate)
    import importlib.util
    spec=importlib.util.spec_from_file_location('quality_compare', ROOT.parent/'compare_candidate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    save(ROOT/'comparison.json',module.compare(ROOT/'candidate_accepted.json', expected_updates=1000))


def tick(client, plan):
    rows=reconcile(client,plan)
    pending,active=pending_and_active(rows)
    status=dict(observed_at=time.time(),state='waiting_checkpoint1000',pending_gpu_jobs=pending,
        active_gpu_jobs=active,accepted_episodes=sum(accepted(rows,id) for id in rows if id.startswith('candidate-')),
        created_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values()),reused_baseline_episodes=20,
        running_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values() if r['observation']['state']=='RUNNING'),
        training_iid_unchanged=plan['training_iid'],training_target_unchanged=5000,training_target_completed=False,
        failures=[id for id,r in rows.items() if r['report'] and r['report']['state']=='failed'])
    save(ROOT/'status.json',status)
    if (ROOT/'STOP').exists() or time.time()>=plan['deadline_epoch']:
        status['state']='stopped_admission';save(ROOT/'status.json',status);return True
    if sha_bytes((ROOT/'baseline_accepted.json').read_bytes())!=plan['baseline_accepted_sha256']:
        raise ValueError('Frozen historical baseline changed')
    if (sha_bytes((ROOT/'step5000_candidate_accepted.json').read_bytes())!=plan['previous_candidate_accepted_sha256']
            or sha_bytes((ROOT/'step5000_comparison.json').read_bytes())!=plan['previous_comparison_sha256']):
        raise ValueError('Frozen5000 comparison evidence changed')
    if (sha_bytes((ROOT/'step10000_candidate_accepted.json').read_bytes())!=plan['additional_candidate_accepted_sha256']
            or sha_bytes((ROOT/'step10000_comparison.json').read_bytes())!=plan['additional_comparison_sha256']):
        raise ValueError('Frozen10000 comparison evidence changed')
    for ref in plan['reused_preflight']:
        raw=Path(ref['path']).read_bytes()
        if sha_bytes(raw)!=ref['sha256'] or json.loads(raw)['state']!='accepted':
            raise ValueError('Previously accepted common preflight changed')
    if 'export' not in rows:
        snap=remote_probe()
        status['completed_training_updates']=snap.get('completed_updates')
        status['candidate_checkpoint_committed']=snap.get('state')=='committed_checkpoint1000'
        save(ROOT/'status.json',status)
        if snap['state']=='committed_checkpoint1000':
            if snap['formal_contract']!=plan['formal_contract']:raise ValueError('Formal training contract changed')
            save(ROOT/'private/export_snapshot.json',snap)
            submit(client,plan,'export','export',snapshot=snap)
        return False
    if not accepted(rows,'export'):
        status['state']='exporting_checkpoint1000';save(ROOT/'status.json',status)
        if rows['export']['report']:raise ValueError('Checkpoint export failed; no automatic rerun')
        return False
    # Human requested parallel admission after the shared export/preflight.
    # Each existing intent remains immutable; candidate00 is not repeated.
    status['state']='parallel_candidate_evaluations'
    status['wait_for_completed_pilot']=False
    save(ROOT/'status.json',status)
    for index,pair in enumerate(plan['pairs']):
        id=f'candidate-{index:02d}'
        if id in rows:continue
        rows=reconcile(client,plan)
        pending,active=pending_and_active(rows)
        if pending>=plan['max_pending'] or active>=plan['max_active_gpu_jobs']:break
        submit(client,plan,id,'evaluate',role='candidate',pair=pair)
        rows=reconcile(client,plan)
    pending,active=pending_and_active(rows)
    status.update(observed_at=time.time(),pending_gpu_jobs=pending,active_gpu_jobs=active,
        running_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values() if r['observation']['state']=='RUNNING'),
        created_gpu_jobs=sum(r['intent']['gpu'] for r in rows.values()),
        accepted_episodes=sum(accepted(rows,id) for id in rows if id.startswith('candidate-')))
    save(ROOT/'status.json',status)
    ids=[f'candidate-{i:02d}' for i in range(20)]
    if all(accepted(rows,id) for id in ids):
        finalize(plan,rows);status.update(state='comparison_complete',accepted_episodes=20)
        save(ROOT/'status.json',status);return True
    if all(id in rows and rows[id]['observation']['state'] in TERMINAL for id in ids):
        grace=ROOT/'terminal_artifact_grace.json'
        if not grace.exists():save(grace,dict(started_at=time.time()))
        if time.time()-load(grace)['started_at']<180:
            status['state']='terminal_jobs_waiting_artifact_visibility';save(ROOT/'status.json',status);return False
        status['state']='finite_jobs_finished_with_missing_or_failed_artifacts'
        status['failed_platform_jobs']=[id for id in ids if rows[id]['observation']['state']!='SUCCEEDED']
        save(ROOT/'status.json',status);return True
    return False


def main():
    for name in ('private','intents','observations','reports'):
        (ROOT/name).mkdir(exist_ok=True)
    with (ROOT/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_bytes=(ROOT/'plan.json').read_bytes();plan=json.loads(plan_bytes)
        if (plan.get('execution_enabled') is not True or plan.get('launch_preflight_complete') is not True
                or not isinstance(plan.get('deadline_epoch'), (int,float))):
            raise ValueError('Prepared only: validate live template and set finite deadline before launch')
        frozen_bytes=(ROOT/'frozen_sha256.json').read_bytes()
        if (len(plan['pairs'])!=20 or len({tuple(p) for p in plan['pairs']})!=20 or plan['episodes']!=20
                or plan['baseline_reused_episodes']!=20 or plan['previous_reused_episodes']!=20
                or plan['candidate_step']!=1000 or plan['previous_candidate_step']!=5000
                or plan['maximum_attempts_per_episode']!=1 or plan['modify_or_stop_training']):
            raise ValueError('Scope differs from the human-authorized paired20')
        save(ROOT/'launch.json',dict(pid=os.getpid(),started_at=time.time(),plan_sha256=sha_bytes(plan_bytes),
                                    controller_sha256=sha_bytes(Path(__file__).read_bytes()),worker_bundle_sha256=sha_bytes(frozen_bytes),
                                    command=str(Path(__file__).resolve()),recurring_automation=False))
        client=aoss_client()
        while True:
            if (ROOT/'plan.json').read_bytes()!=plan_bytes:
                raise ValueError('Plan changed; stop admission')
            if (ROOT/'frozen_sha256.json').read_bytes()!=frozen_bytes:
                raise ValueError('Frozen evaluator overlay changed; stop admission')
            try:
                if tick(client,plan):
                    return
            except Exception as exc:
                # Observation failure may clear after networking returns, but
                # no retry can create a second job because each intent persists.
                save(ROOT/'last_error.json',dict(observed_at=time.time(),error_type=type(exc).__name__,
                     message=str(exc)[:400] if not isinstance(exc,subprocess.SubprocessError) else 'subprocess error; private output withheld'))
                print(json.dumps({'time':time.time(),'error_type':type(exc).__name__}),flush=True)
            time.sleep(plan['poll_seconds'])


if __name__=='__main__':
    main()

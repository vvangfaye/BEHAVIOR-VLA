"""Submit/observe only the explicitly authorized thirty repeatability episodes."""
import base64
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client,environment,CLI

TERMINAL={'SUCCEEDED','FAILED','STOPPED','CANCELLED','DELETED','SUSPENDED'}


def load(p):return json.loads(Path(p).read_text())
def digest(raw):return hashlib.sha256(raw).hexdigest()


def save(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if exclusive:
        with path.open('xb') as stream:stream.write(raw)
    else:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path)
    path.chmod(0o600)


def cli(args,timeout=60):
    r=subprocess.run([CLI]+args,env=environment(),capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError('SCO query failed: exit '+str(r.returncode))
    return json.loads(r.stdout)


def verify(job,name,plan):
    assert job['name']==name and job['workspace_name']==plan['workspace']
    assert job['ownership']['user_name']=='260010165'
    assert job['resource_pool']['name']==plan['pool']
    assert job['scheduling']['quota_type']=='SPOT'
    gpu=sum(int(spec['limits'].get('nvidia.com/gpu',0))*(spec['replicas'] or role.get('total_replicas',1))
            for role in job['roles'] for spec in role['resource_spec'])
    assert gpu==1
    assert all(role['image_path']==plan['image'] for role in job['roles'])
    return dict(name=name,uid=job['uid'],state=job['state'],observed_at=time.time(),requested_gpus=gpu)


def observe(intent,plan):
    job=cli(['acp','jobs','describe',intent['name'],'--workspace-name='+plan['workspace'],'-o','json'])
    result=verify(job,intent['name'],plan)
    if intent.get('uid'):assert intent['uid']==result['uid']
    return result


def validate(plan):
    protocol=load(ROOT/'protocol.json')
    assert digest(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode())==plan['protocol_sha256']
    assert len(plan['pairs'])==30 and len(plan['groups'])==10
    assignments=[m['index'] for g in plan['groups'] for m in g['members']]
    assert sorted(assignments)==list(range(30))
    for g in plan['groups']:
        assert len(g['members'])==3
        assert {m['repeat'] for m in g['members']}=={0,1,2}
        assert len({tuple(plan['pairs'][m['index']]) for m in g['members']})==1
    assert plan['policy_seed']==plan['environment_seed']==0
    assert {p.name:digest(p.read_bytes()) for p in (ROOT/'frozen').glob('*.py')}==load(ROOT/'frozen_sha256.json')
    template=cli(['acp','jobs','describe',plan['template'],'--workspace-name='+plan['workspace'],'-o','json'])
    observed=verify(template,plan['template'],plan)
    assert observed['uid']==plan['template_uid']
    save(ROOT/'template_verified.json',observed)


def submit(plan):
    validate(plan)
    assert time.time()<plan['creation_deadline_epoch']
    client=aoss_client()
    if not (ROOT/'startup_hashes.json').exists():
        spec=importlib.util.spec_from_file_location('repeat_assemble',ROOT/'assemble.py')
        assembly=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembly)
        urls={}
        def signed(key):return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=604800)
        for g in plan['groups']:
            urls[g['id']]={'report':signed(plan['object_prefix']+'/reports/'+g['id']+'.json')}
            for m in g['members']:
                identity=f"baseline-{m['index']:02d}"
                urls[identity]={kind:signed(plan['object_prefix']+'/'+('reports/'+identity+'.json' if kind=='report' else 'telemetry/'+identity+'/'+kind+'.json')) for kind in ['report','heartbeat','diagnostics']}
        files={p.name:p.read_text() for p in (ROOT/'frozen').glob('*.py')}
        data=dict(plan=plan,files=files,urls=urls,bundle_hash=digest((ROOT/'frozen_sha256.json').read_bytes()))
        startups=assembly.assemble(data)
        hashes={}
        for identity,source in startups.items():
            path=ROOT/'private'/f'{identity}.py';path.parent.mkdir(exist_ok=True);path.write_text(source);path.chmod(0o600)
            compile(source,str(path),'exec');hashes[identity]=digest(source.encode())
        save(ROOT/'startup_hashes.json',hashes,True)
        save(ROOT/'startup_metadata.json',dict(assembler_sha256=digest((ROOT/'assemble.py').read_bytes()),
             files=load(ROOT/'frozen_sha256.json'),created_at=time.time(),expiration_seconds=604800))
    hashes=load(ROOT/'startup_hashes.json')
    for group in plan['groups']:
        path=ROOT/'intents'/f"{group['id']}.json"
        if path.exists():
            intent=load(path)
            state=observe(intent,plan);intent.update(state);save(path,intent)
            print(json.dumps(state),flush=True)
            continue
        source=(ROOT/'private'/f"{group['id']}.py").read_bytes()
        assert digest(source)==hashes[group['id']]
        compressed=zlib.compress(source,9)
        key=plan['object_prefix']+'/private/startups/'+group['id']+'.py.zlib'
        compressed_sha=digest(compressed)
        client.put_object(Bucket='260010165',Key=key,Body=compressed,
                          Metadata={'sha256':compressed_sha},ContentType='application/octet-stream')
        uploaded=client.head_object(Bucket='260010165',Key=key)
        assert uploaded['ContentLength']==len(compressed) and uploaded['Metadata']['sha256']==compressed_sha
        url=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=604800)
        launcher=('import hashlib,urllib.request,zlib;'
                  'o=urllib.request.build_opener(urllib.request.ProxyHandler({}));'
                  'b=o.open('+repr(url)+',timeout=120).read(16*1024*1024);'
                  'assert hashlib.sha256(b).hexdigest()=='+repr(compressed_sha)+';'
                  's=zlib.decompress(b);assert hashlib.sha256(s).hexdigest()=='+repr(hashes[group['id']])+';'
                  'exec(compile(s,"repeatability_batch","exec"))')
        command='timeout --signal=TERM --kill-after=90 '+str(plan['job_walltime_seconds'])+' '+shlex.quote(plan['runtime']+'/envs/b1k-eval/bin/python')+' -u -c '+shlex.quote(launcher)
        assert len(command.encode())<100000
        intent=dict(id=group['id'],name=group['name'],uid=None,created_at=time.time(),
                    creation_attempted=True,state='creation_unknown',startup_sha256=hashes[group['id']],
                    command_sha256=digest(command.encode()),members=group['members'])
        save(path,intent,True)
        args=['acp','jobs','copy','--workspace-name='+plan['workspace'],'--copy-job-name='+plan['template'],
              '--name='+group['name'],'--job-name='+group['name'],'--worker-spec='+plan['worker_spec'],
              '--worker-nodes=1','--quota-type=spot','--retry-times=0','--container-image-url='+plan['image'],
              '--storage-mount='+plan['volume_id']+':/mnt/afs','--env=NVIDIA_DRIVER_CAPABILITIES:all',
              '--command='+command]
        try:
            r=subprocess.run([CLI]+args,env=environment(),capture_output=True,text=True,timeout=90)
            intent['cli_returncode']=r.returncode
        except subprocess.TimeoutExpired:
            intent['cli_outcome']='timeout_unknown'
        save(path,intent)
        state=None
        for retry in range(4):
            try:state=observe(intent,plan);break
            except RuntimeError:
                if retry==3:raise
                time.sleep(2)
        intent.update(state);save(path,intent)
        print(json.dumps(state),flush=True)


def fetch(client,plan,identity,folder='reports'):
    try:r=client.get_object(Bucket='260010165',Key=plan['object_prefix']+'/'+folder+'/'+identity+'.json')
    except Exception as exc:
        if getattr(exc,'response',{}).get('Error',{}).get('Code') in ['NoSuchKey','404']:return None
        raise
    try:return json.loads(r['Body'].read(16*1024**2))
    finally:r['Body'].close()


def collect(plan):
    client=aoss_client();states=[];records=[];failures=[];telemetry=[]
    for g in plan['groups']:
        path=ROOT/'intents'/f"{g['id']}.json"
        if path.exists():
            intent=load(path)
            try:
                state=observe(intent,plan);save(ROOT/'observations'/f"{g['id']}.json",state);states.append(state)
            except Exception as exc:
                states.append(dict(name=intent['name'],uid=intent.get('uid'),state='OBSERVATION_UNKNOWN',error_type=type(exc).__name__))
        group=fetch(client,plan,g['id'])
        if group:save(ROOT/'reports'/f"{g['id']}.json",group)
        for m in g['members']:
            identity=f"baseline-{m['index']:02d}"
            report=fetch(client,plan,identity)
            if report:
                assert report['id']==identity and report['protocol_sha256']==plan['protocol_sha256']
                assert report['worker_bundle_sha256']==digest((ROOT/'frozen_sha256.json').read_bytes())
                save(ROOT/'reports'/f'{identity}.json',report)
                if report['state']=='accepted':
                    r=report['record'];assert [r['task'],r['instance_id']]==plan['pairs'][m['index']]
                    assert r['policy_seed']==r['environment_seed']==0 and r['policy_seed_acknowledged']
                    assert report['repeat']==m['repeat']
                    expected=digest(json.dumps(plan['baseline_files'],sort_keys=True).encode())
                    assert report['checkpoint_manifest_sha256']==expected
                    records.append(dict(id=identity,repeat=m['repeat'],**r,
                        initial_physics_sha256=report['initial_state']['physics_sha256'],
                        action_trace_sha256=report['action_trace']['sha256']))
                else:failures.append(dict(id=identity,error_type=report.get('error_type'),state=report['state']))
            heartbeat=fetch(client,plan,identity+'/heartbeat',folder='telemetry')
            if heartbeat:
                save(ROOT/'telemetry'/f'{identity}.json',heartbeat)
                telemetry.append(dict(id=identity,observed_at=heartbeat['observed_at'],state=heartbeat['state'],
                                      episode_files=heartbeat['episode_files']))
    snapshot=dict(observed_at=time.time(),expected=30,completed=len(records),failures=failures,
                  jobs=states,telemetry=telemetry,complete=len(records)==30 and all(s['state'] in TERMINAL for s in states))
    save(ROOT/'status.json',snapshot)
    if records:
        with (ROOT/'per_episode.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if len(records)==30:
        import statistics
        results=[]
        for pair in plan['protocol']['unique_pairs']:
            rr=sorted([r for r in records if [r['task'],r['instance_id']]==pair],key=lambda r:r['repeat'])
            qq=[r['q_score'] for r in rr];ss=[r['model_success'] for r in rr]
            results.append(dict(task=pair[0],instance=pair[1],scores=qq,successes=ss,
                score_mean=statistics.mean(qq),score_std=statistics.pstdev(qq),score_range=max(qq)-min(qq),
                success_count=sum(ss),success_flip=len(set(ss))>1,
                initial_physics_identical=len({r['initial_physics_sha256'] for r in rr})==1,
                action_trace_identical=len({r['action_trace_sha256'] for r in rr})==1))
        save(ROOT/'comparison.json',dict(complete=snapshot['complete'],checkpoint=1688977,
             protocol_sha256=plan['protocol_sha256'],pairs=results,
             score_changed_pairs=sum(r['score_range']>1e-10 for r in results),
             success_flipped_pairs=sum(r['success_flip'] for r in results),
             initial_state_changed_pairs=sum(not r['initial_physics_identical'] for r in results)))
    print(json.dumps({k:snapshot[k] for k in ['observed_at','expected','completed','failures','complete']},ensure_ascii=False),flush=True)
    return snapshot


def main():
    plan=load(ROOT/'plan.json');mode=sys.argv[1] if len(sys.argv)>1 else 'observe'
    if mode=='submit':
        with (ROOT/'submit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);submit(plan)
        return
    if mode=='launch-monitor':
        path=ROOT/'monitor_launch.json'
        if path.exists():
            previous=load(path)
            try:os.kill(previous['pid'],0);raise RuntimeError('Monitor already active')
            except ProcessLookupError:pass
        with (ROOT/'monitor.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__)),'monitor'],stdin=subprocess.DEVNULL,
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        save(path,dict(pid=process.pid,started_at=time.time(),deadline=plan['creation_deadline_epoch']))
        print('monitor_pid',process.pid);return
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                s=collect(plan)
                if mode!='monitor' or s['complete'] or (len(s['jobs'])==10 and all(j['state'] in TERMINAL for j in s['jobs'])):break
            except Exception as exc:
                save(ROOT/'monitor_error.json',dict(at=time.time(),error_type=type(exc).__name__,automatic_resubmission=False))
                if mode!='monitor':raise
            if time.time()>plan['creation_deadline_epoch']:break
            time.sleep(60)


if __name__=='__main__':main()

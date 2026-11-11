"""Finite three-stage launcher. One creation per stage; any failure stops the chain.

Run on the configured submission host. CPU repair requires separate recorded approval.
No polling loop creates a job, no automatic retries, no extension beyond5000 updates.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,os,re,subprocess,sys,tarfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def cpu_validate(proof,fp):
    assert proof['passed'] and proof['real_two_source_batches']==2
    assert proof['model_loaded'] is False and proof['optimizer_initialized'] is False
    assert proof['mixture_fingerprint']==fp and len(proof['batches'])==2
    for b in proof['batches']:
        assert set(b['source_counts'])=={'0','3'} and min(b['source_counts'].values())>0
        assert sum(b['source_counts'].values())==192 and b['full_episode_text_target_tokens']==0
        assert b['supervised_text_target_tokens']>0 and b['actions_shape']==[192,32,32]
        assert len(b['images'])==3 and all(s==[192,224,224,3] for s in b['images'].values())

BOOTSTRAP='''import hashlib,json,os,subprocess,sys,tarfile,time\nfrom pathlib import Path\ncfg=json.loads(Path(__file__).with_name('bootstrap_input.json').read_text())\nroot=Path('/var/tmp/wangfaye/two-source-frozen-bootstrap')/cfg['stage']/str(os.getpid());root.mkdir(parents=True,exist_ok=False)\ntry:\n source=Path('/data/oss_bucket_0')/cfg['runtime_key']\n assert source.stat().st_size==cfg['bytes']\n h=hashlib.sha256()\n with source.open('rb') as f:\n  for block in iter(lambda:f.read(1024**2),b''):h.update(block)\n assert h.hexdigest()==cfg['sha256']\n bundle=root/'bundle';bundle.mkdir()\n with tarfile.open(source) as archive:\n  for member in archive.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts\n  archive.extractall(bundle)\n print('Frozen runtime read from existing OSS mount; SHA256 verified',flush=True)\n result=subprocess.run([sys.executable,str(bundle/cfg['entry'])],cwd=bundle,timeout=cfg['execution_timeout'])\n raise SystemExit(result.returncode)\nexcept Exception as e:\n out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)\n (out/('transport_failure_'+str(os.getpid())+'.json')).write_text(json.dumps(dict(stage=cfg['stage'],error_type=type(e).__name__,observed_at=time.time())))\n raise\n'''

def freeze(folder):
    b=folder/'bundle'
    save(b/'bundle_manifest.json',{p.relative_to(b).as_posix():digest(p) for p in sorted(b.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json' and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')})
    archive=folder/'runtime.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(b.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                tar.add(p,arcname=p.relative_to(b),recursive=False)
    return archive

def package(client,stage,folder,entry,cluster,output,seconds):
    archive=freeze(folder);sha=digest(archive)
    key='wangfaye/embodied/datasets/behavior-1k/retrain_20261015_v2/ratio_trials/r80f20/runtime/'+sha+'/runtime.tar.gz'
    if client.head(key) is None:
        client.bucket.put_object_from_file(key,str(archive),headers={'x-oss-forbid-overwrite':'true'})
    assert client.head(key)==archive.stat().st_size
    # Source bytes must be verified before a GPU task is admitted.
    h=hashlib.sha256();response=client.bucket.get_object(key)
    try:
        while True:
            block=response.read(1024**2)
            if not block:break
            h.update(block)
    finally:response.resp.response.close()
    assert h.hexdigest()==sha
    small=folder/'small_submit';small.mkdir(exist_ok=True)
    (small/'bootstrap.py').write_text(BOOTSTRAP)
    save(small/'bootstrap_input.json',dict(runtime_key=key,bytes=archive.stat().st_size,sha256=sha,entry=entry,stage=stage,output=output,execution_timeout=seconds))
    (small/'cluster.json').write_bytes(cluster.read_bytes())
    save(folder/'runtime_receipt.json',dict(key=key,bytes=archive.stat().st_size,sha256=sha,remote_readback_verified=True))
    return small

def create(client,stage,folder,entry,cluster,output,workers,gpus,seconds,jobname):
    if (HERE/'STOP').exists():raise RuntimeError('Human STOP present')
    if (folder/'intent.json').exists():raise RuntimeError('Existing intent; exact reconciliation required, no retry')
    client.unused(client.key(output))
    small=package(client,stage,folder,entry,cluster,output,seconds)
    if (HERE/'STOP').exists():raise RuntimeError('Human STOP present')
    intent=dict(stage=stage,run_id=jobname,workers=workers,gpus=gpus,created_at=time.time(),output=output,automatic_resubmit=False)
    with (folder/'intent.json').open('x') as f:json.dump(intent,f,indent=2)
    client.put_new(client.key(output)+'.creation_claim.json',folder/'intent.json')
    secrets=[os.environ[k] for k in ('AI_HUB_TOKEN','OSS_ACCESS_ID','OSS_ACCESS_KEY')]
    args=[os.environ.get('AI_HUB_CLI','/Users/faye/.local/bin/ai-hub-cli'),'train','mdl','--token='+secrets[0],
        '--namespace='+os.environ['AI_HUB_NAMESPACE'],'--name=behavior','--queue=demand_general',
        '--entry=bootstrap.py','--worker_count='+str(workers),'--file.cluster_file=cluster.json',
        '--oss_access_id='+secrets[1],'--oss_access_key='+secrets[2],'--oss_bucket='+client.bucket_name,
        '--oss_endpoint='+os.environ['OSS_ENDPOINT'],'--env=OSS_ACCESS_KEY_ID='+secrets[1]+',OSS_ACCESS_KEY_SECRET='+secrets[2],
        '--job_name='+jobname,'--algo_name=pytorch2120','--gpu_card_type=l20z','--oss_appendable=true','--max_failover_times=0','--debug_mode']
    result=dict(intent,state='unknown')
    try:
        cp=subprocess.run(args,cwd=small,capture_output=True,text=True,timeout=180)
        log=cp.stdout+cp.stderr
        for secret in secrets:log=log.replace(secret,'[REDACTED]')
        (folder/'submission.log').write_text(log)
        ids=sorted(set(re.findall(r'(?i)\b(?:iid|instanceId|instance_id)\W+([0-9]+)',log)))
        result.update(state='submitted' if cp.returncode==0 and len(ids)==1 else 'needs_reconciliation',identifiers=ids,exit_code=cp.returncode)
    except Exception as e:result.update(error_type=type(e).__name__,state='needs_reconciliation')
    save(folder/'receipt.json',result)
    if result['state']!='submitted':raise RuntimeError('Creation outcome unknown or failed; no resubmission')
    save(HERE/'gated_training_status.json',dict(stage=stage,state='submitted',receipt=result,observed_at=time.time()))
    print(json.dumps(result),flush=True)
    return result

def status(iid):
    cp=subprocess.run(['/Users/faye/.local/bin/ai-hub-cli','train','status','--iid',str(iid),'--detail','--token='+os.environ['AI_HUB_TOKEN']],capture_output=True,text=True,timeout=45)
    # Never save or print raw status: nested env contains secrets.
    raw=cp.stdout;obj=json.loads(raw[raw.index('{'):]);data=obj.get('data',{})
    app=data.get('application',{});task=data.get('task',{})
    return dict(iid=iid,application_status=app.get('status'),task_status=task.get('status'),task_id=app.get('task_id'),
        containers=[dict(role=c.get('role'),status=c.get('status'),end_time=c.get('end_time'),exitCode=c.get('exitCode')) for c in data.get('containers',[])],observed_at=time.time())

def wait_receipt(client,stage,folder,receipt,success,timeout):
    prefix=client.key(receipt['output']);until=time.time()+timeout;errors=0
    while time.time()<until:
        if (HERE/'STOP').exists():raise RuntimeError('Human STOP present')
        try:
            page=client.bucket.list_objects(prefix=prefix+'/',max_keys=1000)
            bad=[x.key for x in page.object_list if x.key.endswith('failure.json') or '/transport_failure_' in x.key]
            if bad:
                for n,key in enumerate(bad):save(folder/('failure_'+str(n)+'.json'),json.loads(client.read(key)))
                raise RuntimeError('Actual worker failure; no automatic retry')
            if client.head(prefix+'/'+success):
                proof=json.loads(client.read(prefix+'/'+success));save(folder/'accepted_output_raw.json',proof);return proof
            s=status(receipt['identifiers'][0]);save(folder/'last_status.json',s)
            if s['application_status'] in ('FAIL','FAILED','KILLED','STOPPED','SUCCEEDED','SUCCESS'):
                # Successful completion without required proof is also a failed gate.
                raise RuntimeError('Native terminal state without required proof')
            errors=0
        except RuntimeError:raise
        except Exception as e:
            errors+=1;save(folder/'observation_error.json',dict(type=type(e).__name__,consecutive=errors,time=time.time()))
            if errors>=6:raise RuntimeError('Repeated observation failure; no new submissions') from None
        time.sleep(45)
    raise RuntimeError('Finite gate deadline reached; no new submissions')

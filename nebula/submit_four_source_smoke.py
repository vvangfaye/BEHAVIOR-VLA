"""Submit one reviewed two-update smoke; no retries or formal training mode."""
import concurrent.futures,hashlib,json,os,re,subprocess,sys,time
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nebula.submit_oss import OSSClient

def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def verify_checkpoint_source(client,bundle,job):
    details_path=bundle/'inputs/checkpoint_file_manifest.json'
    details={} if not details_path.exists() else {x['relative_path']:x for x in json.loads(details_path.read_text())['files']}
    entries=[]
    for line in (bundle/'inputs/checkpoint.SHA256SUMS').read_text().splitlines():
        digest,name=line.split(None,1);name=name.lstrip('*')
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe checkpoint manifest')
        if details and details[name]['sha256']!=digest:raise ValueError('Checkpoint manifest hash mismatch')
        entries.append((digest,name))
    def one(entry):
        digest,name=entry
        try:h=client.bucket.head_object(job['checkpoint_key'].rstrip('/')+'/'+name)
        except Exception as e:raise ValueError('Initial checkpoint object unavailable: '+name+' ('+type(e).__name__+')') from None
        if details:
            expected=details[name]
            if h.content_length!=expected['size'] or str(h.headers.get('x-oss-hash-crc64ecma'))!=expected['crc64']:
                raise ValueError('Initial checkpoint size/CRC changed: '+name)
        return dict(relative_path=name,bytes=h.content_length,sha256=digest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(one,entries))
    return dict(all_source_objects_present=True,files=files,source_prefix=job['checkpoint_key'])

def main(prepared):
    prepared=Path(prepared);bundle=prepared/'bundle';job=json.loads((bundle/'nebula/smoke_job.json').read_text())
    if job['updates']!=2 or job['formal_training_authorized'] is not False:raise ValueError('Not a two-update smoke')
    workers=job.get('process_count',1)
    if workers not in (1,2):raise ValueError('Smoke requires one or two eight-GPU workers')
    if (prepared/'intent.json').exists():raise ValueError('Prior intent exists; reconcile exact task before retrying')
    if job.get('require_cpu_batch_preflight'):
        proof=json.loads((prepared/'cpu_batch_preflight.json').read_text())
        if not proof.get('passed') or proof.get('real_four_source_batches')!=2 or proof.get('mixture_fingerprint')!=job['mixture_fingerprint']:
            raise ValueError('Actual two-batch CPU preflight not proven')
        for batch in proof['batches']:
            if batch['full_episode_text_target_tokens']!=0 or set(batch['source_counts'])!={'0','1','2','3'} or min(batch['source_counts'].values())<=0 or batch['actions_shape'][0]!=192:
                raise ValueError('CPU batch proof does not match required mixture/masking')
    job['storage_endpoint']=os.environ['OSS_ENDPOINT']
    if (bundle/'nebula/ossutil').exists():
        for path,field in [('nebula/ossutil','ossutil_sha256'),('inputs/paligemma_tokenizer.model','tokenizer_sha256')]:
            if hashlib.sha256((bundle/path).read_bytes()).hexdigest()!=job[field]:raise ValueError('Frozen download asset changed: '+path)
        job['download_endpoint']=os.environ.get('OSS_DOWNLOAD_ENDPOINT') or os.environ['OSS_ENDPOINT']
    client=OSSClient();output=client.key(job['output']);client.unused(output)
    envkey=job['environment']['key'];cachekey=job['cache']['key']
    for field,key in [('environment',envkey),('cache',cachekey)]:
        h=client.bucket.head_object(key)
        if job[field].get('etag') and job[field]['etag']!=h.etag:raise ValueError('Frozen '+field+' object changed')
        job[field].update(bytes=h.content_length,etag=h.etag)
    # Verify unchanged data acceptance and source checkpoint manifest before reservation.
    raw=client.read(job['data_manifest_key'])
    if hashlib.sha256(raw).hexdigest()!=job['data_manifest_sha256']:raise ValueError('Accepted manifest changed')
    proof=verify_checkpoint_source(client,bundle,job)
    (prepared/'checkpoint_source_preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    jobpath=bundle/'nebula/smoke_job.json';jobpath.write_text(json.dumps(job,indent=2)+'\n')
    manifest={p.relative_to(bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in bundle.rglob('*') if p.is_file() and p.name!='bundle_manifest.json'
              and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    (bundle/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    intent=dict(run_id=job['run_id'],time=time.time(),updates=2,formal_training=False,automatic_resubmit=False)
    save(prepared/'intent.json',intent);client.put_new(output+'.claim.json',prepared/'intent.json')
    secret=[os.environ[k] for k in ('AI_HUB_TOKEN','OSS_ACCESS_ID','OSS_ACCESS_KEY')]
    args=[os.environ.get('AI_HUB_CLI','ai-hub-cli'),'train','mdl','--token='+secret[0],
        '--namespace='+os.environ['AI_HUB_NAMESPACE'],'--name=behavior','--queue=demand_general',
        '--entry=nebula/four_source_smoke.py','--worker_count='+str(workers),'--file.cluster_file=nebula/smoke_cluster.json',
        '--oss_access_id='+secret[1],'--oss_access_key='+secret[2],'--oss_bucket='+client.bucket_name,
        '--oss_endpoint='+os.environ['OSS_ENDPOINT'],'--env=OSS_ACCESS_KEY_ID='+secret[1]+',OSS_ACCESS_KEY_SECRET='+secret[2],
        '--job_name='+job['run_id'],'--algo_name=pytorch2120','--gpu_card_type=l20z',
        '--oss_appendable=true','--max_failover_times=0','--debug_mode','--ignore=__pycache__,*.pyc,*.pyo']
    receipt=dict(**intent,state='unknown',output=job['output'],gpus=8*workers,bundle_manifest_sha256=hashlib.sha256((bundle/'bundle_manifest.json').read_bytes()).hexdigest())
    try:
        p=subprocess.run(args,cwd=bundle,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        clean=p.stdout
        for s in secret:clean=clean.replace(s,'[REDACTED]')
        (prepared/'submission.log').write_text(clean)
        receipt.update(state='submitted' if p.returncode==0 else 'failed_or_unknown',exit_code=p.returncode,
            identifiers=sorted(set(re.findall(r'(?i)\b(?:iid|instanceId|instance_id|task_id)\W+([A-Za-z0-9_-]+)',clean))))
    except Exception as e:
        receipt.update(error_type=type(e).__name__)
    finally:
        save(prepared/'receipt.json',receipt)
    print(json.dumps(receipt))
if __name__=='__main__':main(sys.argv[1])

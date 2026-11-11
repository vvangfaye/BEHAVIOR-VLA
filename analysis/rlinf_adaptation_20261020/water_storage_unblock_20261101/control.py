"""Bounded archive supervision and unchanged water-campaign continuation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CAMPAIGN=HERE.parent/'multitask_validation_20261029'
sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
sys.path.insert(0,str(CAMPAIGN))
sys.path.insert(0,str(CAMPAIGN/'capacity_repair_20261030'))
sys.path.insert(0,str(PROJECT/'analysis/serf_multitask_20261029'))
from video_eval_adapter import aoss_client
from internal_storage import internal_aoss_client
from read_quota_exact import query_quota
from capacity_admission import evidence
from cluster_helpers import submit, describe

CLI='/home/faye/.local/bin/sco'
ENV={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
TERMINAL={'SUCCEEDED','FAILED','STOPPED','SUSPENDED','TERMINATED','CANCELED'}


def save(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def audit(plan):
    states=['RUNNING','QUEUEING','WAITING','CREATING','STARTING','RESTARTING','RECOVERING','INIT']
    def fetch(state):
        rows=[]
        for page in range(1,21):
            p=subprocess.run([CLI,'acp','jobs','list','--workspace-name=share-space','--user-name=260010165',
                '--state='+state,'--page-size=100','--page-token='+str(page),'-o','json'],
                env=ENV,capture_output=True,text=True,timeout=40);p.check_returncode()
            group=[] if p.stdout.strip()=='No jobs found' else json.loads(p.stdout)
            assert all(x['state']==state and x['ownership']['user_name']=='260010165' for x in group)
            rows.extend(group)
            if len(group)<100:return rows
        raise RuntimeError('Incomplete active-job pagination')
    jobs={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for group in pool.map(fetch,states):
            for job in group:jobs[job['uid']]=job
    matches=[]
    for job in jobs.values():
        if job['name'].startswith('b1k-water-storage-1101-'):continue
        texts=[r.get('startup_script','') for r in job.get('roles',[])]
        for text in list(texts):
            for encoded in re.findall(r'[A-Za-z0-9+/]{100,}={0,2}',text):
                try:
                    decoder=zlib.decompressobj()
                    raw=decoder.decompress(base64.b64decode(encoded),2*1024**2)
                    if not decoder.unconsumed_tail:texts.append(raw.decode(errors='replace'))
                except Exception:pass
        joined='\n'.join(texts)
        references=set(re.findall(r'/mnt/afs/260010165/[A-Za-z0-9_./-]+',joined))
        references={p for p in references if len(Path(p).relative_to('/mnt/afs/260010165').parts)>=2}
        # Each selected unit is a specific old experiment or model family.
        for source in plan['files']:
            path=source['path']
            if any(Path(path)==Path(p) or Path(path).is_relative_to(p) for p in references):
                matches.append({'job':job['name'],'index':source['index'],'source':path})
    return {'observed_at':time.time(),'jobs':[{'name':j['name'],'uid':j['uid'],'state':j['state']} for j in jobs.values()],
            'conflicts':matches,'passed':not matches,'water_plan_preserved':True}


def put_json(client,key,data):
    raw=(json.dumps(data,ensure_ascii=False,sort_keys=True)+'\n').encode()
    client.put_object(Bucket='260010165',Key=key,Body=raw)
    assert client.get_object(Bucket='260010165',Key=key)['Body'].read()==raw


def launch():
    assert not (HERE/'submission.json').exists(),'Existing submission must be reconciled'
    raw=(HERE/'plan.json').read_bytes();plan=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    inspection=audit(plan);save(HERE/'dependency_audit.json',inspection);assert inspection['passed']
    public=aoss_client();client=internal_aoss_client();prefix=plan['prefix']
    try:
        assert public.list_objects_v2(Bucket='260010165',Prefix=prefix,MaxKeys=1).get('KeyCount',0)==0
        public.put_object(Bucket='260010165',Key=prefix+'plan.json',Body=raw)
        assert public.get_object(Bucket='260010165',Key=prefix+'plan.json')['Body'].read()==raw
        put_json(public,prefix+'dependency_audit.json',inspection)
        for item in plan['files']:
            if item.get('existing_object'):
                assert public.head_object(Bucket='260010165',Key=item['key'])['ContentLength']==item['size']
        lease={'observed_at':time.time(),'expires_at':time.time()+180,'plan_sha256':digest,
               'allowed_indices':[x['index'] for x in plan['files']]}
        put_json(public,prefix+'release_lease.json',lease)
        def signed(method,key):
            return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=6*3600)
        files=[]
        for original in plan['files']:
            item=dict(original);item['get']=signed('get_object',item['key'])
            if plan.get('range_readback'):
                item['get_fallback']=public.generate_presigned_url('get_object',
                    Params={'Bucket':'260010165','Key':item['key']},ExpiresIn=6*3600)
            if not item.get('existing_object'):item['put']=signed('put_object',item['key'])
            if item.get('sharded'):
                item['parts']=[]
                for number,offset in enumerate(range(0,item['size'],item['part_bytes'])):
                    key=item['key']+f'part-{number:05d}'
                    item['parts'].append({'number':number,'offset':offset,
                        'size':min(item['part_bytes'],item['size']-offset),'key':key,
                        'put':signed('put_object',key),'get':signed('get_object',key)})
            for name in ['manifest','complete']:
                key=prefix+f"receipts/{item['index']:04d}/{name}.json"
                item[name+'_put']=signed('put_object',key);item[name+'_get']=signed('get_object',key)
            files.append(item)
        helper=Path(plan.get('helper_file',str(CAMPAIGN/'capacity_repair_20261030/archive_helpers_streaming.py')))
        config={'output':plan.get('output','/mnt/afs/260010165/behavior_1k/operational_cleanup/water_unblock_20261101/a1'),
            'result_key':prefix+'status.json','status_put':signed('put_object',prefix+'status.json'),
            'lease_get':signed('get_object',prefix+'release_lease.json'),'files':files,
            'plan_sha256':digest,'deadline_seconds':6800,'helpers_source':helper.read_text(),
            'range_readback':plan.get('range_readback',False),
            **{k:plan[k] for k in ['allowed_roots','protected_roots','metadata_roots']}}
        config['source_sha256']={'worker':hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest(),
                                 'helpers':hashlib.sha256(helper.read_bytes()).hexdigest(),'plan':digest}
        config['metadata_key']=prefix+'metadata.tar.gz'
        for label,key in [('metadata',config['metadata_key']),('metadata_complete',prefix+'metadata.COMPLETE.json')]:
            config[label+'_put']=signed('put_object',key);config[label+'_get']=signed('get_object',key)
        submit(plan.get('job_name','b1k-water-storage-1101-a1'),HERE/'worker.py',config,HERE/'submission.json',7200,False)
    finally:public.close();client.close()


def monitor():
    receipt=json.loads((HERE/'submission.json').read_text());plan=json.loads((HERE/'plan.json').read_text())
    digest=hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest();client=aoss_client()
    deadline=time.time()+6*3600
    try:
        while time.time()<deadline:
            try:
                job=describe(receipt['name']);assert job['uid']==receipt['uid']
                if job['state'] not in TERMINAL:
                    inspection=audit(plan);save(HERE/'dependency_audit_latest.json',inspection)
                    lease={'observed_at':inspection['observed_at'],'expires_at':time.time()+180,'plan_sha256':digest,
                        'allowed_indices':[] if not inspection['passed'] else [x['index'] for x in plan['files']]}
                    put_json(client,plan['prefix']+'release_lease.json',lease)
                report={'observed_at':time.time(),'archive_job_state':job['state']}
                try:
                    obj=client.get_object(Bucket='260010165',Key=plan['prefix']+'status.json')
                    assert obj['ContentLength']<8*1024**2
                    archive=json.loads(obj['Body'].read());save(HERE/'archive_status.json',archive)
                    report['archive']={k:archive.get(k) for k in ['stage','released_files','released_allocated_bytes','retained','active']}
                except client.exceptions.ClientError as error:
                    if error.response['Error']['Code'] not in ['NoSuchKey','404']:raise
                quota=query_quota();water_plan=json.loads((CAMPAIGN/'water/plan.json').read_text())
                proof=evidence(water_plan,quota,True,False)
                save(HERE/'quota_latest.json',quota);save(CAMPAIGN/'water/training_storage_clearance.json',proof)
                report['free_bytes']=proof['quota_lower_bound_bytes']-proof['account_allocated_bytes']
                report['water_storage_passed']=proof['passed']
                status=json.loads((CAMPAIGN/'water/latest_status.json').read_text())
                nonbaseline=[w for w in status.get('workers',[]) if w['phase']!='baseline']
                report['water_state']=status.get('state');report['water_phase']=status.get('next_phase')
                report['water_jobs']=[{k:w.get(k) for k in ['job','uid','phase','state','progress']} for w in nonbaseline]
                save(HERE/'continuation_status.json',report)
                print(json.dumps({k:v for k,v in report.items() if k!='archive'}),flush=True)
                if job['state'] in TERMINAL:
                    if job['state']!='SUCCEEDED':raise RuntimeError('Archive did not finish; verified releases retained in report')
                    if proof['passed'] and any(w['state']=='RUNNING' for w in nonbaseline):
                        save(HERE/'water_resumed.json',report);return
                    if not proof['passed']:raise RuntimeError('Archive complete but quota still insufficient; inspect before changing scope')
            except Exception as error:
                save(HERE/'monitor_error.json',{'at':time.time(),'type':type(error).__name__,'message':str(error)[:600]})
            time.sleep(30)
        raise TimeoutError('Storage continuation monitoring deadline')
    finally:client.close()


if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    elif sys.argv[1]=='monitor':monitor()
    else:raise ValueError('Expected launch or monitor')

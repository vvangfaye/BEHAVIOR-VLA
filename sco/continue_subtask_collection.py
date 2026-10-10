"""Finish one authorized finite wave after its standard pilot passes acceptance.

This is a bounded local dispatcher, not a recurring scheduler. It never admits
Spot from platform state alone and never starts a second writer for a shard.
Run with the configured AOSS client Python; receipts contain no signed URLs.
"""
import argparse
import base64
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import zlib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-receipt',type=Path,required=True)
    parser.add_argument('--wave-prepared',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--aoss-client-module',type=Path,default=Path('/private/tmp/behavior_aoss_client.py'))
    parser.add_argument('--cli',default='/private/tmp/behavior-sco-runtime/bin/sco')
    parser.add_argument('--max-hours',type=int,default=28)
    args=parser.parse_args()
    assert 1<=args.max_hours<=72
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import importlib.util
    spec=importlib.util.spec_from_file_location('collection_private_aoss',args.aoss_client_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.client()
    env=os.environ.copy();env.update(SCO_CONFIG='/private/tmp/behavior-sco-config',
        SCO_HOME='/private/tmp/behavior-sco-runtime',SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    repo=Path(__file__).resolve().parents[1];bucket='260010165'
    pilot=json.loads(args.pilot_receipt.read_text());wave=json.loads(args.wave_prepared.read_text())
    assert pilot['source_manifest_sha256']==wave['source_manifest_sha256']
    assert wave['state']=='complete' and wave['plan']['expected_trials']<=30
    frozen=root/'frozen_drivers';frozen.mkdir(exist_ok=True)
    for name in ('inspect_subtask_collection.py','verify_collection_training_reader.py',
                 'finalize_subtask_collection.py','submit_subtask_collection.py',
                 'run_subtask_collection.py','public_test_admission.py'):
        target=frozen/name
        if not target.exists():target.write_bytes((repo/'sco'/name).read_bytes())
    started=time.time()
    if (root/'status.json').exists():
        previous=json.loads((root/'status.json').read_text())
        assert previous['wave_fingerprint']==wave['plan']['fingerprint'] and previous['pilot_uid']==pilot['uid']
        started=previous['started_at']
    deadline=started+args.max_hours*3600
    state=dict(state='waiting_standard_pilot',started_at=started,pilot_name=pilot['name'],
        pilot_uid=pilot['uid'],wave_fingerprint=wave['plan']['fingerprint'],expected_trials=wave['plan']['expected_trials'])

    def publish(**updates):
        state.update(updates,updated_at=time.time())
        temp=root/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'status.json')
        print(json.dumps({k:v for k,v in state.items() if k in ('state','error','submitted_shards','completed_shards','dataset_root')}),flush=True)

    def query(receipt):
        job=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe',receipt['name'],
            '--workspace-name=share-space','-o','json'],env=env,text=True,timeout=30))
        assert job['ownership']['user_name']==bucket and job['uid']==receipt['uid']
        return dict(name=job['name'],uid=job['uid'],platform_state=job['state'],owner=bucket)

    def pause():
        if time.time()>deadline:raise TimeoutError('Finite dispatcher deadline reached; existing jobs are preserved')
        time.sleep(45)

    def wait_terminal(receipt):
        while True:
            live=query(receipt)
            if live['platform_state'] in ('SUCCEEDED','FAILED','SUSPENDED'):return live
            pause()

    def cpu_accept(mode,payload):
        receipt_path=root/(mode+'_submission.json')
        if not receipt_path.exists():
            stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            name='behavior-collection-'+mode+'-'+stamp
            report_key='behavior1k/subtask-collection/20261010/'+name+'/accepted.json'
            output='/mnt/afs/260010165/behavior_1k/preparation/'+name
            payload.update(audit_root=output+'/artifacts')
            source=(frozen/'inspect_subtask_collection.py').read_text()
            url=client.generate_presigned_url('put_object',Params={'Bucket':bucket,'Key':report_key},ExpiresIn=172800)
            bundle=base64.b64encode(zlib.compress(json.dumps(dict(source=source,payload=payload,
                output=output,report_put=url)).encode())).decode()
            startup='import base64,json,os,traceback,urllib.request,zlib;from pathlib import Path;d=json.loads(zlib.decompress(base64.b64decode('+repr(bundle)+')));r=Path(d["output"]);r.mkdir(parents=True,exist_ok=False);os.environ.update(CUDA_VISIBLE_DEVICES="",OMNIGIBSON_NO_OMNIVERSE="1",PYTHONNOUSERSITE="1");exec(compile(d["source"],"artifact_acceptance","exec"));result=run(d["payload"]);body=json.dumps(result).encode();(r/"accepted.json").write_bytes(body);urllib.request.urlopen(urllib.request.Request(d["report_put"],data=body,method="PUT"),timeout=60).close()'
            record=dict(name=name,state='submitting',report_key=report_key,output_root=output,
                source_manifest_sha256=payload['source_manifest_sha256'],
                driver_sha256=hashlib.sha256(source.encode()).hexdigest(),gpu_count=0)
            with receipt_path.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
            (root/(mode+'_payload.json')).write_text(json.dumps(payload,indent=2)+'\n')
            command=[args.cli,'acp','jobs','create','--workspace-name=share-space','--aec2-name=share-cluster',
                '--name='+name,'--job-name='+name,'--training-framework=pytorch',
                '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
                '--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--command=timeout --signal=TERM --kill-after=30 14400 '+payload['runtime_root']+'/envs/b1k-eval/bin/python -c '+shlex.quote(startup)]
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
            assert result.returncode==0,'CPU submission unknown: query exact saved name'
            job=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe',name,
                '--workspace-name=share-space','-o','json'],env=env,text=True,timeout=30))
            assert job['ownership']['user_name']==bucket
            record.update(uid=job['uid'],state='submitted');receipt_path.write_text(json.dumps(record,indent=2)+'\n')
        receipt=json.loads(receipt_path.read_text())
        assert receipt.get('uid'),'Reconcile exact CPU job name before resuming dispatcher'
        live=wait_terminal(receipt);assert live['platform_state']=='SUCCEEDED',mode+' CPU acceptance failed'
        report=json.loads(client.get_object(Bucket=bucket,Key=receipt['report_key'])['Body'].read())
        (root/(mode+'_accepted.json')).write_text(json.dumps(report,indent=2)+'\n')
        assert report.get('artifact_acceptance_complete') is True
        return report

    def quota():
        table=subprocess.check_output([args.cli,'afs','dir-quota','list',
            '--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5','--dirPath=/260010165'],env=env,text=True,timeout=30)
        numbers=re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*(TB|GB|MB)',table)
        assert len(numbers)==2,'Unknown quota table: no admission'
        def lower(value,unit,high=False):
            scale={'TB':10**12,'GB':10**9,'MB':10**6}[unit]
            precision=len(value.split('.')[1]) if '.' in value else 0
            rounding=0.5*10**(-precision)
            return (float(value)+(rounding if high else -rounding))*scale
        free=max(0,int(lower(*numbers[0])-lower(*numbers[1],high=True)))
        observed=datetime.now(timezone.utc).isoformat()
        (root/'quota_current.json').write_text(json.dumps(dict(table=table,observed_at=observed,
            free_lower_bound_bytes=free),indent=2)+'\n')
        return free,observed

    def submit_shard(shard,index,recover=None):
        generation=recover.get('generation',0)+1 if recover else 0
        path=root/('spot_%03d_g%d.json'%(index,generation))
        if not path.exists():
            free,observed=quota()
            name='behavior-collection-spot-'+wave['plan']['fingerprint'][:8]+'-a%03d-g%d'%(index,generation)
            command=['python3',str(frozen/'submit_subtask_collection.py'),'--prepared',str(root/'spot_prepared.json'),
                '--receipt',str(path),'--name',name,'--quota-type','spot','--task-name',shard['task'],
                '--shared-results-root',wave['results_root'],'--personal-quota-free-lower-bound-bytes',str(free),
                '--quota-observed-at',observed]
            if recover:command.extend(['--recover-receipt',str(root/('spot_%03d_g%d.json'%(index,generation-1)))])
            subprocess.run(command,env=env,check=True)
        receipt=json.loads(path.read_text());assert receipt.get('uid'),'Reconcile saved exact shard name before resume'
        return receipt

    try:
        publish()
        live=wait_terminal(pilot);assert live['platform_state']=='SUCCEEDED','Standard pilot did not succeed'
        publish(state='accepting_standard_pilot')
        common=dict(code_root=wave['code_root'],source_manifest_sha256=wave['source_manifest_sha256'],
            runtime_root=pilot['runtime_root'],reader_source=(frozen/'verify_collection_training_reader.py').read_text())
        accepted=cpu_accept('pilot-audit',dict(common,mode='pilot',prepared_root=pilot['prepared_root'],
            plan_fingerprint=pilot['plan_fingerprint'],pilot=dict(pilot,**live)))
        wave['accepted_standard_pilot']=dict(live,source_manifest_sha256=wave['source_manifest_sha256'],
            trainable_dataset_verified=True,export=accepted['export'],dataset_root=accepted['dataset_root'],
            actual_training_reader=accepted['actual_training_reader'])
        (root/'spot_prepared.json').write_text(json.dumps(wave,indent=2)+'\n')
        receipts=[]
        for index,shard in enumerate(wave['shards']):
            receipts.append(submit_shard(shard,index));publish(state='submitting_spot',submitted_shards=len(receipts))
        while True:
            completed=0;executions=[]
            for index,receipt in enumerate(receipts):
                live=query(receipt)
                if live['platform_state']=='SUCCEEDED':
                    completed+=1;executions.append(dict(receipt,**live))
                elif live['platform_state'] in ('FAILED','SUSPENDED'):
                    receipts[index]=submit_shard(wave['shards'][index],index,recover=receipt)
            publish(state='collecting_spot',completed_shards=completed,submitted_shards=len(receipts),
                jobs=[dict(name=r['name'],uid=r['uid']) for r in receipts])
            if completed==len(receipts):break
            pause()
        repo_id='behavior-collection-'+wave['plan']['fingerprint'][:12]
        finalizer=dict(common,prepared_root=wave['output_root'],results_root=wave['results_root'],
            output_root=wave['output_root']+'/exports/'+repo_id,repo_id=repo_id,
            plan_fingerprint=wave['plan']['fingerprint'],executions=executions)
        finalizer.pop('reader_source')
        publish(state='finalizing')
        final=cpu_accept('wave-finalize',dict(common,mode='finalize',prepared_root=wave['output_root'],
            plan_fingerprint=wave['plan']['fingerprint'],finalizer_payload=finalizer,
            finalizer_source=(frozen/'finalize_subtask_collection.py').read_text()))
        publish(state='complete',dataset_root=final['dataset_root'],export=final['export'])
    except BaseException as error:
        publish(state='needs_attention',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        lock.close()


if __name__=='__main__':main()

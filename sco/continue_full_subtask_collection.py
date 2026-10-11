"""One bounded dispatcher: accept reserved gates, submit all 299 Spot shards, recover progress."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time

def require_automatic_collection_authorized(root):
    marker = Path(root) / 'USER_CANCEL_NONRUNNING.json'
    if marker.exists():
        raise RuntimeError('Automatic collection submission and recovery cancelled by the user: '+str(marker))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--gate-receipts',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-hours',type=int,default=144)
    p.add_argument('--aoss-client-module',default='/private/tmp/behavior_aoss_client.py')
    p.add_argument('--cli',default='/private/tmp/behavior-sco-runtime/bin/sco');args=p.parse_args()
    require_automatic_collection_authorized(args.output_dir)
    from submit_full_collection_shard import BUCKET,describe,environment,submit
    assert 1<=args.max_hours<=144
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=importlib.util.spec_from_file_location('private_aoss',args.aoss_client_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.client()
    campaign=json.loads(args.campaign.read_text());assert campaign['state']=='complete'
    assert campaign['prepared_group_count']==campaign['requested_group_count']==299
    groups=campaign['groups'];assert len({(e['group']['task'],e['group']['subtask']) for e in groups})==299
    # Prioritize demonstrated yield while retaining every zero-success group.
    groups=sorted(groups,key=lambda e:(e['group']['successes']==0,-e['group']['rate'],
                                      -e['group']['eligible'],e['index']))
    gates_input=json.loads(args.gate_receipts.read_text());env=environment()
    started=time.time();state=dict(state='waiting_standard_gates',started_at=started,
        source_manifest_sha256=campaign['source_manifest_sha256'],requested_groups=299,expected_trials=campaign['expected_trials'])
    if (root/'status.json').exists():
        previous=json.loads((root/'status.json').read_text());assert previous['source_manifest_sha256']==campaign['source_manifest_sha256']
        started=previous['started_at'];state.update(started_at=started)
    deadline=started+args.max_hours*3600
    def publish(**updates):
        state.update(updates,updated_at=time.time());temp=root/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'status.json')
        print(json.dumps({k:v for k,v in state.items() if k in ('state','submitted_groups','accepted_groups','accepted_trials','error')}),flush=True)
    def pause():
        if time.time()>deadline:raise TimeoutError('Finite dispatcher deadline reached; jobs and checkpoints preserved')
        time.sleep(45)
    def report(receipt):
        try:value=client.get_object(Bucket=BUCKET,Key=receipt['final_report_key'])['Body'].read()
        except client.exceptions.NoSuchKey:return None
        except Exception as e:
            response=getattr(e,'response',{})
            if response.get('Error',{}).get('Code') in ('NoSuchKey','404'):return None
            raise RuntimeError('Acceptance observation failed: '+type(e).__name__) from None
        result=json.loads(value)
        assert result['source_manifest_sha256']==campaign['source_manifest_sha256'] and result['plan_fingerprint']==receipt['plan_fingerprint']
        assert result['artifact_acceptance_complete'] and result['object_checkpoints_complete']
        assert result['collection_summary']['complete'] and result['collection_summary']['completed_trials']==receipt['expected_trials']
        head=client.head_object(Bucket=BUCKET,Key=result['archive']['object_key'])
        assert head['ContentLength']==result['archive']['bytes'] and head['ETag']==result['archive']['etag']
        if result['collection_summary']['accepted_trials']:
            assert result['actual_training_reader']['complete'] and result['actual_training_reader']['actual_openpi_training_reader']
        return result
    try:
        publish()
        accepted_gates=[]
        while len(accepted_gates)!=len(gates_input):
            accepted_gates=[]
            for item in gates_input:
                receipt=json.loads(Path(item['receipt']).read_text());assert receipt.get('uid')
                live=describe(args.cli,env,receipt['name']);assert live['uid']==receipt['uid']
                if live['state'] in ('FAILED','SUSPENDED'):raise RuntimeError('Standard gate failed: '+item['kind']+' '+receipt['name'])
                if live['state']!='SUCCEEDED':continue
                acceptance=report(receipt);assert acceptance,'Standard gate lacks verified object deliverables'
                assert acceptance['checkpoint_restore_verified'] and acceptance['collection_summary']['eligible_trials']>0
                (root/('gate_'+item['kind']+'_acceptance.json')).write_text(json.dumps(acceptance,indent=2)+'\n')
                accepted_gates.append(dict(receipt,kind=item['kind']))
            if len(accepted_gates)!=len(gates_input):pause()
        assert {g['kind'] for g in accepted_gates}=={'native','history','additional'}
        assert sum(json.loads((root/('gate_'+g['kind']+'_acceptance.json')).read_text())['collection_summary']['accepted_trials'] for g in accepted_gates)>0
        identities=accepted_gates[0]['driver_identities'];assert all(g['driver_identities']==identities for g in accepted_gates)
        gates=dict(state='complete',source_manifest_sha256=campaign['source_manifest_sha256'],driver_identities=identities,
            protocols_verified=['history','native'],additional_start_verified=True,gates=accepted_gates,completed_at=time.time())
        gate_path=root/'standard_gates.json';gate_path.write_text(json.dumps(gates,indent=2)+'\n')
        campaign_id=hashlib.sha256(json.dumps(sorted((e['index'],e['plan']['fingerprint']) for e in groups),sort_keys=True).encode()).hexdigest()[:12]
        prepared=root/'prepared';prepared.mkdir(exist_ok=True)
        def execute(entry,recover=None):
            require_automatic_collection_authorized(root)
            generation=recover.get('generation',0)+1 if recover else 0
            path=root/('spot_%03d_g%d.json'%(entry['index'],generation))
            if path.exists():
                receipt=json.loads(path.read_text())
                if not receipt.get('uid'):
                    live=describe(args.cli,env,receipt['name'])
                    assert live['scheduling']['quota_type']=='SPOT'
                    receipt.update(uid=live['uid'],state='submitted',platform_state=live['state'])
                    path.write_text(json.dumps(receipt,indent=2)+'\n')
                return receipt
            ready=prepared/('%03d.json'%entry['index'])
            if not ready.exists():ready.write_text(json.dumps(entry,indent=2)+'\n')
            name='behavior-collect299-'+campaign_id+'-a%03d-g%d'%(entry['index'],generation)
            return submit(SimpleNamespace(receipt=str(path),prepared=str(ready),name=name,quota_type='spot',gates=str(gate_path),
                object_prefix='behavior1k/subtask-collection/full299-20261011/'+campaign_id+'/group-%03d'%entry['index'],
                timeout_hours=24,aoss_client_module=args.aoss_client_module,cli=args.cli,
                recover_receipt=str(root/('spot_%03d_g%d.json'%(entry['index'],generation-1))) if recover else None))
        receipts=[]
        for entry in groups:
            receipts.append(execute(entry));publish(state='submitting_spot',submitted_groups=len(receipts))
        while True:
            accepted=[];observations=[];exhausted=[]
            for entry,receipt in zip(groups,list(receipts)):
                saved=root/('accepted_%03d.json'%entry['index'])
                if saved.exists():accepted.append(json.loads(saved.read_text()));continue
                live=describe(args.cli,env,receipt['name']);assert live['uid']==receipt['uid']
                observations.append(dict(index=entry['index'],name=receipt['name'],uid=receipt['uid'],platform_state=live['state'],generation=receipt['generation']))
                if live['state'] not in ('SUCCEEDED','FAILED','SUSPENDED'):continue
                result=report(receipt)
                if result:
                    result.update(platform_state=live['state'],job_uid=receipt['uid'],index=entry['index'])
                    saved.write_text(json.dumps(result,indent=2)+'\n');accepted.append(result)
                elif receipt['generation']<2:
                    if live['state']=='SUCCEEDED':raise RuntimeError('Succeeded job lacks object acceptance: '+receipt['name'])
                    receipts[groups.index(entry)]=execute(entry,recover=receipt)
                else:exhausted.append(observations[-1])
            manifest=dict(kind='verified_low_subtask_success_training_collection',source_manifest_sha256=campaign['source_manifest_sha256'],
                accepted_groups=len(accepted),requested_groups=299,accepted_trials=sum(r['collection_summary']['accepted_trials'] for r in accepted),
                training_episodes=sum((r.get('export') or {}).get('episodes',0) for r in accepted),
                training_frames=sum((r.get('export') or {}).get('new_frames',0) for r in accepted),
                standard_gate_datasets=[json.loads((root/('gate_'+g['kind']+'_acceptance.json')).read_text()) for g in accepted_gates],
                completed_trials=sum(r['collection_summary']['completed_trials'] for r in accepted),groups=accepted,updated_at=time.time())
            (root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            client.put_object(Bucket=BUCKET,Key='behavior1k/subtask-collection/full299-20261011/'+campaign_id+'/dataset_manifest.json',Body=(json.dumps(manifest,indent=2)+'\n').encode())
            publish(state='collecting_spot',submitted_groups=len(receipts),accepted_groups=len(accepted),accepted_trials=manifest['accepted_trials'],jobs=observations,exhausted_groups=exhausted)
            if len(accepted)+len(exhausted)==299:
                publish(state='complete' if len(accepted)==299 else 'needs_attention');break
            pause()
    except BaseException as e:
        publish(state='needs_attention',error_type=type(e).__name__,error=str(e));raise
    finally:lock.close()


if __name__=='__main__':main()

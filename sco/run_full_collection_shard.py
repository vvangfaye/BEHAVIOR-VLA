"""Collect on worker scratch, checkpoint every attempt, and archive verified training data."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET


class MultipartArchive:
    def __init__(self, urls, request, part_bytes=128*1024**2):
        self.urls,self.request,self.part_bytes=urls,request,part_bytes
        self.buffer=bytearray();self.parts=[];self.hash=hashlib.sha256();self.count=0
    def writable(self):return True
    def write(self, data):
        self.hash.update(data);self.count+=len(data);self.buffer.extend(data)
        while len(self.buffer)>=self.part_bytes:
            self.send(bytes(self.buffer[:self.part_bytes]));del self.buffer[:self.part_bytes]
        return len(data)
    def send(self, body):
        number=len(self.parts)+1
        if number>len(self.urls['parts']):raise ValueError('Archive exceeds its frozen object budget')
        md5=hashlib.md5(body).digest()
        _,headers=self.request(self.urls['parts'][number-1],method='PUT',body=body,
            headers={'Content-Length':str(len(body)),'Content-MD5':base64.b64encode(md5).decode()})
        headers={k.lower():v for k,v in headers.items()}
        etag=headers.get('etag')
        if not etag:raise ValueError('Stored archive part lacks its receipt')
        self.parts.append((number,etag,md5))
    def finish(self):
        if self.buffer:self.send(bytes(self.buffer));self.buffer.clear()
        xml=ET.Element('CompleteMultipartUpload')
        for number,etag,_ in self.parts:
            part=ET.SubElement(xml,'Part');ET.SubElement(part,'PartNumber').text=str(number);ET.SubElement(part,'ETag').text=etag
        body=ET.tostring(xml)
        response,_=self.request(self.urls['complete'],method='POST',body=body,headers={'Content-Type':'application/xml'})
        tree=ET.fromstring(response)
        if tree.tag.rsplit('}',1)[-1]=='Error':raise ValueError('Object store rejected multipart completion')
        _,head=self.request(self.urls['head'],method='HEAD')
        head={k.lower():v for k,v in head.items()}
        if int(head.get('content-length','-1'))!=self.count:raise ValueError('Archived byte count differs')
        expected=hashlib.md5(b''.join(p[2] for p in self.parts)).hexdigest()+'-'+str(len(self.parts))
        actual=head.get('etag','').strip('"').lower()
        if actual!=expected:raise ValueError('Object-store multipart digest differs')
        return dict(object_key=self.urls['key'],bytes=self.count,sha256=self.hash.hexdigest(),
            etag=head.get('etag'),parts=len(self.parts),part_content_md5_verified=True)


def verify_trial_objects(storage,plan,results,task,request):
    """Require committed receipts even if a simulator swallows an upload exception."""
    verified=[]
    for trial in plan['trials']:
        trial_id=trial['trial_id'];urls=storage['trials'][trial_id]
        body,_=request(urls['receipt']['get']);receipt=json.loads(body)
        directory=Path(results)/'pi05'/task/'trials'/trial_id
        if (receipt.get('trial_id')!=trial_id or receipt.get('plan_fingerprint')!=plan['fingerprint'] or
                receipt.get('source_manifest_sha256')!=storage['source_manifest_sha256'] or
                receipt.get('object_key')!=urls['archive']['key'] or
                receipt.get('completed')!=json.loads((directory/'completed.json').read_text()) or
                receipt.get('accepted')!=(directory/'accepted.json').is_file()):
            raise ValueError('Object checkpoint differs from its completed attempt')
        _,head=request(urls['archive']['head'],method='HEAD');head={k.lower():v for k,v in head.items()}
        if (int(head.get('content-length','-1'))!=receipt['archive']['bytes'] or
                head.get('etag')!=receipt['archive']['etag']):
            raise ValueError('Stored attempt archive differs from its committed receipt')
        verified.append(trial_id)
    return verified


def run(payload):
    code=Path(payload['code_root']);persistent=Path(payload['persistent_root'])
    persistent.mkdir(parents=True,exist_ok=False)
    log=(persistent/'driver.log').open('x',buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    sys.path[:0]=[str(code/'scripts'),str(code/'sco')]
    from collection_checkpoint import request,restore_trials,upload_json,file_hash
    marker=(code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(marker).hexdigest()==payload['source_manifest_sha256']
    storage_bytes,_=request(payload['storage_manifest_url'])
    assert hashlib.sha256(storage_bytes).hexdigest()==payload['storage_manifest_sha256']
    storage=json.loads(storage_bytes)
    assert storage['source_manifest_sha256']==payload['source_manifest_sha256']
    assert storage['plan_fingerprint']==payload['plan_fingerprint']
    prepared=Path(payload['prepared_root'])
    plan=json.loads((prepared/'collection_plan.json').read_text())
    assert plan['fingerprint']==payload['plan_fingerprint']
    scratch=Path('/var/tmp/behavior-collection-data')/payload['execution_name']
    scratch.parent.mkdir(exist_ok=True)
    assert not scratch.exists()
    required=2*plan['maximum_output_bytes']+72*1024**3
    assert shutil.disk_usage(scratch.parent).free>=required,'Insufficient worker scratch budget'
    storage_path=persistent/'storage.private.json';storage_path.write_bytes(storage_bytes);storage_path.chmod(0o600)
    os.environ['B1K_COLLECTION_OBJECT_MANIFEST']=str(storage_path)
    state=dict(state='starting',name=payload['execution_name'],plan_fingerprint=plan['fingerprint'],
        source_manifest_sha256=payload['source_manifest_sha256'],expected_trials=plan['expected_trials'],
        checkpoint_identity=plan['checkpoint_identity'],generation=payload.get('generation',0),started_at=time.time())
    def publish(**changes):
        state.update(changes,updated_at=time.time())
        temp=persistent/'status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(persistent/'status.json')
        print('CAMPAIGN_STATUS='+json.dumps(state),flush=True)
    publish()
    try:
        group=plan['selected_groups'][0]
        restore=Path('/var/tmp/behavior-collection-restored')/payload['execution_name']
        restored=restore_trials(storage,restore,group['task'])
        driver={k:v for k,v in payload.items() if k not in ('persistent_root','storage_manifest_url','storage_manifest_sha256','reader_source')}
        driver.update(output_root=str(scratch))
        if payload.get('recovery'):
            scratch.mkdir()
            if restore.exists():restore.rename(scratch/'results')
            else:(scratch/'results').mkdir()
            contract=dict(run_id=payload['execution_name'],kind=plan['kind'],official_benchmark=False,
                plan_fingerprint=plan['fingerprint'],parent_panel_fingerprint=plan['panel_fingerprint'],
                checkpoint_identity=plan['checkpoint_identity'],max_steps=plan['max_steps'])
            (scratch/'results/collection_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
        elif restored:
            raise ValueError('Unexpected existing checkpoints without a verified terminal recovery receipt')
        publish(state='collecting',restored_trial_ids=restored)
        from run_subtask_collection import run as collect
        collect(driver)
        completed=json.loads((scratch/'status.json').read_text())
        assert completed['state']=='complete'
        summary=completed['collection_summary']
        assert summary['complete'] and summary['completed_trials']==plan['expected_trials']
        committed=verify_trial_objects(storage,plan,scratch/'results',group['task'],request)
        assert set(committed)==set(summary['completed_trial_ids'])
        restore_verified = False
        if payload.get('verify_checkpoint_restore'):
            verification = scratch/'checkpoint_readback'
            recovered = restore_trials(storage, verification, group['task'])
            assert set(recovered)=={t['trial_id'] for t in plan['trials']}
            from collect_b1k_subtasks import validate_collection
            from subtask_sample_panel import load_panel
            assert validate_collection(verification,load_panel(prepared/'parent_panel.json'),plan)==summary
            restore_verified = True
        reader=None
        if summary['accepted_trials']:
            assert completed['trainable_dataset_verified']
            dataset=Path(completed['dataset_root'])
            script=scratch/'training_reader.py'
            script.write_text(payload['reader_source']+'\nr=run('+repr(dict(dataset_root=str(dataset),repo_id=payload['execution_name']))+')\n'+
                'Path('+repr(str(scratch/'training_reader.json'))+').write_text(json.dumps(r,indent=2)+"\\n")\n')
            env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',
                PYTHONPATH=str(code/'openpi/src')+':'+str(code/'BEHAVIOR-1K')+':'+str(code/'BEHAVIOR-1K/OmniGibson'))
            env.pop('PYTHONHOME',None)
            env['LD_LIBRARY_PATH']=str(Path(payload['runtime_root'])/'envs/b1k-pi05/lib')+':'+env.get('LD_LIBRARY_PATH','')
            with (scratch/'training_reader.log').open('x') as reader_log:
                subprocess.run([str(Path(payload['runtime_root'])/'envs/b1k-pi05/bin/python'),str(script)],
                    env=env,cwd=code,stdout=reader_log,stderr=subprocess.STDOUT,check=True)
            reader=json.loads((scratch/'training_reader.json').read_text())
            assert reader['complete'] and reader['actual_openpi_training_reader']
        publish(state='archiving',collection_summary=summary,actual_training_reader=reader)
        sink=MultipartArchive(storage['final_archive'],request)
        with tarfile.open(fileobj=sink,mode='w|') as archive:
            # Trial HDF5/videos have independent object checkpoints; avoid a second raw copy.
            for name in ('exports','training_reader.json','training_reader.log','status.json','source_executions'):
                path=scratch/name
                if path.exists():archive.add(path,arcname=name,recursive=True)
            archive.add(prepared/'collection_plan.json',arcname='collection_plan.json')
            for path in scratch.glob('*.driver.log'):archive.add(path,arcname=path.name)
        archived=sink.finish()
        report=dict(state='complete',artifact_acceptance_complete=True,name=payload['execution_name'],
            source_manifest_sha256=payload['source_manifest_sha256'],plan_fingerprint=plan['fingerprint'],
            collection_summary=summary,actual_training_reader=reader,export=completed.get('export'),
            archive=archived,object_checkpoints_complete=True,group=group,completed_at=time.time())
        report['checkpoint_restore_verified']=restore_verified
        upload_json(report,storage['final_report'])
        (persistent/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
        publish(state='complete',archive=archived,accepted_trials=summary['accepted_trials'],
            actual_training_reader=reader,artifact_acceptance_complete=True)
    except BaseException as e:
        diagnostics=persistent/'failure_diagnostics';diagnostics.mkdir(exist_ok=True)
        for path in [*scratch.glob('*.log'),scratch/'status.json',
                     *scratch.glob('results/pi05/*.log')]:
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-256*1024))
                    (diagnostics/path.name).write_bytes(stream.read())
        records=[]
        for path in scratch.glob('results/pi05/*/trials/*/completed.json'):
            records.append(dict(path=str(path.relative_to(scratch)),completed=json.loads(path.read_text())))
        (diagnostics/'completed_trials.json').write_text(json.dumps(records,indent=2)+'\n')
        # Network helpers sanitize their exceptions so signed URLs cannot enter this log.
        (persistent/'error.log').write_text(traceback.format_exc())
        publish(state='failed',error_type=type(e).__name__,error=str(e))
        raise

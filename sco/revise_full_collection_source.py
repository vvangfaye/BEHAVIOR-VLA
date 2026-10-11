"""Revise storage/runtime helpers while retaining an independently audited source scope."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def run(payload):
    parent,code,output=(Path(payload[k]) for k in ('parent_code_root','code_root','output_root'))
    output.mkdir(parents=True,exist_ok=False)
    parent_bytes=(parent/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(parent_bytes).hexdigest()==payload['parent_source_manifest_sha256']
    old=json.loads(Path(payload['parent_campaign_path']).read_text());assert old['state']=='complete' and old['prepared_group_count']==299
    assert old['source_manifest_sha256']==payload['parent_source_manifest_sha256']
    overlay=base64.b64decode(payload['overlay']);assert hashlib.sha256(overlay).hexdigest()==payload['overlay_sha256']
    shutil.copytree(parent,code,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    allowed={'scripts/collection_checkpoint.py','sco/run_full_collection_shard.py','sco/test_collection_object_checkpoint.py',
             'sco/test_full_subtask_collection.py','sco/revise_full_collection_source.py'}
    assert set(payload['files_sha256'])<=allowed
    with tarfile.open(fileobj=io.BytesIO(overlay),mode='r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in payload['files_sha256']
            data=archive.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==payload['files_sha256'][member.name]
            (code/member.name).write_bytes(data)
    marker=json.loads(parent_bytes);marker['overlay_files_sha256'].update(payload['files_sha256'])
    marker.update(code_root=str(code),parent_code_root=str(parent),parent_source_manifest_sha256=payload['parent_source_manifest_sha256'],
                  created_at=time.time(),revision='case_insensitive_object_receipts_and_persistent_failure_diagnostics')
    for n,sha in marker['overlay_files_sha256'].items():assert hashlib.sha256((code/n).read_bytes()).hexdigest()==sha
    body=(json.dumps(marker,indent=2)+'\n').encode();(code/'_CODE_SNAPSHOT_READY').write_bytes(body);identity=hashlib.sha256(body).hexdigest()
    runtime=Path(payload['runtime_root']);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMNIGIBSON_NO_OMNIVERSE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=str(code/'sco')+':'+str(code/'scripts'),LD_LIBRARY_PATH=str(runtime/'envs/b1k-eval/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    env.pop('PYTHONHOME',None)
    with (output/'pytest.log').open('x') as log:
        subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),'-m','pytest','sco/test_accepted_subtask_collection.py',
            'sco/test_full_subtask_collection.py','sco/test_additional_training_collection.py','sco/test_collection_object_checkpoint.py',
            'sco/test_collection_resume.py','sco/test_history_artifact_accounting.py','scripts/test_subtask_improvement.py','-q'],
            cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    campaign=output/'campaign';campaign.mkdir();oldroot=Path(old['output_root']);groups=[]
    for n in ('parent_panel.json','additional_panel.json','source_evidence.json'):
        shutil.copyfile(oldroot/n,campaign/n)
    assert hashlib.sha256((campaign/'source_evidence.json').read_bytes()).hexdigest()==old['source_evidence_sha256']
    for entry in old['groups']:
        directory=campaign/'groups'/('%03d'%entry['index']);directory.mkdir(parents=True)
        olddir=Path(entry['output_root']);shutil.copyfile(olddir/'collection_plan.json',directory/'collection_plan.json')
        original_panel=olddir/'parent_panel.json';target='additional_panel.json' if entry['plan'].get('additional_start_source_evidence_sha256') else 'parent_panel.json'
        assert original_panel.read_bytes()==(campaign/target).read_bytes();(directory/'parent_panel.json').symlink_to(campaign/target)
        revised=dict(entry,output_root=str(directory),code_root=str(code),source_manifest_sha256=identity,
            source_audit_parent_source_manifest_sha256=old['source_manifest_sha256'])
        (directory/'prepared.json').write_text(json.dumps(revised,indent=2)+'\n');groups.append(revised)
    result=dict(old,groups=groups,code_root=str(code),output_root=str(campaign),source_manifest_sha256=identity,
        source_audit_parent_source_manifest_sha256=old['source_manifest_sha256'],source_audit_parent_campaign_path=str(oldroot/'campaign.json'),
        completed_at=time.time())
    (campaign/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    # The physics, annotation and baseline audit code is unchanged by the storage-only revision.
    gates=[]
    import sys
    sys.path[:0]=[str(code/'scripts'),str(code/'sco')]
    from full_subtask_collection import freeze
    from subtask_sample_panel import load_panel
    for kind in ('native','history','additional'):
        previous=Path(payload['parent_gates_root'])/kind;ready=json.loads((previous/'prepared.json').read_text());panel=load_panel(previous/'parent_panel.json')
        prior=ready['plan'];sid=prior['trials'][0]['sample_id'];sample=next(s for s in panel['samples'] if s['sample_id']==sid)
        evidence=json.loads((previous/'source_evidence.json').read_text());proof=evidence['samples'][sid]
        directory=output/'standard_gates'/kind;directory.mkdir(parents=True)
        evidence_bytes=(previous/'source_evidence.json').read_bytes();(directory/'source_evidence.json').write_bytes(evidence_bytes)
        plan=freeze(panel,prior['selected_groups'][0],[sample],{sid:proof},prior['checkpoint_identity'],prior['held_out_instances'],
            accepted_input_sha256=prior['accepted_input_sha256'],evidence_manifest_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
            repeats=1,max_steps=2048,seed=1010)
        (directory/'collection_plan.json').write_text(json.dumps(plan,indent=2)+'\n');shutil.copyfile(previous/'parent_panel.json',directory/'parent_panel.json')
        item=dict(ready,output_root=str(directory),code_root=str(code),source_manifest_sha256=identity,plan=plan)
        (directory/'prepared.json').write_text(json.dumps(item,indent=2)+'\n');gates.append(item)
    report=dict(state='complete',source_manifest_sha256=identity,parent_source_manifest_sha256=old['source_manifest_sha256'],
                campaign_path=str(campaign/'campaign.json'),requested_groups=299,expected_trials=old['expected_trials'],gates=gates)
    (output/'ready.json').write_text(json.dumps(report,indent=2)+'\n');return report

"""CPU artifact acceptance for a completed pilot or a complete collection wave."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(payload):
    code,prepared=Path(payload['code_root']),Path(payload['prepared_root'])
    source=(code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(source).hexdigest()==payload['source_manifest_sha256']
    for name,expected in json.loads(source)['overlay_files_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected
    sys.path[:0]=[str(code/'scripts'),str(code/'sco')]
    from subtask_sample_panel import load_panel
    from subtask_improvement import load_plan
    panel=load_panel(prepared/'parent_panel.json')
    plan,_=load_plan(prepared/'collection_plan.json',panel)
    assert plan['fingerprint']==payload['plan_fingerprint']
    if payload['mode']=='pilot':
        pilot=payload['pilot']
        assert pilot['platform_state']=='SUCCEEDED' and pilot['owner']=='260010165'
        root=Path(pilot['output_root'])
        status=json.loads((root/'status.json').read_text())
        assert status['state']=='complete' and status['trainable_dataset_verified'] is True
        assert status['source_manifest_sha256']==payload['source_manifest_sha256']
        assert status['plan_fingerprint']==plan['fingerprint']
        execution=json.loads((root/'source_executions'/(pilot['name']+'.json')).read_text())
        assert execution['driver_sha256']==pilot['driver_sha256']
        from collect_b1k_subtasks import validate_collection
        summary=validate_collection(root/'results',panel,plan)
        assert summary['ready_for_export']
        dataset=Path(status['dataset_root'])
        assert dataset==root/'exports'/pilot['name']
        from verify_subtask_collection_export import verify_export
        verification=verify_export(dataset,pilot['name'],collection_root=root/'results')
        result=dict(status,collection_summary=summary,cpu_export_verification=verification)
        repo_id=pilot['name']
    else:
        assert payload['mode']=='finalize'
        namespace={};exec(compile(payload['finalizer_source'],'frozen_finalizer','exec'),namespace)
        result=namespace['run'](payload['finalizer_payload'])
        if not result['trainable_dataset_verified']:return result
        dataset=Path(result['dataset_root']);repo_id=payload['finalizer_payload']['repo_id']
    audit_root=Path(payload['audit_root']);audit_root.mkdir(parents=True,exist_ok=False)
    script=audit_root/'training_reader.py'
    script.write_text(payload['reader_source']+'\nresult=run('+repr(dict(dataset_root=str(dataset),repo_id=repo_id))+')\n'+
        'Path('+repr(str(audit_root/'training_reader.json'))+').write_text(json.dumps(result,indent=2)+"\\n")\n')
    runtime=Path(payload['runtime_root'])
    env=os.environ.copy();env.pop('PYTHONHOME',None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=str(code/'openpi/src')+':'+str(code/'BEHAVIOR-1K')+':'+str(code/'BEHAVIOR-1K/OmniGibson'),
        LD_LIBRARY_PATH=str(runtime/'envs/b1k-pi05/lib')+':'+env.get('LD_LIBRARY_PATH',''))
    with (audit_root/'training_reader.log').open('x') as log:
        subprocess.run([str(runtime/'envs/b1k-pi05/bin/python'),str(script)],env=env,
            cwd=code,stdout=log,stderr=subprocess.STDOUT,check=True)
    reader=json.loads((audit_root/'training_reader.json').read_text())
    assert reader['complete'] and reader['actual_openpi_training_reader']
    result.update(actual_training_reader=reader,artifact_acceptance_complete=True)
    (audit_root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

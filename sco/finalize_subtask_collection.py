"""Verify completed disjoint task shards and export one dense LeRobot dataset."""
import json
import hashlib
import fcntl
import os
from pathlib import Path
import subprocess
import sys


def run(payload):
    code, prepared, results, output = (Path(payload[k]) for k in
        ('code_root','prepared_root','results_root','output_root'))
    assert not output.exists()
    marker=(code/'_CODE_SNAPSHOT_READY').read_bytes()
    assert hashlib.sha256(marker).hexdigest()==payload['source_manifest_sha256']
    for name, expected in json.loads(marker)['overlay_files_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected
    lock=(results/'.export-writer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    executions = payload['executions']
    assert executions and len({r['uid'] for r in executions}) == len(executions)
    sys.path.insert(0, str(code/'scripts'))
    from subtask_sample_panel import load_panel
    from subtask_improvement import load_plan
    from collect_b1k_subtasks import validate_collection
    panel = load_panel(prepared/'parent_panel.json')
    plan, samples = load_plan(prepared/'collection_plan.json',panel)
    assert plan['fingerprint']==payload['plan_fingerprint']
    covered = set()
    for execution in executions:
        assert execution['platform_state']=='SUCCEEDED' and execution['owner']=='260010165'
        assert execution['source_manifest_sha256']==payload['source_manifest_sha256']
        receipt=json.loads((Path(execution['output_root'])/'source_executions'/
            (execution['name']+'.json')).read_text())
        assert receipt['execution_name']==execution['name']
        assert receipt['source_manifest_sha256']==payload['source_manifest_sha256']
        assert receipt['plan_fingerprint']==plan['fingerprint']
        assert receipt['shared_results_root']==str(results)
        assert receipt['task_name']==execution['task_name']
        shard = json.loads((Path(execution['output_root'])/'_SHARD_SUCCESS').read_text())
        assert shard['plan_fingerprint']==plan['fingerprint'] and shard['shard_complete'] is True
        ids = set(shard['assigned_trial_ids'])
        assert ids=={t['trial_id'] for t in plan['trials'] if samples[t['sample_id']]['task']==execution['task_name']}
        assert not covered & ids
        covered |= ids
    assert covered == {t['trial_id'] for t in plan['trials']}
    summary = validate_collection(results,panel,plan)
    (results/'collection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (results/'_SUCCESS').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['ready_for_export']:
        return dict(state='complete',zero_success=True,trainable_dataset_verified=False,summary=summary)
    runtime = Path(payload['runtime_root'])
    env=os.environ.copy()
    env.pop('PYTHONHOME',None)
    env.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(code/'BEHAVIOR-1K')+':'+str(code/'BEHAVIOR-1K/OmniGibson'),
        LD_LIBRARY_PATH=str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
    subprocess.run([str(runtime/'envs/b1k-eval/bin/python'),str(code/'scripts/export_subtask_collection.py'),
        '--collection-root',str(results),'--panel',str(prepared/'parent_panel.json'),
        '--collection-plan',str(prepared/'collection_plan.json'),'--output',str(output),
        '--repo-id',payload['repo_id']],cwd=code,env=env,check=True)
    assert (output/'_VERIFIED_EXPORT').is_file()
    return dict(state='complete',trainable_dataset_verified=True,dataset_root=str(output),
                summary=summary,export=json.loads((output/'export_report.json').read_text()))

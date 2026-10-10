"""Exercise partial-attempt validation and process-level resume on CPU fixtures."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import collect_b1k_subtasks as collector
from test_subtask_improvement import inputs,plan_for,write_trial


def test_native_source_binding_imports_its_deployed_dependencies(monkeypatch):
    import recorded_native_state as native
    monkeypatch.setattr(native,'preview_source_role_bindings',lambda *args:{})
    proof={'contract':'recorded_native_fields_v1','recorded_native_fields_verified_before_policy':True}
    assert native.source_role_bindings({},proof,{})=={'runtime_native_fields_verified':True}


def test_partial_success_cannot_be_exported_and_tampering_prevents_resume(tmp_path):
    panel,baseline=inputs(); plan=plan_for(panel,baseline)
    trial=plan['trials'][0]
    directory=write_trial(tmp_path,plan,panel,trial)
    result=collector.validate_collection(tmp_path,panel,plan,require_complete=False)
    assert result['completed_trial_ids']==[trial['trial_id']]
    assert result['accepted_trials']==1 and not result['complete'] and not result['ready_for_export']
    with pytest.raises(ValueError,match='incomplete'):
        collector.validate_collection(tmp_path,panel,plan)
    receipt=directory/'accepted.json'
    altered=json.loads(receipt.read_text());altered['content_sha256']='0'*64
    receipt.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='acceptance receipt'):
        collector.validate_collection(tmp_path,panel,plan,require_complete=False)


def test_resume_does_not_launch_completed_source_episodes(tmp_path,monkeypatch):
    panel,baseline=inputs(); plan=plan_for(panel,baseline)
    samples={s['sample_id']:s for s in panel['samples']}
    first=plan['trials'][0]['sample_id']
    for trial in plan['trials']:
        if trial['sample_id']==first:write_trial(tmp_path,plan,panel,trial)
    monkeypatch.setattr(collector.runner,'load_panel',lambda path:panel)
    monkeypatch.setattr(collector,'load_plan',lambda path,panel:(plan,samples))
    commands=[]
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda command,**kwargs:commands.append(command))
    collector.main(['--subtask-panel','panel.json','--collection-plan','plan.json',
        '--raw-data-root','raw','--sample-max-steps','2048','--task-name','task',
        '--output-dir',str(tmp_path/'pi05/task'),'--resume-completed'])
    remaining={samples[t['sample_id']]['episode_index'] for t in plan['trials'] if t['sample_id']!=first}
    assert {int(c[-1]) for c in commands}==remaining
    assert all(c[-2]=='--episode-index' for c in commands)

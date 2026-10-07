"""Pilot scope, source identity, held-out separation and real control recording."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import subtask_improvement as plans
from collect_b1k_subtasks import AuditedTrajectoryWriter
from subtask_trajectory import TrajectoryWriter, CAMERAS
from test_subtask_improvement import inputs


def pilot():
    panel, summary=inputs()
    ids=['s0','s2']
    identity=dict(checkpoint_step=1375000, checkpoint_manifest_sha256='a'*64,
                  checkpoint_source='oss://fixture/base')
    for s in panel['samples']: s['start']=32
    proofs={sid:dict(contract='recorded_native_fields_v1',sample_id=sid,source_start=32,
                    raw_sha256='b'*64,selected_state_sha256='c'*64) for sid in ids}
    evidence=dict(verified=True,complete=True,panel_fingerprint=panel['fingerprint'],
        **identity,samples=[dict(sample_id='s0',metric_eligible=True,steps=3,
                                raw_native_fields_independently_compared=True)])
    # One subtask group with two distinct source instances, one GPU-proven start.
    return panel, ids, proofs, evidence, identity


def freeze(*,change=None,**kwargs):
    panel, ids, proofs, evidence, identity=pilot()
    if change: change(panel,ids,proofs,evidence,identity)
    return panel, plans.freeze_pilot_plan(panel,ids,proofs,evidence,
        checkpoint_identity=identity,held_out_instances=[['task',99]],**kwargs)


def test_explicit_pilot_does_not_require_or_invent_a_difficulty_baseline(tmp_path):
    panel,p=freeze()
    assert p['evaluation_summary_sha256'] is None
    assert p['difficulty_selection_verified'] is False
    assert p['production_difficulty_baseline'] is False
    assert p['selected_groups'][0]['rate'] is None
    assert p['expected_trials']==6 and p['expected_starts']==2
    assert p['maximum_control_steps']==6*768
    path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    assert plans.load_plan(path,panel)[0]==p


@pytest.mark.parametrize('change',[
    lambda p,ids,s,e,i: e.update(verified=False),
    lambda p,ids,s,e,i: e.update(checkpoint_manifest_sha256='d'*64),
    lambda p,ids,s,e,i: e.update(samples=[]),
    lambda p,ids,s,e,i: s['s2'].update(selected_state_sha256='invalid'),
    lambda p,ids,s,e,i: s['s2'].update(source_start=0),
    lambda p,ids,s,e,i: p['samples'][2].update(task_instance_id=99),
])
def test_pilot_keeps_native_source_checkpoint_runtime_and_holdout_gates(change):
    with pytest.raises(ValueError):freeze(change=change)


@pytest.mark.parametrize('limits',[dict(trials=7),dict(max_steps=1025),dict(max_steps=True)])
def test_pilot_is_bounded(limits):
    with pytest.raises(ValueError):freeze(**limits)


@pytest.mark.parametrize('field,value',[('difficulty_selection_verified',True),('production_difficulty_baseline',True),('expected_starts',3)])
def test_rehash_does_not_turn_pilot_into_formal_evidence(tmp_path,field,value):
    panel,p=freeze();p[field]=value;p['fingerprint']=plans.digest({k:v for k,v in p.items() if k!='fingerprint'})
    path=tmp_path/'plan.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):plans.load_plan(path,panel)


def test_audited_writer_records_real_arrays_and_rejects_invalid_control(tmp_path):
    writer=TrajectoryWriter(tmp_path/'trajectory.hdf5',{'purpose':'test'})
    audited=AuditedTrajectoryWriter(writer)
    frame={'observation.state':np.zeros(61,dtype=np.float32),
           **{k:np.zeros((224,224,3),dtype=np.uint8) for k in CAMERAS}}
    action=np.ones(23,dtype=np.float32)
    try:
        audited.record(frame,action,False,False)
        assert writer.count==1 and audited.audit.rows[0]['executed_action']==action.tolist()
        with pytest.raises(ValueError):audited.record(frame,np.full(23,np.nan),False,False)
        assert writer.count==1 and len(audited.audit.rows)==1
    finally:writer.close()

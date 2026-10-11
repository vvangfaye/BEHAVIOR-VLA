"""Audit tests use fake simulator/CLI output; never treat fixtures as real success."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "sco")]
spec = importlib.util.spec_from_file_location("pilot002_attempt_audit_builder", ROOT / "eval_results/codex_pilot_20261011/build_pilot002_attempt_audit_v5.py")
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))


@pytest.fixture
def worker():
    namespace={}
    source=("import hashlib,json,os,re,signal,sys,time\nfrom collections import Counter\nfrom pathlib import Path\n"
            + builder.shared_helpers() + "\n" + builder.AUDIT)
    exec(compile(source,"offline_audit","exec"),namespace)
    return namespace


@pytest.fixture
def attempt(tmp_path):
    from collect_b1k_codex import collect_attempt
    from test_collect_b1k_codex import Evaluator,metadata
    from codex_collection_teacher import CodexTeacher
    from test_codex_collection_teacher import FakeRunner
    root=tmp_path / "collection";root.mkdir()
    result=collect_attempt(Evaluator(terminal_step=1),CodexTeacher(runner=FakeRunner()),root / "attempt",
                           {**metadata(),"attempt_id":"pilot-002"},max_steps=10,max_decisions=2,teacher_interval=256)
    completed=dict(state="complete",result=result,training_eligible=True,official_benchmark=False)
    dump(root / "_COMPLETE",completed);dump(root / "status.json",completed)
    return dict(run_root=str(root),audit_root=str(tmp_path / "audit"),wrapper_root=str(tmp_path / "wrapper"),
                code=str(ROOT),source_files_sha256={},source_sha256="a"*64,
                job=dict(name=builder.JOB_NAME,uid=builder.JOB_UID,state="SUCCEEDED"))


def test_actual_collector_evidence_audits_without_mutating_or_calling_model(worker,attempt,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:pytest.fail("Audit cannot invoke models or external processes"))
    before=worker["snapshot"](Path(attempt["run_root"]))
    report=worker["run"](attempt)
    assert report["state"]=="audit_complete" and report["outcome"]=="verified_success"
    assert report["training_eligible"] is True and report["source_unchanged_during_audit"] is True
    assert worker["snapshot"](Path(attempt["run_root"]))==before
    assert report["teachers"]["observed_token_totals"]==dict(input_tokens=1234,cached_input_tokens=1000,output_tokens=42,reasoning_output_tokens=20)
    assert report["teachers"]["actions"]=={"continue":1}
    assert report["teachers"]["entries"][0]["event_audit_valid"] is True
    assert report["actions"]["sources"]=={"policy_conditioned_on_teacher_subtask":1}
    assert report["model_calls"]==report["network_calls"]==0
    assert report["model_calls_scope"]=="audit_process_only"
    assert report["teachers"]["model_calls_observed"]==1
    assert report["teachers"]["requests_with_unknown_model_call"]==0


def test_missing_usage_stays_unknown(worker,attempt):
    path=next((Path(attempt["run_root"])/"attempt/teacher").glob("step-*/decision/*"))
    (path/"stdout.log").write_text('{"type":"turn.completed"}\n')
    receipt=json.loads((path/"receipt.json").read_text());receipt["usage"]=None;dump(path/"receipt.json",receipt)
    report=worker["run"](attempt)
    assert report["teachers"]["requests_without_usage"]==1
    assert report["teachers"]["observed_token_totals"] is None
    assert report["teachers"]["missing_usage_is_zero"] is False
    assert report["teachers"]["model_calls_observed"]==1


@pytest.mark.parametrize("stage,expected_unknown",[("authentication",False),("tool_configuration",False),("execution",True),(None,True)])
def test_model_call_unknown_is_separate_from_pre_inference_failure(worker,tmp_path,stage,expected_unknown):
    receipt={"status":"failed","stage":stage} if stage else {}
    result=worker["model_call_evidence"](tmp_path/"absent_stdout.log",receipt)
    assert result["observed_model_turns"]==0 and result["invocation_unknown"] is expected_unknown
    assert result["failed_before_model"] is not expected_unknown


def test_model_turn_observed_without_usage_is_not_zero(worker,tmp_path):
    path=tmp_path/"events.log"
    path.write_text('{"type":"turn.started"}\n{"type":"turn.failed"}\n')
    result=worker["model_call_evidence"](path,{"status":"failed","stage":"execution"})
    assert result["observed_model_turns"]==1 and result["invocation_unknown"] is False


def test_main_failure_marker_disqualifies_otherwise_complete_attempt(worker,attempt):
    dump(Path(attempt["run_root"])/"collector_failure.json",dict(stage="shutdown",error_type="RuntimeError",training_eligible=False))
    report=worker["run"](attempt)
    assert report["training_eligible"] is False
    assert "collector_main_failure_receipt_present" in report["issues"]
    assert report["receipts"]["collector_failure.json"]["stage"]=="shutdown"


def test_tool_event_rejected_after_saved_receipt(worker,attempt):
    path=next((Path(attempt["run_root"])/"attempt/teacher").glob("step-*/decision/*"))/"stdout.log"
    with path.open("a") as stream:stream.write('{"type":"item.completed","item":{"type":"command_execution"}}\n')
    report=worker["run"](attempt)
    assert report["training_eligible"] is False
    assert "teacher_event_audit_failed" in report["issues"]


def test_corrupted_trajectory_is_not_training_data(worker,attempt):
    import h5py
    with h5py.File(Path(attempt["run_root"])/"attempt/trajectory.hdf5","r+") as stream:stream["action"][0,0]=9
    report=worker["run"](attempt)
    assert report["state"]=="audit_complete" and report["training_eligible"] is False
    assert report["trajectory"]["valid_complete_trajectory"] is False


def test_failed_correction_message_and_incomplete_shapes_are_readonly_diagnostics(worker,attempt):
    import h5py
    root=Path(attempt['run_root'])
    with h5py.File(root/'attempt/trajectory.hdf5','r+') as stream:stream.attrs['complete']=False
    dump(root/'attempt/failed.json',dict(steps=1,error_type='CorrectionError',training_eligible=False,
        message='IK solution exceeds joint position/step bounds\nTOKEN=private\nhttps://private.invalid/secret'))
    before=worker['snapshot'](root);report=worker['run'](attempt)
    assert report['state']=='audit_complete' and report['training_eligible'] is False
    message=report['receipts']['attempt/failed.json']['message']
    assert 'IK solution exceeds joint position/step bounds' in message
    assert 'private' not in message and len(message.encode())<=2048
    trajectory=report['trajectory']
    assert trajectory['valid_complete_trajectory'] is False
    assert trajectory['incomplete_schema']['diagnostic_only'] is True
    assert trajectory['incomplete_schema']['attributes']['complete'] is False
    assert trajectory['incomplete_schema']['datasets']['action']['shape']==[1,23]
    assert worker['snapshot'](root)==before


def test_last_teacher_correction_and_executed_diagnostics_are_bounded_and_safe(worker,attempt):
    root=Path(attempt['run_root'])
    path=next((root/'attempt/teacher').glob('step-*/decision/*'))
    correction=dict(arm='right',delta_position_m=[0.001,0,0],delta_rotation_axis_angle_rad=[0,0,0],gripper='keep',steps=2)
    for name in ('response.json','response.raw.json'):
        value=json.loads((path/name).read_text());value.update(action='correct',correction=correction,reason='Fine alignment')
        dump(path/name,value)
    diagnostics=dict(arm='right',max_joint_delta_rad=0.01,linearized_position_residual_m=0.0001,
        linearized_orientation_residual_rad=0.001,server_token='never_report')
    dump(root/'attempt/action_sources.jsonl',dict(step=0,source='teacher_cartesian_correction',diagnostics=diagnostics))
    report=worker['run'](attempt)
    decision=report['teachers']['recent_decisions'][-1]
    assert decision['correction']==correction and decision['reason']=='Fine alignment'
    assert report['actions']['recent_actions'][-1]['diagnostics']['max_joint_delta_rad']==0.01
    assert 'never_report' not in json.dumps(report)
    assert worker['correction_diagnostics']({'max_joint_delta_rad':float('inf'),'other':'secret'})=={}


@pytest.mark.parametrize("target",["source","subtask"])
def test_execution_must_match_teacher_decision(worker,attempt,target):
    root=Path(attempt["run_root"])
    if target=="source":
        dump(root/"attempt/action_sources.jsonl",dict(step=0,source="teacher_cartesian_correction"))
    else:
        path=next((root/"attempt/teacher").glob("step-*/decision/*"))
        for name in ["response.json","response.raw.json"]:
            value=json.loads((path/name).read_text());value['subtask']='a different legal subtask';dump(path/name,value)
    report=worker["run"](attempt)
    assert report["training_eligible"] is False
    issue="executed_action_source_differs_from_teacher" if target=="source" else "executed_subtask_differs_from_teacher"
    assert issue in report["issues"]


def test_teacher_subtask_whitespace_uses_the_collectors_normalization(worker,attempt):
    path=next((Path(attempt["run_root"])/"attempt/teacher").glob("step-*/decision/*"))
    for name in ["response.json","response.raw.json"]:
        value=json.loads((path/name).read_text());value['subtask']='grasp  the\n cup';dump(path/name,value)
    report=worker["run"](attempt)
    assert report["training_eligible"] is True
    assert report["teachers"]["subtasks"]=={"grasp the cup":1}


def test_startup_failure_preserves_evidence_and_redacts_logs(worker,tmp_path):
    root=tmp_path/"collection";root.mkdir();secret="f"*64
    dump(root/"status.json",dict(state="failed",error_type="RuntimeError"))
    (root/"driver.log").write_text("startup failed\nAuthorization: Bearer secret\nhttps://private.invalid/key\n"+secret)
    config=dict(run_root=str(root),audit_root=str(tmp_path/"audit"),wrapper_root=str(tmp_path/"wrapper"),
        code=str(ROOT),source_files_sha256={},source_sha256="a"*64,job=dict(name=builder.JOB_NAME,uid=builder.JOB_UID,state="FAILED"))
    before=worker["snapshot"](root);report=worker["run"](config)
    assert report["state"]=="audit_complete" and report["training_eligible"] is False
    assert report["teachers"]["observed_token_totals"] is None
    raw=json.dumps(report)
    assert secret not in raw and "private.invalid" not in raw and "Bearer secret" not in raw
    assert worker["snapshot"](root)==before


def test_deadline_escapes_per_evidence_exception_handlers(worker,attempt,monkeypatch):
    def stop(path):raise worker["AuditDeadline"]("Fixture deadline")
    monkeypatch.setitem(worker,"trajectory_summary",stop)
    report=worker["run"](attempt)
    assert report["state"]=="audit_failed" and report["error_type"]=="AuditDeadline"
    assert report["training_eligible"] is False


def test_output_cannot_be_inside_original_attempt(worker,attempt):
    attempt["audit_root"]=str(Path(attempt["run_root"])/"audit")
    with pytest.raises(ValueError,match="separate"):worker["run"](attempt)
    assert not Path(attempt["audit_root"]).exists()


def test_builder_requires_terminal_exact_job_and_frozen_submitted_source(tmp_path):
    collection=dict(task="turning_on_radio",instance_id=311,attempt_id="pilot-002",run_root=builder.AFS+"/collection/radio-311-pilot002",
                    code=builder.AFS+"/code-v7",source_files_sha256={})
    config=dict(collection=collection,report_root=builder.AFS+"/first-radio311-gpu-v5")
    source=tmp_path/"gpu.py";source.write_text("BOOTSTRAP_REPORT = run("+repr(config)+")\n")
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    job=dict(name=builder.JOB_NAME,uid=builder.JOB_UID,state="RUNNING")
    job_path=tmp_path/"terminal.json";dump(job_path,job)
    submission_path=tmp_path/"submission.json";dump(submission_path,{**job,"source_sha256":digest})
    target=tmp_path/"audit.py"
    with pytest.raises(ValueError,match="terminal"):
        builder.build(job_path,digest,output=target,source_path=source,submission_path=submission_path)
    job["state"]="FAILED";dump(job_path,job)
    with pytest.raises(ValueError,match="SHA256"):
        builder.build(job_path,"b"*64,output=target,source_path=source,submission_path=submission_path)
    result=builder.build(job_path,digest,output=target,source_path=source,submission_path=submission_path)
    assert result["actual_job_submitted"] is False and result["max_seconds"]==600
    parsed=ast.parse(target.read_text());compile(parsed,str(target),"exec")
    imports={name.name.split('.')[0] for node in ast.walk(parsed) if isinstance(node,ast.Import) for name in node.names}
    assert not imports & {"urllib","socket","subprocess","requests"}
    assert "BOOTSTRAP_REPORT = run(" in target.read_text()

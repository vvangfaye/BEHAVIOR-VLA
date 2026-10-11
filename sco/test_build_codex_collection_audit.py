"""Offline synthetic evidence tests; fixtures never assert real robot success."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sco"), str(ROOT / "scripts")]
import build_codex_collection_audit as builder
import build_codex_next_collection as next_builder
import test_build_codex_next_collection as next_tests


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    previous = next_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    source = tmp_path / "next_collection.py"
    generated = next_builder.build(source, **previous)
    job = dict(name="owned-shoes311-followup001", uid="actual-submitted-uid", state="FAILED", observed_at=1000)
    job_path, submission = tmp_path / "terminal.json", tmp_path / "submitted.json"
    dump(job_path, job); dump(submission, {**job, "source_sha256": generated["sha256"]})
    return dict(job_receipt=job_path, source_sha256=generated["sha256"], source_path=source,
                submission_path=submission, output=tmp_path / "audit.py", now=1000)


def rewrite_source(inputs, edit):
    path = inputs["source_path"]
    config = next_builder.frozen_config(path); edit(config)
    path.write_text(next_builder.WORKER + "\nBOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
    inputs["source_sha256"] = next_builder.sha(path)
    submission = next_builder.load(inputs["submission_path"])
    submission["source_sha256"] = inputs["source_sha256"]; dump(inputs["submission_path"], submission)


@pytest.mark.parametrize("state", sorted(builder.pilot.TERMINAL))
def test_real_next_builder_source_with_any_terminal_state_can_build(inputs, state):
    job = next_builder.load(inputs["job_receipt"]); job["state"] = state; dump(inputs["job_receipt"], job)
    result = builder.build(**inputs)
    assert result["identity"] == dict(task="putting_shoes_on_rack", instance_id=311, attempt_id="followup-001")
    assert result["max_steps"] == 10000 and result["max_decisions"] == 32
    assert result["model_calls"] == result["network_calls"] == 0 and result["actual_job_submitted"] is False
    assert result["source_job"]["uid"] == job["uid"]
    tree = ast.parse(inputs["output"].read_text()); compile(tree, str(inputs["output"]), "exec")
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not imports & {"urllib", "requests", "socket", "subprocess"}
    for path in (inputs["output"], inputs["output"].with_suffix(".receipt.json")):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("damage", ["active", "stale", "future", "uid", "source_sha", "worker", "wrong_steps", "too_many_decisions", "instance", "code", "root", "inventory", "admission"])
def test_boundaries_reject_before_writing_audit(inputs, damage):
    if damage in {"active", "stale", "future", "uid"}:
        job = next_builder.load(inputs["job_receipt"])
        job.update({"active": {"state": "RUNNING"}, "stale": {"observed_at": 99},
                    "future": {"observed_at": 1001}, "uid": {"uid": "other"}}[damage])
        dump(inputs["job_receipt"], job)
    elif damage == "source_sha": inputs["source_sha256"] = "b" * 64
    elif damage == "worker":
        path = inputs["source_path"]; path.write_text("print('unreviewed')\n" + path.read_text())
        inputs["source_sha256"] = next_builder.sha(path)
        submission = next_builder.load(inputs["submission_path"])
        submission["source_sha256"] = inputs["source_sha256"]; dump(inputs["submission_path"], submission)
    else:
        def edit(config):
            if damage == "wrong_steps": config["collection"]["max_steps"] = 4500
            elif damage == "too_many_decisions": config["collection"]["max_decisions"] = 33
            elif damage == "instance": config["collection"]["instance_id"] = 301
            elif damage == "code": config["collection"]["code"] = builder.pilot.AFS + "/code-v6"
            elif damage == "root": config["collection"]["run_root"] = builder.pilot.AFS + "/collection/unrelated"
            elif damage == "inventory": config["collection"]["source_files_sha256"]["../outside"] = "f" * 64
            elif damage == "admission": config["admission"]["reserved_decisions"] = 0
        rewrite_source(inputs, edit)
    with pytest.raises(ValueError): builder.build(**inputs)
    assert not inputs["output"].exists()


def test_generated_audit_is_accepted_by_next_admission_contract(inputs, tmp_path):
    result = builder.build(**inputs)
    config = ast.literal_eval(ast.parse(inputs["output"].read_text()).body[-1].value.args[0])
    # This is an admission contract fixture, not an actual audit or cloud receipt.
    audit = dict(kind=builder.KIND, state="audit_complete", source_unchanged_during_audit=True,
        source_job=config["job"], collection_source_sha256=config["source_sha256"],
        run_root=config["run_root"], audit_root=result["audit_root"], model_calls=0, network_calls=0,
        model_calls_scope="audit_process_only", training_eligible=False, issues=["failed_attempt"], actions={}, trajectory={},
        teachers=dict(requests=1, model_calls_observed=0, requests_with_unknown_model_call=1, requests_failed_before_model=0))
    path = tmp_path / "audit_report.json"; dump(path, audit)
    previous = next_builder.checked_previous(inputs["job_receipt"], inputs["submission_path"],
        inputs["source_path"], path, now=1000)
    assert previous["audit"]["teachers"]["requests_with_unknown_model_call"] == 1


def test_real_radio_next_source_uses_4500_steps_and_its_exact_identity(tmp_path, monkeypatch):
    previous = next_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    previous.update(task="turning_on_radio", instance_id=320, attempt_id="radio-next", demo_episode=28)
    source = tmp_path / "radio_next.py"; result = next_builder.build(source, **previous)
    job = dict(name="owned-radio320", uid="actual-radio-uid", state="SUCCEEDED", observed_at=1000)
    job_path, submission = tmp_path / "terminal.json", tmp_path / "submitted.json"
    dump(job_path, job); dump(submission, {**job, "source_sha256": result["sha256"]})
    audited = builder.build(job_path, result["sha256"], submission_path=submission,
                            source_path=source, output=tmp_path / "audit.py", now=1000)
    assert audited["max_steps"] == 4500
    assert audited["identity"] == dict(task="turning_on_radio", instance_id=320, attempt_id="radio-next")


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.IMPORTS + builder.pilot.shared_helpers() + "\n" + builder.AUDIT,
                 "generic_readonly_audit", "exec"), namespace)
    return namespace


@pytest.fixture(params=[("turning_on_radio", 314, "radio-followup", 4500), ("putting_shoes_on_rack", 320, "shoes-followup", 10000)])
def attempt(tmp_path, request):
    from collect_b1k_codex import collect_attempt
    from test_collect_b1k_codex import Evaluator, metadata
    from codex_collection_teacher import CodexTeacher
    from test_codex_collection_teacher import FakeRunner
    task, instance, name, limit = request.param
    identity = dict(task=task, instance_id=instance, attempt_id=name)
    root = tmp_path / "collection"; root.mkdir()
    result = collect_attempt(Evaluator(terminal_step=1), CodexTeacher(runner=FakeRunner()), root / "attempt",
                             {**metadata(), **identity}, max_steps=limit, max_decisions=32, teacher_interval=256)
    completed = dict(state="complete", result=result, training_eligible=True, official_benchmark=False)
    dump(root / "_COMPLETE", completed); dump(root / "status.json", completed)
    return dict(run_root=str(root), audit_root=str(tmp_path / "audit"), wrapper_root=str(tmp_path / "wrapper"),
        code=str(ROOT), source_files_sha256={}, source_sha256="a" * 64,
        job=dict(name="owned-generic", uid="submitted-uid", state="SUCCEEDED"), identity=identity, max_steps=limit)


def test_generic_actual_collector_contract_remains_readonly(worker, attempt, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Audit must not run a model or process"))
    before = worker["snapshot"](Path(attempt["run_root"]))
    report = worker["run"](attempt)
    assert report["kind"] == builder.KIND and report["state"] == "audit_complete"
    assert report["training_eligible"] is True and report["outcome"] == "verified_success"
    assert report["identity"] == attempt["identity"] and report["max_steps"] == attempt["max_steps"]
    assert report["teachers"]["model_calls_observed"] == 1
    assert report["teachers"]["requests_with_unknown_model_call"] == 0
    assert report["model_calls"] == report["network_calls"] == 0
    assert report['diagnostic_pngs_capture']['state'] == 'complete'
    assert [row['camera'] for row in report['diagnostic_pngs']] == [0, 1, 2]
    for row in report['diagnostic_pngs']:
        image = base64.b64decode(row['base64'], validate=True)
        assert image.startswith(b'\x89PNG\r\n\x1a\n') and len(image) == row['size']
        assert hashlib.sha256(image).hexdigest() == row['sha256']
    assert worker["snapshot"](Path(attempt["run_root"])) == before


def test_optional_png_failure_does_not_change_training_eligibility(worker, attempt):
    snapshot = next((Path(attempt['run_root']) / 'attempt/teacher').glob('step-*'))
    (snapshot / 'camera-0.png').write_bytes(b'not a PNG')
    report = worker['run'](attempt)
    assert report['training_eligible'] is True and report['issues'] == []
    assert report['diagnostic_pngs'] == [] and report['diagnostic_pngs_capture']['state'] == 'unavailable'


@pytest.fixture
def camera_snapshots(tmp_path):
    attempt = tmp_path / 'collection/attempt'
    for name in ('step-00000000', 'step-00000256-decision-0001', 'step-00000256-decision-0002'):
        directory = attempt / 'teacher' / name; directory.mkdir(parents=True)
        for camera in range(3):
            (directory / f'camera-{camera}.png').write_bytes(b'\x89PNG\r\n\x1a\n' + name.encode() + bytes([camera]))
        # The diagnostic reader must not inspect decision/context attachments.
        (directory / 'decision').mkdir()
        (directory / 'decision' / 'outside.png').symlink_to('/no-such-diagnostic-source')
    return attempt


def test_png_capture_selects_last_step_and_last_same_step_decision_only(worker, camera_snapshots):
    before = worker['snapshot'](camera_snapshots)
    result = worker['capture_last_teacher_pngs'](camera_snapshots)
    assert result['diagnostic_pngs_capture'] == dict(state='complete', step=256, decision_index=2,
        directory='step-00000256-decision-0002')
    for row in result['diagnostic_pngs']:
        raw = (camera_snapshots / 'teacher/step-00000256-decision-0002' / f"camera-{row['camera']}.png").read_bytes()
        assert base64.b64decode(row['base64'], validate=True) == raw
        assert row['size'] == len(raw) and row['sha256'] == hashlib.sha256(raw).hexdigest()
    assert worker['snapshot'](camera_snapshots) == before


@pytest.mark.parametrize('damage', ['file_link', 'step_link', 'teacher_link', 'attempt_link', 'fifo', 'missing',
    'not_png', 'oversize', 'encoded_budget', 'extra_step', 'out_of_range'])
def test_png_capture_is_bounded_and_never_follows_links(worker, camera_snapshots, damage):
    attempt = camera_snapshots; directory = attempt / 'teacher/step-00000256-decision-0002'
    image = directory / 'camera-0.png'
    if damage in {'file_link', 'step_link', 'teacher_link', 'attempt_link'}:
        path = {'file_link':image, 'step_link':directory, 'teacher_link':attempt / 'teacher', 'attempt_link':attempt}[damage]
        target = path.with_name(path.name + '.moved'); path.rename(target); path.symlink_to(target)
    elif damage == 'fifo': image.unlink(); os.mkfifo(image)
    elif damage == 'missing': image.unlink()
    elif damage == 'not_png': image.write_bytes(b'diagnostic private bytes must never be returned')
    elif damage == 'oversize': image.write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * (512 * 1024))
    elif damage == 'encoded_budget':
        for camera in range(3):
            (directory / f'camera-{camera}.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * (512 * 1024 - 8))
    elif damage == 'extra_step':
        for step in range(32): (attempt / 'teacher' / f'step-{step + 500:08d}').mkdir()
    else: (attempt / 'teacher/step-00010001').mkdir()
    result = worker['capture_last_teacher_pngs'](attempt)
    assert result['diagnostic_pngs'] == [] and result['diagnostic_pngs_capture']['state'] == 'unavailable'
    assert len(json.dumps(result)) < 512


def test_png_capture_preserves_global_audit_deadline(worker, camera_snapshots, monkeypatch):
    def deadline(*args): raise worker['AuditDeadline']('actual deadline')
    monkeypatch.setattr(os, 'listdir', deadline)
    with pytest.raises(worker['AuditDeadline']): worker['capture_last_teacher_pngs'](camera_snapshots)


def test_generic_missing_model_events_is_unknown_and_rejected_for_training(worker, attempt):
    path = next((Path(attempt["run_root"]) / "attempt/teacher").glob("step-*/decision/*"))
    (path / "stdout.log").unlink()
    receipt = json.loads((path / "receipt.json").read_text()); receipt.update(status="failed", stage="execution", usage=None)
    dump(path / "receipt.json", receipt)
    report = worker["run"](attempt)
    assert report["state"] == "audit_complete" and report["training_eligible"] is False
    assert report["teachers"]["requests_with_unknown_model_call"] == 1
    assert report["teachers"]["observed_token_totals"] is None
    assert report["teachers"]["requests_without_usage"] == 1 and report["teachers"]["missing_usage_is_zero"] is False


def test_generic_wrong_attempt_identity_rejects_teacher_and_trajectory(worker, attempt):
    attempt["identity"]["instance_id"] -= 1
    report = worker["run"](attempt)
    assert report["training_eligible"] is False
    assert "teacher_decision_unverified" in report["issues"]
    assert report["trajectory"]["valid_complete_trajectory"] is False


@pytest.mark.parametrize("max_steps", [4500, 10000])
def test_action_scan_uses_exact_task_step_limit(worker, tmp_path, max_steps):
    path = tmp_path / "action_sources.jsonl"
    rows = [json.dumps(dict(step=i, source="policy_conditioned_on_teacher_subtask")) + "\n" for i in range(max_steps)]
    path.write_text("".join(rows))
    result = worker["action_summary"](path, max_steps)
    assert result["rows"] == max_steps and not result["issues"]
    with path.open("a") as stream: stream.write(json.dumps(dict(step=max_steps, source="policy_conditioned_on_teacher_subtask")) + "\n")
    assert "action_budget_exceeded" in worker["action_summary"](path, max_steps)["issues"]


def test_generic_startup_failure_can_complete_audit_without_trajectory(worker, tmp_path):
    root = tmp_path / "failed"; root.mkdir()
    dump(root / "collector_failure.json", dict(stage="connect", error_type="TimeoutError"))
    config = dict(run_root=str(root), audit_root=str(tmp_path / "audit"), wrapper_root=str(tmp_path / "wrapper"),
        code=str(ROOT), source_files_sha256={}, source_sha256="a" * 64, job=dict(state="FAILED"), max_steps=10000,
        identity=dict(task="putting_shoes_on_rack", instance_id=311, attempt_id="followup-001"))
    report = worker["run"](config)
    assert report["state"] == "audit_complete" and report["training_eligible"] is False
    assert report["teachers"]["requests"] == 0 and report["teachers"]["observed_token_totals"] is None
    assert "collector_main_failure_receipt_present" in report["issues"]


def test_audit_outputs_cannot_overwrite_input_or_existing_source(inputs):
    before = inputs["source_path"].read_bytes()
    with pytest.raises(ValueError, match="collides"):
        builder.build(**{**inputs, "output": inputs["source_path"]})
    assert inputs["source_path"].read_bytes() == before
    builder.build(**inputs)
    with pytest.raises(ValueError, match="fresh"): builder.build(**inputs)


@pytest.fixture
def recovery_inputs(tmp_path, monkeypatch):
    import build_codex_radio_recovery_v8 as recovery
    import test_build_codex_radio_recovery_v8 as recovery_tests
    previous = recovery_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    source = tmp_path / 'code8_recovery.py'
    result = recovery.build(source, **previous)
    job = dict(name='owned-radio311-pilot003', uid='actual-code8-uid', state='SUCCEEDED', observed_at=1000)
    job_path, submission = tmp_path / 'code8_terminal.json', tmp_path / 'code8_submitted.json'
    dump(job_path, job); dump(submission, {**job, 'source_sha256': result['sha256']})
    return dict(job_receipt=job_path, source_sha256=result['sha256'], source_path=source,
                submission_path=submission, output=tmp_path / 'code8_audit.py', now=1000)


def test_explicit_code8_recovery_worker_and_validator_admitted(recovery_inputs):
    result = builder.build(**recovery_inputs)
    assert result['identity'] == dict(task='turning_on_radio', instance_id=311, attempt_id='pilot-003')
    config = next_builder.frozen_config(recovery_inputs['output'])
    assert config['code'].endswith('/code-v8') and config['run_root'].endswith('/radio-311-pilot003')
    assert result['model_calls'] == result['network_calls'] == 0


@pytest.mark.parametrize('damage', ['worker', 'teacher', 'admission', 'prior_calls', 'scope', 'interval'])
def test_v8_source_and_frozen_contract_cannot_be_forged(recovery_inputs, damage):
    import build_codex_radio_recovery_v8 as recovery
    path = recovery_inputs['source_path']; config = next_builder.frozen_config(path)
    if damage == 'teacher': config['manifest']['source_files_sha256']['scripts/codex_collection_teacher.py'] = 'f'*64
    elif damage == 'admission': config['admission']['total_reserved_decisions'] = 32
    elif damage == 'prior_calls': config['recovery_from']['observed_model_calls'] = 0
    elif damage == 'scope': config['collection']['instance_id'] = 312
    elif damage == 'interval': config['collection']['teacher_interval'] = 320
    path.write_text(('print("unreviewed")\n' if damage == 'worker' else '') + recovery.WORKER + '\nBOOTSTRAP_REPORT = run(' + repr(config) + ')\n')
    recovery_inputs['source_sha256'] = next_builder.sha(path)
    submitted = next_builder.load(recovery_inputs['submission_path'])
    submitted['source_sha256'] = recovery_inputs['source_sha256']; dump(recovery_inputs['submission_path'], submitted)
    with pytest.raises(ValueError): builder.build(**recovery_inputs)
    assert not recovery_inputs['output'].exists()


@pytest.fixture
def installed_code8_inputs(recovery_inputs):
    import build_codex_next_collection_v8 as installed
    data = recovery_inputs
    config = next_builder.frozen_config(data['source_path'])
    config.pop('installer'); config.pop('recovery_from')
    config.update(builder_kind=installed.KIND, expected_bundle_sha256=installed.CODE8_SHA, demo_episode=4424,
        report_root=builder.pilot.AFS+'/next-code8-shoes-312-followup-004')
    config['manifest']['bundle_sha256'] = installed.CODE8_SHA
    config['collection'].update(task='putting_shoes_on_rack', instance_id=312, attempt_id='followup-004',
        max_steps=10000, teacher_interval=320, run_root=builder.pilot.AFS+'/collection/shoes-312-followup-004',
        demonstration_manifest=builder.pilot.AFS+'/original-reader-v4/demonstrations/4424/manifest.json')
    config['admission'].update(sequence=4, total_reserved_decisions=128, task_attempt_number=1,
        all_continue_action_ceiling=10240, step_limit_may_exceed_teacher_budget=False,
        correction_can_reduce_action_ceiling=True)
    data['source_path'].write_text(installed.WORKER+'\nBOOTSTRAP_REPORT = run('+repr(config)+')\n')
    data['source_sha256'] = next_builder.sha(data['source_path'])
    submission = next_builder.load(data['submission_path']); submission['source_sha256'] = data['source_sha256']
    dump(data['submission_path'], submission)
    return data


def test_installed_code8_successor_is_explicit_third_worker_contract(installed_code8_inputs):
    result = builder.build(**installed_code8_inputs)
    assert result['identity'] == dict(task='putting_shoes_on_rack', instance_id=312, attempt_id='followup-004')
    assert result['max_steps'] == 10000


@pytest.mark.parametrize('damage', ['marker', 'worker', 'installed_sha', 'scope', 'demo', 'installer', 'interval'])
def test_installed_code8_branch_preserves_source_scope_and_no_installer(installed_code8_inputs, damage):
    import build_codex_next_collection_v8 as installed
    data = installed_code8_inputs; config = next_builder.frozen_config(data['source_path'])
    if damage == 'marker': config['builder_kind'] = 'unknown_code8'
    elif damage == 'installed_sha': config['expected_bundle_sha256'] = 'f'*64
    elif damage == 'scope': config['collection']['instance_id'] = 301
    elif damage == 'demo': config['demo_episode'] = 28
    elif damage == 'installer': config['installer'] = 'unexpected'
    elif damage == 'interval': config['collection']['teacher_interval'] = 256
    source = installed.WORKER if damage != 'worker' else next_builder.WORKER
    data['source_path'].write_text(source+'\nBOOTSTRAP_REPORT = run('+repr(config)+')\n')
    data['source_sha256'] = next_builder.sha(data['source_path'])
    submission = next_builder.load(data['submission_path']); submission['source_sha256'] = data['source_sha256']
    dump(data['submission_path'], submission)
    with pytest.raises(ValueError): builder.build(**data)
    assert not data['output'].exists()


@pytest.fixture
def native_code9_inputs(tmp_path, monkeypatch):
    import build_codex_native_collection_v9 as native
    import test_build_codex_native_collection_v9 as native_tests
    inputs = native_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    source = tmp_path / 'native_collection.py'
    result = native.build(source, **{**inputs, 'task':'putting_shoes_on_rack', 'instance_id':312,
                                   'demo_episode':4424})
    job = dict(name='owned-native-shoes312', uid='actual-native-uid', state='SUCCEEDED', observed_at=1000)
    job_path, submission = tmp_path / 'native_terminal.json', tmp_path / 'native_submitted.json'
    dump(job_path, job); dump(submission, {**job, 'source_sha256':result['sha256']})
    return dict(job_receipt=job_path, source_sha256=result['sha256'], source_path=source,
                submission_path=submission, output=tmp_path / 'native_audit.py', now=1000)


def test_native_code9_frozen_contract_is_explicit_and_binds_mode_inventory(native_code9_inputs):
    import build_codex_native_collection_v9 as native
    result = builder.build(**native_code9_inputs)
    config = next_builder.frozen_config(native_code9_inputs['output'])
    frozen = next_builder.frozen_config(native_code9_inputs['source_path'])
    assert config['collection_mode'] == native.MODE and config['teacher_interval'] == 320
    assert config['collection_bundle_sha256'] == native.CODE9_SHA
    assert config['source_files_sha256'] == frozen['manifest']['source_files_sha256']
    assert config['code'].endswith('/code-v9') and config['wrapper_root'].endswith('/native-code9-shoes-312-native-001')
    assert result['max_steps'] == 10000 and result['model_calls'] == result['network_calls'] == 0


@pytest.mark.parametrize('damage', ['marker','worker','mode','bundle','teacher','interval','inventory','scope','budget'])
def test_native_generic_audit_rejects_unreviewed_version_or_semantics(native_code9_inputs, damage):
    import build_codex_native_collection_v9 as native
    data = native_code9_inputs; config = next_builder.frozen_config(data['source_path'])
    if damage == 'marker': config['builder_kind'] = 'codex_next_collection_v8'
    elif damage == 'mode': config['collection']['collection_mode'] = 'teacher_conditioned'
    elif damage == 'bundle': config['expected_bundle_sha256'] = 'f'*64
    elif damage == 'teacher': config['teacher_sha256'] = 'f'*64
    elif damage == 'interval': config['collection']['teacher_interval'] = 256
    elif damage == 'inventory': config['collection']['source_files_sha256']['scripts/codex_collection_actions.py'] = 'f'*64
    elif damage == 'scope': config['collection']['instance_id'] = 301
    elif damage == 'budget': config['admission']['total_reserved_decisions'] = 32
    source = native.WORKER + ('\nprint("unreviewed")\n' if damage == 'worker' else '\n')
    data['source_path'].write_text(source + 'BOOTSTRAP_REPORT = run(' + repr(config) + ')\n')
    data['source_sha256'] = next_builder.sha(data['source_path'])
    submitted = next_builder.load(data['submission_path']); submitted['source_sha256'] = data['source_sha256']
    dump(data['submission_path'], submitted)
    with pytest.raises(ValueError): builder.build(**data)
    assert not data['output'].exists()


def replan_fixture(tmp_path, *, midburst=False, budget=32, always_reject=False, joint_bounds=False):
    import numpy as np
    from collect_b1k_codex import collect_attempt
    from codex_collection_actions import BoundedCartesianCorrection
    from codex_collection_teacher import CodexTeacher
    from test_codex_collection_teacher import FakeRunner
    from test_collect_b1k_codex import Evaluator, metadata
    from test_codex_collection_actions import Robot, Backend
    evaluator = Evaluator(terminal_step=3 if midburst else 1); evaluator.robot = Robot()
    backend = Backend()
    if joint_bounds:
        backend.solve = lambda **kw: kw['q'] + np.full((1, 7), 0.2, dtype=np.float32)
    backend.jacobian = lambda *_: np.c_[np.eye(6) if midburst and evaluator.steps == 0 else np.diag([0,1,1,1,1,1]), np.zeros(6)]
    calls = []
    def response(value):
        calls.append(1)
        if len(calls) == 1 or always_reject:
            value.update(action='correct', subtask='align the radio', correction=dict(arm='left',
                delta_position_m=[0.01,0,0], delta_rotation_axis_angle_rad=[0,0,0], gripper='keep', steps=3))
        return value
    identity = dict(task='turning_on_radio', instance_id=311, attempt_id='pilot-003')
    root = tmp_path / 'collection'; root.mkdir()
    result = collect_attempt(evaluator, CodexTeacher(runner=FakeRunner(response=response)), root / 'attempt',
        {**metadata(), **identity}, max_steps=4500, max_decisions=budget, teacher_interval=256,
        correction_factory=lambda robot: BoundedCartesianCorrection(robot, backend=backend))
    completed = dict(state='complete', result=result, training_eligible=result['success'], official_benchmark=False)
    dump(root / '_COMPLETE', completed); dump(root / 'status.json', completed)
    return dict(run_root=str(root), audit_root=str(tmp_path / 'audit'), wrapper_root=str(tmp_path / 'wrapper'),
        code=str(ROOT), source_files_sha256={}, source_sha256='a'*64, job=dict(state='SUCCEEDED'),
        identity=identity, max_steps=4500)


@pytest.mark.parametrize('midburst', [False, True])
def test_real_collector_rejection_and_replan_evidence_keeps_only_executed_frames(worker, tmp_path, midburst):
    config = replan_fixture(tmp_path, midburst=midburst)
    before = worker['snapshot'](Path(config['run_root']))
    report = worker['run'](config)
    assert report['issues'] == [] and report['training_eligible'] is True
    assert report['teachers']['requests'] == 2 and report['correction_rejections']['rows'] == 1
    rejected = report['correction_rejections']['entries'][0]
    assert rejected['executed'] is False and rejected['step'] == (1 if midburst else 0)
    assert report['actions']['rows'] == (3 if midburst else 1)
    assert report['actions']['sources'].get('teacher_cartesian_correction', 0) == int(midburst)
    assert worker['snapshot'](Path(config['run_root'])) == before


def test_original_joint_step_bound_rejection_can_be_audited_without_executing_candidate(worker, tmp_path):
    config = replan_fixture(tmp_path, joint_bounds=True)
    report = worker['run'](config)
    assert report['training_eligible'] is True and not report['issues']
    rejected = report['correction_rejections']['entries'][0]
    assert rejected['reason_code'] == 'ik_joint_bounds' and rejected['executed'] is False
    assert rejected['diagnostics']['max_joint_delta_rad'] > 0.15
    assert report['actions']['sources'] == {'policy_conditioned_on_teacher_subtask': 1}


@pytest.mark.parametrize('budget', [1, 32])
def test_zero_action_rejection_budget_has_no_fake_success_or_unknown_model_calls(worker, tmp_path, budget):
    config = replan_fixture(tmp_path, budget=budget, always_reject=True)
    report = worker['run'](config)
    assert report['state'] == 'audit_complete' and report['training_eligible'] is False
    assert report['teachers']['model_calls_observed'] == budget
    assert report['correction_rejections']['rows'] == budget and report['actions']['rows'] == 0
    assert report['trajectory']['valid_complete_trajectory'] is False
    assert 'collector_failure_receipt_present' not in report['issues']
    assert not report['correction_rejections']['issues']


@pytest.mark.parametrize('damage', ['missing', 'deleted_event', 'duplicate', 'executed', 'wrong_request', 'wrong_index',
    'future_step', 'below_boundary', 'relaxed_boundary', 'nonfinite', 'unsafe_diagnostic', 'stale_action', 'unbound_action'])
def test_rejection_and_actual_action_binding_tampering_is_not_training_data(worker, tmp_path, damage):
    config = replan_fixture(tmp_path)
    attempt = Path(config['run_root']) / 'attempt'; path = attempt / 'correction_rejections.jsonl'
    row = json.loads(path.read_text()); rows = [row]
    if damage == 'missing': path.unlink()
    elif damage == 'deleted_event': path.write_text('')
    elif damage in ('stale_action', 'unbound_action'):
        action_path = attempt / 'action_sources.jsonl'; action = json.loads(action_path.read_text())
        if damage == 'stale_action': action.update(request_id=row['request_id'], decision_index=row['decision_index'])
        else:
            action.pop('request_id'); action.pop('decision_index')
        action_path.write_text(json.dumps(action)+'\n')
    else:
        if damage == 'duplicate': rows.append(dict(row))
        elif damage == 'executed': row['executed'] = True
        elif damage == 'wrong_request': row['request_id'] = 'f'*64
        elif damage == 'wrong_index': row['decision_index'] = 1
        elif damage == 'future_step': row['step'] = 1
        elif damage == 'below_boundary': row['diagnostics']['linearized_position_residual_m'] = 0
        elif damage == 'relaxed_boundary': row['diagnostics']['linearized_position_limit_m'] = 0.02
        elif damage == 'nonfinite': row['diagnostics']['max_joint_delta_rad'] = float('nan')
        else: row['diagnostics']['secret_token'] = 'private-evidence-not-for-report'
        path.write_text(''.join(json.dumps(item)+'\n' for item in rows))
    report = worker['run'](config)
    assert report['state'] == 'audit_complete' and report['training_eligible'] is False and report['issues']
    assert 'private-evidence-not-for-report' not in json.dumps(report)


def test_legacy_step_only_directories_and_unbound_actions_still_audit(worker, attempt):
    from codex_collection_teacher import request_identity
    root = Path(attempt['run_root']) / 'attempt'
    path = next((root / 'teacher').glob('step-*/decision/*'))
    captured = json.loads((path / 'request.json').read_text())
    captured['request'].pop('decision_index')
    identity = request_identity(captured['request'], [path / image['file'] for image in captured['images']])
    captured['request_id'] = identity; dump(path / 'request.json', captured)
    for name, key in [('receipt.json', 'request_id'), ('response.json', 'request_id'), ('response.raw.json', 'response_id')]:
        value = json.loads((path / name).read_text()); value[key] = identity; dump(path / name, value)
    new = path.with_name(identity); path.rename(new)
    step_root = new.parent.parent; step_root.rename(step_root.with_name('step-00000000'))
    action_path = root / 'action_sources.jsonl'; action = json.loads(action_path.read_text())
    action.pop('decision_index'); action.pop('request_id'); action_path.write_text(json.dumps(action)+'\n')
    (root / 'correction_rejections.jsonl').unlink()
    report = worker['run'](attempt)
    assert report['training_eligible'] is True and not report['issues']
    assert report['teachers']['entries'][0]['layout'] == 'legacy'


@pytest.mark.parametrize('indices,steps,expected', [([0,2],[0,1],'teacher_decision_indices_not_contiguous'),
    ([0,0],[0,1],'teacher_decision_indices_not_contiguous'), ([0,1],[1,0],'teacher_steps_not_monotonic')])
def test_decision_sequence_cannot_skip_duplicate_or_reverse(worker, indices, steps, expected):
    rows = [dict(decision_valid=True, layout='indexed', decision_index=index, step=step) for index,step in zip(indices,steps)]
    assert expected in worker['decision_order'](rows)


def native_fixture(tmp_path, *, correction=False, reject=False, midburst=False, terminal_step=5, budget=32):
    import numpy as np
    from collect_b1k_codex import collect_attempt
    from codex_collection_actions import BoundedCartesianCorrection
    from codex_collection_teacher import CodexTeacher
    from test_codex_collection_teacher import FakeRunner
    from test_collect_b1k_codex import Evaluator, NativePolicy, metadata
    from test_codex_collection_actions import Robot, Backend
    evaluator=Evaluator(terminal_step=terminal_step);evaluator.policy=NativePolicy();evaluator.robot=Robot()
    backend=Backend()
    if reject:
        backend.jacobian=lambda *_: np.c_[np.eye(6) if midburst and evaluator.steps==2 else np.diag([0,1,1,1,1,1]),np.zeros(6)]
    decisions=[]
    def response(value):
        decisions.append(True)
        value.update(action='continue',subtask='',correction=None)
        if correction and len(decisions)==1:
            value.update(action='correct',subtask='teacher align radio',correction=dict(arm='left',
                delta_position_m=[0.01,0,0],delta_rotation_axis_angle_rad=[0,0,0],gripper='keep',steps=3 if midburst else 2))
        return value
    identity=dict(task='turning_on_radio',instance_id=311,attempt_id='native-audit')
    root=tmp_path/'collection';root.mkdir()
    result=collect_attempt(evaluator,CodexTeacher(runner=FakeRunner(response=response)),root/'attempt',
        {**metadata(),**identity},max_steps=4500,max_decisions=budget,teacher_interval=2,
        correction_factory=lambda robot:BoundedCartesianCorrection(robot,backend=backend),collection_mode='native_with_corrections')
    completed=dict(state='complete',result=result,training_eligible=result['success'],official_benchmark=False)
    dump(root/'_COMPLETE',completed);dump(root/'status.json',completed)
    return dict(run_root=str(root),audit_root=str(tmp_path/'audit'),wrapper_root=str(tmp_path/'wrapper'),
        code=str(ROOT),source_files_sha256={},source_sha256='a'*64,job=dict(state='SUCCEEDED'),
        identity=identity,max_steps=4500,collection_mode='native_with_corrections',teacher_interval=2)


@pytest.mark.parametrize('options',[{},dict(terminal_step=1),dict(correction=True),
    dict(correction=True,reject=True),dict(correction=True,reject=True,midburst=True)])
def test_native_actual_collector_audit_preserves_native_labels_and_bounded_corrections(worker,tmp_path,options):
    config=native_fixture(tmp_path,**options)
    report=worker['run'](config)
    assert report['state']=='audit_complete' and report['issues']==[] and report['training_eligible'] is True
    assert report['collection_mode']==report['trajectory']['collection_mode']=='native_with_corrections'
    assert report['trajectory']['subtask_source']=='mixed_native_and_teacher'
    assert report['actions']['sources']['policy_native_subtask']>0
    first=report['actions']['segments'][0]
    assert first['decision_index'] is None and first['request_id'] is None
    assert first['subtask_source']=='generated'
    if report['teachers']['requests']:
        assert report['teachers']['entries'][0]['step']==2
    if options.get('reject'):assert report['correction_rejections']['rows']==1


def test_native_teacher_budget_exhaustion_still_stops_and_never_makes_success(worker,tmp_path):
    config=native_fixture(tmp_path,terminal_step=100,budget=32)
    report=worker['run'](config)
    assert report['issues']==[] and report['training_eligible'] is False
    assert report['teachers']['requests']==32 and report['actions']['rows']==66
    assert report['trajectory']['result']['stop_reason']=='teacher_decision_budget'


@pytest.mark.parametrize('damage',['mode','native_label','correction_label','native_source','correction_source',
    'prefix_binding','late_null_binding','stale_binding','missing_binding','partial_null','legacy_source'])
def test_native_labels_modes_and_every_action_binding_are_checked(worker,tmp_path,damage):
    config=native_fixture(tmp_path,correction=True)
    path=Path(config['run_root'])/'attempt/action_sources.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if damage=='mode':config['collection_mode']='teacher_conditioned'
    elif damage=='native_label':rows[4]['subtask']='invented label'
    elif damage=='correction_label':rows[2]['subtask']='native turn dial'
    elif damage=='native_source':rows[4]['subtask_source']='provided'
    elif damage=='correction_source':rows[2]['subtask_source']='generated'
    elif damage=='prefix_binding':rows[0].update(decision_index=rows[2]['decision_index'],request_id=rows[2]['request_id'])
    elif damage=='late_null_binding':rows[4].update(decision_index=None,request_id=None)
    elif damage=='stale_binding':rows[4].update(decision_index=rows[2]['decision_index'],request_id=rows[2]['request_id'])
    elif damage=='missing_binding':rows[0].pop('decision_index');rows[0].pop('request_id')
    elif damage=='partial_null':rows[0]['request_id']='a'*64
    else:rows[4]['source']='policy_conditioned_on_teacher_subtask'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report=worker['run'](config)
    assert report['training_eligible'] is False and report['issues']


def test_native_initial_interval_is_pinned_even_if_actions_and_h5_are_valid(worker,tmp_path):
    config=native_fixture(tmp_path);config['teacher_interval']=3
    report=worker['run'](config)
    assert report['training_eligible'] is False and 'native_first_teacher_step_differs' in report['issues']

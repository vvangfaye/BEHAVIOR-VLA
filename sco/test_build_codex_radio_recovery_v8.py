"""Offline recovery installer, exact audit and cross-version budget boundaries."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_next_collection as shared
import build_codex_radio_recovery_v8 as builder
import test_build_codex_next_collection as next_tests
import test_build_codex_first_collection_v5 as v5


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    previous = next_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    source = previous["prior_source_path"]
    config = shared.frozen_config(source)
    config.update(manifest=shared.load(previous["manifest_path"]),
                  preparation_manifest=shared.load(previous["preparation_manifest_path"]), ciphertext_sha256="c"*64)
    config["manifest"]["original_data"] = shared.AFS + "/original-v2"
    source.write_text("BOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
    source_sha = shared.sha(source); monkeypatch.setattr(builder, "PRIOR_SOURCE_SHA", source_sha)
    submission = shared.load(previous["prior_submission_path"]);submission["source_sha256"] = source_sha
    dump(previous["prior_submission_path"], submission)
    audit = shared.load(previous["prior_audit_path"])
    audit["collection_source_sha256"] = source_sha; audit["actions"]["rows"] = 3339
    audit["teachers"].update(requests=17, model_calls_observed=17, requests_with_unknown_model_call=0,
        entries=[dict(receipt_status="validated", decision_valid=True, event_audit_valid=True,
                      pinned_model_verified=True, chatgpt_login_verified=True) for _ in range(17)])
    dump(previous["prior_audit_path"], audit)
    manifest = copy.deepcopy(config["manifest"])
    bundle = tmp_path / "code-v8.tar.gz";bundle.write_bytes(b"offline code-v8 fixture")
    manifest.update(version="v8", code=shared.AFS + "/code-v8", bundle=str(bundle), bundle_sha256=shared.sha(bundle))
    for name in builder.ALLOWED_CHANGES:manifest["source_files_sha256"][name] = "e" * 64
    manifest_path = tmp_path / "code8_manifest.json";dump(manifest_path, manifest)
    url = tmp_path / "private_url.txt";url.write_text("https://fixture.invalid/code8?Signature=private-fixture");url.chmod(0o600)
    kwargs = {key: previous[key] for key in ("prior_job_path", "prior_submission_path", "prior_source_path", "prior_audit_path", "ledger_path", "now")}
    kwargs.update(manifest_path=manifest_path, expected_bundle_sha256=manifest["bundle_sha256"], bundle_url_file=url)
    return kwargs


def test_pilot003_keeps_17_calls_and_reserves_96_without_real_execution(tmp_path, inputs):
    target = tmp_path / "recovery.py"
    result = builder.build(target, **inputs)
    config = shared.frozen_config(target)
    assert builder.validate_frozen_config(config)["attempt_id"] == "pilot-003"
    assert config["recovery_from"]["observed_model_calls"] == 17
    assert result["code_installation"] is True and result["actual_job_submitted"] is False
    assert result["code"].endswith("/code-v8") and result["recommended_cloud_max_seconds"] == 9600
    assert result["budget_admission"]["total_reserved_decisions"] == 96
    assert config["collection"]["run_root"] == builder.RUN_ROOT and config["report_root"] == builder.REPORT_ROOT
    ledger = shared.load(inputs["ledger_path"])
    assert [r["attempt_id"] for r in ledger["entries"]] == ["pilot-001", "pilot-002", "pilot-003"]
    assert ledger["entries"][1]["teacher_accounting"]["model_calls_observed"] == 17
    assert ledger["entries"][1]["reserved_decisions"] == 32
    assert ledger["entries"][-1]["collection_code"].endswith("/code-v8")
    assert "private-fixture" not in target.with_suffix(".receipt.json").read_text()
    assert "signal.alarm(min(600,remaining))" in config["installer"] and "socket.EAI_AGAIN" in config["installer"]
    compile(target.read_text(), str(target), "exec")


@pytest.mark.parametrize("damage", ["source_sha", "steps", "calls", "unknown", "unvalidated", "state", "uid", "stale"])
def test_only_exact_terminal_failed_actual_audit_can_recover(tmp_path, inputs, damage):
    if damage in ("state", "uid", "stale"):
        path = inputs["prior_job_path"];value = shared.load(path)
        value.update({"state": {"state": "RUNNING"}, "uid": {"uid": "other"}, "stale": {"observed_at": 0}}[damage])
    else:
        path = inputs["prior_audit_path"];value = shared.load(path)
        if damage == "source_sha":value["collection_source_sha256"] = "f" * 64
        elif damage == "steps":value["actions"]["rows"] = 0
        elif damage == "calls":value["teachers"]["model_calls_observed"] = 0
        elif damage == "unknown":value["teachers"]["requests_with_unknown_model_call"] = 1
        else:value["teachers"]["entries"][0]["decision_valid"] = False
    dump(path, value)
    with pytest.raises(ValueError):builder.build(tmp_path / "bad.py", **inputs)
    assert not inputs["ledger_path"].exists()


@pytest.mark.parametrize("damage", ["expected_sha", "bundle", "teacher", "base", "extra_change", "missing_change", "version", "account"])
def test_new_bundle_and_only_two_source_changes_are_required(tmp_path, inputs, damage):
    manifest = shared.load(inputs["manifest_path"])
    if damage == "expected_sha":inputs["expected_bundle_sha256"] = "f" * 64
    elif damage == "bundle":Path(manifest["bundle"]).write_bytes(b"changed")
    elif damage == "teacher":manifest["source_files_sha256"]["scripts/codex_collection_teacher.py"] = "f" * 64
    elif damage == "base":manifest["base_source_manifest_sha256"] = "f" * 64
    elif damage == "extra_change":manifest["source_files_sha256"]["sco/run_codex_collection.py"] = "f" * 64
    elif damage == "missing_change":manifest["source_files_sha256"]["scripts/codex_collection_actions.py"] = "d" * 64
    elif damage == "version":manifest["version"] = "v9"
    else:
        source = inputs["prior_source_path"];config = shared.frozen_config(source)
        config["collection"]["codex_home"] = "/fixture/another-account"
        source.write_text("BOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
    dump(inputs["manifest_path"], manifest)
    with pytest.raises(ValueError):builder.build(tmp_path / "bad.py", **inputs)
    assert not inputs["ledger_path"].exists()


def test_existing_pending_reservation_prevents_budget_restart(tmp_path, inputs):
    builder.build(tmp_path / "one.py", **inputs)
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError, match="latest reserved"):
        builder.build(tmp_path / "two.py", **inputs)
    assert inputs["ledger_path"].read_bytes() == before


def test_completed_code8_audit_can_settle_in_same_ledger(tmp_path, inputs):
    source = tmp_path / "recovery.py";receipt = builder.build(source, **inputs)
    job = dict(name="owned-code8-job", uid="code8-uid", state="SUCCEEDED", observed_at=1000)
    job_path, submission_path, audit_path = [tmp_path / (name + ".json") for name in ("code8_job", "code8_submission", "code8_audit")]
    dump(job_path, job);dump(submission_path, {**job, "source_sha256": receipt["sha256"]})
    audit = shared.load(inputs["prior_audit_path"])
    audit.update(kind="codex_collection_readonly_audit_v1", source_job=job,
                 collection_source_sha256=receipt["sha256"], run_root=builder.RUN_ROOT,
                 audit_root=shared.AFS + "/audit-radio311-pilot003-v8")
    audit["teachers"].update(model_calls_observed=3, requests=4, requests_with_unknown_model_call=1)
    dump(audit_path, audit)
    previous = shared.checked_previous(job_path, submission_path, source, audit_path, now=1000)
    ledger = shared.load(inputs["ledger_path"])
    shared.settle(ledger, previous)
    assert ledger["entries"][-1]["state"] == "audited"
    assert ledger["entries"][-1]["teacher_accounting"]["requests_with_unknown_model_call"] == 1
    assert sum(row["reserved_decisions"] for row in ledger["entries"]) == 96


@pytest.mark.parametrize("damage", ["identity", "path", "budget", "teacher", "prior_calls"])
def test_public_frozen_validator_rejects_changed_contract(tmp_path, inputs, damage):
    path = tmp_path / "recovery.py";builder.build(path, **inputs);config = shared.frozen_config(path)
    if damage == "identity":config["collection"]["instance_id"] = 312
    elif damage == "path":config["report_root"] = shared.AFS + "/old-report"
    elif damage == "budget":config["admission"]["total_reserved_decisions"] = 32
    elif damage == "teacher":config["manifest"]["source_files_sha256"]["scripts/codex_collection_teacher.py"] = "f"*64
    else:config["recovery_from"]["observed_model_calls"] = 0
    with pytest.raises(ValueError):builder.validate_frozen_config(config)


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.WORKER, "recovery_worker", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("installer_fails", [False, True])
def test_installer_is_bounded_and_restores_global_remaining_alarm(tmp_path, worker, monkeypatch, installer_fails):
    namespace = worker.run.__globals__;alarms = []
    monkeypatch.setattr(worker.signal, "alarm", lambda seconds: alarms.append(seconds))
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.delenv("CODEX_PILOT_REPORT_PUT", raising=False)
    for name in ("verify_predecessor", "preparation_config", "preparation_gates"):
        monkeypatch.setitem(namespace, name, lambda config: {})
    def fail(config):raise ValueError("stop before actual collection")
    monkeypatch.setitem(namespace, "gates", fail)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch a collector"))
    installer = "import signal\nsignal.alarm(600)\ntry:\n " + ("raise ValueError('install failed')" if installer_fails else "pass") + "\nfinally:signal.alarm(0)"
    collection = dict(run_root=str(tmp_path / "attempt"), task="turning_on_radio", instance_id=311, attempt_id="pilot-003", max_steps=4500)
    report = worker.run(dict(collection=collection, report_root=str(tmp_path / "report"), installer=installer,
                             recovery_from={}, admission={}))
    assert report["state"] == "failed"
    assert report["stage"] == ("code_install" if installer_fails else "gates")
    assert alarms[:3] == [9000, 600, 0] and 8900 <= alarms[3] <= 9000 and alarms[-1] == 0
    assert not issubclass(worker.WrapperDeadline, Exception)


def test_runtime_hash_checks_require_installed_code8_with_code6_preparation(tmp_path, worker):
    config = v5.gate_inputs.__wrapped__(tmp_path, worker)
    base = Path(config["pilot_root"]);code7 = base / "code-v7";code8 = base / "code-v8"
    shutil.copytree(code7, code8)
    config["manifest"].update(version="v8", code=str(code8))
    config["collection"].update(code=str(code8), task="turning_on_radio");config["demo_episode"] = 28
    ready = shared.load(code8 / "_CODEX_PILOT_CODE_READY");ready.update(version="v8", code=str(code8));dump(code8 / "_CODEX_PILOT_CODE_READY", ready)
    assert worker.gates(config)["collection_code"] == str(code8)
    (code8 / "scripts/fixture_0.py").write_text("changed after install")
    with pytest.raises(ValueError):worker.gates(config)


@pytest.mark.parametrize("now,expected", [(100, 600), (8500, 500), (9000, None)])
def test_actual_installer_never_extends_total_deadline(tmp_path, inputs, worker, monkeypatch, now, expected):
    manifest = shared.load(inputs["manifest_path"])
    source = builder.bounded_installer(manifest, "https://fixture.invalid/bundle")
    tree = ast.parse(source); statements = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):break
        statements.append(node)
    calls = []
    monkeypatch.setattr(worker.signal, "alarm", lambda value: calls.append(value))
    monkeypatch.setattr(worker.time, "monotonic", lambda: now)
    execute = lambda: exec(compile(ast.Module(body=statements, type_ignores=[]), "installer_deadline", "exec"), {"outer_deadline": 9000})
    if expected is None:
        with pytest.raises(TimeoutError):execute()
        assert not calls
    else:
        execute();assert calls == [expected]

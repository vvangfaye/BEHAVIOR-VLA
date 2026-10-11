"""Offline native admission, shared budgets and finite installer boundaries."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_native_collection_v9 as builder
import build_codex_next_collection as shared
import build_codex_next_collection_v8 as code8
import build_codex_radio_recovery_v8 as recovery
import test_build_codex_radio_recovery_v8 as recovery_tests
from test_build_codex_next_collection_v8 import audit_next, dump


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    prior = recovery_tests.inputs.__wrapped__(tmp_path, monkeypatch)
    # The old synthetic fixture predates the bridge file. Add that real inventory
    # member before the first reservation, keeping all 29-member manifests aligned.
    path = prior["prior_source_path"]; frozen = shared.frozen_config(path)
    manifests = [frozen["manifest"], frozen["preparation_manifest"], shared.load(prior["manifest_path"])]
    for manifest in manifests:
        values = manifest["source_files_sha256"]
        values["scripts/codex_teacher_bridge.py"] = values.pop("scripts/fixture-0.py")
    frozen["collection"]["source_files_sha256"] = manifests[0]["source_files_sha256"]
    path.write_text("BOOTSTRAP_REPORT = run(" + repr(frozen) + ")\n")
    previous_sha = shared.sha(path); monkeypatch.setattr(recovery, "PRIOR_SOURCE_SHA", previous_sha)
    for key, field in (("prior_submission_path", "source_sha256"), ("prior_audit_path", "collection_source_sha256")):
        value = shared.load(prior[key]); value[field] = previous_sha; dump(prior[key], value)
    dump(prior["manifest_path"], manifests[-1])
    monkeypatch.setattr(code8, "CODE8_SHA", prior["expected_bundle_sha256"])
    source = tmp_path / "pilot003.py"; receipt = recovery.build(source, **prior)
    manifest = copy.deepcopy(manifests[-1])
    bundle = tmp_path / "code9.tar.gz"; bundle.write_bytes(b"offline native bundle fixture")
    manifest.update(version="v9", code=shared.AFS + "/code-v9", bundle=str(bundle), bundle_sha256=shared.sha(bundle))
    for name in builder.ALLOWED_CHANGES: manifest["source_files_sha256"][name] = "9" * 64
    manifest_path = tmp_path / "code9_manifest.json"; dump(manifest_path, manifest)
    monkeypatch.setattr(builder, "CODE9_SHA", manifest["bundle_sha256"])
    return dict(**audit_next(tmp_path, source, receipt), task="turning_on_radio", instance_id=317,
        attempt_id="native-001", demo_episode=28, manifest_path=manifest_path,
        expected_bundle_sha256=manifest["bundle_sha256"], bundle_url_file=prior["bundle_url_file"],
        ledger_path=prior["ledger_path"], now=1000)


@pytest.mark.parametrize("task,instance,demo", [("turning_on_radio", 311, 68), ("turning_on_radio", 320, 102),
    ("putting_shoes_on_rack", 311, 4424), ("putting_shoes_on_rack", 320, 4517)])
def test_native_install_admission_preserves_history_and_all_limits(tmp_path, inputs, task, instance, demo):
    source = tmp_path / "native.py"
    result = builder.build(source, **{**inputs, "task": task, "instance_id": instance, "demo_episode": demo})
    frozen = shared.frozen_config(source); collection = builder.validate_frozen_config(frozen)
    assert collection["collection_mode"] == builder.MODE
    assert collection["max_decisions"] == 32 and collection["max_steps"] == shared.TASKS[task][0]
    assert collection["teacher_interval"] == builder.TEACHER_INTERVALS[task]
    assert result["code_installation"] is True and result["actual_job_submitted"] is False
    assert result["max_seconds"] == 9000 and result["recommended_cloud_max_seconds"] == 9600
    assert result["budget_admission"]["total_reserved_decisions"] == 128
    assert result["budget_admission"]["all_continue_action_ceiling"] == 33 * builder.TEACHER_INTERVALS[task]
    ledger = shared.load(inputs["ledger_path"])
    assert ledger["entries"][1]["teacher_accounting"]["model_calls_observed"] == 17
    assert ledger["entries"][2]["teacher_accounting"]["requests_with_unknown_model_call"] == 2
    assert [row["reserved_decisions"] for row in ledger["entries"]] == [32] * 4
    assert "private-fixture" not in source.with_suffix(".receipt.json").read_text()
    assert frozen["teacher_sha256"] != frozen["preparation_manifest"]["source_files_sha256"]["scripts/codex_collection_teacher.py"]
    compile(source.read_text(), "native-worker", "exec")


def audited_native(tmp_path, source, receipt):
    value = audit_next(tmp_path, source, receipt, number=2)
    audit = shared.load(value["prior_audit_path"]); audit["collection_mode"] = builder.MODE
    dump(value["prior_audit_path"], audit)
    return value


def test_audited_native_reuses_installed_exact_bundle_without_second_installer(tmp_path, inputs):
    source = tmp_path / "first.py"; receipt = builder.build(source, **inputs)
    result = builder.build(tmp_path / "second.py", **{**inputs, **audited_native(tmp_path, source, receipt),
        "attempt_id": "native-002", "bundle_url_file": None})
    config = shared.frozen_config(tmp_path / "second.py")
    assert config["installation_required"] is False and "installer" not in config
    assert result["budget_admission"]["sequence"] == 5 and result["budget_admission"]["total_reserved_decisions"] == 160


@pytest.mark.parametrize("damage", ["unfinished_freeze", "active", "stale", "source", "audit_mode", "mode", "unknown_worker"])
def test_bad_predecessor_cannot_write_a_reservation(tmp_path, inputs, monkeypatch, damage):
    if damage in {"audit_mode", "mode", "unknown_worker"}:
        source = tmp_path / "first.py"; receipt = builder.build(source, **inputs)
        inputs.update(audited_native(tmp_path, source, receipt), attempt_id="native-002", bundle_url_file=None)
    if damage == "unfinished_freeze": monkeypatch.setattr(builder, "CODE9_SHA", None)
    elif damage in {"active", "stale"}:
        path = inputs["prior_job_path"]; value = shared.load(path)
        value.update({"state": "RUNNING"} if damage == "active" else {"observed_at": 0}); dump(path, value)
    elif damage == "audit_mode":
        path = inputs["prior_audit_path"]; value = shared.load(path); value.pop("collection_mode"); dump(path, value)
    elif damage in {"source", "mode", "unknown_worker"}:
        path = inputs["prior_source_path"]; config = shared.frozen_config(path)
        worker = builder.WORKER if damage != "source" else recovery.WORKER
        if damage == "mode": config["collection"]["collection_mode"] = "teacher_conditioned"
        elif damage == "unknown_worker": config["builder_kind"] = "unreviewed"
        path.write_text(worker + ("\nx = 'unreviewed mutation'\n" if damage == "source" else "\n") + "BOOTSTRAP_REPORT = run(" + repr(config) + ")\n")
        for key, field in (("prior_submission_path", "source_sha256"), ("prior_audit_path", "collection_source_sha256")):
            value = shared.load(inputs[key]); value[field] = shared.sha(path); dump(inputs[key], value)
    before = inputs["ledger_path"].read_bytes()
    with pytest.raises(ValueError): builder.build(tmp_path / "blocked.py", **inputs)
    assert inputs["ledger_path"].read_bytes() == before


@pytest.mark.parametrize("damage", ["mode", "budget", "ceiling", "teacher", "extra_change", "missing_change", "missing_actions_fix", "superseded_candidate", "account",
    "root", "installer", "installer_deadline", "installation", "version", "runtime", "interval", "demo", "predecessor_bundle"])
def test_frozen_native_contract_rejects_tampering(tmp_path, inputs, damage):
    source = tmp_path / "native.py"; builder.build(source, **inputs); frozen = shared.frozen_config(source)
    if damage == "mode": frozen["collection"]["collection_mode"] = "teacher_conditioned"
    elif damage == "budget": frozen["admission"]["total_reserved_decisions"] = 32
    elif damage == "ceiling": frozen["admission"]["all_continue_action_ceiling"] = 8192
    elif damage == "teacher": frozen["teacher_sha256"] = "f" * 64
    elif damage in {"extra_change", "missing_change", "missing_actions_fix"}:
        name = ("scripts/codex_collection_demonstrations.py" if damage == "extra_change"
                else "scripts/codex_collection_actions.py" if damage == "missing_actions_fix"
                else "scripts/codex_teacher_mailbox.py")
        digest = "f" * 64 if damage == "extra_change" else frozen["previous_manifest"]["source_files_sha256"][name]
        frozen["manifest"]["source_files_sha256"][name] = digest
        frozen["collection"]["source_files_sha256"][name] = digest
    elif damage == "superseded_candidate":
        frozen["expected_bundle_sha256"] = frozen["manifest"]["bundle_sha256"] = "bb92c47c5ba669fdd7e054d84b4bcd759473ef32085e55594e981e7d139cac3d"
    elif damage == "account": frozen["collection"]["codex_home"] = "/other"
    elif damage == "root": frozen["report_root"] = recovery.REPORT_ROOT
    elif damage == "installer": frozen["installer"] += "\nprint('unreviewed')\n"
    elif damage == "installer_deadline": frozen["installer"] = frozen["installer"].replace("min(600,remaining)", "600")
    elif damage == "installation": frozen["installation_required"] = False
    elif damage == "version": frozen["manifest"]["version"] = "v8"
    elif damage == "runtime": frozen["python"] = frozen["python"].replace("/envs/", "/")
    elif damage == "interval": frozen["collection"]["teacher_interval"] = 320
    elif damage == "demo": frozen["demo_episode"] = 4517
    else: frozen["predecessor"]["bundle_sha256"] = "e" * 64
    with pytest.raises(ValueError): builder.validate_frozen_config(frozen)


@pytest.mark.parametrize("now,expected", [(100, 600), (8500, 500), (9000, None)])
def test_actual_native_installer_respects_remaining_global_deadline(tmp_path, inputs, monkeypatch, now, expected):
    import signal
    import time
    source = builder.bounded_installer(shared.load(inputs["manifest_path"]), "https://fixture.invalid/bundle", "/fixture/report")
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef): break
        nodes.append(node)
    calls = []; monkeypatch.setattr(signal, "alarm", calls.append); monkeypatch.setattr(time, "monotonic", lambda: now)
    run = lambda: exec(compile(ast.Module(body=nodes, type_ignores=[]), "deadline-fixture", "exec"), {"outer_deadline": 9000})
    if expected is None:
        with pytest.raises(TimeoutError): run()
        assert calls == []
    else: run(); assert calls == [expected]


@pytest.mark.parametrize("install", [False, True])
def test_native_worker_install_and_reuse_restore_bounded_failure_report(tmp_path, monkeypatch, install):
    ns = {}; exec(compile(builder.WORKER, "worker", "exec"), ns)
    alarms = []; monkeypatch.setattr(ns["signal"], "alarm", alarms.append)
    monkeypatch.setattr(ns["signal"], "signal", lambda *args: None)
    monkeypatch.delenv("CODEX_PILOT_REPORT_PUT", raising=False)
    for name in ("verify_predecessor", "preparation_config", "preparation_gates"): ns[name] = lambda config: {}
    def fail(config): raise ValueError("stop before collector")
    ns["gates"] = fail
    monkeypatch.setattr(ns["subprocess"], "Popen", lambda *a, **k: pytest.fail("No collector allowed"))
    config = dict(collection=dict(collection_mode=builder.MODE, run_root=str(tmp_path / "attempt"),
        task="turning_on_radio", instance_id=317, attempt_id="native-001", max_steps=4500),
        report_root=str(tmp_path / "report"), installation_required=install, admission={})
    if install: config["installer"] = "import signal\nsignal.alarm(600)\ntry: pass\nfinally: signal.alarm(0)"
    report = ns["run"](config)
    assert report["state"] == "failed" and report["stage"] == "gates" and report["collection_mode"] == builder.MODE
    assert alarms[0] == 9000 and alarms[-1] == 0
    if install: assert alarms[1:3] == [600, 0] and 8900 < alarms[3] <= 9000


def test_new_worker_still_uses_original_account_lease_and_finite_limits():
    assert "Codex-home lease" in builder.WORKER
    assert "signal.alarm(9000)" in builder.WORKER
    assert "code_v9_installer" in builder.WORKER
    assert "remaining=int(deadline-time.monotonic())" in builder.WORKER

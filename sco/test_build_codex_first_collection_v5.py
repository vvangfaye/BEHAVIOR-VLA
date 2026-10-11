"""Offline second-attempt gates and installer lifecycle; no network or GPU."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import stat
import sys
import time
from types import SimpleNamespace
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_first_collection_v5 as builder
import test_build_codex_first_collection_v4 as v4
from test_build_codex_first_collection_v3 import dump


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.WORKER, "wrapper_v5_test", "exec"), namespace)
    return SimpleNamespace(**namespace)


def failed_report():
    return dict(state="failed", training_eligible=False, task="turning_on_radio", instance_id=311,
                attempt_id="pilot-001", collector_progress=dict(executed_steps_observed=0),
                teacher_usage={key: 0 for key in ("request_directories", "validated_decisions", "failed_decisions",
                                                "model_calls_with_usage", "requests_without_usage")})


@pytest.fixture
def gate_inputs(tmp_path, worker):
    config = v4.gate_inputs.__wrapped__(tmp_path, worker)
    old = config["manifest"];base = Path(config["pilot_root"]);old_code = Path(old["code"])
    old.update(base_code=config["collection"]["base_code"], bundle_sha256=builder.OLD_BUNDLE_SHA256)
    for index in range(28):
        name = f"scripts/fixture_{index}.py";path = old_code / name;path.write_text(str(index))
        old["source_files_sha256"][name] = worker.sha(path)
    dump(old_code / "_CODEX_PILOT_CODE_READY", {**old, "state": "code_verified"})
    prep_path = base / "preparation-v14/report.json";prep = json.loads(prep_path.read_text())
    prep["code"] = dict(path=str(old_code), bundle_sha256=old["bundle_sha256"], files=29);dump(prep_path, prep)
    new = copy.deepcopy(old);new.update(version="v7", code=str(base / "code-v7"), bundle_sha256="d" * 64)
    shutil.copytree(old_code, new["code"])
    dump(Path(new["code"]) / "_CODEX_PILOT_CODE_READY", {**new, "state": "code_verified"})
    config.update(preparation_manifest=old, manifest=new, old_bundle_sha256=builder.OLD_BUNDLE_SHA256)
    config["collection"].update(code=new["code"], source_files_sha256=new["source_files_sha256"])
    return config


def test_reuses_original_code6_preparation_and_independently_checks_code7(worker, gate_inputs):
    result = worker.gates(gate_inputs)
    assert result["collection_code"].endswith("/code-v7")
    assert result["preparation_code_ready_sha256"] != result["code_ready_sha256"]
    assert result["selected_episodes"] == builder.EPISODES


@pytest.mark.parametrize("damage", ["old_prep", "new_source", "new_ready", "teacher_changed", "old_count"])
def test_either_old_or_new_code_mismatch_blocks(worker, gate_inputs, damage):
    base = Path(gate_inputs["pilot_root"])
    if damage == "old_prep":
        path = base / "preparation-v14/report.json";data = json.loads(path.read_text());data["code"]["path"] = str(base / "code-v7");dump(path, data)
    elif damage == "new_source":
        (base / "code-v7/scripts/fixture_0.py").write_text("tampered")
    elif damage == "new_ready":
        path = base / "code-v7/_CODEX_PILOT_CODE_READY";data = json.loads(path.read_text());data["bundle_sha256"] = "0" * 64;dump(path, data)
    elif damage == "teacher_changed":
        gate_inputs["manifest"]["source_files_sha256"]["scripts/codex_collection_teacher.py"] = "0" * 64
    else:
        gate_inputs["preparation_manifest"]["source_files_sha256"].pop("scripts/fixture_0.py")
    with pytest.raises(ValueError):
        worker.gates(gate_inputs)


def test_terminal_platform_success_can_recover_actual_failed_zero_work_attempt():
    job = dict(name=builder.OLD_NAME, uid=builder.OLD_UID, state="SUCCEEDED", observed_at=1000)
    observed, identity = builder.checked_prior(job, job, failed_report(), now=1050)
    assert observed == job and identity["executed_steps_observed"] == 0


@pytest.mark.parametrize("damage", ["stale", "running", "uid", "action", "teacher"])
def test_recovery_rejects_stale_unknown_or_executed_previous_attempt(damage):
    job = dict(name=builder.OLD_NAME, uid=builder.OLD_UID, state="SUCCEEDED", observed_at=1000)
    submission = dict(job);report = failed_report()
    if damage == "stale":job["observed_at"] = 0
    elif damage == "running":job["state"] = "RUNNING"
    elif damage == "uid":job["uid"] = "f" * 32
    elif damage == "action":report["collector_progress"]["executed_steps_observed"] = 1
    else:report["teacher_usage"]["request_directories"] = 1
    with pytest.raises(ValueError):builder.checked_prior(job, submission, report, now=1050)


@pytest.mark.parametrize("evidence", [None, "actions", "teacher", "trajectory"])
def test_worker_rechecks_old_afs_evidence_without_writing_it(tmp_path, worker, evidence):
    base = tmp_path / "pilot";report_path = base / "first-radio311-gpu-v4/report.json"
    report = failed_report();dump(report_path, report);before = report_path.read_bytes()
    attempt = base / "collection/radio-311-pilot001/attempt";attempt.mkdir(parents=True)
    if evidence == "actions":(attempt / "action_sources.jsonl").write_text('{"step":0}')
    elif evidence == "teacher":(attempt / "teacher/step-00000000").mkdir(parents=True)
    elif evidence == "trajectory":(attempt / "trajectory.hdf5").write_bytes(b"incomplete")
    job = dict(name=builder.OLD_NAME, uid=builder.OLD_UID, state="SUCCEEDED")
    config = dict(pilot_root=str(base), old_name=builder.OLD_NAME, old_uid=builder.OLD_UID,
                  terminal_states=sorted(builder.TERMINAL),
                  recovery_from=dict(job=job, report_identity=worker.recovery_identity(report)))
    if evidence:
        with pytest.raises(ValueError):worker.verify_recovery(config)
    else:
        assert worker.verify_recovery(config)["afs_report_sha256"] == worker.sha(report_path)
    assert report_path.read_bytes() == before


@pytest.mark.parametrize("installer_fails", [False, True])
def test_installer_alarm_is_restored_and_failures_stop_before_any_collector(tmp_path, worker, monkeypatch, installer_fails):
    namespace = worker.run.__globals__;alarms = []
    monkeypatch.setattr(worker.signal, "alarm", lambda seconds: alarms.append(seconds))
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.delenv("CODEX_PILOT_REPORT_PUT", raising=False)
    monkeypatch.setitem(namespace, "verify_recovery", lambda config: {})
    monkeypatch.setitem(namespace, "preparation_config", lambda config: config)
    monkeypatch.setitem(namespace, "preparation_gates", lambda config: {})
    def fail_gates(config):raise ValueError("stop before collection")
    monkeypatch.setitem(namespace, "gates", fail_gates)
    def forbid(*args, **kwargs):raise AssertionError("No collector may start")
    monkeypatch.setattr(worker.subprocess, "Popen", forbid)
    installer = "import signal\nsignal.alarm(600)\ntry:\n " + ("raise ValueError('installer failed')" if installer_fails else "pass") + "\nfinally:signal.alarm(0)"
    config = dict(collection=dict(run_root=str(tmp_path / "new-attempt")), report_root=str(tmp_path / "report"), installer=installer)
    report = worker.run(config)
    assert report["state"] == "failed" and report["stage"] == ("code_install" if installer_fails else "gates")
    assert alarms[0:3] == [9000, 600, 0] and 8900 <= alarms[3] <= 9000 and alarms[-1] == 0
    assert not issubclass(worker.WrapperDeadline, Exception)
    if installer_fails:assert json.loads((tmp_path / "report/code_install_report.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("kind,expected_calls", [("dns", 4), ("timeout", 1), ("http", 1)])
def test_private_installer_retries_only_transient_dns(tmp_path, worker, monkeypatch, kind, expected_calls):
    manifest = dict(version="v7", code=str(tmp_path / "code-v7"), original_data="/fixture/data", base_code="/fixture/base",
                    base_source_manifest_sha256="a" * 64, bundle_sha256="b" * 64)
    source = builder.private_installer(manifest, "https://fixture.invalid/bundle", tmp_path / "report")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "open_bundle")
    namespace = {"urllib": worker.urllib, "time": worker.time}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "dns_helper", "exec"), namespace)
    calls = []
    def opening(*args, **kwargs):
        calls.append(1)
        if kind == "dns" and len(calls) == 4:return "response"
        if kind == "http":raise urllib.error.HTTPError("https://fixture.invalid", 403, "blocked", {}, None)
        reason = socket.gaierror(socket.EAI_AGAIN, "temporary") if kind == "dns" else TimeoutError("bounded")
        raise urllib.error.URLError(reason)
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=opening))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    if kind == "dns":assert namespace["open_bundle"]("https://fixture.invalid") == "response"
    else:
        with pytest.raises(urllib.error.URLError):namespace["open_bundle"]("https://fixture.invalid")
    assert len(calls) == expected_calls


def test_offline_builder_freezes_pilot002_with_private_source_and_no_url_in_receipt(tmp_path, monkeypatch):
    import run_codex_collection
    required = ["scripts/run_codex_collection.sh", "scripts/collect_b1k_codex.py", "scripts/codex_collection_teacher.py",
                "scripts/codex_teacher_mailbox.py", "scripts/codex_collection_actions.py", "scripts/codex_collection_trajectory.py",
                "scripts/codex_collection_demonstrations.py", "sco/run_codex_collection.py", "sco/codex_cluster_proxy.py", run_codex_collection.ROBOT_CONFIG]
    names = required + [f"scripts/fixture_{index}.py" for index in range(29-len(required))]
    collection = dict(code="/fixture/old-code", base_code="/fixture/base", runtime="/fixture/runtime", checkpoint="/fixture/checkpoint",
        run_root="/fixture/pilot-001", task="turning_on_radio", instance_id=311, attempt_id="pilot-001", execution_name="offline-fixture",
        teacher_backend="codex_local", codex_binary="/fixture/codex", codex_home="/fixture/home", mihomo_binary="/fixture/mihomo",
        mihomo_config_dir="/fixture/mihomo-config", demonstration_manifest="/fixture/demo", robot_config="/fixture/robot",
        max_steps=4500, max_decisions=32, max_seconds=7200, wall_timeout_seconds=8400, teacher_interval=256,
        source_files_sha256={name: "d" * 64 for name in names})
    manifests = []
    for version in ("v6", "v7"):
        bundle = tmp_path / (version + ".tar.gz");bundle.write_bytes(version.encode())
        manifest = dict(version=version, code=builder.AFS + "/code-" + version, bundle=str(bundle),
            bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(), base_code=collection["base_code"],
            base_source_manifest_sha256="a" * 64, original_data=builder.AFS + "/original-v2",
            source_files_sha256=collection["source_files_sha256"])
        path = tmp_path / (version + ".json");dump(path, manifest);manifests.append((path, manifest))
    monkeypatch.setattr(builder, "OLD_BUNDLE_SHA256", manifests[0][1]["bundle_sha256"])
    old_source = tmp_path / "old.py";old_source.write_text("# immutable previous source")
    job = dict(name=builder.OLD_NAME, uid=builder.OLD_UID, state="SUCCEEDED", observed_at=time.time())
    kwargs = dict(manifest_path=manifests[1][0], preparation_manifest_path=manifests[0][0], previous_source_path=old_source)
    for key, value in {
        "collection_path": collection, "encryption_path": dict(selected_auth_file="auth_weizhi.json", ciphertext_sha256="c" * 64),
        "prior_job_path": job, "previous_submission_path": {**job, "source_sha256": hashlib.sha256(old_source.read_bytes()).hexdigest()},
        "previous_report_path": failed_report(),
    }.items():
        path = tmp_path / (key + ".json");dump(path, value);kwargs[key] = path
    url = tmp_path / "bundle-url";url.write_text("https://fixture.invalid/bundle?Signature=private-canary");url.chmod(0o600)
    output = tmp_path / "job.py";receipt = builder.build(output, bundle_url_file=url, **kwargs)
    assert receipt["attempt_id"] == "pilot-002" and receipt["collection_run_root"].endswith("/radio-311-pilot002")
    assert receipt["report_root"].endswith("/first-radio311-gpu-v5") and receipt["actual_job_submitted"] is False
    assert receipt["recommended_cloud_max_seconds"] == 9600 and receipt["max_seconds"] == 9000
    tree = ast.parse(output.read_text());config = ast.literal_eval(tree.body[-1].value.args[0])
    assert config["collection"]["attempt_id"] == "pilot-002" and config["manifest"]["version"] == "v7"
    assert config["preparation_manifest"]["version"] == "v6"
    assert "private-canary" in output.read_text() and "private-canary" not in output.with_suffix(".receipt.json").read_text()
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(output.with_suffix(".receipt.json").stat().st_mode) == 0o600
    assert old_source.read_text() == "# immutable previous source"
    with pytest.raises(ValueError, match="fresh"):builder.build(output, bundle_url_file=url, **kwargs)


def test_bundle_url_requires_private_regular_file(tmp_path):
    path = tmp_path / "url";path.write_text("https://fixture.invalid/bundle");path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):builder.private_bundle_url(path)
    path.chmod(0o600);assert builder.private_bundle_url(path).startswith("https://")
    link = tmp_path / "link";link.symlink_to(path)
    with pytest.raises(ValueError, match="0600"):builder.private_bundle_url(link)

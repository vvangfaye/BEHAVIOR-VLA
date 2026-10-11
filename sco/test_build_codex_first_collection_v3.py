"""Offline wrapper contracts; never launch a GPU, model, network or cloud job."""
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import urllib.error

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scripts")]
import build_codex_first_collection_v3 as builder


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.WORKER, "wrapper_test_definitions", "exec"), namespace)
    return SimpleNamespace(**namespace)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def gate_inputs(tmp_path, worker):
    base, code, original = tmp_path / "pilot", tmp_path / "pilot/code-v5", tmp_path / "base"
    for path in (code / "scripts", original): path.mkdir(parents=True)
    (code / "scripts/codex_collection_teacher.py").write_text("offline fixture auditor")
    (original / "_PUBLIC_TEST_SOURCE_READY").write_text("offline source fixture")
    dump(base / "cluster_validation_v9_report.json", {"kind": "offline fixture"})
    dump(base / "connectivity-v9/stdout.log", {"kind": "offline event fixture"})
    manifest = dict(version="v5", code=str(code), bundle_sha256="a" * 64,
                    base_source_manifest_sha256=worker.sha(original / "_PUBLIC_TEST_SOURCE_READY"),
                    source_files_sha256={"scripts/codex_collection_teacher.py": worker.sha(code / "scripts/codex_collection_teacher.py")})
    dump(code / "_CODEX_PILOT_CODE_READY", {**manifest, "state": "code_verified"})
    model = dict(status="verified", requested_model="gpt-6.1-sol", reasoning_effort="medium",
                 event_audit=dict(status="no_tool_events_observed", completed_turns=1))
    validation = dict(state="validation_complete", selected_auth="auth_weizhi.json", account_failover=False,
        authentication=dict(ciphertext_sha256="b" * 64, readback_verified=True, live_login_verified=True), model=model,
        new_model_calls=0, source_report_sha256=worker.sha(base / "cluster_validation_v9_report.json"),
        event_log_sha256=worker.sha(base / "connectivity-v9/stdout.log"),
        auditor_sha256=manifest["source_files_sha256"]["scripts/codex_collection_teacher.py"])
    dump(base / "cluster_validation_v13_report.json", validation)
    demo_path = base / "original-v3/demonstrations/28/manifest.json"
    dump(demo_path, dict(metadata=dict(task="turning_on_radio", episode_index=28), fingerprint="c" * 64))
    data = dict(state="complete", original_reader_verified=True, raw_success_verified=True,
        selected_episodes=builder.EPISODES,
        demonstrations=[dict(episode_index=episode, manifest=str(base / f"original-v3/demonstrations/{episode}/manifest.json"),
                             fingerprint="c" * 64) for episode in builder.EPISODES])
    data_path = base / "original-v3/_ORIGINAL_DATA_READY.json"; dump(data_path, data)
    prep = dict(state="complete", ready_for_collection=True,
        code=dict(path=str(code), bundle_sha256=manifest["bundle_sha256"], files=1),
        data_root=str(base / "original-v3"), data_ready_sha256=worker.sha(data_path))
    dump(base / "preparation-v13/report.json", prep)
    collection = dict(code=str(code), base_code=str(original), source_files_sha256=manifest["source_files_sha256"],
                      demonstration_manifest=str(demo_path), run_root=str(base / "collection/owned-attempt"))
    return dict(pilot_root=str(base), manifest=manifest, collection=collection, episodes=builder.EPISODES,
                ciphertext_sha256="b" * 64, report_root=str(base / "first-radio311-gpu-v3"), python="/fixture/python")


def test_all_preparation_validation_code_and_original_data_gates(worker, gate_inputs):
    result = worker.gates(gate_inputs)
    assert result["selected_episodes"] == builder.EPISODES
    assert result["selected_auth"] == "auth_weizhi.json"


@pytest.mark.parametrize("target,key,value", [
    ("preparation-v13/report.json", "state", "failed"),
    ("preparation-v13/report.json", "ready_for_collection", False),
    ("preparation-v13/report.json", "data_ready_sha256", "0" * 64),
    ("cluster_validation_v13_report.json", "state", "incomplete"),
    ("cluster_validation_v13_report.json", "selected_auth", "another-account.json"),
    ("cluster_validation_v13_report.json", "account_failover", True),
    ("cluster_validation_v13_report.json", "event_log_sha256", "0" * 64),
    ("cluster_validation_v13_report.json", "auditor_sha256", "0" * 64),
    ("original-v3/_ORIGINAL_DATA_READY.json", "raw_success_verified", False),
    ("original-v3/_ORIGINAL_DATA_READY.json", "selected_episodes", [28]),
])
def test_missing_or_mismatched_gate_blocks_collection(worker, gate_inputs, target, key, value):
    path = Path(gate_inputs["pilot_root"]) / target
    data = json.loads(path.read_text()); data[key] = value; dump(path, data)
    with pytest.raises(ValueError): worker.gates(gate_inputs)


def test_pinned_model_and_live_login_are_required(worker, gate_inputs):
    path = Path(gate_inputs["pilot_root"]) / "cluster_validation_v13_report.json"
    original = json.loads(path.read_text())
    for section, key, value in [("authentication", "live_login_verified", False),
                                ("model", "requested_model", "other"), ("model", "reasoning_effort", "ultra")]:
        data = json.loads(json.dumps(original)); data[section][key] = value; dump(path, data)
        with pytest.raises(ValueError): worker.gates(gate_inputs)


def test_usage_reads_actual_collector_and_teacher_evidence_layout(tmp_path, worker):
    from test_collect_b1k_codex import Evaluator, run
    from test_codex_collection_teacher import FakeRunner
    from codex_collection_teacher import CodexTeacher
    teacher = CodexTeacher(runner=FakeRunner())
    run(tmp_path, Evaluator(terminal_step=1), teacher)
    summary = worker.teacher_usage(tmp_path / "attempt")
    assert summary["request_directories"] == summary["validated_decisions"] == summary["model_calls_with_usage"] == 1
    assert summary["observed_token_totals"] == dict(input_tokens=1234, cached_input_tokens=1000,
                                                   output_tokens=42, reasoning_output_tokens=20)
    assert summary["requests_without_usage"] == 0


def test_inflight_usage_is_counted_once_and_missing_usage_is_unknown(tmp_path, worker):
    attempt = tmp_path / "attempt"
    for index in range(3): (attempt / f"teacher/step-{index:08d}/decision/fixture").mkdir(parents=True)
    usage = dict(input_tokens=100, cached_input_tokens=50, output_tokens=10)
    for index in range(2):
        path = attempt / f"teacher/step-{index:08d}/decision/fixture"
        (path / "stdout.log").write_text(json.dumps(dict(type="turn.completed", usage=usage)) + "\n")
    dump(attempt / "teacher/step-00000000/decision/fixture/receipt.json", dict(status="validated", usage=usage))
    summary = worker.teacher_usage(attempt)
    assert summary["model_calls_with_usage"] == 2 and summary["requests_without_usage"] == 1
    assert summary["observed_token_totals"]["input_tokens"] == 200
    assert worker.teacher_usage(tmp_path / "absent")["observed_token_totals"] is None


def test_collector_progress_tolerates_partial_last_action(tmp_path, worker):
    attempt = tmp_path / "attempt"; attempt.mkdir()
    (attempt / "action_sources.jsonl").write_text('{"step":9,"source":"teacher_cartesian_correction"}\n{"step":10')
    summary = worker.progress(tmp_path)
    assert summary["collector_progress"]["executed_steps_observed"] == 10


def test_failure_tails_redact_credentials_urls_and_respect_byte_budget(tmp_path, worker):
    secret = "a" * 48
    text = "ordinary failure\n--expected-server-token " + secret + "\nhttps://private.invalid/file?Signature=private\n"
    text += "Authorization: Bearer short-secret\nauth.json contents must not appear\n" + secret + "\n"
    text += "保持 Unicode 边界\n" * 2000
    path = tmp_path / "log"; path.write_text(text)
    result = worker.safe_text(text, 65536)
    assert secret not in result and "private.invalid" not in result and "short-secret" not in result
    assert "auth.json contents" not in result and "ordinary failure" in result
    assert len(worker.tail(path)["tail"].encode()) <= 4096


@pytest.mark.parametrize("kind,calls", [("dns", 4), ("timeout", 1), ("http", 1)])
def test_report_retries_only_eai_again_at_most_four_times(worker, monkeypatch, kind, calls):
    observed=[]
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def opening(request, **kwargs):
        observed.append(1)
        if kind == "dns" and len(observed) == 4: return Response()
        if kind == "http": raise urllib.error.HTTPError("https://fixture.invalid", 403, "blocked", {}, None)
        reason = socket.gaierror(socket.EAI_AGAIN, "temporary") if kind == "dns" else TimeoutError("bounded")
        raise urllib.error.URLError(reason)
    monkeypatch.setattr(worker.threading, "Event", lambda: SimpleNamespace(wait=lambda seconds: None))
    if kind == "dns":
        assert worker.direct_put(SimpleNamespace(open=opening), "https://fixture.invalid", b"{}", worker.time.monotonic()+30) == 4
    else:
        with pytest.raises(urllib.error.URLError):
            worker.direct_put(SimpleNamespace(open=opening), "https://fixture.invalid", b"{}", worker.time.monotonic()+30)
    assert len(observed) == calls


def test_report_transport_failure_never_replaces_local_gate_failure(worker, gate_inputs, monkeypatch):
    path = Path(gate_inputs["pilot_root"]) / "preparation-v13/report.json"
    dump(path, {"state": "failed"})
    def fail(*args, **kwargs): raise TimeoutError("fixture no network")
    monkeypatch.setattr(worker.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=fail))
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker.signal, "alarm", lambda *args: None)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Gate must block process launch"))
    monkeypatch.setenv("CODEX_PILOT_REPORT_PUT", "https://fixture.invalid/private")
    report = worker.run(gate_inputs)
    assert report["state"] == "failed" and report["error_type"] == "ValueError"
    assert report["report_transport_error"]["error_type"] == "TimeoutError"
    assert json.loads((Path(gate_inputs["report_root"]) / "report.json").read_text())["state"] == "failed"
    assert "fixture.invalid" not in json.dumps(report)


def test_build_does_not_submit_and_keeps_bootstrap_report_assignment(tmp_path):
    names = ["scripts/run_codex_collection.sh", "scripts/collect_b1k_codex.py",
             "scripts/codex_collection_teacher.py", "scripts/codex_teacher_mailbox.py",
             "scripts/codex_collection_actions.py", "scripts/codex_collection_trajectory.py",
             "scripts/codex_collection_demonstrations.py", "sco/run_codex_collection.py",
             "sco/codex_cluster_proxy.py", "BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml"]
    initial = dict(code=builder.AFS+"/code-v4", base_code="/fixture/base", runtime="/fixture/runtime",
        checkpoint="/fixture/checkpoint", run_root="/fixture/collection/radio", task="turning_on_radio",
        instance_id=311, attempt_id="pilot-001", execution_name="offline-fixture", teacher_backend="codex_local",
        codex_binary="/fixture/codex", codex_home="/fixture/private-home", mihomo_binary="/fixture/mihomo",
        mihomo_config_dir="/fixture/mihomo-config", robot_config="/fixture/robot.yaml",
        demonstration_manifest="/fixture/demo.json", max_steps=4500, max_decisions=32, max_seconds=7200,
        wall_timeout_seconds=8400, teacher_interval=256, source_files_sha256={name:"d"*64 for name in names})
    collection_path = tmp_path / "initial.json"; dump(collection_path, initial)
    bundle = tmp_path / "fixture-bundle.tar.gz"; bundle.write_bytes(b"offline builder fixture")
    manifest = dict(version="v5", code=builder.AFS+"/code-v5", bundle=str(bundle),
                    bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    source_files_sha256=initial["source_files_sha256"])
    manifest_path = tmp_path / "manifest.json"; dump(manifest_path, manifest)
    encryption_path = tmp_path / "encryption.json"; dump(encryption_path, dict(selected_auth_file="auth_weizhi.json", ciphertext_sha256="c"*64))
    target = tmp_path / "first_collection_job_v3.py"
    receipt = builder.build(target, manifest_path=manifest_path, collection_path=collection_path, encryption_path=encryption_path)
    source = target.read_text(); compile(source, str(target), "exec")
    assert receipt["actual_job_submitted"] is False and receipt["max_seconds"] == 9000
    assert "BOOTSTRAP_REPORT = run(" in source and "os.execve" not in source and "with MihomoProxy" not in source
    assert "code-v5/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml" in source
    assert "original-v3/demonstrations/28/manifest.json" in source and builder.PYTHON in source
    with pytest.raises(ValueError, match="fresh"):
        builder.build(target, manifest_path=manifest_path, collection_path=collection_path, encryption_path=encryption_path)

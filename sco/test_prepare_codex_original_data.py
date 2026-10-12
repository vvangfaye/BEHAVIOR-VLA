"""Finite selected-file CPU staging tests; fake subprocesses never decode video."""
import json
import io
from pathlib import Path
import sys
import stat
import socket
import urllib.error
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sco import prepare_codex_original_data as stage
from test_stage_subtask_replay import inputs as replay_inputs


@pytest.fixture
def payload(replay_inputs, tmp_path):
    metadata, bucket, plan = replay_inputs
    originals = tmp_path / "source"
    for key, content in bucket.objects.items():
        target = originals / key.removeprefix("base/")
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    retained = tmp_path / "retained.txt"; retained.write_text("0\n1\n")
    labels = tmp_path / "segments.parquet"; labels.write_bytes(b"synthetic fixture labels, no decoder")
    return dict(kind=stage.KIND, output_root=str(tmp_path / "output"),
                code_root=str(Path(__file__).resolve().parents[1]), episodes=[0, 1],
                metadata_source=str(metadata), local_source_roots=[str(originals)],
                segments_path=str(labels), retained_episodes_path=str(retained), reserve_bytes=0,
                segments_sha256=stage.sha(labels), retained_episodes_sha256=stage.sha(retained),
                demonstrations=[dict(episode_index=ep, raw_path=f"/fixture/source-{ep}.hdf5") for ep in (0, 1)])


def demo_process(arguments, **kwargs):
    assert "codex_collection_demonstrations.py" in arguments[1]
    assert kwargs["timeout"] <= 900
    target = Path(arguments[arguments.index("--output-dir") + 1]); target.mkdir(parents=True)
    value = dict(fingerprint="synthetic-contract-fixture", examples=[{}], metadata={"task": "task"})
    (target / "manifest.json").write_text(json.dumps(value))
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_exact_shared_files_are_staged_once_and_source_is_untouched(payload, monkeypatch):
    monkeypatch.setattr(stage.subprocess, "run", demo_process)
    result = stage.run(payload)
    root = Path(payload["output_root"])
    assert len(result["transfers"]) == 4 and result["selected_episodes"] == [0, 1]
    assert result["ready_for_training"] is False
    assert result["original_reader_verified"] is True  # Synthetic subprocess proof in this unit fixture only.
    assert (root / "_ORIGINAL_DATA_READY.json").is_file() and not (root / "_RUNNING").exists()
    assert "depth" not in json.loads((root / "meta/info.json").read_text())["features"]
    assert "depth" in json.loads((Path(payload["metadata_source"]) / "meta/info.json").read_text())["features"]
    assert all(stage.sha(root / item["path"]) == item["sha256"] for item in result["transfers"])


def test_frozen_raw_hashes_are_passed_without_changing_demonstration_scope(payload, monkeypatch):
    original_scope = [dict(row) for row in payload["demonstrations"]]
    payload["raw_source_sha256"] = {row["raw_path"]: str(i + 1) * 64
                                   for i, row in enumerate(original_scope)}
    called = []
    def process(arguments, **kwargs):
        raw = arguments[arguments.index("--raw-path") + 1]
        digest = arguments[arguments.index("--expected-raw-sha256") + 1]
        assert digest == payload["raw_source_sha256"][raw]
        called.append(raw)
        return demo_process(arguments, **kwargs)
    monkeypatch.setattr(stage.subprocess, "run", process)
    result = stage.run(payload)
    assert result["state"] == "complete" and len(called) == len(original_scope)
    assert payload["demonstrations"] == original_scope


def test_predeclared_required_subtask_is_passed_to_real_demonstration_reader(payload, monkeypatch):
    payload["required_demonstration_subtask"] = "place on tupperware to countertop"
    calls = []
    def process(arguments, **kwargs):
        assert arguments[arguments.index("--required-subtask") + 1] == payload["required_demonstration_subtask"]
        calls.append(arguments)
        return demo_process(arguments, **kwargs)
    monkeypatch.setattr(stage.subprocess, "run", process)
    assert stage.run(payload)["state"] == "complete"
    assert len(calls) == len(payload["demonstrations"])


@pytest.mark.parametrize("damage", ["missing", "extra", "invalid"])
def test_raw_hash_mapping_must_exactly_pin_selected_sources(payload, damage):
    payload["raw_source_sha256"] = {row["raw_path"]: "a" * 64 for row in payload["demonstrations"]}
    first = payload["demonstrations"][0]["raw_path"]
    if damage == "missing": del payload["raw_source_sha256"][first]
    elif damage == "extra": payload["raw_source_sha256"]["/unselected.hdf5"] = "b" * 64
    else: payload["raw_source_sha256"][first] = "not-a-sha"
    with pytest.raises(ValueError, match="Frozen raw SHA256"):
        stage.run(payload)
    assert not Path(payload["output_root"]).exists()


@pytest.mark.parametrize("damage", ["duplicate", "missing_demo", "changed_source", "unbounded_time"])
def test_invalid_scope_is_rejected_before_output(payload, damage):
    if damage == "duplicate": payload["episodes"] = [0, 0]
    elif damage == "missing_demo": payload["demonstrations"].pop()
    elif damage == "changed_source": payload["source"] = "oss://another-bucket/all-data"
    else: payload["max_seconds"] = 999999
    with pytest.raises(ValueError): stage.run(payload)
    assert not Path(payload["output_root"]).exists()


@pytest.mark.parametrize("damage", ["budget", "labels", "retained", "reader"])
def test_partial_staging_cannot_publish_ready(payload, monkeypatch, damage):
    if damage == "budget": payload["max_download_bytes"] = 1
    elif damage == "labels": Path(payload["segments_path"]).write_bytes(b"changed")
    elif damage == "retained": Path(payload["retained_episodes_path"]).write_text("0\n")
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="secret must not be logged", stderr="signed request"))
    with pytest.raises((ValueError, RuntimeError)): stage.run(payload)
    root = Path(payload["output_root"])
    assert (root / "_FAILED").is_file() and not (root / "_ORIGINAL_DATA_READY.json").exists()
    text = (root / "status.json").read_text()
    assert "secret" not in text and "signed request" not in text


def test_oss_stat_extracts_only_identity_and_rejects_missing_size():
    assert stage.stat_values('Content-Length: 124\nETag: "abcdef"\n') == dict(bytes=124, etag="abcdef")
    with pytest.raises(ValueError): stage.stat_values("ETag: abcdef\n")


def test_existing_output_is_never_reused(payload):
    Path(payload["output_root"]).mkdir()
    with pytest.raises(ValueError, match="fresh"): stage.run(payload)


def test_paths_cannot_escape_selected_dataset():
    for value in ("../outside", "/outside", "data/../../outside", "data\\outside"):
        with pytest.raises(ValueError): stage.relative(value)


@pytest.fixture
def signed_payload(payload, replay_inputs):
    _, bucket, _ = replay_inputs
    payload["local_source_roots"] = []
    payload["signed_files"] = [dict(relative_path=key.removeprefix("base/"),
                                     url=f"https://fixture.invalid/{i}?signature=DO_NOT_PERSIST",
                                     bytes=len(content), etag=f"fixture-{i}")
                                for i, (key, content) in enumerate(bucket.objects.items())]
    return payload, list(bucket.objects.values())


def test_signed_selected_files_need_no_credentials_and_never_persist_urls(signed_payload, monkeypatch):
    payload, contents = signed_payload
    monkeypatch.setattr(stage, "find_ossutil", lambda *a: pytest.fail("Signed path must not discover credentials"))
    monkeypatch.setattr(stage.subprocess, "run", demo_process)
    calls = []
    def download(request, **kwargs):
        i = int(request.full_url.split("?")[0].rsplit("/", 1)[1])
        assert kwargs["timeout"] <= 60
        assert request.get_header("If-match") == f'"fixture-{i}"'
        calls.append(i)
        response = io.BytesIO(contents[i])
        response.headers = {"Content-Length": str(len(contents[i])), "ETag": f'"fixture-{i}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, "urlopen", download)
    result = stage.run(payload)
    assert len(calls) == 4 and result["state"] == "complete"
    root = Path(payload["output_root"])
    for name in ("payload.json", "status.json", "_ORIGINAL_DATA_READY.json"):
        assert "DO_NOT_PERSIST" not in (root / name).read_text()
        assert "fixture.invalid" not in (root / name).read_text()


@pytest.mark.parametrize("damage", ["length", "etag", "truncated"])
def test_changed_signed_transfer_cannot_publish_ready(signed_payload, monkeypatch, damage):
    payload, contents = signed_payload
    def download(request, **kwargs):
        i = int(request.full_url.split("?")[0].rsplit("/", 1)[1])
        response = io.BytesIO(contents[i][:-1] if damage == "truncated" else contents[i])
        response.headers = {"Content-Length": str(len(contents[i]) + (damage == "length")),
                            "ETag": "wrong" if damage == "etag" else f'"fixture-{i}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, "urlopen", download)
    with pytest.raises(ValueError): stage.run(payload)
    assert not (Path(payload["output_root"]) / "_ORIGINAL_DATA_READY.json").exists()


def test_decoder_failure_has_private_bounded_sanitized_evidence(payload, monkeypatch):
    stderr = "Traceback: DecoderError frame mismatch https://example.invalid/video?Signature=private\nOSS_ACCESS_KEY_SECRET=hidden\n"
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=7, stdout="", stderr=stderr))
    with pytest.raises(RuntimeError): stage.run(payload)
    root = Path(payload["output_root"])
    failure = json.loads((root / "child_failure.json").read_text())
    assert failure["returncode"] == 7 and failure["phase"] == "real_reader_and_demonstrations"
    assert "DecoderError frame mismatch" in failure["diagnostic"]
    assert "example.invalid" not in failure["diagnostic"] and "hidden" not in failure["diagnostic"]
    assert len(failure["diagnostic"]) <= 8192
    assert stat.S_IMODE((root / "child_failure.json").stat().st_mode) == 0o600
    assert json.loads((root / "status.json").read_text())["failure_phase"] == "real_reader_and_demonstrations"


def test_oss_cli_failure_streams_are_completely_discarded(payload, monkeypatch):
    payload["local_source_roots"] = []
    monkeypatch.setattr(stage, "find_ossutil", lambda *a: ("ossutil", []))
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="OSS_SECRET=private", stderr="private config dump"))
    with pytest.raises(RuntimeError): stage.run(payload)
    root = Path(payload["output_root"])
    assert not (root / "child_failure.json").exists()
    assert "OSS_SECRET" not in (root / "status.json").read_text()
    assert "private config dump" not in (root / "status.json").read_text()


@pytest.mark.parametrize("http", [True, False])
def test_transport_failure_diagnostics_do_not_dump_urls_or_nested_credentials(signed_payload, monkeypatch, http):
    payload, _ = signed_payload
    private = 'https://fixture.invalid/object?Signature=PRIVATE_QUERY OSS_ACCESS_KEY_SECRET=PRIVATE_CREDENTIAL'
    error = (urllib.error.HTTPError(private, 403, private, {}, None) if http else
             urllib.error.URLError(TimeoutError(110, private)))
    def download(*args, **kwargs):
        raise error
    monkeypatch.setattr(stage.urllib.request, 'urlopen', download)
    with pytest.raises(urllib.error.URLError):
        stage.run(payload)
    result = json.loads((Path(payload['output_root']) / 'status.json').read_text())
    assert result['failure_phase'] == 'selected_files' and result['failure_detail']
    if http:
        assert result['http_status'] == 403
    else:
        assert result['url_reason_type'] == 'TimeoutError' and result['url_reason_errno'] == 110
    assert all(value not in json.dumps(result) for value in ('fixture.invalid', 'PRIVATE_QUERY', 'PRIVATE_CREDENTIAL'))


def test_general_failure_detail_uses_existing_bounded_redactor():
    result = stage.failure_diagnostic(ValueError('Download https://fixture.invalid/file?Signature=secret OSS_ACCESS_KEY_SECRET=private'))
    assert 'fixture.invalid' not in result['failure_detail'] and 'private' not in result['failure_detail']
    assert '<URL redacted>' in result['failure_detail'] and '<redacted>' in result['failure_detail']


@pytest.mark.parametrize('succeeds', [True, False])
def test_only_initial_transient_dns_open_is_retried_at_most_four_times(monkeypatch, succeeds):
    calls, waits = [], []
    response = io.BytesIO(b'body')
    def opening(*args, **kwargs):
        calls.append(kwargs['timeout'])
        if succeeds and len(calls) == 4:
            return response
        raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, 'temporary DNS failure'))
    monkeypatch.setattr(stage.urllib.request, 'urlopen', opening)
    monkeypatch.setattr(stage.threading, 'Event', lambda: SimpleNamespace(wait=waits.append))
    if succeeds:
        actual, attempts = stage.open_signed_request(object(), deadline=stage.time.monotonic()+30)
        assert actual is response and attempts == 4
    else:
        with pytest.raises(urllib.error.URLError):
            stage.open_signed_request(object(), deadline=stage.time.monotonic()+30)
    assert len(calls) == 4 and waits == [1, 2, 4]


@pytest.mark.parametrize('error', [urllib.error.HTTPError('https://fixture.invalid', 403, 'Forbidden', {}, None),
    urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, 'permanent DNS failure'))])
def test_http_or_nontransient_dns_failure_is_not_retried(monkeypatch, error):
    calls = []
    def opening(*args, **kwargs):
        calls.append(1); raise error
    monkeypatch.setattr(stage.urllib.request, 'urlopen', opening)
    monkeypatch.setattr(stage.threading, 'Event', lambda: pytest.fail('Unexpected retry wait'))
    with pytest.raises(urllib.error.URLError):
        stage.open_signed_request(object(), deadline=stage.time.monotonic()+30)
    assert len(calls) == 1


def test_dns_retry_wait_cannot_extend_total_staging_deadline(monkeypatch):
    ticks = iter([0, 0, 2]); calls, waits = [], []
    monkeypatch.setattr(stage.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(stage.threading, 'Event', lambda: SimpleNamespace(wait=waits.append))
    def opening(*args, **kwargs):
        calls.append(1); raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, 'temporary DNS failure'))
    monkeypatch.setattr(stage.urllib.request, 'urlopen', opening)
    with pytest.raises(TimeoutError):
        stage.open_signed_request(object(), deadline=1)
    assert calls == [1] and waits == [1]


def test_response_body_dns_error_is_not_retried(signed_payload, monkeypatch):
    payload, contents = signed_payload
    calls = []
    class BrokenBody(io.BytesIO):
        def read(self, *args):
            raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, 'body interrupted'))
    def opening(request, **kwargs):
        index = int(request.full_url.split('?')[0].rsplit('/', 1)[1]); calls.append(index)
        response = BrokenBody(contents[index])
        response.headers = {'Content-Length': str(len(contents[index])), 'ETag': f'"fixture-{index}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, 'urlopen', opening)
    monkeypatch.setattr(stage.threading, 'Event', lambda: pytest.fail('Body must not be retried'))
    with pytest.raises(urllib.error.URLError): stage.run(payload)
    assert len(calls) == 1
    assert not (Path(payload['output_root']) / '_ORIGINAL_DATA_READY.json').exists()


def test_local_reuse_keeps_frozen_signed_etag_for_a_later_resume(signed_payload, tmp_path, monkeypatch):
    payload, _ = signed_payload
    payload['local_source_roots'] = [str(tmp_path / 'source')]
    monkeypatch.setattr(stage.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Verified local files need no GET'))
    monkeypatch.setattr(stage.subprocess, 'run', demo_process)
    result = stage.run(payload)
    expected = {row['relative_path']: row['etag'] for row in payload['signed_files']}
    assert all(row['etag'] == expected[row['path']] for row in result['transfers'])
    assert all(row['signed_request_attempts'] is None for row in result['transfers'])

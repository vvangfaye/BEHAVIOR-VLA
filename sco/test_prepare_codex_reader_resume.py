"""Offline recovery tests; no fixture is accepted as actual reader/raw success."""
import copy
import errno
import io
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_codex_reader_resume as resume
from test_prepare_codex_original_data import payload, replay_inputs, signed_payload, stage


JOB = {"name": "behavior-codex-original-v2-fixture", "uid": "1" * 32, "state": "FAILED"}
SUBMISSION = {"name": JOB["name"], "uid": JOB["uid"], "state": "submitted"}


@pytest.fixture
def downloaded(signed_payload, monkeypatch, tmp_path):
    original_payload, contents = signed_payload
    def download(request, **kwargs):
        index = int(request.full_url.split("?")[0].rsplit("/", 1)[1])
        response = io.BytesIO(contents[index])
        response.headers = {"Content-Length": str(len(contents[index])), "ETag": f'"fixture-{index}"'}
        return response
    monkeypatch.setattr(stage.urllib.request, "urlopen", download)
    # Force the reader stage to fail. Nothing in this test suite certifies raw
    # success or marks a fixture as a verified demonstration.
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout="", stderr="fixture reader failure"))
    with pytest.raises(RuntimeError):
        stage.run(original_payload)
    old = Path(original_payload["output_root"])
    refreshed = copy.deepcopy(original_payload)
    refreshed["output_root"] = str(tmp_path / "new-staging")
    for row in refreshed["signed_files"]:
        row["url"] = row["url"].replace("DO_NOT_PERSIST", "REFRESHED_PRIVATE_TOKEN")
    return old, refreshed


def prepare(downloaded, tmp_path, **overrides):
    old, payload = downloaded
    args = dict(previous_job=JOB, previous_submission=SUBMISSION,
                reuse_root=tmp_path / "verified-reuse", payload_path=tmp_path / "private-next-payload.json")
    args.update(overrides)
    return resume.prepare_resume(old, payload, **args)


def files(root):
    return {p.relative_to(root).as_posix(): stage.sha(p) for p in root.rglob("*") if p.is_file()}


def test_pristine_metadata_and_complete_data_prepare_without_mutating_or_claiming_success(downloaded, tmp_path):
    old, payload = downloaded
    source = Path(payload["metadata_source"])
    old_before, source_before = files(old), files(source)
    assert "depth" not in json.loads((old / "meta/info.json").read_text())["features"]
    assert (old / "meta/info.json.orig").exists()
    result = prepare(downloaded, tmp_path)
    reuse = tmp_path / "verified-reuse"
    assert "depth" in json.loads((reuse / "meta/info.json").read_text())["features"]
    assert not (reuse / "meta/info.json.orig").exists()
    assert stage.sha(reuse / "meta/info.json") == source_before["meta/info.json"]
    assert (reuse / "meta/info.json").stat().st_ino != (source / "meta/info.json").stat().st_ino
    assert result["reused_files"] == 4 and result["remaining_files"] == result["remaining_download_bytes"] == 0
    assert result["state"] == "prepared_only" and result["previous_job_matches_submission"]
    assert not result["ready_for_training"] and not result["raw_success_verified"] and not result["original_reader_verified"]
    assert not Path(payload["output_root"]).exists()
    assert not (reuse / "_ORIGINAL_DATA_READY.json").exists()
    private = tmp_path / "private-next-payload.json"
    prepared = json.loads(private.read_text())
    assert prepared["metadata_source"] == str(reuse) and prepared["local_source_roots"] == [str(reuse)]
    assert all(len(row["sha256"]) == 64 for row in prepared["signed_files"])
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(reuse.stat().st_mode) == 0o700
    assert "REFRESHED_PRIVATE_TOKEN" not in json.dumps(result)
    assert "fixture.invalid" not in (reuse / "reader_resume_receipt.json").read_text()
    for row in result["source_files"]:
        assert (old / row["path"]).stat().st_ino == (reuse / row["path"]).stat().st_ino
    assert files(old) == old_before and files(source) == source_before


def test_prepared_stager_uses_local_roots_and_still_requires_real_reader(downloaded, tmp_path, monkeypatch):
    old, payload = downloaded
    before = files(old)
    prepare(downloaded, tmp_path)
    monkeypatch.setattr(stage.urllib.request, "urlopen", lambda *a, **k: pytest.fail("No download is allowed"))
    monkeypatch.setattr(stage, "find_ossutil", lambda *a: pytest.fail("No credential or OSS lookup is allowed"))
    prepared = json.loads((tmp_path / "private-next-payload.json").read_text())
    with pytest.raises(RuntimeError):  # The intentionally failing reader is still mandatory.
        stage.run(prepared)
    fresh = Path(payload["output_root"])
    status = json.loads((fresh / "status.json").read_text())
    assert status["failure_phase"] == "real_reader_and_demonstrations" and len(status["transfers"]) == 4
    assert not (fresh / "_ORIGINAL_DATA_READY.json").exists()
    assert files(old) == before


@pytest.mark.parametrize("fault", ["running", "ready", "phase", "missing_transfer", "extra_transfer",
    "duplicate_transfer", "data_corruption", "metadata_corruption", "metadata_extra", "metadata_source",
    "signed_etag", "signed_size", "signed_sha", "signed_missing", "raw_scope", "labels", "selection",
    "required_paths", "data_symlink", "metadata_symlink"])
def test_corrupt_or_changed_evidence_fails_before_creating_resume(downloaded, tmp_path, fault):
    old, payload = downloaded
    state = json.loads((old / "status.json").read_text())
    if fault == "running": (old / "_RUNNING").touch()
    elif fault == "ready": (old / "_ORIGINAL_DATA_READY.json").write_text("{}")
    elif fault == "phase": state["failure_phase"] = "selected_files"
    elif fault == "missing_transfer": state["transfers"].pop()
    elif fault == "extra_transfer": state["transfers"].append({**state["transfers"][0], "path": "data/extra.parquet"})
    elif fault == "duplicate_transfer": state["transfers"].append(state["transfers"][0])
    elif fault == "data_corruption": (old / state["transfers"][0]["path"]).write_bytes(b"corrupted")
    elif fault == "metadata_corruption": (Path(payload["metadata_source"]) / "meta/info.json").write_text("{}")
    elif fault == "metadata_extra": (Path(payload["metadata_source"]) / "meta/unreceipted.json").write_text("{}")
    elif fault == "metadata_source": payload["metadata_source"] = str(old)
    elif fault == "signed_etag": payload["signed_files"][0]["etag"] = "changed"
    elif fault == "signed_size": payload["signed_files"][0]["bytes"] += 1
    elif fault == "signed_sha": payload["signed_files"][0]["sha256"] = "0" * 64
    elif fault == "signed_missing": payload["signed_files"].pop()
    elif fault == "raw_scope": payload["demonstrations"][0]["raw_path"] = "/other/raw.hdf5"
    elif fault == "labels": Path(payload["segments_path"]).write_bytes(b"changed")
    elif fault == "selection": state["selected_metadata"][0]["task_instance_id"] = 301
    elif fault == "required_paths": state["required_files"].append("data/unselected.parquet")
    elif fault == "data_symlink":
        target = old / state["transfers"][0]["path"]
        other = tmp_path / "linked-data"; other.write_bytes(target.read_bytes()); target.unlink(); target.symlink_to(other)
    elif fault == "metadata_symlink":
        target = Path(payload["metadata_source"]) / "meta/info.json"
        other = tmp_path / "linked-info"; other.write_bytes(target.read_bytes()); target.unlink(); target.symlink_to(other)
    (old / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        prepare(downloaded, tmp_path)
    assert not (tmp_path / "verified-reuse").exists()
    assert not (tmp_path / "private-next-payload.json").exists()


@pytest.mark.parametrize("job", [{**JOB, "state": "RUNNING"}, {**JOB, "uid": "2" * 32},
                                  {**JOB, "name": "different-job"}, {**JOB, "uid": ""}])
def test_exact_terminal_job_is_bound_to_separate_submission(downloaded, tmp_path, job):
    with pytest.raises(ValueError, match="terminal job"):
        prepare(downloaded, tmp_path, previous_job=job)
    assert not (tmp_path / "verified-reuse").exists()


def test_cross_filesystem_copy_fallback_is_verified(downloaded, tmp_path, monkeypatch):
    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "fixture different filesystem")
    monkeypatch.setattr(resume.os, "link", cross_device)
    result = prepare(downloaded, tmp_path)
    assert result["local_copy_fallback_bytes"] == result["reused_bytes"]


@pytest.mark.parametrize("bad", ["reuse_in_old", "output_in_old", "payload_in_reuse", "output_exists", "payload_exists"])
def test_output_paths_cannot_mutate_existing_evidence(downloaded, tmp_path, bad):
    old, payload = downloaded
    kwargs = {}
    if bad == "reuse_in_old": kwargs["reuse_root"] = old / "new-reuse"
    elif bad == "output_in_old": payload["output_root"] = str(old / "new-output")
    elif bad == "payload_in_reuse": kwargs["payload_path"] = tmp_path / "verified-reuse/payload.json"
    elif bad == "output_exists": Path(payload["output_root"]).mkdir()
    elif bad == "payload_exists": (tmp_path / "private-next-payload.json").write_text("existing")
    with pytest.raises(ValueError):
        prepare(downloaded, tmp_path, **kwargs)
    assert not (old / "new-reuse").exists() and not (old / "new-output").exists()


def test_preparation_deadline_cannot_exceed_one_hour(downloaded, tmp_path):
    for seconds in (0, 3601, True):
        with pytest.raises(ValueError, match="one hour"):
            prepare(downloaded, tmp_path, max_seconds=seconds)


def test_expired_deadline_stops_before_writing_resume(downloaded, tmp_path, monkeypatch):
    ticks = iter([0, 2])
    monkeypatch.setattr(resume.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        prepare(downloaded, tmp_path, max_seconds=1)
    assert not (tmp_path / "verified-reuse").exists()

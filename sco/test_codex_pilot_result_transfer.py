"""Offline exact-byte native panel transfer; no network or cloud jobs."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sco"), str(ROOT / "scripts")]
import codex_pilot_result_transfer as transfer
import test_evaluate_codex_pilot as fixtures

URL = "https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/behavior1k/codex-pilot-20261011/results/fixture.tar?Signature=PRIVATE-FIXTURE"


def save(path, value):
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    before, after, candidate, _ = fixtures.panels.__wrapped__(root / "evidence")
    monkeypatch.setattr(transfer, "AFS", str(root))
    code = root / "code";code.mkdir()
    refs = {}
    for key, source in (("evaluation_worker", ROOT / "sco/run_codex_pilot_evaluation.py"),
                        ("report_helper", ROOT / "scripts/evaluate_codex_pilot.py")):
        path = code / source.name;path.write_bytes(source.read_bytes());refs[key] = transfer.file_ref(path)
    receipts = []
    for ref in candidate["run_receipts"]:
        receipt = transfer.load(ref["path"])
        receipt.update(driver_sha256=refs["evaluation_worker"]["sha256"], report_helper=refs["report_helper"],
                       training_receipt=candidate["checkpoint"]["training_receipt"])
        save(ref["path"], receipt);receipts.append(transfer.file_ref(ref["path"]))
    config = dict(run_receipts=receipts, training_receipt=candidate["checkpoint"]["training_receipt"],
                  **refs, report_root=str(root / "result-transfer-fixture"), archive_put_url=URL)
    # The existing historical fixture uses a synthetic source snapshot. Keep
    # real assemble_panel/evidence validation, changing only that fixture ID.
    actual_loader = transfer.load_module
    def loader(ref, name):
        module = actual_loader(ref, name)
        if name == "frozen_native_evaluation":module.SOURCE_SHA256 = "a" * 64
        return module
    monkeypatch.setattr(transfer, "load_module", loader)
    uploaded = []
    monkeypatch.setattr(transfer, "upload_archive", lambda path, url: uploaded.append((Path(path), url)))
    return config, before, candidate, uploaded


def run(inputs):
    report = transfer.transfer(inputs[0])
    assert report["state"] == "complete", report
    return report


def test_full_raw_transfer_then_relocation_preserves_failed_outcomes_and_receipts(inputs, tmp_path):
    config, baseline, original, uploaded = inputs
    report = run(inputs)
    assert len(uploaded) == 1 and report["artifact_count"] == 29
    assert "PRIVATE-FIXTURE" not in json.dumps(report)
    with tarfile.open(report["archive_path"], "r:") as archive:
        assert len(archive.getmembers()) == 29 and all(item.isfile() for item in archive)
        for member in archive.getmembers():
            assert archive.extractfile(member).read() == Path(report["members"][member.name]["path"]).read_bytes()
    before_receipts = {ref["path"]: Path(ref["path"]).read_bytes() for ref in config["run_receipts"] + [config["training_receipt"]]}
    # Downloaded receipts must work even when their remote result paths are gone.
    for row in original["results"]:
        Path(row["json"]["path"]).unlink()
        if row.get("video"):Path(row["video"]["path"]).unlink()
    output = tmp_path / "local"
    compared = transfer.unpack_and_compare(report["archive_path"], report, baseline, output)
    assert compared["comparison_complete"] and len(compared["pairs"]) == 20
    assert (compared["overall"]["finetuned_successes"], compared["overall"]["both_failure"]) == (3, 16)
    assert compared["baseline_verification_levels"]["a001"].endswith("full_shard_audit_pending")
    for name, ref in report["members"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == ref["sha256"]
    for path, raw in before_receipts.items():assert Path(path).read_bytes() == raw
    local = transfer.load(output / "candidate_local_panel.json")
    assert Path(local["checkpoint"]["checkpoint_manifest"]["path"]).is_file()
    assert transfer.load(output / "training/training_receipt.json")["checkpoint_manifest"]["path"] != local["checkpoint"]["checkpoint_manifest"]["path"]


@pytest.mark.parametrize("damage", ["order", "missing", "duplicate", "driver", "teacher", "changed_json", "missing_video", "symlink", "manifest", "foreign_path"])
def test_missing_or_unverified_evidence_never_uploads(inputs, damage, tmp_path):
    config, _, original, uploaded = inputs
    if damage == "order":config["run_receipts"][0:2] = reversed(config["run_receipts"][0:2])
    elif damage == "missing":config["run_receipts"].pop()
    elif damage == "duplicate":config["run_receipts"][-1] = config["run_receipts"][0]
    elif damage in {"driver", "teacher", "foreign_path"}:
        ref = config["run_receipts"][0];receipt = transfer.load(ref["path"])
        if damage == "driver":receipt["driver_sha256"] = "f" * 64
        elif damage == "teacher":receipt["teacher_imported"] = True
        else:
            outside = tmp_path / "outside.json";outside.write_bytes(Path(receipt["results"][0]["json"]["path"]).read_bytes())
            receipt["results"][0]["json"] = transfer.file_ref(outside)
        save(ref["path"], receipt);config["run_receipts"][0] = transfer.file_ref(ref["path"])
    elif damage == "manifest":
        training = transfer.load(config["training_receipt"]["path"])
        Path(training["checkpoint_manifest"]["path"]).write_text("changed")
    else:
        row = original["results"][0];path = Path(row["video" if damage == "missing_video" else "json"]["path"])
        if damage == "missing_video":path.unlink()
        elif damage == "changed_json":path.write_text(path.read_text() + " ")
        else:
            target = path.with_suffix(".original");path.rename(target);path.symlink_to(target)
    if damage in {"missing", "duplicate"}:
        with pytest.raises(ValueError):transfer.transfer(config)
    else:assert transfer.transfer(config)["state"] == "failed"
    assert not uploaded


def test_byte_ceiling_rejects_before_upload(inputs, monkeypatch):
    monkeypatch.setattr(transfer, "MAX_BYTES", 100)
    with pytest.raises(ValueError):transfer.transfer(inputs[0])
    assert not inputs[3]


def test_bounded_deadline_is_not_swallowed_as_success(inputs, monkeypatch):
    def timed_out(*args):raise transfer.TransferDeadline()
    monkeypatch.setattr(transfer, "pack_archive", timed_out)
    report = transfer.transfer(inputs[0])
    assert report["state"] == "failed" and report["error_type"] == "TransferDeadline" and not inputs[3]


def test_upload_failure_report_never_discloses_private_url(inputs, monkeypatch):
    monkeypatch.setattr(transfer, "upload_archive", lambda *args: (_ for _ in ()).throw(RuntimeError(URL)))
    report = transfer.transfer(inputs[0])
    assert report["state"] == "failed" and report["uploaded"] is False
    assert "PRIVATE-FIXTURE" not in json.dumps(report)


def rewrite_tar(path, change):
    with tarfile.open(path, "r:") as stream:
        rows = [(copy.copy(item), stream.extractfile(item).read()) for item in stream]
    header, raw = rows[-1]
    if change == "traversal":header.name = "../escape"
    elif change == "duplicate":header.name = rows[0][0].name
    elif change == "symlink":header.type = tarfile.SYMTYPE;header.linkname = "/outside";header.size = 0;raw = b""
    elif change == "hardlink":header.type = tarfile.LNKTYPE;header.linkname = rows[0][0].name;header.size = 0;raw = b""
    elif change == "bytes":raw = b"X" + raw[1:]
    elif change == "extra":rows.append((tarfile.TarInfo("extra"), b""))
    rows[-1 if change != "extra" else -2] = header, raw
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as stream:
        for item, data in rows:stream.addfile(item, io.BytesIO(data))


@pytest.mark.parametrize("damage", ["traversal", "duplicate", "symlink", "hardlink", "bytes", "extra"])
def test_tar_boundary_validation_rejects_injection(inputs, tmp_path, damage):
    report = run(inputs);path = Path(report["archive_path"]);rewrite_tar(path, damage)
    report["archive"] = transfer.identity(transfer.file_ref(path))
    with pytest.raises(ValueError):transfer.unpack_and_compare(path, report, inputs[1], tmp_path / "unpack")
    assert not (tmp_path / "escape").exists()


def test_archive_hash_and_complete_report_are_required_before_extraction(inputs, tmp_path):
    report = run(inputs)
    bad = copy.deepcopy(report);bad["archive"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):transfer.unpack_and_compare(report["archive_path"], bad, inputs[1], tmp_path / "out")
    bad = copy.deepcopy(report);bad["state"] = "failed"
    with pytest.raises(ValueError):transfer.unpack_and_compare(report["archive_path"], bad, inputs[1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_build_is_offline_private_and_self_contained(inputs, tmp_path):
    source = tmp_path / "worker.py"
    result = transfer.build_transfer_source(source, **inputs[0])
    compile(source.read_text(), "worker", "exec")
    assert result["actual_job_submitted"] is False and result["gpu_count"] == 0 and result["max_seconds"] == 600
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    assert "PRIVATE-FIXTURE" not in json.dumps(result)
    assert "PRIVATE-FIXTURE" in source.read_text()
    assert "def unpack_and_compare" not in source.read_text() and "def build_transfer_source" not in source.read_text()
    assert not inputs[3] and not Path(inputs[0]["report_root"]).exists()


def test_pinned_file_at_experiment_root_does_not_overlap_fresh_output(inputs):
    config = copy.deepcopy(inputs[0])
    config["evaluation_worker"]["path"] = transfer.AFS + "/frozen_evaluation.py"
    transfer.validate_config(config)
    config["evaluation_worker"]["path"] = config["report_root"] + "/frozen_evaluation.py"
    with pytest.raises(ValueError, match="overlaps"):
        transfer.validate_config(config)


@pytest.mark.parametrize("url", [URL.replace("https:", "http:"), URL.replace("/260010165/", "/other/"),
    URL.replace("behavior1k/codex-pilot-20261011", "unrelated"), URL.replace("aoss.cn-sh-01b.sensecoreapi-oss.cn", "example.com"),
    URL.replace("fixture.tar", "../fixture.tar"), URL.split("?")[0]])
def test_foreign_unsigned_or_unsafe_archive_url_rejected(url):
    with pytest.raises(ValueError):transfer.private_url(url)


def test_download_is_size_and_hash_bounded_without_network(inputs, tmp_path, monkeypatch):
    report = run(inputs);raw = Path(report["archive_path"]).read_bytes()
    class Opener:
        def open(self, *args, **kwargs):return io.BytesIO(raw)
    monkeypatch.setattr(transfer, "opener", Opener)
    target = tmp_path / "download.tar"
    assert transfer.download_archive(URL, report, target)["sha256"] == report["archive"]["sha256"]
    assert target.read_bytes() == raw
    raw += b"extra"
    with pytest.raises(RuntimeError):transfer.download_archive(URL, report, tmp_path / "bad.tar")

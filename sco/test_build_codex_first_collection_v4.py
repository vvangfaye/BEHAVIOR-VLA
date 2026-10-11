"""Offline v4 target gates; reuse the v3 worker evidence fixtures."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_codex_first_collection_v4 as builder
import test_build_codex_first_collection_v3 as previous


@pytest.fixture
def worker():
    namespace = {}
    exec(compile(builder.WORKER, "wrapper_v4_test_definitions", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def gate_inputs(tmp_path, worker):
    config = previous.gate_inputs.__wrapped__(tmp_path, worker)
    base = Path(config["pilot_root"])
    config = json.loads(json.dumps(config).replace("code-v5", "code-v6").replace('"v5"', '"v6"')
                        .replace("original-v3", "original-reader-v4").replace("gpu-v3", "gpu-v4"))
    (base / "code-v5").rename(base / "code-v6")
    (base / "original-v3").rename(base / "original-reader-v4")
    (base / "preparation-v13").rename(base / "preparation-v14")
    (base / "cluster_validation_v13_report.json").rename(base / "cluster_validation_v14_report.json")
    for relative in ("code-v6/_CODEX_PILOT_CODE_READY", "preparation-v14/report.json",
                     "original-reader-v4/_ORIGINAL_DATA_READY.json"):
        path = base / relative
        path.write_text(path.read_text().replace("code-v5", "code-v6").replace('"v5"', '"v6"')
                        .replace("original-v3", "original-reader-v4"))
    data_path = base / "original-reader-v4/_ORIGINAL_DATA_READY.json"
    data = json.loads(data_path.read_text())
    data.update(root=str(base / "original-reader-v4"), dataset_root=str(base / "original-v3"))
    previous.dump(data_path, data)
    prep_path = base / "preparation-v14/report.json"
    prep = json.loads(prep_path.read_text()); prep["data_ready_sha256"] = worker.sha(data_path)
    previous.dump(prep_path, prep)
    return config


def test_v4_accepts_new_reader_receipt_with_original_v3_dataset(worker, gate_inputs):
    result = worker.gates(gate_inputs)
    assert result["selected_episodes"] == builder.EPISODES
    assert result["model"] == "gpt-6.1-sol" and result["reasoning_effort"] == "medium"


@pytest.mark.parametrize("target,key,value", [
    ("preparation-v14/report.json", "ready_for_collection", False),
    ("preparation-v14/report.json", "data_root", "/fixture/old-original-v3"),
    ("cluster_validation_v14_report.json", "state", "failed"),
    ("original-reader-v4/_ORIGINAL_DATA_READY.json", "root", "/fixture/other-reader"),
    ("original-reader-v4/_ORIGINAL_DATA_READY.json", "dataset_root", "/fixture/other-dataset"),
    ("original-reader-v4/_ORIGINAL_DATA_READY.json", "raw_success_verified", False),
])
def test_v4_rejects_wrong_preparation_reader_or_dataset(worker, gate_inputs, target, key, value):
    path = Path(gate_inputs["pilot_root"]) / target
    data = json.loads(path.read_text()); data[key] = value; previous.dump(path, data)
    with pytest.raises(ValueError): worker.gates(gate_inputs)


def test_v4_freezes_new_targets_without_submission(tmp_path, monkeypatch):
    # Reuse v3's fully local collection fixture while pinning this test's v6 manifest.
    import run_codex_collection
    names = ["scripts/run_codex_collection.sh", "scripts/collect_b1k_codex.py",
             "scripts/codex_collection_teacher.py", "scripts/codex_teacher_mailbox.py",
             "scripts/codex_collection_actions.py", "scripts/codex_collection_trajectory.py",
             "scripts/codex_collection_demonstrations.py", "sco/run_codex_collection.py",
             "sco/codex_cluster_proxy.py", run_codex_collection.ROBOT_CONFIG]
    collection = dict(code="/fixture/old-code", base_code="/fixture/base", runtime="/fixture/runtime",
        checkpoint="/fixture/checkpoint", run_root="/fixture/collection/radio-311-pilot001",
        task="turning_on_radio", instance_id=311, attempt_id="pilot-001", execution_name="offline-fixture",
        teacher_backend="codex_local", codex_binary="/fixture/codex", codex_home="/fixture/private-home",
        mihomo_binary="/fixture/mihomo", mihomo_config_dir="/fixture/mihomo-config",
        demonstration_manifest="/fixture/old-demo.json", robot_config="/fixture/robot.yaml",
        max_steps=4500, max_decisions=32, max_seconds=7200, wall_timeout_seconds=8400, teacher_interval=256,
        source_files_sha256={name:"d"*64 for name in names})
    collection_path = tmp_path / "collection.json"; previous.dump(collection_path, collection)
    bundle = tmp_path / "bundle.tar.gz"; bundle.write_bytes(b"offline v6 code fixture")
    manifest = dict(version="v6", code=builder.AFS + "/code-v6", bundle=str(bundle),
                    bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                    source_files_sha256=collection["source_files_sha256"])
    manifest_path = tmp_path / "manifest.json"; previous.dump(manifest_path, manifest)
    encryption_path = tmp_path / "encryption.json"
    previous.dump(encryption_path, dict(selected_auth_file="auth_weizhi.json", ciphertext_sha256="c"*64))
    output = tmp_path / "first_collection_job_v4.py"
    receipt = builder.build(output, manifest_path=manifest_path, collection_path=collection_path,
                            encryption_path=encryption_path)
    source = output.read_text(); compile(source, str(output), "exec")
    assert receipt["actual_job_submitted"] is False
    assert receipt["report_root"] == builder.AFS + "/first-radio311-gpu-v4"
    assert receipt["original_data"] == builder.AFS + "/original-reader-v4"
    assert receipt["dataset_root"] == builder.AFS + "/original-v3"
    assert receipt["collection_run_root"] == collection["run_root"]
    assert "preparation-v14/report.json" in source and "cluster_validation_v14_report.json" in source
    assert "/code-v6" in source and "original-reader-v4/demonstrations/28/manifest.json" in source
    assert "BOOTSTRAP_REPORT = run(" in source and "os.execve" not in source


def test_v4_teacher_usage_keeps_actual_evidence_layout(tmp_path, worker):
    previous.test_usage_reads_actual_collector_and_teacher_evidence_layout(tmp_path, worker)


def test_radio_package_relocation_retains_fingerprint_and_uses_copied_images(tmp_path):
    import shutil
    from PIL import Image
    import codex_collection_demonstrations as demos
    from test_codex_collection_demonstrations import frame
    dataset = tmp_path / "original-v3"
    old = dataset / "demonstrations/28"
    metadata = dict(source_kind="original_train", task="turning_on_radio", instance_id=18,
                    episode_index=28, source_dataset=str(dataset))
    made = demos._prepare(old, metadata, [dict(start=0, end=2, subtask="turn on radio")], 2, frame)
    new = tmp_path / "original-reader-v4/demonstrations/28"
    shutil.copytree(old, new)
    before = (old / "manifest.json").read_bytes()
    shutil.rmtree(old)
    context, images = demos.demonstration_context(new / "manifest.json", "turning_on_radio")
    assert context["demonstration_fingerprint"] == made["fingerprint"]
    assert (new / "manifest.json").read_bytes() == before
    assert json.loads(before)["metadata"]["source_dataset"] == str(dataset)
    assert all(image.parent == new.resolve() for image in images)
    for image in images:
        with Image.open(image) as picture: picture.verify()
    images[0].write_bytes(b"damaged copied montage")
    with pytest.raises(ValueError, match="montage changed"):
        demos.demonstration_context(new / "manifest.json", "turning_on_radio")

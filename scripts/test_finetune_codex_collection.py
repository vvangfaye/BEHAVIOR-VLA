"""Finite1688977 training checks; no GPU training or real codec is simulated."""
import json
import importlib.util
import shutil
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_codex_collection as train
from test_finetune_subtask_collection import checkpoint as legacy_checkpoint


@pytest.fixture
def checkpoint(legacy_checkpoint):
    root, identity = legacy_checkpoint
    identity["checkpoint_step"] = 1688977
    (root / "_READY").write_text(json.dumps(dict(step=1688977, manifest_sha256=identity["checkpoint_manifest_sha256"])))
    return root, identity


@pytest.fixture
def prepared(checkpoint, tmp_path, monkeypatch):
    base, identity = checkpoint
    output = tmp_path / "dataset"; output.mkdir()
    (output / "retained_episodes.txt").write_text("0\n1\n")
    report = dict(kind=train.EXPORT_KIND, conditioning="task_and_subtask", fingerprint="fixture-data", frames=7,
                  tasks=["turning_on_radio"], new_frame_fraction=3 / 7)
    verification = dict(complete=True, real_lerobot_reader=True, real_lerobot_writer=True, action_horizon=32)
    monkeypatch.setattr(train, "audit_export", lambda *a, **k: (report, [], verification))
    return train.prepare(output, base, "base", tmp_path / "run", run_id="pilot", steps=128,
                         expected_manifest_sha256=identity["checkpoint_manifest_sha256"])


def test_isolated_defaults_preserve_joint_supervision_and_baseline_norm(prepared, monkeypatch):
    monkeypatch.setenv("RESUME", "1"); monkeypatch.setenv("SUBTASK_LOSS_WEIGHT", "0")
    monkeypatch.setenv("PYTHONPATH", "/old-runtime/code")
    env = train.environment(prepared, "smoke")
    assert prepared["init_checkpoint_identity"]["checkpoint_step"] == 1688977
    assert prepared["evaluation_instances"] == list(range(301, 311))
    assert prepared["evaluation_agent_enabled"] is False
    assert env["NUM_GPUS"] == env["BATCH_SIZE"] == env["FSDP_DEVICES"] == "4"
    assert prepared["memory_plan"]["requires_same_device_count_smoke"] is True
    assert env["TRAIN_STEPS"] == "2" and env["RESUME"] == "0"
    assert env["SUBTASK_ACTION_CONDITIONING"] == "task_and_subtask"
    assert env["SUBTASK_LOSS_WEIGHT"] == "1.0" and env["SUBTASK_PREDICTION"] == "1"
    assert env["INIT_PARAMS"] == str(Path(prepared["checkpoint"]) / "params")
    assert env["PYTHONPATH"].split(":") == [str(train.REPO / "openpi/src"), str(train.REPO / "openpi/packages/openpi-client/src")]
    assert env["CODEX_TRAINING_RUNTIME"] == "1"
    assert prepared["base_norm_sha256"] == train.sha(Path(prepared["checkpoint"]) / prepared["base_norm_path"])


def test_cannot_initialize_old1375000_or_wrong_manifest(checkpoint, tmp_path):
    base, identity = checkpoint
    with pytest.raises(ValueError, match="manifest"):
        train.prepare(tmp_path / "data", base, "base", tmp_path / "run", run_id="pilot", expected_manifest_sha256="wrong")
    (base / "_READY").write_text(json.dumps(dict(step=1375000, manifest_sha256=identity["checkpoint_manifest_sha256"])))
    with pytest.raises(ValueError, match="1688977"):
        train.prepare(tmp_path / "data", base, "base", tmp_path / "run", run_id="pilot")


@pytest.mark.parametrize("damage", ["data", "plan", "already_exists", "unsmoked_full"])
def test_refuses_mutations_and_unsmoked_training_before_gpu_query(prepared, monkeypatch, damage):
    monkeypatch.setattr(train.subprocess, "check_output", lambda *a, **k: pytest.fail("GPU query before integrity gate"))
    phase = "smoke"
    if damage == "data": (Path(prepared["dataset_root"]) / "retained_episodes.txt").write_text("0\n")
    elif damage == "plan": prepared["steps"] += 1
    elif damage == "already_exists": (Path(prepared["run_root"]) / "smoke").mkdir()
    else:
        phase = "full"; path = Path(prepared["run_root"]) / "smoke"; path.mkdir()
        (path / "phase_receipt.json").write_text('{"complete":false}')
    with pytest.raises(ValueError): train.run(prepared, phase)


def test_busy_gpu_is_never_used_or_stopped(prepared, monkeypatch):
    monkeypatch.setattr(train.subprocess, "check_output", lambda args, **k: "123\n" if "--query-compute-apps=pid" in args else "".join(f"{i}, L20, 48000\n" for i in range(4)))
    with pytest.raises(ValueError, match="existing compute"):
        train.run(prepared, "smoke")
    assert not (Path(prepared["run_root"]) / "smoke").exists()


@pytest.mark.parametrize("field", ["complete", "real_lerobot_reader", "real_lerobot_writer", "action_horizon"])
def test_injected_or_incomplete_reader_proof_cannot_prepare_training(tmp_path, monkeypatch, field):
    root = tmp_path / "data"; root.mkdir()
    (root / "export_report.json").write_text(json.dumps(dict(kind=train.EXPORT_KIND, conditioning="task_and_subtask")))
    verification = dict(complete=True, real_lerobot_reader=True, real_lerobot_writer=True, action_horizon=32)
    verification[field] = 16 if field == "action_horizon" else False
    monkeypatch.setattr(train, "verify_mixed", lambda *a: verification)
    with pytest.raises(ValueError, match="actual LeRobot"):
        train.audit_export(root)


@pytest.fixture
def aoss_checkpoint(checkpoint, tmp_path):
    root, identity = checkpoint
    (root / "_READY").write_text(json.dumps(dict(checkpoint_step=1688977)))
    files = [dict(relative_path=name, size=(root / name).stat().st_size, sha256=checksum)
             for name, checksum in train.tree(root).items() if name.startswith(("params/", "assets/")) or name == "_CHECKPOINT_METADATA"]
    manifest = tmp_path / "upload.json"
    manifest.write_text(json.dumps(dict(checkpoint_step=1688977, files=files)))
    identity.update(manifest_format="aoss_upload_plan_v1", checkpoint_manifest_path=str(manifest),
                    checkpoint_manifest_sha256=train.sha(manifest))
    return root, identity, manifest


def test_real_deployed_aoss_checkpoint_format_is_supported(aoss_checkpoint):
    root, identity, manifest = aoss_checkpoint
    entries, norm = train.verify_checkpoint(root, identity, "base")
    assert len(entries) == 5 and norm == "assets/base/norm_stats.json"


@pytest.mark.parametrize("damage", ["weights", "extra_param", "manifest", "ready", "norm"])
def test_aoss_content_identity_and_normalization_cannot_drift(aoss_checkpoint, damage):
    root, identity, manifest = aoss_checkpoint
    if damage == "weights": (root / "params/array").write_bytes(b"changed")
    elif damage == "extra_param": (root / "params/uncovered").write_bytes(b"changed")
    elif damage == "manifest": manifest.write_text("{}")
    elif damage == "ready": (root / "_READY").write_text('{"checkpoint_step":1375000}')
    else: (root / "assets/base/norm_stats.json").write_text("{}")
    with pytest.raises(ValueError): train.verify_checkpoint(root, identity, "base")


@pytest.fixture
def completed_phase(prepared):
    phase = Path(prepared["run_root"]) / "full"
    source = phase / "checkpoints/pi05_b1k/pilot_full/127"
    source.mkdir(parents=True)
    shutil.copytree(Path(prepared["checkpoint"]) / "params", source / "params")
    (source / "params/array").write_bytes(b"fixture updated model, no GPU execution")
    shutil.copytree(Path(prepared["checkpoint"]) / "assets", source / "assets")
    shutil.copytree(source / "assets/base", source / "assets" / prepared["repo_id"])
    shutil.copyfile(Path(prepared["checkpoint"]) / "_CHECKPOINT_METADATA", source / "_CHECKPOINT_METADATA")
    proof = dict(restored_step=128, action_horizon=32, conditioning_mode="joint", global_devices=4, processes=1)
    (phase / "checkpoint_restore_verified.json").write_text(json.dumps(proof))
    (phase / "train.log").write_text("CPU fixture; never actual GPU training\n")
    receipt = dict(complete=True, kind=train.KIND, phase="full",
                   training_plan_fingerprint=prepared["fingerprint"], saved_checkpoint=str(source),
                   checkpoint_restore_evidence=proof, training_log_sha256=train.sha(phase / "train.log"))
    (phase / "phase_receipt.json").write_text(json.dumps(receipt))
    return phase, source


def test_completed_stage_links_new_inference_to1688977_and_exact_norm(prepared, completed_phase, tmp_path):
    destination = tmp_path / "inference"
    result = train.stage_final(prepared, destination)
    ready = train.read(destination / "_READY")
    assert ready["init_checkpoint_step"] == 1688977 and ready["step"] == 127
    assert ready["restored_train_state_step"] == result["optimizer_steps"] == 128
    assert result["inference_staging_complete"] is True
    assert result["kind"] == "codex_pilot_trained_checkpoint_v1"
    assert result["checkpoint_id"] == "pi05_b1k_codex_pilot_128"
    assert result["parent_checkpoint_step"] == 1688977
    assert result["ready_for_evaluation"] is result["real_lerobot_readback_verified"] is result["base_normalization_preserved"] is True
    assert result["checkpoint_manifest"] == train.file_ref(destination / "SHA256SUMS.inference")
    assert result["base_norm_sha256"] == prepared["base_norm_sha256"]
    assert train.sha(destination / "assets" / prepared["repo_id"] / "norm_stats.json") == prepared["base_norm_sha256"]
    assert not (destination / "train_state").exists()
    spec = importlib.util.spec_from_file_location("eval_worker", Path(__file__).resolve().parents[1] / "sco/run_codex_pilot_evaluation.py")
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    assert worker.verify_training(dict(checkpoint=str(destination), checkpoint_id=result["checkpoint_id"],
        repo_id=prepared["repo_id"], training_receipt=train.file_ref(Path(prepared["run_root"]) / "training_receipt.json"))) == result


@pytest.mark.parametrize("fault", ["weights_unchanged", "readback", "smoke", "restore", "normalization"])
def test_stage_never_admits_untrained_or_unverified_checkpoint(prepared, completed_phase, tmp_path, monkeypatch, fault):
    phase, source = completed_phase
    if fault == "weights_unchanged":
        shutil.copyfile(Path(prepared["checkpoint"]) / "params/array", source / "params/array")
    elif fault == "readback":
        report = dict(fingerprint=prepared["dataset_fingerprint"])
        monkeypatch.setattr(train, "audit_export", lambda *a, **kw:
            (report, [], {**prepared["export_verification"], "real_lerobot_reader": False}))
    elif fault == "smoke":
        receipt = train.read(phase / "phase_receipt.json"); receipt["phase"] = "smoke"
        (phase / "phase_receipt.json").write_text(json.dumps(receipt))
    elif fault == "restore":
        proof = train.read(phase / "checkpoint_restore_verified.json"); proof["restored_step"] = 2
        (phase / "checkpoint_restore_verified.json").write_text(json.dumps(proof))
    else:
        (source / "assets" / prepared["repo_id"] / "norm_stats.json").write_text("{}")
    destination = tmp_path / "inference"
    with pytest.raises(ValueError):
        train.stage_final(prepared, destination)
    assert not destination.exists()
    assert not (Path(prepared["run_root"]) / "training_receipt.json").exists()

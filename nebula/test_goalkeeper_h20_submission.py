import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEBULA = ROOT / "nebula"
WORKER_PATH = NEBULA / "train_goalkeeper_h20.py"
SUBMIT = NEBULA / "submit_goalkeeper_h20.sh"


def _load_worker():
    spec = importlib.util.spec_from_file_location("train_goalkeeper_h20", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _feature_root(tmp_path: Path):
    root = tmp_path / "features"
    (root / "records").mkdir(parents=True)
    (root / "records" / "one.npz").write_bytes(b"immutable-feature-record")
    manifest_row = {
        "id": "record-1",
        "group_id": "group-1",
        "source": "training_demo",
        "record": "records/one.npz",
        "bddl": "task.bddl",
        "feature_id": "encoder@revision",
        "action_spec": "robot-action-v1",
    }
    (root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
    (root / "task.bddl").write_text("(define (problem fixture))\n", encoding="utf-8")
    manifest_hash = worker.file_sha256(root / "manifest.jsonl")
    files = {
        "manifest.jsonl": manifest_hash,
        "records/one.npz": worker.file_sha256(root / "records" / "one.npz"),
        "task.bddl": worker.file_sha256(root / "task.bddl"),
    }
    (root / "dataset_contract.json").write_text(
        json.dumps({"format": "goalkeeper-dataset-v1", "manifest": "manifest.jsonl",
                    "manifest_sha256": manifest_hash, "files": files, "row_count": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract_hash = worker.file_sha256(root / "dataset_contract.json")
    (root / "_READY").write_text(
        json.dumps({"format": "goalkeeper-ready-v1", "dataset_contract_sha256": contract_hash}) + "\n",
        encoding="utf-8",
    )
    tree_hash, _, _ = worker.feature_tree_sha256(root)
    return root, tree_hash, manifest_hash, contract_hash


def _submit_env(tmp_path: Path, feature):
    root, tree_hash, manifest_hash, contract_hash = feature
    env = os.environ.copy()
    for name in (
        "QUEUE",
        "WORKER_COUNT",
        "NUM_GPUS",
        "ALGO_NAME",
        "CLUSTER_FILE",
        "GLOBAL_BATCH_SIZE",
        "BATCH_PER_RANK",
        "GPU_CARD_TYPE",
    ):
        env.pop(name, None)
    env.update(
        {
            "DRY_RUN": "1",
            "KEEP_BUNDLE": "1",
            "BUNDLE_PARENT": str(tmp_path / "bundles"),
            "MODE": "preflight",
            "FEATURE_ROOT": str(root),
            "EXPECTED_TREE_SHA256": tree_hash,
            "EXPECTED_MANIFEST_SHA256": manifest_hash,
            "EXPECTED_CONTRACT_SHA256": contract_hash,
            "RUN_ID": "goalkeeper-test-20260930",
            "ATTEMPT_ID": "attempt-001",
            "JOB_NAME": "goalkeeper-h20-test-20260930",
            "OUTPUT_ROOT": str(tmp_path / "fresh-output"),
            "AI_HUB_TOKEN": "DO_NOT_PRINT_TOKEN",
            "OSS_ACCESS_KEY": "DO_NOT_PRINT_OSS_KEY",
        }
    )
    return env


def test_feature_validation_rejects_content_and_ready_mismatches(tmp_path):
    root, tree_hash, manifest_hash, contract_hash = _feature_root(tmp_path)
    result = worker.validate_feature_root(root, tree_hash, manifest_hash, contract_hash)
    assert result["manifest_rows"] == 1
    assert result["tree_sha256"] == tree_hash

    (root / "_READY").write_text(json.dumps({"tree_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="_READY does not bind"):
        worker.validate_feature_root(root, tree_hash, manifest_hash, contract_hash)

    (root / "_READY").write_text("ready\n", encoding="utf-8")
    (root / "records" / "one.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tree_sha256 mismatch"):
        worker.validate_feature_root(root, tree_hash, manifest_hash, contract_hash)


def test_submitter_dry_run_builds_only_allowlisted_bundle_without_secrets(tmp_path):
    feature = _feature_root(tmp_path)
    env = _submit_env(tmp_path, feature)
    result = subprocess.run(
        ["bash", str(SUBMIT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "DRY_RUN=1: submission skipped." in combined
    assert "DO_NOT_PRINT_TOKEN" not in combined
    assert "DO_NOT_PRINT_OSS_KEY" not in combined
    kept_line = next(line for line in combined.splitlines() if line.startswith("Kept submission bundle: "))
    bundle = Path(kept_line.split(": ", 1)[1])
    assert bundle.is_dir()

    expected_nebula = {
        "multihost_cluster.json",
        "submit_goalkeeper_h20.sh",
        "train_goalkeeper_h20.py",
        "train_goalkeeper_h20.sh",
    }
    assert {path.name for path in (bundle / "nebula").iterdir()} == expected_nebula
    assert {path.name for path in (bundle / "goalkeeper").iterdir()} == {
        "pyproject.toml",
        "scripts",
        "src",
    }
    assert not list(bundle.rglob("config.sh"))
    assert not list(bundle.rglob("*.pyc"))
    assert "QUEUE: demand_specific_na175_h20" in combined
    assert "TOPOLOGY: 1 worker x 8 H20 GPUs" in combined
    assert "BATCH: global=64 per-rank=8" in combined


def test_submitter_rejects_fixed_contract_override(tmp_path):
    feature = _feature_root(tmp_path)
    env = _submit_env(tmp_path, feature)
    env["QUEUE"] = "demand_general"
    result = subprocess.run(
        ["bash", str(SUBMIT)], cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 2
    assert "QUEUE is fixed at demand_specific_na175_h20" in result.stderr
    assert not (tmp_path / "fresh-output").exists()


def test_worker_and_submitter_keep_fixed_launch_contract():
    submit_text = SUBMIT.read_text(encoding="utf-8")
    worker_text = (NEBULA / "train_goalkeeper_h20.sh").read_text(encoding="utf-8")
    ddp_text = (ROOT / "goalkeeper" / "scripts" / "train_head_ddp.sh").read_text(encoding="utf-8")
    assert "status --detail" not in submit_text
    assert 'GK_NNODES=1' in worker_text
    assert 'GK_GPUS_PER_NODE=8' in worker_text
    assert 'GK_GLOBAL_BATCH_SIZE=64' in worker_text
    assert 'GK_DATASET_CONTRACT="$LOCAL_FEATURE_ROOT/dataset_contract.json"' in worker_text
    assert 'bash "$GOALKEEPER_DIR/scripts/train_head_ddp.sh"' in worker_text
    assert "torch.distributed.run" in ddp_text

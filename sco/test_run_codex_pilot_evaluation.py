import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_codex_pilot_evaluation as worker


def payload():
    return dict(code="/fixture/code", runtime="/fixture/runtime", checkpoint="/fixture/checkpoint",
                run_root="/fixture/results", checkpoint_id="trained-pilot", repo_id="b1k_codex_pilot",
                execution_name="native-eval-pilot", source_manifest_sha256=worker.SOURCE_SHA256,
                training_receipt={}, report_helper={}, driver_sha256="a"*64)


def helper():
    source = Path(__file__).resolve().parents[1] / "scripts/evaluate_codex_pilot.py"
    return worker.load_verifier(worker.file_ref(source))


def checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    contents = {"params/_METADATA": "orbax metadata", "params/shard": "actual updated weights",
                "assets/subtask_conditioning.json": json.dumps({"action_conditioning": "task_and_subtask"}),
                "assets/b1k_codex_pilot/norm_stats.json": '{"action":{"mean":[1,2]}}'}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = root / "SHA256SUMS.inference"
    manifest.write_text("".join(worker.sha(root / name) + "  " + name + "\n" for name in sorted(contents)))
    worker.write_new(root / "_READY", dict(init_checkpoint_step=1688977, restored_train_state_step=256,
                                         manifest_sha256=worker.sha(manifest)))
    receipt = dict(kind="codex_pilot_trained_checkpoint_v1", checkpoint_id="trained-pilot",
                   parent_checkpoint_step=1688977, optimizer_steps=256, ready_for_evaluation=True,
                   real_lerobot_readback_verified=True, base_normalization_preserved=True,
                   base_norm_sha256=worker.sha(root / "assets/b1k_codex_pilot/norm_stats.json"),
                   checkpoint_manifest=worker.file_ref(manifest))
    path = tmp_path / "training_receipt.json"
    worker.write_new(path, receipt)
    value = payload()
    value.update(checkpoint=str(root), training_receipt=worker.file_ref(path))
    return value, receipt


def test_fixed_panel_has_original_budget_and_order():
    data = worker.validate_payload(payload())
    assert data["shard_ids"] == list(worker.SHARD_IDS)
    contract = helper().runtime_contract(data["source_manifest_sha256"])
    assert contract["resolved_step_budgets"] == {"turning_on_radio": 3225, "putting_shoes_on_rack": 11590}
    assert contract["evaluation_seed"] == 0
    for shard in helper().SHARDS:
        server, evaluator = worker.commands(data, shard, Path("/results"), "identity")
        assert server[:2] == ["/fixture/runtime/envs/b1k-pi05/bin/python", "/fixture/code/openpi/scripts/b1k/serve_b1k.py"]
        assert evaluator[1:3] == ["-m", "omnigibson.eval.eval"]
        assert "--max-steps" not in evaluator and "--external-subtask" not in evaluator
        first = evaluator.index("--instance-indices") + 1
        last = evaluator.index("--expected-server-token")
        assert evaluator[first:last] == [str(i-301) for i in shard["instance_ids"]]
        assert ("--write-video" in evaluator) == (shard["id"] == "radio-pilot")
        assert "--collection-seeding" not in server


@pytest.mark.parametrize("key,value", [
    ("shard_ids", []), ("shard_ids", ["a000", "radio-pilot"]), ("shard_ids", ["a000", "a000"]),
    ("shard_ids", ["other"]), ("code", "relative"), ("source_manifest_sha256", "b"*64),
    ("checkpoint_id", "1688977"), ("repo_id", "../base"), ("execution_name", "../other"),
    ("port", 65536), ("startup_timeout_seconds", 1801), ("shard_timeout_seconds", True),
    ("shard_timeout_seconds", 100), ("max_steps", 1), ("teacher_mailbox_config", "/mailbox"),
    ("run_root", "/fixture/code/output"), ("driver_sha256", "missing"),
])
def test_invalid_or_collection_scope_is_rejected(key, value):
    source = payload()
    source[key] = value
    with pytest.raises(ValueError):
        worker.validate_payload(source)


def test_environment_removes_teacher_queue_and_stale_source_settings():
    source = payload()
    inherited = dict(PATH="/bin", PYTHONPATH="/wrong", PYTHONHOME="/wrong", CODEX_HOME="/account",
                     CODEX_TOKEN="private", TEACHER_CONFIG="/mailbox", PI05_EVAL_QUEUE_DB="/queue",
                     EXTERNAL_SUBTASK="instruction", PROVIDED_SUBTASK="instruction", OPENAI_API_KEY="private",
                     ANTHROPIC_API_KEY="private", SUBTASK_RAW_DATA_ROOT="/raw", GOALKEEPER_CONFIG_DIR="/overlay",
                     MAX_STEPS="100", PYTHONHASHSEED="17", XLA_FLAGS="--unexpected")
    for process in ("policy", "evaluator"):
        env = worker.build_env(source, Path("/cache"), Path("/vk"), Path("/egl"), "/graphics",
                               process=process, inherited=inherited)
        assert not set(inherited).intersection(env) - {"PATH", "PYTHONPATH"}
        assert not any(key.startswith(("CODEX_", "TEACHER_", "PI05_", "SUBTASK_")) for key in env)
        assert env["PYTHONPATH"].startswith("/fixture/code/")
        if process == "policy":
            assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.35"
            assert env["CUDA_VISIBLE_DEVICES"] == "0"
        else:
            assert env["WARP_CACHE_PATH"] == "/cache/warp"
    assert inherited["CODEX_TOKEN"] == "private"


def test_complete_checkpoint_admitted_only_after_real_readback(tmp_path):
    source, receipt = checkpoint(tmp_path)
    assert worker.verify_training(source) == receipt
    receipt["real_lerobot_readback_verified"] = False
    Path(source["training_receipt"]["path"]).write_text(json.dumps(receipt))
    source["training_receipt"] = worker.file_ref(source["training_receipt"]["path"])
    with pytest.raises(ValueError, match="completed training/readback"):
        worker.verify_training(source)


@pytest.mark.parametrize("fault", ["weights", "norm", "extra", "ready", "directory", "traversal"])
def test_checkpoint_manifest_is_not_merely_a_label(tmp_path, fault):
    source, receipt = checkpoint(tmp_path)
    root = Path(source["checkpoint"])
    if fault == "weights":
        (root / "params/shard").write_text("old weights")
    elif fault == "extra":
        (root / "params/unmanifested").write_text("new shard")
    elif fault == "ready":
        (root / "_READY").write_text(json.dumps(dict(init_checkpoint_step=1688977,
            restored_train_state_step=1, manifest_sha256=receipt["checkpoint_manifest"]["sha256"])))
    elif fault == "directory":
        source["checkpoint"] = str(tmp_path / "other")
    elif fault == "norm":
        receipt["base_norm_sha256"] = "f"*64
    else:
        manifest = root / "SHA256SUMS.inference"
        manifest.write_text("f"*64 + "  ../outside\n")
        receipt["checkpoint_manifest"] = worker.file_ref(manifest)
    Path(source["training_receipt"]["path"]).write_text(json.dumps(receipt))
    source["training_receipt"] = worker.file_ref(source["training_receipt"]["path"])
    with pytest.raises(ValueError):
        worker.verify_training(source)


def test_worker_does_not_import_teacher_or_collection_code():
    parsed = ast.parse(Path(worker.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(parsed) if isinstance(node, ast.Import) for alias in node.names]
    assert not any("teacher" in name or "collection" in name for name in imports)


def test_owned_process_is_bounded_and_terminated(tmp_path):
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait(60)"],
                                   stdout=log, stderr=log, env=os.environ.copy(), start_new_session=True)
        worker.stop_process(process)
    assert process.poll() is not None


def test_verifier_requires_frozen_bytes(tmp_path):
    source = tmp_path / "verifier.py"
    source.write_text("SHARDS = []\n")
    ref = worker.file_ref(source)
    source.write_text("raise Exception('must not execute')\n")
    with pytest.raises(ValueError, match="does not match"):
        worker.load_verifier(ref)

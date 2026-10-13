import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from run_codex_subtask_collection import (
    CONDITIONING_SHA256, NORM_RELATIVE, NORM_SHA256, REQUIRED_SOURCES,
    collector_command, environment, policy_command, sha, stop_owned,
    validate_completion, validate_payload, verify_checkpoint_inventory, verify_inputs,
)


def payload():
    paths = ("code", "base_code", "runtime", "checkpoint", "run_root", "plan_path", "raw_data_root",
             "demonstration_manifest", "robot_config", "codex_binary", "codex_home", "mihomo_binary",
             "mihomo_config_dir")
    return {**{key: "/fixture/" + key for key in paths}, "execution_name": "freeze-zero-pilot001",
            "plan_sha256": "a" * 64, "checkpoint_upload_plan": "/fixture/checkpoint-inventory.json",
            "source_files_sha256": {name: "a" * 64 for name in REQUIRED_SOURCES}}


@pytest.mark.parametrize("key,value", [
    ("code", "/fixture/base_code/nested"), ("code", "/fixture/base_code"),
    ("run_root", "/fixture/checkpoint/attempt"), ("run_root", "/fixture"),
    ("plan_path", "relative"), ("port", 65536), ("wall_timeout_seconds", 10801),
    ("checkpoint_upload_plan", "relative"),
    ("teacher_timeout", 181), ("policy_startup_timeout", 0),
    ("execution_name", "../other"), ("teacher_backend", "mailbox"), ("plan_sha256", "wrong"),
])
def test_rejects_wrong_scope_paths_and_unbounded_resources(key, value):
    with pytest.raises(ValueError):
        validate_payload({**payload(), key: value})


def test_freeze_required_overlay_cannot_omit_new_collector_or_traverse():
    source = payload()
    source["source_files_sha256"].pop("scripts/collect_b1k_codex_subtask.py")
    with pytest.raises(ValueError, match="sources"):
        validate_payload(source)
    source = payload()
    source["source_files_sha256"]["../other"] = "b" * 64
    with pytest.raises(ValueError, match="inventory"):
        validate_payload(source)


def test_policy_and_collector_bind_the_same_private_server_and_seed_contract():
    source = validate_payload(payload())
    server = policy_command(source, "freeze_fruit", "ephemeral-token")
    client = collector_command(source, "ephemeral-token")
    assert server[0].endswith("envs/b1k-pi05/bin/python")
    assert client[0].endswith("envs/b1k-eval/bin/python")
    assert "--collection-seeding" in server
    assert server[server.index("--policy.dir") + 1] == source["checkpoint"]
    assert server[server.index("--server-token") + 1] == client[client.index("--policy-server-token") + 1]
    assert server[server.index("--port") + 1] == client[client.index("--policy-port") + 1]
    assert client[client.index("--plan") + 1] == source["plan_path"]
    assert client[client.index("--output-dir") + 1] == "/fixture/run_root/attempt"


def test_new_attempt_does_not_inherit_other_collection_or_eval_overrides():
    source = validate_payload(payload())
    inherited = {"PI05_EVAL_QUEUE_DB": "/other/queue", "SUBTASK_LABEL": "wrong", "PROVIDED_SUBTASK": "wrong",
                 "B1K_LEGACY_HISTORY_RECOVERY": "wrong", "CODEX_HOME": "/other/account",
                 "CODEX_API_KEY": "never", "OPENAI_API_KEY": "never", "PYTHONPATH": "/other/source",
                 "PYTHONHOME": "/other/python", "PATH": "/bin", "LD_LIBRARY_PATH": "/base/lib"}
    actual = environment(source, Path("/var/tmp/behavior-1k-runtime/test"), Path("/gpu/vk"),
                         Path("/gpu/egl"), "/graphics", inherited=inherited)
    assert not {"PI05_EVAL_QUEUE_DB", "SUBTASK_LABEL", "PROVIDED_SUBTASK", "B1K_LEGACY_HISTORY_RECOVERY",
                "CODEX_API_KEY", "OPENAI_API_KEY", "PYTHONPATH", "PYTHONHOME"} & actual.keys()
    assert actual["CODEX_HOME"] == source["codex_home"]
    assert inherited["CODEX_HOME"] == "/other/account"


def test_completion_accepts_actual_subtask_success_even_without_whole_task_success():
    plan = {"fingerprint": "plan", "budgets": {"max_steps": 2048}}
    result = {"official_benchmark": False, "success": True, "metric_eligible": True,
              "task_success": False, "steps": 20}
    completed = {"official_benchmark": False, "training_eligible": True, "plan_fingerprint": "plan", "result": result}
    assert validate_completion(completed, plan) == result
    completed["result"] = {**result, "success": False}
    with pytest.raises(ValueError, match="eligibility"):
        validate_completion(completed, plan)
    completed["training_eligible"] = False
    assert validate_completion(completed, plan)["success"] is False


@pytest.mark.parametrize("patch", [
    {"plan_fingerprint": "other"}, {"official_benchmark": True},
    {"result": {"official_benchmark": False, "success": True, "metric_eligible": False, "steps": 20}},
    {"result": {"official_benchmark": False, "success": True, "metric_eligible": True, "steps": 0}},
    {"result": {"official_benchmark": False, "success": True, "metric_eligible": True, "steps": 2049}},
])
def test_completion_rejects_unrelated_or_non_trainable_claims(patch):
    completed = {"official_benchmark": False, "training_eligible": True, "plan_fingerprint": "plan",
                 "result": {"official_benchmark": False, "success": True, "metric_eligible": True, "steps": 20}}
    with pytest.raises(ValueError):
        validate_completion({**completed, **patch}, {"fingerprint": "plan", "budgets": {"max_steps": 2048}})


def test_stop_owned_terminates_new_process_group_without_touching_another_process(tmp_path):
    owned = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    try:
        stop_owned(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        stop_owned(owned)
        stop_owned(unrelated)


def test_verify_inputs_rejects_changed_plan_before_importing_or_opening_model(tmp_path):
    source = payload()
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    source["plan_path"] = str(plan)
    with pytest.raises(ValueError, match="plan changed"):
        verify_inputs(source)


def checkpoint_fixture(tmp_path):
    root = tmp_path / "checkpoint"
    names = ["params/_METADATA", "params/ocdbt.process_0/data/parameter",
             "_CHECKPOINT_METADATA", "assets/subtask_conditioning.json", NORM_RELATIVE]
    rows = []
    for index, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index + 1]) * 8)
        rows.append({"relative_path": name, "size": path.stat().st_size, "sha256": sha(path)})
    inventory = {"checkpoint_step": 1688977, "bucket": "bucket", "prefix": "verified/",
                 "files": rows, "file_count": len(rows), "total_bytes": sum(row["size"] for row in rows)}
    return root, inventory, {"checkpoint_source": "oss://bucket/verified/"}


def test_inventory_verifies_all_actual_inference_parameter_bytes(tmp_path):
    root, inventory, identity = checkpoint_fixture(tmp_path)
    verify_checkpoint_inventory(root, inventory, identity)
    parameter = root / "params/ocdbt.process_0/data/parameter"
    parameter.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="bytes"):
        verify_checkpoint_inventory(root, inventory, identity)


@pytest.mark.parametrize("change", ["duplicate", "missing_parameter", "symlink", "source", "size"])
def test_inventory_rejects_incomplete_or_unsafe_checkpoint_proof(tmp_path, change):
    root, inventory, identity = checkpoint_fixture(tmp_path)
    if change == "duplicate":
        inventory["files"].append(inventory["files"][0])
    elif change == "missing_parameter":
        inventory["files"].pop(1)
        inventory["file_count"] -= 1
        inventory["total_bytes"] -= 8
    elif change == "symlink":
        parameter = root / "params/ocdbt.process_0/data/parameter"
        other = tmp_path / "other"
        parameter.rename(other)
        parameter.symlink_to(other)
    elif change == "source":
        identity["checkpoint_source"] = "oss://bucket/another/"
    elif change == "size":
        inventory["files"][0]["size"] += 1
    with pytest.raises(ValueError):
        verify_checkpoint_inventory(root, inventory, identity)


@pytest.mark.parametrize("instance,step,raw_root", [(301, 1688977, "/fixture/raw_data_root"),
    (33, 1375000, "/fixture/raw_data_root"), (33, 1688977, "/wrong/raw")])
def test_verify_inputs_does_not_reuse_eval_instance_or_old_model_plan(tmp_path, monkeypatch, instance, step, raw_root):
    plan = {"sample": {"task": "freeze_fruit", "subtask": "place on tupperware to countertop",
                       "task_instance_id": instance}, "raw_data_root": raw_root,
            "checkpoint_identity": {"checkpoint_step": step}}
    module = types.ModuleType("codex_subtask_collection")
    module.load_plan = lambda _: plan
    module.SELECTED_SUBTASKS = {("freeze_fruit", "place on tupperware to countertop"),
                               ("turning_on_radio", "place on radio to coffee table koagbh")}
    monkeypatch.setitem(sys.modules, "codex_subtask_collection", module)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    source = {**payload(), "plan_path": str(path), "plan_sha256": sha(path)}
    with pytest.raises(ValueError):
        verify_inputs(source)


@pytest.mark.parametrize('value', [-1, 3, True, 1.0])
def test_capacity_retry_budget_cannot_escape_bounds(value):
    with pytest.raises(ValueError):
        validate_payload({**payload(), 'teacher_capacity_retries': value})


def test_capacity_retry_budget_is_forwarded_without_changing_policy():
    old = validate_payload(payload())
    new = validate_payload({**payload(), 'teacher_capacity_retries': 2})
    argv = collector_command(new, 'ephemeral-token')
    assert argv[argv.index('--codex-capacity-retries') + 1] == '2'
    assert policy_command(old, 'freeze_fruit', 'ephemeral-token') == policy_command(new, 'freeze_fruit', 'ephemeral-token')


def test_recovery_cannot_reuse_qualification_without_persisting_state():
    base={**payload(),'reuse_qualification_root':'/previous/trial'}
    with pytest.raises(ValueError):validate_payload(base)
    new=validate_payload({**base,'persist_preparation':True,'teacher_capacity_retries':2})
    command=collector_command(new,'token')
    assert '--persist-preparation' in command
    assert command[command.index('--reuse-qualification-root')+1]=='/previous/trial'
    with pytest.raises(ValueError):validate_payload({**base,'persist_preparation':True,'reuse_qualification_root':'/fixture/run_root'})

import os
from pathlib import Path
import subprocess
import sys

import pytest

from run_codex_collection import ROBOT_CONFIG, bounded_process, environment, robot_contract, validate_payload


def payload():
    keys = ("code", "base_code", "runtime", "checkpoint", "run_root", "teacher_mailbox_config",
            "demonstration_manifest", "robot_config")
    files = ("run_codex_collection.sh", "collect_b1k_codex.py", "codex_collection_teacher.py",
             "codex_teacher_mailbox.py", "codex_collection_actions.py", "codex_collection_trajectory.py",
             "codex_collection_demonstrations.py")
    inventory = {"scripts/" + name: "a" * 64 for name in files}
    inventory.update({ROBOT_CONFIG: "a" * 64, "sco/run_codex_collection.py": "a" * 64})
    return dict(**{key: "/fixture/" + key for key in keys}, task="turning_on_radio",
                instance_id=311, attempt_id="pilot-001", execution_name="teacher-pilot-001",
                source_files_sha256=inventory)


def test_real_robot_yaml_preserves_the_required_controller_contract():
    yaml = pytest.importorskip("yaml", reason="Actual YAML is checked again in the simulator runtime CPU preflight")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / ROBOT_CONFIG).read_text())
    assert robot_contract(config)["arm_and_trunk_mode"] == "absolute_joint_position"
    config["controller_config"]["trunk"]["use_delta_commands"] = True
    with pytest.raises(ValueError, match="trunk"): robot_contract(config)


def test_default_finite_pilot_contract():
    source = payload()
    actual = validate_payload(source)
    assert actual["max_steps"] == 4500
    assert actual["max_decisions"] == 32
    assert actual["max_seconds"] == 7200
    assert actual["teacher_interval"] == 256
    assert actual["wall_timeout_seconds"] == 8400
    assert "max_steps" not in source
    assert actual["collection_mode"] == "teacher_conditioned"


@pytest.mark.parametrize("mode", ["teacher_conditioned", "native_with_corrections"])
def test_explicit_collection_mode_reaches_isolated_worker_environment(mode):
    source = validate_payload({**payload(), "collection_mode": mode})
    env = environment(source, Path("/var/tmp/behavior-1k-runtime/pilot"), Path("/gpu/vk"),
                      Path("/gpu/egl"), "/graphics/lib", inherited={"CODEX_COLLECTION_MODE": "wrong"})
    assert env["CODEX_COLLECTION_MODE"] == mode


@pytest.mark.parametrize("mode", [None, "native", "", False, []])
def test_unknown_collection_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        validate_payload({**payload(), "collection_mode": mode})


def test_shell_rejects_unknown_mode_before_launching_anything():
    shell = Path(__file__).resolve().parents[1] / "scripts/run_codex_collection.sh"
    env = {name: "/no-such-fixture" for name in (
        "PI05_ENV", "EVAL_ENV", "PI05_CKPT", "CODEX_COLLECTION_TASK", "CODEX_COLLECTION_INSTANCE",
        "CODEX_COLLECTION_ATTEMPT", "CODEX_COLLECTION_OUTPUT", "CODEX_COLLECTION_LOG_ROOT",
        "CODEX_DEMONSTRATION_MANIFEST", "CODEX_ROBOT_CONFIG", "SCO_EPHEMERAL_RUNTIME_ROOT", "OMNIGIBSON_DATA_PATH")}
    result = subprocess.run(["bash", str(shell)], env={**env, "PATH": "/usr/bin:/bin", "CODEX_COLLECTION_MODE": "wrong"},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "Unsupported collection mode" in result.stderr
    subprocess.run(["bash", "-n", str(shell)], check=True, timeout=5)


@pytest.mark.parametrize("key,value", [
    ("instance_id", 301), ("instance_id", 310), ("instance_id", 321), ("instance_id", True),
    ("task", "a; touch /tmp/unwanted"), ("attempt_id", "../existing"), ("execution_name", "../../shared"),
    ("code", "relative"), ("max_steps", 0), ("max_decisions", -1), ("max_seconds", True),
    ("port", 65536), ("wall_timeout_seconds", 7200), ("source_files_sha256", {}),
])
def test_invalid_scope_or_unbounded_inputs_are_rejected(key, value):
    source = payload()
    source[key] = value
    with pytest.raises(ValueError): validate_payload(source)


def test_code_tree_cannot_be_the_original():
    source = payload()
    source["code"] = source["base_code"]
    with pytest.raises(ValueError, match="independent"): validate_payload(source)


def test_output_cannot_write_into_existing_runtime():
    source = payload()
    source["run_root"] = source["runtime"] + "/pilot"
    with pytest.raises(ValueError, match="separate"): validate_payload(source)


def test_overlay_inventory_rejects_traversal():
    source = payload()
    source["source_files_sha256"]["../outside.py"] = "b" * 64
    with pytest.raises(ValueError): validate_payload(source)


def test_worker_environment_isolates_collection_from_live_evaluation():
    source = validate_payload(payload())
    inherited = dict(PATH="/bin", PYTHONPATH="/wrong/source", PYTHONHOME="/wrong/python",
                     PI05_EVAL_QUEUE_DB="/live/queue", PI05_CKPT="/old/model",
                     GOALKEEPER_CONFIG_DIR="/unrelated", EXTERNAL_SUBTASK="old instruction",
                     SUBTASK_RAW_DATA_ROOT="/old/raw", OPENAI_API_KEY="never-forward",
                     CODEX_API_KEY="never-forward", CODEX_HOME="/local/account", LD_LIBRARY_PATH="/base/lib")
    actual = environment(source, Path("/var/tmp/behavior-1k-runtime/pilot"), Path("/gpu/vk"),
                         Path("/gpu/egl"), "/graphics/lib", inherited=inherited)
    assert inherited["PI05_EVAL_QUEUE_DB"] == "/live/queue"
    assert not {"PI05_EVAL_QUEUE_DB", "PYTHONHOME", "PYTHONPATH", "GOALKEEPER_CONFIG_DIR", "EXTERNAL_SUBTASK",
                "SUBTASK_RAW_DATA_ROOT", "OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_HOME"} & actual.keys()
    assert actual["CODEX_COLLECTION_INSTANCE"] == "311"
    assert actual["CODEX_COLLECTION_OUTPUT"] == "/fixture/run_root/attempt"
    assert actual["CODEX_MAX_DECISIONS"] == "32"
    assert actual["CODEX_TEACHER_INTERVAL"] == "256"
    assert actual["PI05_CKPT"] == source["checkpoint"]
    assert actual["LD_LIBRARY_PATH"].startswith("/graphics/lib:")


def test_timeout_terminates_only_new_worker_process(tmp_path):
    with (tmp_path / "log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            bounded_process([sys.executable, "-c", "import time; time.sleep(60)"],
                            env=os.environ.copy(), cwd=tmp_path, log=log, timeout=.1)


def test_direct_teacher_requires_explicit_cluster_runtime_and_sanitizes_inherited_auth():
    source = payload()
    source.pop("teacher_mailbox_config")
    source.update(teacher_backend="codex_local", codex_binary="/usr/local/bin/codex",
                  codex_home="/private/pilot-account", mihomo_binary="/usr/local/bin/mihomo",
                  mihomo_config_dir="/existing/mihomo")
    with pytest.raises(ValueError, match="inventory"):
        validate_payload(source)
    source["source_files_sha256"]["sco/codex_cluster_proxy.py"] = "b" * 64
    validated = validate_payload(source)
    env = environment(validated, Path("/var/tmp/behavior-1k-runtime/pilot"), Path("/gpu/vk"),
                      Path("/gpu/egl"), "/graphics", inherited={"CODEX_HOME": "/wrong", "CODEX_API_KEY": "never"})
    assert env["CODEX_HOME"] == "/private/pilot-account"
    assert env["CODEX_TEACHER_BACKEND"] == "codex_local"
    assert env["CODEX_TEACHER_TIMEOUT"] == "120"
    assert env["CODEX_CHECKPOINT_ID"] == "pi05_b1k_full100_joint_1688977"
    assert not {"CODEX_API_KEY", "CODEX_TEACHER_MAILBOX_CONFIG"} & env.keys()


def test_worker_process_preserves_exit_status(tmp_path):
    with (tmp_path / "log").open("w") as log:
        status = bounded_process([sys.executable, "-c", "raise SystemExit(7)"],
                                 env=os.environ.copy(), cwd=tmp_path, log=log, timeout=5)
    assert status == 7

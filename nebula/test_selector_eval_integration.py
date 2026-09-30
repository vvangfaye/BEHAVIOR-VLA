import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from nebula.summarize_pi05_eval import load_worker_metrics, load_worker_statuses

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = REPO_ROOT / "nebula" / "summarize_pi05_eval.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_b1k.sh"
SUBMIT_SCRIPT = REPO_ROOT / "nebula" / "submit_pi05_eval.sh"
WORKER_SCRIPT = REPO_ROOT / "nebula" / "eval_pi05_worker.sh"
SELECTOR = "state_machine_v1"
SELECTOR_LABELS_SHA256 = (
    "fd785101426544b24ce756a84e72e2f5971776cfe6ebf7bde18e5ff7fb58ac04"
)
LEGACY_SEGMENTS_SHA256 = (
    "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d"
)
TRAINING_ANNOTATION_SHA256 = (
    "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d"
)
TASK = "carrying_in_groceries"


def selector_result() -> dict:
    return {
        "task": TASK,
        "instance_id": 301,
        "rollout_id": 0,
        "steps": 2,
        "success": False,
        "q_score": {"final": 0.25},
        "conditioning": {
            "selector": SELECTOR,
            "label_format": "legacy",
            "selector_labels_sha256": SELECTOR_LABELS_SHA256,
            "legacy_segments_sha256": LEGACY_SEGMENTS_SHA256,
            "training_annotation_sha256": TRAINING_ANNOTATION_SHA256,
            "source": "provided",
            "subtask_source": "provided",
            "oracle": True,
            "privileged": True,
            "official_benchmark": False,
        },
        "selector_events": [
            {
                "step": 0,
                "current_phase": "move_to_sack_in_car",
                "current_text": "move to paper bag",
            },
            {
                "step": 1,
                "current_phase": "pick_sack_from_car",
                "current_text": "pick up from paper bag to car ssxsje",
            },
        ],
        "selector_summary": {
            "valid": True,
            "mode": SELECTOR,
            "task_name": TASK,
            "recoveries": 2,
            "stalls": [{"phase": "pick_sack_from_car"}],
            "phase_completion_counts": {"move_to_sack_in_car": 1},
            "validated_action_requests": 2,
        },
        "subtask_events": [
            {"step": 0, "subtask": "move to paper bag", "source": "provided"},
            {
                "step": 1,
                "subtask": "pick up from paper bag to car ssxsje",
                "source": "provided",
            },
        ],
    }


def generated_result() -> dict:
    result = selector_result()
    result["conditioning"] = {
        "mode": "whole_task",
        "oracle": False,
        "privileged": False,
        "comparable_to_whole_task": True,
    }
    result.pop("selector_events")
    result.pop("selector_summary")
    result["subtask_events"] = [
        {"step": 0, "subtask": "move to paper bag", "source": "generated"},
        {"step": 1, "subtask": "move to paper bag", "source": "cached"},
    ]
    return result


def run_summarizer(
    tmp_path: Path,
    result: dict,
    *,
    require_selector: bool,
    require_generated: bool = False,
) -> tuple[subprocess.CompletedProcess, dict]:
    results_root = tmp_path / "results"
    result_dir = results_root / "pi05" / TASK / "json"
    result_dir.mkdir(parents=True)
    (result_dir / f"{TASK}_301_0.json").write_text(json.dumps(result))
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        str(SUMMARIZER),
        "--results-root",
        str(results_root),
        "--output",
        str(output),
        "--run-id",
        "test-run",
        "--profile",
        "two_tasks",
        "--checkpoint",
        "/checkpoint",
        "--expected-rollouts",
        "1",
        "--expected-task",
        TASK,
        "--expected-instance-ids",
        "301",
        "--expected-rollout-ids",
        "0",
        "--require-complete",
    ]
    if require_selector:
        command.append("--require-selector-events")
    if require_generated:
        command.extend(("--require-subtask-events", "--require-generated-subtasks"))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed, json.loads(output.read_text())


def test_selector_diagnostics_are_validated_and_aggregated(tmp_path):
    completed, summary = run_summarizer(
        tmp_path, selector_result(), require_selector=True
    )

    assert completed.returncode == 0, completed.stderr
    assert summary["selector_diagnostics"] == {
        "selector": SELECTOR,
        "valid_rollouts": 1,
        "invalid_rollout_keys": [],
        "invalid_rollouts": [],
        "total_recoveries": 2,
        "total_stalls": 1,
        "per_phase": {
            "move_to_sack_in_car": {"entered": 1, "completed": 1, "stalled": 0},
            "pick_sack_from_car": {"entered": 1, "completed": 0, "stalled": 1},
        },
    }


def test_default_summary_schema_omits_selector_diagnostics(tmp_path):
    result = selector_result()
    for key in (
        "conditioning",
        "selector_events",
        "selector_summary",
        "subtask_events",
    ):
        result.pop(key)

    completed, summary = run_summarizer(tmp_path, result, require_selector=False)

    assert completed.returncode == 0, completed.stderr
    assert "selector_diagnostics" not in summary


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: result["conditioning"].update(selector="other"),
        lambda result: result["conditioning"].update(oracle=1),
        lambda result: result.update(selector_events=[]),
        lambda result: result["selector_summary"].update(valid=False),
        lambda result: result["selector_summary"].update(validated_action_requests=1),
        lambda result: result["subtask_events"][0].update(source="generated"),
    ],
)
def test_selector_diagnostic_violation_fails_completeness(tmp_path, mutate):
    result = copy.deepcopy(selector_result())
    mutate(result)

    completed, summary = run_summarizer(tmp_path, result, require_selector=True)

    assert completed.returncode == 2
    assert summary["selector_diagnostics"]["invalid_rollout_keys"] == [[TASK, 301, 0]]
    assert summary["selector_diagnostics"]["valid_rollouts"] == 0


@pytest.mark.parametrize(
    "events",
    [
        [{"step": 0, "subtask": "move to paper bag", "source": "generated"}],
        [
            {"step": 0, "subtask": "move to paper bag", "source": "generated"},
            {"step": 3, "subtask": "move to paper bag", "source": "cached"},
            {"step": 8, "subtask": "move to paper bag", "source": "cached"},
            {"step": 13, "subtask": "pick up paper bag", "source": "generated"},
            {"step": 21, "subtask": "close car", "source": "generated"},
            {"step": 34, "subtask": "close car", "source": "cached"},
        ],
    ],
)
def test_generated_subtasks_are_validated(tmp_path, events):
    result = generated_result()
    result["subtask_events"] = events

    completed, summary = run_summarizer(
        tmp_path,
        result,
        require_selector=False,
        require_generated=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert summary["invalid_generated_subtasks"] == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda result: result["conditioning"].update(oracle=True),
        lambda result: result["conditioning"].update(privileged=True),
        lambda result: result["conditioning"].update(selector=SELECTOR),
        lambda result: result["subtask_events"][0].update(source="cached"),
        lambda result: result["subtask_events"][1].update(subtask="different skill"),
        lambda result: result["subtask_events"][1].update(source="provided"),
        lambda result: result["subtask_events"][1].update(source="unknown"),
        lambda result: result["subtask_events"][1].pop("source"),
        lambda result: result["subtask_events"][0].update(subtask=None),
        lambda result: result["subtask_events"][0].update(subtask=""),
        lambda result: result["subtask_events"][1].update(subtask="  "),
    ],
)
def test_generated_subtask_violation_fails_completeness(tmp_path, mutate):
    result = generated_result()
    mutate(result)

    completed, summary = run_summarizer(
        tmp_path,
        result,
        require_selector=False,
        require_generated=True,
    )

    assert completed.returncode == 2
    assert len(summary["invalid_generated_subtasks"]) == 1


def run_shell(
    script: Path, *args: str, **overrides: str
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for name in (
        "EXTERNAL_SUBTASK",
        "BASE_PORT",
        "CLUSTER_FILE",
        "EVAL_WORKERS",
        "EXTERNAL_SUBTASK_SELECTOR",
        "PI05_CKPT",
        "PI05_MODEL_FORMAT",
        "SUBTASK_INTERVAL",
    ):
        env.pop(name, None)
    env.update(overrides)
    with tempfile.TemporaryDirectory() as temporary:
        if script == SUBMIT_SCRIPT:
            # Validation tests must not depend on a developer's private Nebula credentials.
            nebula_dir = Path(temporary) / "nebula"
            nebula_dir.mkdir()
            copied_script = nebula_dir / script.name
            shutil.copy(script, copied_script)
            (nebula_dir / "config.sh").write_text("")
            script = copied_script
        return subprocess.run(
            ["bash", str(script), *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.mark.parametrize(
    ("overrides", "task", "message"),
    [
        (
            {
                "EXTERNAL_SUBTASK_SELECTOR": SELECTOR,
                "EXTERNAL_SUBTASK": "fixed",
                "SUBTASK_PREDICTION": "1",
            },
            TASK,
            "mutually exclusive",
        ),
        (
            {"EXTERNAL_SUBTASK_SELECTOR": SELECTOR, "SUBTASK_PREDICTION": "0"},
            TASK,
            "requires a subtask conditioning mode",
        ),
        (
            {
                "EXTERNAL_SUBTASK_SELECTOR": SELECTOR,
                "SUBTASK_PREDICTION": "1",
                "PROMPT_MODE": "oracle_unsatisfied_goal",
                "PI05_CKPT": "/missing",
            },
            TASK,
            "requires PROMPT_MODE=whole_task",
        ),
        (
            {
                "EXTERNAL_SUBTASK_SELECTOR": SELECTOR,
                "SUBTASK_PREDICTION": "1",
                "PI05_CKPT": "/missing",
            },
            "turning_on_radio",
            "only supports carrying_in_groceries and make_rose_centerpieces",
        ),
    ],
)
def test_eval_script_rejects_invalid_selector_configuration(overrides, task, message):
    completed = run_shell(EVAL_SCRIPT, task, **overrides)

    assert completed.returncode == 2
    assert message in completed.stderr


def test_submit_script_rejects_selector_outside_supported_profiles():
    completed = run_shell(
        SUBMIT_SCRIPT,
        EVAL_PROFILE="reduced",
        EXTERNAL_SUBTASK_SELECTOR=SELECTOR,
        SUBTASK_PREDICTION="1",
    )

    assert completed.returncode == 2
    assert "supported only for EVAL_PROFILE=smoke or two_tasks" in completed.stderr


def test_submit_script_rejects_unknown_model_format():
    completed = run_shell(SUBMIT_SCRIPT, PI05_MODEL_FORMAT="unknown")

    assert completed.returncode == 2
    assert "PI05_MODEL_FORMAT must be jax or torch" in completed.stderr


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_submit_script_rejects_invalid_subtask_interval(value):
    completed = run_shell(
        SUBMIT_SCRIPT,
        SUBTASK_PREDICTION="1",
        SUBTASK_INTERVAL=value,
    )

    assert completed.returncode == 2
    assert "SUBTASK_INTERVAL must be a positive integer" in completed.stderr


def test_submit_script_rejects_subtask_interval_without_prediction():
    completed = run_shell(
        SUBMIT_SCRIPT,
        SUBTASK_PREDICTION="0",
        SUBTASK_INTERVAL="8",
    )

    assert completed.returncode == 2
    assert "SUBTASK_INTERVAL requires SUBTASK_PREDICTION=1" in completed.stderr


def test_submit_script_requires_explicit_torch_checkpoint():
    completed = run_shell(SUBMIT_SCRIPT, PI05_MODEL_FORMAT="torch")

    assert completed.returncode == 2
    assert "CHECKPOINT_SOURCE is required" in completed.stderr


def test_nebula_propagates_and_records_subtask_interval():
    submit_source = SUBMIT_SCRIPT.read_text()
    worker_source = WORKER_SCRIPT.read_text()

    assert 'write_setting SUBTASK_INTERVAL "$SUBTASK_INTERVAL"' in submit_source
    assert "printf 'SUBTASK_INTERVAL: %s\\n'" in submit_source
    assert (
        '"subtask_interval": int(subtask_interval) if subtask_interval else None'
        in submit_source
    )
    assert (
        '"subtask_interval_source": "explicit" if subtask_interval else "policy_default"'
        in submit_source
    )
    assert worker_source.count('SUBTASK_INTERVAL="$SUBTASK_INTERVAL"') == 2
    assert "int(requested_subtask_interval)" in worker_source
    assert 'action_conditioning == "subtask_only"' in worker_source
    assert '"subtask_interval": subtask_interval' in worker_source
    assert '"subtask_interval_source": subtask_interval_source' in worker_source
    assert (
        'grep -Fq "subtask refresh every $SUBTASK_INTERVAL requests"' in worker_source
    )


def test_submit_script_supports_no_dependency_bootstrap():
    source = SUBMIT_SCRIPT.read_text()

    assert 'BOOTSTRAP_ENV_ARGS=("--env=PIP_NO_DEPS=1")' in source
    assert '"nebula_pip_no_deps": nebula_pip_no_deps == "1"' in source


def test_two_tasks_supports_parameterized_eight_gpu_topology():
    submit_source = SUBMIT_SCRIPT.read_text()
    worker_source = WORKER_SCRIPT.read_text()

    assert 'EVAL_WORKERS="${EVAL_WORKERS:-4}"' in submit_source
    assert 'if [[ "$EVAL_WORKERS" == 8 ]]' in submit_source
    assert (
        'CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster.json}"'
        in submit_source
    )
    assert "two_tasks profile requires EVAL_WORKERS=4 or 8" in submit_source
    assert '"carrying_in_groceries|0 4 8 12 16"' in worker_source
    assert '"make_rose_centerpieces|0 4 8 12 16"' in worker_source
    assert (
        '"expected_rollouts": len(worker_tasks) * len(shard) * $NUM_ROLLOUTS'
        in worker_source
    )


def test_submit_script_rejects_unsupported_worker_count_for_two_tasks():
    completed = run_shell(
        SUBMIT_SCRIPT,
        EVAL_PROFILE="two_tasks",
        EVAL_WORKERS="6",
    )

    assert completed.returncode == 2
    assert "two_tasks profile requires EVAL_WORKERS=4 or 8" in completed.stderr


def test_full4_supports_parameterized_eight_gpu_topology():
    submit_source = SUBMIT_SCRIPT.read_text()
    worker_source = WORKER_SCRIPT.read_text()

    assert 'if [[ "$EVAL_PROFILE" == full4 && "$EVAL_WORKERS" == 8 ]]' in submit_source
    assert "full4 profile requires EVAL_WORKERS=4 or 8" in submit_source
    assert (
        '[[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks || "$EVAL_PROFILE" == full4 ]]'
        in worker_source
    )
    assert (
        "task_shards = [all_tasks[worker::num_workers] for worker in range(num_workers)]"
        in worker_source
    )


def test_submit_script_rejects_unsupported_worker_count_for_full4():
    completed = run_shell(
        SUBMIT_SCRIPT,
        EVAL_PROFILE="full4",
        CONFIRM_FULL="1",
        EVAL_WORKERS="6",
    )

    assert completed.returncode == 2
    assert "full4 profile requires EVAL_WORKERS=4 or 8" in completed.stderr


def _write_two_task_worker_statuses(status_root: Path, worker_count: int) -> None:
    status_root.mkdir()
    tasks = ("carrying_in_groceries", "make_rose_centerpieces")
    for worker in range(worker_count):
        if worker_count == 4:
            worker_tasks = list(tasks)
            indices = list(range(worker, 20, 4))
            expected_rollouts = 30
        else:
            worker_tasks = [tasks[worker // 4]]
            indices = list(range(worker % 4, 20, 4))
            expected_rollouts = 15
        value = {
            "worker": worker,
            "tasks": worker_tasks,
            "instance_indices": indices,
            "gpu": worker,
            "port": 8200 + worker,
            "expected_rollouts": expected_rollouts,
            "completed_rollouts": expected_rollouts,
            "completed_videos": expected_rollouts,
            "rollouts_with_subtask_events": expected_rollouts,
            "rollouts_with_selector_events": 0,
            "exit_code": 0,
        }
        (status_root / f"worker-{worker}.json").write_text(json.dumps(value))


@pytest.mark.parametrize("worker_count", [4, 8])
def test_two_task_worker_status_layouts_are_valid(tmp_path, worker_count):
    status_root = tmp_path / "status"
    _write_two_task_worker_statuses(status_root, worker_count)

    statuses, missing, unexpected, invalid, mismatches = load_worker_statuses(
        status_root,
        worker_count,
        "two_tasks",
        ["carrying_in_groceries", "make_rose_centerpieces"],
        120,
        True,
        True,
        False,
    )

    assert len(statuses) == worker_count
    assert not missing
    assert not unexpected
    assert not invalid
    assert not mismatches


def test_two_task_worker_status_rejects_wrong_eight_worker_assignment(tmp_path):
    status_root = tmp_path / "status"
    _write_two_task_worker_statuses(status_root, 8)
    path = status_root / "worker-4.json"
    value = json.loads(path.read_text())
    value["tasks"] = ["carrying_in_groceries"]
    path.write_text(json.dumps(value))

    *_, mismatches = load_worker_statuses(
        status_root,
        8,
        "two_tasks",
        ["carrying_in_groceries", "make_rose_centerpieces"],
        120,
        True,
        True,
        False,
    )

    assert mismatches


def test_nonzero_worker_metric_is_reported(tmp_path):
    path = tmp_path / "eval_metrics_8200.json"
    path.write_text(
        json.dumps(
            {
                "port": 8200,
                "server_gpu": 0,
                "eval_gpu": 0,
                "elapsed_seconds": 1,
                "exit_code": 1,
            }
        )
    )

    metrics, invalid, failed, duplicates = load_worker_metrics(tmp_path)

    assert len(metrics) == 1
    assert not invalid
    assert failed == [{"path": str(path), "exit_code": 1}]
    assert not duplicates


def test_eval_launcher_verifies_server_identity():
    source = EVAL_SCRIPT.read_text()

    assert '--server-token "$SERVER_TOKEN"' in source
    assert '--expected-server-token "$SERVER_TOKEN"' in source
    assert 'metadata.get("server_instance_token") == sys.argv[2]' in source
    assert "connect_ex" not in source


def test_submit_records_eight_worker_port_contract():
    source = SUBMIT_SCRIPT.read_text()

    assert 'write_setting CLUSTER_FILE "$CLUSTER_FILE"' in source
    assert '"eval_workers": int(eval_workers)' in source
    assert '"base_port": int(base_port)' in source
    assert '"cluster_file": cluster_file' in source
    assert "worker port range exceeds 65535" in source


def run_worker_validation(tmp_path, *, subtask_prediction: str, subtask_interval: str):
    nebula_dir = tmp_path / "repo" / "nebula"
    nebula_dir.mkdir(parents=True)
    worker = nebula_dir / WORKER_SCRIPT.name
    shutil.copy(WORKER_SCRIPT, worker)
    (nebula_dir / "eval_job.env").write_text(
        "\n".join(
            (
                "EVAL_PROFILE=smoke",
                "CONFIRM_FULL=0",
                "EVAL_WORKERS=1",
                "BASE_PORT=8000",
                "WRITE_VIDEO=1",
                f"SUBTASK_PREDICTION={subtask_prediction}",
                f"SUBTASK_INTERVAL={subtask_interval}",
                "SMOKE_EVAL_GPU=1",
            )
        )
        + "\n"
    )
    return subprocess.run(
        ["bash", str(worker)], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_worker_rejects_invalid_subtask_interval(tmp_path, value):
    completed = run_worker_validation(
        tmp_path, subtask_prediction="1", subtask_interval=value
    )

    assert completed.returncode == 1
    assert "SUBTASK_INTERVAL must be a positive integer" in completed.stderr


def test_worker_rejects_subtask_interval_without_prediction(tmp_path):
    completed = run_worker_validation(
        tmp_path, subtask_prediction="0", subtask_interval="8"
    )

    assert completed.returncode == 1
    assert "SUBTASK_INTERVAL requires SUBTASK_PREDICTION=1" in completed.stderr


def test_worker_rejects_selector_outside_supported_profiles(tmp_path):
    nebula_dir = tmp_path / "repo" / "nebula"
    nebula_dir.mkdir(parents=True)
    worker = nebula_dir / WORKER_SCRIPT.name
    shutil.copy(WORKER_SCRIPT, worker)
    (nebula_dir / "eval_job.env").write_text(
        "\n".join(
            (
                "EVAL_PROFILE=reduced",
                "CONFIRM_FULL=0",
                "EVAL_WORKERS=1",
                "BASE_PORT=8000",
                "WRITE_VIDEO=1",
                "SUBTASK_PREDICTION=1",
                f"EXTERNAL_SUBTASK_SELECTOR={SELECTOR}",
            )
        )
        + "\n"
    )

    completed = run_shell(worker)

    assert completed.returncode == 1
    assert "supported only for EVAL_PROFILE=smoke or two_tasks" in completed.stderr


def test_worker_rejects_unknown_model_format(tmp_path):
    nebula_dir = tmp_path / "repo" / "nebula"
    nebula_dir.mkdir(parents=True)
    worker = nebula_dir / WORKER_SCRIPT.name
    shutil.copy(WORKER_SCRIPT, worker)
    (nebula_dir / "eval_job.env").write_text(
        "PI05_MODEL_FORMAT=unknown\n"
        "EVAL_PROFILE=smoke\n"
        "CONFIRM_FULL=0\n"
        "EVAL_WORKERS=1\n"
        "BASE_PORT=8000\n"
        "WRITE_VIDEO=1\n"
        "SUBTASK_PREDICTION=1\n"
    )

    completed = run_shell(worker)

    assert completed.returncode == 1
    assert "PI05_MODEL_FORMAT must be jax or torch" in completed.stderr

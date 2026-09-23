import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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
        "EXTERNAL_SUBTASK_SELECTOR",
        "PI05_CKPT",
        "PI05_MODEL_FORMAT",
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
            "requires SUBTASK_PREDICTION=1",
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


def test_submit_script_requires_explicit_torch_checkpoint():
    completed = run_shell(SUBMIT_SCRIPT, PI05_MODEL_FORMAT="torch")

    assert completed.returncode == 2
    assert "CHECKPOINT_SOURCE is required" in completed.stderr


def test_worker_uses_checkpoint_factorized_interval():
    source = WORKER_SCRIPT.read_text()

    assert "SUBTASK_INTERVAL=1" not in source
    assert 'action_conditioning == "subtask_only"' in source
    assert '"subtask_interval": subtask_interval' in source


def test_submit_script_supports_no_dependency_bootstrap():
    source = SUBMIT_SCRIPT.read_text()

    assert 'BOOTSTRAP_ENV_ARGS=("--env=PIP_NO_DEPS=1")' in source
    assert '"nebula_pip_no_deps": nebula_pip_no_deps == "1"' in source


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

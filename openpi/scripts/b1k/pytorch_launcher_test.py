"""Exercise real shell entrypoints without downloading assets or starting training."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
LAUNCHER = REPOSITORY / "scripts/train_pi05_pytorch_b1k.sh"
SUBTASK_LAUNCHER = REPOSITORY / "scripts/train_pi05_subtask_pytorch_b1k.sh"


@pytest.fixture
def launch_environment(tmp_path):
    openpi, environment = tmp_path / "code space", tmp_path / "torch environment"
    data, weights = tmp_path / "dataset space", tmp_path / "base weights"
    (environment / "bin").mkdir(parents=True)
    openpi.mkdir()
    (data / "meta").mkdir(parents=True)
    (data / "meta/info.json").write_text("{}")
    weights.mkdir()
    (weights / "model.safetensors").touch()
    episodes = tmp_path / "episodes.txt"
    episodes.write_text("5\n6\n7\n")
    capture = tmp_path / "calls.jsonl"
    interpreter = environment / "bin/python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        r"""
import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
with open(os.environ["CAPTURE"], "a") as stream:
    stream.write(json.dumps({"args": arguments, "cwd": os.getcwd(), "jax": os.environ.get("JAX_PLATFORMS"),
                             "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
def option(flag):
    return arguments[arguments.index(flag) + 1]
if arguments[0] == "scripts/b1k/check_subtask_data.py":
    Path(option("--write-episodes-file")).write_text("5\n7\n")
    Path(option("--report-file")).write_text("{}")
elif arguments[0] == "scripts/compute_norm_stats.py":
    path = Path(option("--assets-base-dir")) / option("--config-name") / option("--repo-id") / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
"""
    )
    interpreter.chmod(0o755)
    variables = {
        **os.environ,
        "CAPTURE": str(capture),
        "OPENPI_DIR": str(openpi),
        "PYTORCH_ENV": str(environment),
        "DATASET_ROOT": str(data),
        "PYTORCH_WEIGHT_PATH": str(weights),
        "EPISODES_FILE": str(episodes),
        "ASSETS_BASE_DIR": str(tmp_path / "assets"),
        "CHECKPOINT_BASE_DIR": str(tmp_path / "checkpoints"),
        "EXP_NAME": "test_run",
        "NUM_GPUS": "1",
        "CUDA_DEVICES": "0",
        "BATCH_SIZE": "2",
        "GRAD_ACCUM_STEPS": "3",
        "NUM_WORKERS": "0",
        "TRAIN_STEPS": "4",
        "FRAME_STRIDE": "1",
        "SUBTASK_PREDICTION": "0",
        "RESUME": "0",
        "OVERWRITE": "0",
        "GRAD_CHECKPOINTING": "1",
        "NORM_MAX_FRAMES": "32",
        "SAVE_INTERVAL": "5",
        "REPO_ID": "behavior/demos",
    }

    def launch(mode="full", *, subtask=False, overrides=None):
        capture.unlink(missing_ok=True)
        overrides = {**variables, **(overrides or {})}
        result = subprocess.run(
            ["bash", str(SUBTASK_LAUNCHER if subtask else LAUNCHER), mode],
            cwd=tmp_path,
            env={key: value for key, value in overrides.items() if value is not None},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, calls

    return tmp_path, variables, launch


def _value(arguments, flag):
    return arguments[arguments.index(flag) + 1]


def _values(arguments, flag):
    result = []
    for argument in arguments[arguments.index(flag) + 1 :]:
        if argument.startswith("--"):
            break
        result.append(argument)
    return result


def test_public_data_cli_fields_and_global_microbatch(launch_environment):
    _, environment, launch = launch_environment
    result, calls = launch()
    assert result.returncode == 0, result.stderr
    arguments = calls[-1]["args"]
    assert arguments[:2] == ["scripts/b1k/train_b1k_pytorch.py", "pi05_b1k_pytorch"]
    assert _value(arguments, "--data.dataset-root") == environment["DATASET_ROOT"]
    assert _values(arguments, "--data.episodes-index") == ["5", "6", "7"]
    assert not any("base-config" in argument for argument in arguments)
    assert _value(arguments, "--batch-size") == "2"
    assert _value(arguments, "--pytorch-gradient-accumulation-steps") == "3"
    assert calls[-1]["jax"] == "cpu"
    assert calls[-1]["cwd"] == environment["OPENPI_DIR"]


@pytest.mark.parametrize("subtask", [False, True])
def test_launcher_arguments_parse_with_real_training_config(launch_environment, monkeypatch, subtask):
    from openpi.training import config as config_lib

    _, environment, launch = launch_environment
    result, calls = launch(subtask=subtask)
    assert result.returncode == 0, result.stderr
    monkeypatch.setattr(sys, "argv", calls[-1]["args"])
    config = config_lib.cli()
    assert config.name == "pi05_b1k_pytorch"
    assert config.ema_decay is None
    assert config.model.train_subtask_prediction == subtask
    assert config.model.subtask_action_conditioning == ("subtask_only" if subtask else "task_and_subtask")
    assert config.data.dataset_root == environment["DATASET_ROOT"]
    assert config.data.episodes_index == ([5, 7] if subtask else [5, 6, 7])
    assert config.batch_size == 2
    assert config.pytorch_gradient_accumulation_steps == 3


def test_two_gpu_dispatches_torchrun(launch_environment):
    _, _, launch = launch_environment
    result, calls = launch(overrides={"NUM_GPUS": "2", "CUDA_DEVICES": "0,1", "BATCH_SIZE": "4"})
    assert result.returncode == 0, result.stderr
    arguments = calls[-1]["args"]
    assert arguments[:2] == ["-m", "torch.distributed.run"]
    assert _value(arguments, "--nproc-per-node") == "2"
    assert _value(arguments, "--batch-size") == "4"
    assert calls[-1]["cuda"] == "0,1"


def test_filtered_episodes_shared_by_norm_stats_and_training(launch_environment):
    _, _, launch = launch_environment
    result, calls = launch(subtask=True)
    assert result.returncode == 0, result.stderr
    statistics = next(call["args"] for call in calls if call["args"][0] == "scripts/compute_norm_stats.py")
    training = calls[-1]["args"]
    assert _values(statistics, "--episodes-index") == ["5", "7"]
    assert _values(training, "--data.episodes-index") == ["5", "7"]
    assert "--model.train-subtask-prediction" in training
    assert _value(training, "--model.max-token-len") == "512"
    assert _value(training, "--model.subtask-action-conditioning") == "subtask_only"
    assert _value(training, "--data.subtask-whole-task-prob") == "0"


def test_check_does_not_load_weights_compute_stats_or_train(launch_environment):
    _, _, launch = launch_environment
    result, calls = launch("check", subtask=True, overrides={"PYTORCH_WEIGHT_PATH": None})
    assert result.returncode == 0, result.stderr
    assert [call["args"][0] for call in calls] == [
        "scripts/install_transformers_patch.py",
        "scripts/b1k/check_subtask_data.py",
    ]


def test_resume_needs_no_base_weights_or_norm_cache(launch_environment):
    _, _, launch = launch_environment
    result, calls = launch(overrides={"RESUME": "1", "PYTORCH_WEIGHT_PATH": None})
    assert result.returncode == 0, result.stderr
    assert all(call["args"][0] != "scripts/compute_norm_stats.py" for call in calls)
    assert "--resume" in calls[-1]["args"]
    assert "--pytorch-weight-path" not in calls[-1]["args"]


def test_relative_paths_resolve_before_entering_openpi(launch_environment):
    directory, environment, launch = launch_environment
    paths = {
        name: str(Path(environment[name]).relative_to(directory))
        for name in (
            "OPENPI_DIR",
            "PYTORCH_ENV",
            "DATASET_ROOT",
            "PYTORCH_WEIGHT_PATH",
            "EPISODES_FILE",
            "ASSETS_BASE_DIR",
            "CHECKPOINT_BASE_DIR",
        )
    }
    result, calls = launch(subtask=True, overrides=paths)
    assert result.returncode == 0, result.stderr
    preflight = next(call["args"] for call in calls if call["args"][0] == "scripts/b1k/check_subtask_data.py")
    assert _value(preflight, "--report-file").startswith(environment["OPENPI_DIR"] + "/outputs/")
    assert _value(calls[-1]["args"], "--data.dataset-root") == environment["DATASET_ROOT"]
    assert _value(calls[-1]["args"], "--pytorch-weight-path") == environment["PYTORCH_WEIGHT_PATH"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"NUM_GPUS": "2", "CUDA_DEVICES": "0,1", "BATCH_SIZE": "3"},
        {"RESUME": "1", "OVERWRITE": "1"},
        {"GRAD_CHECKPOINTING": "maybe"},
        {"SAVE_INTERVAL": "0"},
    ],
)
def test_invalid_settings_fail_before_python(launch_environment, overrides):
    _, _, launch = launch_environment
    result, calls = launch(overrides=overrides)
    assert result.returncode != 0
    assert calls == []

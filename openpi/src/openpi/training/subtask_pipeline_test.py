"""Integration of the B1K config, dataset adapters, and both JAX train steps."""
# ruff: noqa: SLF001 -- replace external I/O and inspect a local tokenizer in tests.

import dataclasses
import functools
import json
import sys
from types import SimpleNamespace

from flax import nnx
import jax
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.configs import ROBOT_REGISTRY
from openpi.models import tokenizer as tokenizer_lib
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import utils
from scripts import train
from scripts.b1k import train_b1k

pytest_plugins = ["openpi.models.subtask_test"]


def test_b1k_pipeline_preserves_task_and_original_frame_indices(tmp_path, monkeypatch, tokenizer):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task_name": "turning_on_radio",
                "task": "turn on radio",
            }
        )
        + "\n"
    )
    pq.write_table(
        pa.table(
            {
                # A bad label outside the selected subset must not block loading.
                "episode_index": [5, 42, 42],
                "start": [0, 0, 5],
                "end": [576, 5, 10],
                "subtask": ["", "move to radio", "press radio"],
            }
        ),
        tmp_path / "meta" / "subtask_segments.parquet",
    )
    robot = ROBOT_REGISTRY["b1k/R1Pro"]

    class Frames:
        def __init__(self, *, episodes, delta_timestamps, **kwargs):
            assert episodes == [42]
            assert len(delta_timestamps[robot.action_key]) == 2

        def __len__(self):
            return 10

        def __getitem__(self, index):
            row = {
                "episode_index": np.int64(42),
                "frame_index": np.int64(index),
                "task_index": np.int64(0),
                "observation.state": np.zeros(57),
                "action": np.zeros((2, 23)),
            }
            row.update(
                {obs.dataset_key: np.zeros((3, 224, 224), dtype=np.float32) for obs in robot.observations.values()}
            )
            return row

    monkeypatch.setattr(
        data_loader._lerobot_compat,
        "LeRobotDatasetMetadata",
        lambda **kwargs: SimpleNamespace(fps=30, tasks={0: "turning_on_radio"}),
    )
    tokenizer._max_len = 128
    monkeypatch.setattr(tokenizer_lib, "PaligemmaTokenizer", lambda max_len: tokenizer)
    config = config_lib.get_config("pi05_b1k")
    model_config = dataclasses.replace(config.model, train_subtask_prediction=True, max_token_len=128, action_horizon=2)
    factory = dataclasses.replace(
        config.data,
        repo_id="test/b1k",
        dataset_root=str(tmp_path),
        episodes_index=[42],
        frame_stride=3,
        base_config=dataclasses.replace(config.data.base_config, data_cls=Frames),
    )
    data_config = factory.create(tmp_path / "assets", model_config)
    assert data_config.subtask_prediction
    dataset = data_loader.create_b1k_dataset(data_config, 2)
    assert len(dataset) == 4
    assert dataset[2]["frame_index"] == 6
    assert dataset[2]["prompt"] == "turn on radio"
    assert dataset[2]["subtask"] == "press radio"
    converted = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)[2]
    assert converted["state"].shape == (32,)
    assert converted["actions"].shape == (2, 32)
    assert converted["token_loss_mask"].any()
    assert (
        tokenizer._tokenizer.decode(converted["tokenized_prompt"][converted["token_loss_mask"]].tolist())
        == "press radio"
    )
    with pytest.raises(ValueError, match="subtask_whole_task_prob"):
        dataclasses.replace(factory, subtask_whole_task_prob=0.15).create(tmp_path, model_config)
    with pytest.raises(ValueError, match="one dataset root"):
        data_loader.create_b1k_dataset(dataclasses.replace(data_config, repo_id=["a", "b"]), 2)


@pytest.mark.parametrize("trainer", [train_b1k, train])
def test_real_train_step_returns_separate_metrics_and_updates_state(model_setup, trainer):
    model, obs, actions, model_config = model_setup
    config = dataclasses.replace(config_lib.get_config("pi05_b1k"), model=model_config, ema_decay=None)
    model_def, params = nnx.split(model)
    tx = optax.adam(1e-3)
    state = utils.TrainState(
        step=0,
        params=params,
        model_def=model_def,
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
    )
    step = jax.jit(functools.partial(trainer.train_step, config))
    updated, info = step(jax.random.key(9), state, (obs, actions))
    assert int(updated.step) == 1
    assert {"action_loss", "subtask_loss", "subtask_token_accuracy", "subtask_tokens"} <= info.keys()
    assert all(np.isfinite(value).all() for value in info.values())
    np.testing.assert_allclose(info["loss"], info["action_loss"] + info["subtask_loss"], rtol=1e-5)
    assert float(info["grad_norm"]) > 0


def test_subtask_cli_flags_parse_with_existing_pi05_b1k_config(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_b1k.py",
            "pi05_b1k",
            "--exp-name",
            "test-subtask",
            "--model.train-subtask-prediction",
            "--model.max-token-len",
            "512",
            "--model.max-subtask-len",
            "64",
            "--model.subtask-loss-weight",
            "0.5",
            "--model.subtask-loss-chunk-size",
            "4",
            "--data.subtask-prompts-path",
            "/tmp/segments.parquet",
            "--data.subtask-whole-task-prob",
            "0",
            "--weight-loader.params-path",
            "/tmp/pi05_base/params",
            "--no-wandb-enabled",
        ],
    )
    config = config_lib.cli()
    assert config.model.train_subtask_prediction
    assert config.model.subtask_loss_weight == 0.5
    assert config.model.max_token_len == 512
    assert config.model.subtask_loss_chunk_size == 4
    assert config.weight_loader.params_path == "/tmp/pi05_base/params"

"""Integration of the B1K config, dataset adapters, and both JAX train steps."""
# ruff: noqa: SLF001 -- replace external I/O and inspect a local tokenizer in tests.

import dataclasses
import functools
import json
import sys
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openpi import transforms
from openpi.configs import ROBOT_REGISTRY
from openpi.models import tokenizer as tokenizer_lib
from openpi.models.subtask_test import with_action_context
from openpi.shared import subtask_contract
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import utils
from scripts import train
from scripts.b1k import train_b1k

pytest_plugins = ["openpi.models.subtask_test"]


@pytest.mark.parametrize("framework", ["jax", "pytorch"])
@pytest.mark.parametrize("conditioning", ["task_and_subtask", "subtask_only"])
def test_b1k_pipeline_preserves_task_and_original_frame_indices(
    tmp_path, monkeypatch, tokenizer, framework, conditioning
):
    if framework == "pytorch":

        def no_jax_backend(*args, **kwargs):
            raise AssertionError("PyTorch data loading must not initialize a JAX backend")

        monkeypatch.setattr(jax, "devices", no_jax_backend)
        monkeypatch.setattr(jax, "process_count", no_jax_backend)
        monkeypatch.setattr(jax, "process_index", no_jax_backend)
    (tmp_path / "meta").mkdir()
    episode_meta = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_meta.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [42], "length": [10], "dataset_from_index": [0]}),
        episode_meta / "file-000.parquet",
    )
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
                "is_annotated": [True, True, True],
                "provenance": ["annotated", "annotated", "annotated"],
            }
        ),
        tmp_path / "meta" / "subtask_segments.parquet",
    )
    robot = ROBOT_REGISTRY["b1k/R1Pro"]

    class IndexView:
        def __init__(self, columns):
            self.columns = columns

        def select_columns(self, names):
            assert names == ["episode_index", "frame_index"]
            return IndexView({name: self.columns[name] for name in names})

        def __getitem__(self, name):
            return self.columns[name]

    class Frames:
        def __init__(self, *, episodes, delta_timestamps, **kwargs):
            assert episodes == [42]
            assert len(delta_timestamps[robot.action_key]) == 2
            self.hf_dataset = IndexView(
                {"episode_index": np.full(10, 42, dtype=np.int64), "frame_index": np.arange(10)}
            )
            self.decode_count = 0

        def __len__(self):
            return 10

        def __getitem__(self, index):
            self.decode_count += 1
            row = {
                "episode_index": np.int64(42),
                "frame_index": np.int64(index),
                "task_index": np.int64(0),
                "observation.state": np.zeros(57),
                "action": np.zeros((2, 23)),
                "action_is_pad": np.asarray([False, index == 9]),
            }
            row.update(
                {obs.dataset_key: np.zeros((3, 32, 48), dtype=np.float32) for obs in robot.observations.values()}
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
    model_config = dataclasses.replace(
        config.model,
        train_subtask_prediction=True,
        max_token_len=128,
        action_horizon=2,
        subtask_action_conditioning=conditioning,
    )
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
    if conditioning == "subtask_only":
        action_text = tokenizer.decode_subtask(
            converted["action_tokenized_prompt"][converted["action_tokenized_prompt_mask"]]
        )
        assert "press radio" in action_text
        assert "turn on radio" not in action_text
    assert (
        tokenizer._tokenizer.decode(converted["tokenized_prompt"][converted["token_loss_mask"]].tolist())
        == "press radio"
    )

    if framework == "pytorch":
        train_config = dataclasses.replace(config, data=factory, model=model_config, batch_size=2, num_workers=0)
        loader = data_loader.create_data_loader(train_config, framework="pytorch", skip_norm_stats=True, num_batches=1)
        observation, actions = next(iter(loader))
        assert all(isinstance(leaf, torch.Tensor) for leaf in jax.tree.leaves((observation, actions)))
        assert observation.tokenized_prompt.dtype == torch.int32
        assert observation.token_ar_mask.dtype == torch.int32
        assert observation.token_loss_mask.dtype == torch.bool
        assert observation.tokenized_prompt_mask.dtype == torch.bool
        assert observation.token_loss_mask.any(dim=1).all()
        if conditioning == "subtask_only":
            assert observation.action_tokenized_prompt.dtype == torch.int32
            assert observation.action_tokenized_prompt_mask.dtype == torch.bool
        assert observation.images["base_0_rgb"].dtype == torch.float32
        assert actions.shape == (2, 2, 32)
        prompt = tokenizer._tokenizer.decode(
            observation.tokenized_prompt[0, observation.tokenized_prompt_mask[0]].tolist()
        )
        assert "turn on radio" in prompt
        assert "move to radio" in prompt
        plain_config = dataclasses.replace(
            train_config, model=dataclasses.replace(model_config, train_subtask_prediction=False)
        )
        plain_observation, plain_actions = next(
            iter(data_loader.create_data_loader(plain_config, framework="pytorch", skip_norm_stats=True, num_batches=1))
        )
        assert plain_observation.token_loss_mask is None
        assert plain_observation.token_ar_mask is None
        assert plain_actions.shape == actions.shape
        plain_prompt = tokenizer._tokenizer.decode(
            plain_observation.tokenized_prompt[0, plain_observation.tokenized_prompt_mask[0]].tolist()
        )
        assert "turn on radio" in plain_prompt
        assert "move to radio" not in plain_prompt
    with pytest.raises(ValueError, match="subtask_whole_task_prob"):
        dataclasses.replace(factory, subtask_whole_task_prob=0.15).create(tmp_path, model_config)
    with pytest.raises(ValueError, match="one dataset root"):
        data_loader.create_b1k_dataset(dataclasses.replace(data_config, repo_id=["a", "b"]), 2)


@pytest.mark.parametrize("trainer", [train_b1k, train])
@pytest.mark.parametrize("conditioning", ["task_and_subtask", "subtask_only"])
def test_real_train_step_returns_separate_metrics_and_updates_state(model_setup, trainer, tokenizer, conditioning):
    model, obs, actions, model_config = model_setup
    model.subtask_action_conditioning = conditioning
    model_config = dataclasses.replace(model_config, subtask_action_conditioning=conditioning)
    if conditioning == "subtask_only":
        obs = with_action_context(obs, tokenizer)
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


def test_action_only_train_step_reports_only_action_metrics(model_setup, tokenizer):
    _, _, actions, joint_model_config = model_setup
    model_config = dataclasses.replace(joint_model_config, train_subtask_prediction=False)
    arrays = [tokenizer.tokenize(prompt, np.zeros(4)) for prompt in ("move to radio", "press radio")]
    tokens, masks = [jnp.asarray(np.stack([item[index] for item in arrays])) for index in range(2)]
    observation = dataclasses.replace(
        model_config.fake_obs(2),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=masks,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    model = model_config.create(jax.random.key(10))
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

    updated, info = jax.jit(functools.partial(train_b1k.train_step, config))(
        jax.random.key(11), state, (observation, actions)
    )

    assert int(updated.step) == 1
    assert {"loss", "action_loss", "grad_norm", "param_norm"} == info.keys()
    np.testing.assert_allclose(info["loss"], info["action_loss"])
    assert all(np.isfinite(value).all() for value in info.values())


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


def test_validation_episode_split_overrides_factory_and_reuses_repo(monkeypatch, tmp_path, tokenizer):
    monkeypatch.setattr(config_lib.LeRobotB1KDataConfig, "_load_norm_stats", lambda *args: None)
    monkeypatch.setattr(tokenizer_lib, "PaligemmaTokenizer", lambda max_len: tokenizer)
    config = config_lib.get_config("pi05_b1k")
    train_factory = dataclasses.replace(
        config.data,
        repo_id="test/same-repo",
        dataset_root=str(tmp_path),
        episodes_index=[42],
        base_config=dataclasses.replace(config.data.base_config, episodes_index=[42]),
    )
    config = dataclasses.replace(
        config,
        data=train_factory,
        val_log_interval=10,
        val_episodes_index=[99],
    )
    norm_stats = {"state": object()}

    val_config, repo_id, use_norm_stats, data_config = train_b1k._prepare_validation_config(config, norm_stats)

    assert repo_id == "test/same-repo"
    assert val_config.data.episodes_index == [99]
    assert val_config.data.base_config.episodes_index == [99]
    assert data_config.episodes_index == [99]
    assert data_config.norm_stats is norm_stats
    assert use_norm_stats


def test_jax_checkpoint_persists_conditioning_and_guards_resume(model_setup, tokenizer, tmp_path):
    model, _, _, _ = model_setup
    model.subtask_action_conditioning = "subtask_only"
    model_def, params = nnx.split(model)
    tx = optax.sgd(0.1)
    state = utils.TrainState(
        step=jnp.asarray(1),
        params=params,
        model_def=model_def,
        tx=tx,
        opt_state=tx.init(params),
        ema_decay=None,
    )
    transform = transforms.TokenizeSubtaskPrompt(tokenizer, subtask_action_conditioning="subtask_only")
    data_config = config_lib.DataConfig(model_transforms=transforms.Group(inputs=[transform]))
    loader = SimpleNamespace(data_config=lambda: data_config)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, state, loader, 1)
        manager.wait_until_finished()
        assert subtask_contract.load(tmp_path / "checkpoints" / "1" / "assets") == "subtask_only"
        restored = checkpoints.restore_state(manager, state, loader)
        for actual, expected in zip(jax.tree.leaves(restored.params), jax.tree.leaves(params), strict=True):
            np.testing.assert_array_equal(actual, expected)
        data_config = dataclasses.replace(
            data_config,
            model_transforms=transforms.Group(
                inputs=[dataclasses.replace(transform, subtask_action_conditioning="task_and_subtask")]
            ),
        )
        with pytest.raises(ValueError, match="Cannot resume"):
            checkpoints.restore_state(manager, state, loader)
    finally:
        manager.close()


def test_validation_rejects_overlapping_episode_split():
    config = config_lib.get_config("pi05_b1k")
    train_factory = dataclasses.replace(config.data, repo_id="test/same-repo", episodes_index=[42, 99])
    config = dataclasses.replace(config, data=train_factory, val_episodes_index=[99])

    with pytest.raises(ValueError, match="overlap"):
        train_b1k._prepare_validation_config(config, None)


def test_validation_rejects_implicit_all_episode_training_split():
    config = config_lib.get_config("pi05_b1k")
    train_factory = dataclasses.replace(config.data, repo_id="test/same-repo", episodes_index=None)
    config = dataclasses.replace(config, data=train_factory, val_episodes_index=[99])

    with pytest.raises(ValueError, match="explicit training episodes_index"):
        train_b1k._prepare_validation_config(config, None)


def test_validation_rejects_different_repo_without_separate_dataset_root():
    config = config_lib.get_config("pi05_b1k")
    train_factory = dataclasses.replace(config.data, repo_id="test/train", episodes_index=[42])
    config = dataclasses.replace(
        config,
        data=train_factory,
        val_repo_id="test/validation",
        val_episodes_index=[99],
    )

    with pytest.raises(ValueError, match="only an explicit episode split"):
        train_b1k._prepare_validation_config(config, None)


def test_enabled_validation_requires_explicit_split():
    with pytest.raises(ValueError, match="requires val_repo_id or val_episodes_index"):
        dataclasses.replace(config_lib.get_config("pi05_b1k"), val_log_interval=10)

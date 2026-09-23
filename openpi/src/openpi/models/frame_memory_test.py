# ruff: noqa: SLF001
import dataclasses
import functools
import sys
from types import SimpleNamespace

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi import transforms
from openpi.models import frame_memory
from openpi.models.subtask_test import with_action_context
from openpi.shared import frame_memory as memory_contract
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import utils
from openpi.training import weight_loaders
from openpi.training.config import DataConfig
from openpi.training.config import TrainConfig
from scripts.b1k import train_b1k

pytest_plugins = ["openpi.models.subtask_test"]


def test_memory_launcher_cli_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_b1k.py",
            "pi05_b1k",
            "--exp-name",
            "memory-cli",
            "--model.framesamp-modul",
            "--model.memory-num-frames",
            "32",
            "--model.memory-grid-size",
            "4",
            "--model.memory-frame-stride",
            "16",
            "--model.memory-camera",
            "base_0_rgb",
            "--data.memory-cache-dir",
            "/tmp/memory",
            "--model.train-subtask-prediction",
            "--model.subtask-action-conditioning",
            "subtask_only",
        ],
    )
    config = config_lib.cli()
    assert config.model.framesamp_modul
    assert config.model.train_subtask_prediction
    assert config.model.subtask_action_conditioning == "subtask_only"
    assert config.data.memory_cache_dir == "/tmp/memory"


@pytest.fixture
def memory_setup(model_setup, tokenizer):
    _, obs, actions, config = model_setup
    config = dataclasses.replace(
        config,
        framesamp_modul=True,
        memory_num_frames=3,
        memory_grid_size=1,
        subtask_action_conditioning="subtask_only",
    )
    model = config.create(jax.random.key(0))
    obs = with_action_context(obs, tokenizer)
    obs = dataclasses.replace(
        obs,
        memory_features=jax.random.normal(jax.random.key(1), (2, 3, 64)),
        memory_positions=jnp.array([[[0, 0, 0], [16, 0, 0], [32, 0, 0]]] * 2),
        memory_mask=jnp.array([[True, True, False]] * 2),
    )
    return model, obs, actions, config


def test_pooling_grid_and_padding():
    tokens = jnp.arange(16).reshape(1, 16, 1).astype(jnp.float32)
    np.testing.assert_allclose(frame_memory.pool_tokens(tokens, 2)[0, :, 0], [2.5, 4.5, 10.5, 12.5])
    with pytest.raises(ValueError, match="square"):
        frame_memory.pool_tokens(tokens[:, :15], 2)


def test_memory_training_gradients_and_conditioning(memory_setup):
    model, obs, actions, config = memory_setup
    training = TrainConfig(name="test", exp_name="memory", model=config)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=training.trainable_filter)

    @nnx.jit
    def step(model, optimizer):
        grads = nnx.grad(
            lambda m: m.compute_loss(jax.random.key(3), obs, actions).mean(),
            argnums=nnx.DiffState(0, training.trainable_filter),
        )(model)
        optimizer.update(grads)
        return grads

    # A pretrained checkpoint has open action gates; the tiny random fixture
    # needs one optimizer step to open its initially zero AdaRMS gates.
    frozen = jax.tree.map(np.array, nnx.state(model.PaliGemma.img).to_pure_dict())
    step(model, optimizer)
    grads = step(model, optimizer)
    flat_grads = traverse_util.flatten_dict(grads.to_pure_dict(), sep="/")
    assert not any(key.startswith("PaliGemma/img/") for key in flat_grads)
    assert sum(float(jnp.sum(jnp.abs(value))) for key, value in flat_grads.items() if "memory_modulator" in key) > 0
    assert float(optax.tree.norm(grads["memory_proj"])) > 0
    for before, after in zip(
        jax.tree.leaves(frozen), jax.tree.leaves(nnx.state(model.PaliGemma.img).to_pure_dict()), strict=True
    ):
        np.testing.assert_array_equal(before, after)

    evaluate = nnx_utils.module_jit(model.compute_loss_with_metrics)
    _, original = evaluate(jax.random.key(4), obs, actions)
    changed = dataclasses.replace(obs, memory_features=obs.memory_features.at[:, 0].add(3))
    _, with_memory_change = evaluate(jax.random.key(4), changed, actions)
    np.testing.assert_array_equal(original["subtask_loss"], with_memory_change["subtask_loss"])
    assert not np.isclose(original["action_loss"], with_memory_change["action_loss"], rtol=1e-7)
    task_changed = dataclasses.replace(obs, tokenized_prompt=obs.tokenized_prompt.at[:, 1].set(42))
    _, task_metrics = evaluate(jax.random.key(4), task_changed, actions)
    np.testing.assert_array_equal(original["action_loss"], task_metrics["action_loss"])

    sample = nnx_utils.module_jit(model.sample_actions)
    predicted = sample(jax.random.key(5), obs, num_steps=2)
    assert np.isfinite(predicted).all()
    assert not np.allclose(predicted, sample(jax.random.key(5), changed, num_steps=2), rtol=1e-7, atol=1e-7)
    # Masked feature values, even NaNs, must never affect actions or loss.
    padded_changed = dataclasses.replace(obs, memory_features=obs.memory_features.at[:, 2].set(jnp.nan))
    np.testing.assert_array_equal(predicted, sample(jax.random.key(5), padded_changed, num_steps=2))
    with pytest.raises(ValueError, match="requires memory"):
        model.compute_loss(jax.random.key(0), dataclasses.replace(obs, memory_mask=None), actions)


def test_migrate_checkpoint_retains_backbone_and_initializes_only_memory(memory_setup, monkeypatch):
    model, obs, _, config = memory_setup
    base = dataclasses.replace(config, framesamp_modul=False).create(jax.random.key(2))
    old = nnx.state(base).to_pure_dict()
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **k: old)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    new = weight_loaders.CheckpointWeightLoader("unused").load(nnx.state(model).to_pure_dict())
    reference = traverse_util.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    flat = traverse_util.flatten_dict(new, sep="/")
    assert flat.keys() == reference.keys()
    for key, value in traverse_util.flatten_dict(old, sep="/").items():
        np.testing.assert_array_equal(flat[key], value)
    migrated = config.load(new)
    empty = dataclasses.replace(obs, memory_mask=jnp.zeros_like(obs.memory_mask))
    np.testing.assert_allclose(
        migrated.sample_actions(jax.random.key(4), empty, num_steps=2),
        base.sample_actions(jax.random.key(4), empty, num_steps=2),
        atol=1e-6,
    )


def test_production_modulator_bfloat16_and_empty_mask():
    layer = frame_memory.MemoryModulator(64)
    x = jax.random.normal(jax.random.key(1), (1, 2, 64)).astype(jnp.bfloat16)
    memory = jnp.ones((1, 3, 64), dtype=jnp.bfloat16)
    mask = jnp.array([[True, False, False]])
    variables = layer.init(jax.random.key(2), x, memory, mask)
    output = layer.apply(variables, x, memory, mask)
    assert output.dtype == jnp.bfloat16
    assert np.isfinite(output.astype(jnp.float32)).all()
    np.testing.assert_array_equal(layer.apply(variables, x, memory, jnp.zeros_like(mask)), x)


def test_b1k_train_step_and_checkpoint_round_trip(memory_setup, tokenizer, tmp_path):
    model, obs, actions, config = memory_setup
    training = TrainConfig(name="test", exp_name="memory", model=config, ema_decay=None)
    graph, params = nnx.split(model)
    tx = optax.adam(1e-3)
    state = utils.TrainState(
        step=0,
        params=params,
        model_def=graph,
        tx=tx,
        opt_state=tx.init(params.filter(training.trainable_filter)),
        ema_decay=None,
    )
    step = jax.jit(functools.partial(train_b1k.train_step, training))
    updated, metrics = step(jax.random.key(6), state, (obs, actions))
    assert int(updated.step) == 1
    assert all(np.isfinite(v).all() for v in metrics.values())
    assert {"action_loss", "subtask_loss", "subtask_token_accuracy"} <= metrics.keys()
    data_config = DataConfig(
        memory_contract=memory_contract.model_contract(config),
        model_transforms=transforms.Group(
            inputs=[transforms.TokenizeSubtaskPrompt(tokenizer, subtask_action_conditioning="subtask_only")]
        ),
    )
    loader = SimpleNamespace(data_config=lambda: data_config)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, updated, loader, 1)
        manager.wait_until_finished()
        restored = checkpoints.restore_state(manager, updated, loader)
        for a, b in zip(jax.tree.leaves(restored.params), jax.tree.leaves(updated.params), strict=True):
            np.testing.assert_array_equal(a, b)
        data_config = dataclasses.replace(data_config, memory_contract=None)
        with pytest.raises(ValueError, match="Cannot resume"):
            checkpoints.restore_state(manager, updated, loader)
    finally:
        manager.close()

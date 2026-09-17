"""CPU regressions for optimizer semantics and inference-ready Torch checkpoints."""

import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")
_SPEC = importlib.util.spec_from_file_location(
    "openpi_torch_trainer_test_target", Path(__file__).with_name("train_pytorch.py")
)
trainer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trainer
_SPEC.loader.exec_module(trainer)


class TinyJoint(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(4, 2)
        self.token_head = nn.Linear(4, 5)
        self.unused = nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, observation, actions, *, return_metrics=False):
        if getattr(self, "checkpoint_encoder", False):
            from torch.utils.checkpoint import checkpoint

            encoded = checkpoint(self.encoder, observation["x"], use_reentrant=False)
        else:
            encoded = self.encoder(observation["x"])
        hidden = self.dropout(encoded.tanh())
        action_loss = (self.action_head(hidden) - actions).square().mean(-1)
        ce = nn.functional.cross_entropy(self.token_head(hidden), observation["labels"], reduction="none")
        loss = action_loss + ce
        metrics = {"action_loss": action_loss.mean(), "subtask_loss": ce.mean()}
        return (loss, metrics) if return_metrics else loss


def _batch(n=4):
    generator = torch.Generator().manual_seed(731)
    return {"x": torch.randn(n, 3, generator=generator), "labels": torch.arange(n) % 5}, torch.randn(
        n, 2, generator=generator
    )


def _split_batch(batch):
    observation, actions = batch
    return [
        ({key: value[index : index + 2] for key, value in observation.items()}, actions[index : index + 2])
        for index in range(0, len(actions), 2)
    ]


def _optimizer(model):
    return torch.optim.AdamW(trainer.get_model_parameters(model), lr=0.003, weight_decay=0.01, foreach=False)


def _update(model, optimizer, batches, accumulation=1, precision="float32"):
    return trainer.train_one_update(
        model,
        optimizer,
        iter(batches),
        device=torch.device("cpu"),
        accumulation_steps=accumulation,
        precision=precision,
        clip_gradient_norm=1e6,
    )


def _config(tmp_path, steps=3, interval=2, accumulation=1):
    return SimpleNamespace(
        checkpoint_dir=tmp_path,
        batch_size=4,
        pytorch_gradient_accumulation_steps=accumulation,
        num_train_steps=steps,
        save_interval=interval,
        wandb_enabled=False,
    )


def test_joint_gradient_accumulation_matches_full_batch():
    torch.manual_seed(41)
    full = TinyJoint()
    split = copy.deepcopy(full)
    full_optimizer, split_optimizer = _optimizer(full), _optimizer(split)
    initial_token = full.token_head.weight.detach().clone()
    initial_encoder = full.encoder.weight.detach().clone()
    full_metrics = _update(full, full_optimizer, [_batch()])
    split_metrics = _update(split, split_optimizer, _split_batch(_batch()), accumulation=2)
    for name, parameter in full.named_parameters():
        torch.testing.assert_close(parameter, dict(split.named_parameters())[name], rtol=1e-5, atol=1e-7)
        assert parameter.grad is None
    assert not torch.equal(full.token_head.weight, initial_token)
    assert not torch.equal(full.encoder.weight, initial_encoder)
    assert full_metrics["subtask_loss"] == pytest.approx(split_metrics["subtask_loss"], rel=1e-6)
    assert full_metrics["loss"] == pytest.approx(split_metrics["loss"], rel=1e-6)
    assert full.unused not in full_optimizer.state


def test_plain_tensor_loss_is_supported():
    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 2)

        def forward(self, observation, actions, *, return_metrics=False):
            return (self.layer(observation["x"]) - actions).square()

    model = Plain()
    metrics = _update(model, _optimizer(model), [_batch()])
    assert metrics.keys() == {"loss", "grad_norm"}
    assert metrics["loss"] > 0


def test_cpu_bf16_selection_keeps_weights_and_moments_float32():
    model = TinyJoint()
    optimizer = _optimizer(model)
    _update(model, optimizer, [_batch()], precision="bfloat16")
    for parameter, state in optimizer.state.items():
        assert parameter.dtype == torch.float32
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_subtask_accuracy_is_token_weighted_across_microbatches():
    class Counted(TinyJoint):
        def forward(self, observation, actions, *, return_metrics=False):
            loss, metrics = super().forward(observation, actions, return_metrics=True)
            metrics.update(subtask_token_accuracy=observation["accuracy"], subtask_tokens=observation["count"])
            return loss, metrics

    model = Counted()
    batches = _split_batch(_batch())
    batches[0][0].update(accuracy=torch.tensor(1.0), count=torch.tensor(1.0))
    batches[1][0].update(accuracy=torch.tensor(0.0), count=torch.tensor(3.0))
    metrics = _update(model, _optimizer(model), batches, accumulation=2)
    assert metrics["subtask_token_accuracy"] == pytest.approx(0.25)
    assert metrics["subtask_tokens"] == pytest.approx(2.0)


def test_accumulation_uses_no_sync_until_last_microbatch():
    class NoSyncModel(TinyJoint):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def no_sync(self):
            import contextlib

            self.calls += 1
            return contextlib.nullcontext()

    model = NoSyncModel()
    _update(model, _optimizer(model), _split_batch(_batch()), accumulation=2)
    assert model.calls == 1


def test_nonfinite_loss_never_updates_parameters():
    model = TinyJoint()
    initial = copy.deepcopy(model.state_dict())
    observation, actions = _batch()
    actions[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        _update(model, _optimizer(model), [(observation, actions)])
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, initial[name])


def test_final_checkpoint_is_exactly_completed_step(tmp_path):
    config = _config(tmp_path, steps=3, interval=100)
    model = TinyJoint()
    optimizer = _optimizer(model)
    for step in range(1, 4):
        _update(model, optimizer, [_batch()])
        trainer.save_checkpoint(model, optimizer, step, config)
    assert not (tmp_path / "2").exists()
    assert trainer.get_latest_checkpoint_step(tmp_path) == 3
    assert (tmp_path / "3" / "model.safetensors").is_file()
    assert json.loads((tmp_path / "3" / "metadata.json").read_text())["completed_steps"] == 3


def test_resume_matches_uninterrupted_optimizer_and_rng(tmp_path):
    trainer.set_seed(101, 0)
    original = TinyJoint(dropout=0.25)
    original_optimizer = _optimizer(original)
    config = _config(tmp_path)
    for _ in range(2):
        _update(original, original_optimizer, [_batch()])
    trainer.save_checkpoint(original, original_optimizer, 2, config)
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    expected_metrics = _update(original, original_optimizer, [_batch()])
    restored = TinyJoint(dropout=0.25)
    restored_optimizer = _optimizer(restored)
    assert trainer.load_checkpoint(restored, restored_optimizer, tmp_path, torch.device("cpu"), config) == 2
    actual_random = (random.random(), np.random.rand(), torch.rand(3))
    assert actual_random[:2] == expected_random[:2]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
    actual_metrics = _update(restored, restored_optimizer, [_batch()])
    assert actual_metrics == expected_metrics
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name], rtol=0, atol=0)
    for expected_state, actual_state in zip(
        original_optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        for key in expected_state:
            torch.testing.assert_close(expected_state[key], actual_state[key], rtol=0, atol=0)


def test_resume_rejects_batch_change_and_ignores_incomplete_checkpoint(tmp_path):
    config = _config(tmp_path)
    model = TinyJoint()
    optimizer = _optimizer(model)
    _update(model, optimizer, [_batch()])
    trainer.save_checkpoint(model, optimizer, 2, config)
    (tmp_path / "999").mkdir()
    (tmp_path / "999" / "model.safetensors").touch()
    assert trainer.get_latest_checkpoint_step(tmp_path) == 2
    config.batch_size = 8
    with pytest.raises(ValueError, match="original batch size"):
        trainer.load_checkpoint(model, optimizer, tmp_path, torch.device("cpu"), config)


def test_checkpoint_never_silently_overwrites(tmp_path):
    config = _config(tmp_path)
    model = TinyJoint()
    optimizer = _optimizer(model)
    trainer.save_checkpoint(model, optimizer, 2, config)
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        trainer.save_checkpoint(model, optimizer, 2, config)


def test_resume_configuration_comparison_handles_shared_python_callables(tmp_path):
    config = _config(tmp_path)
    config.model = {"train_subtask_prediction": True, "max_token_len": 512}
    model = TinyJoint()
    optimizer = _optimizer(model)
    trainer.save_checkpoint(model, optimizer, 2, config)
    config.model["max_token_len"] = 200
    with pytest.raises(ValueError, match="changed model"):
        trainer.load_checkpoint(model, optimizer, tmp_path, torch.device("cpu"), config)


def test_resume_accepts_new_disabled_subtask_defaults(tmp_path):
    config = _config(tmp_path)
    config.model = {"train_subtask_prediction": True}
    config.data = {"base_config": {"prompt_from_task": True}}
    model = TinyJoint()
    optimizer = _optimizer(model)
    trainer.save_checkpoint(model, optimizer, 2, config)

    config.model["train_subtask_executor"] = False
    config.data.update(subtask_executor_max_samples_per_segment=None, strict_subtask_action_chunks=False)
    config.data["base_config"].update(
        subtask_executor=False,
        subtask_executor_max_samples_per_segment=None,
        strict_subtask_action_chunks=False,
    )
    restored = TinyJoint()
    restored_optimizer = _optimizer(restored)
    assert trainer.load_checkpoint(restored, restored_optimizer, tmp_path, torch.device("cpu"), config) == 2


def test_resume_rejects_new_subtask_mode_when_enabled(tmp_path):
    config = _config(tmp_path)
    config.model = {"train_subtask_prediction": True}
    model = TinyJoint()
    optimizer = _optimizer(model)
    trainer.save_checkpoint(model, optimizer, 2, config)
    config.model["train_subtask_executor"] = True
    with pytest.raises(ValueError, match="changed model"):
        trainer.load_checkpoint(model, optimizer, tmp_path, torch.device("cpu"), config)


def test_resume_uses_checkpoint_norm_stats_without_mutating_training_config(tmp_path):
    @dataclasses.dataclass(frozen=True)
    class Assets:
        assets_dir: str = "/old/cache"
        asset_id: str = "behavior"

    @dataclasses.dataclass(frozen=True)
    class Data:
        repo_id: str = "behavior"
        assets: Assets = dataclasses.field(default_factory=Assets)

    @dataclasses.dataclass(frozen=True)
    class Config:
        checkpoint_dir: Path
        data: Data = dataclasses.field(default_factory=Data)
        resume: bool = True

    checkpoint = tmp_path / "12"
    checkpoint.mkdir()
    # This regression exercises internal checkpoint asset selection without a 3B model.
    for name in trainer._CHECKPOINT_FILES:  # noqa: SLF001
        (checkpoint / name).touch()
    config = Config(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing normalization assets"):
        trainer._data_config_for_resume(config)  # noqa: SLF001
    (checkpoint / "assets" / "behavior").mkdir(parents=True)
    resumed = trainer._data_config_for_resume(config)  # noqa: SLF001
    assert resumed.data.assets.assets_dir == str(checkpoint / "assets")
    assert resumed.data.assets.asset_id == "behavior"
    assert config.data.assets.assets_dir == "/old/cache"
    fresh = dataclasses.replace(config, resume=False)
    assert trainer._data_config_for_resume(fresh) is fresh  # noqa: SLF001


def test_move_observation_keeps_optional_masks_and_token_dtype():
    @dataclasses.dataclass(frozen=True)
    class Observation:
        tokenized_prompt: torch.Tensor
        token_loss_mask: torch.Tensor | None
        images: dict

    observation = Observation(torch.ones(2, dtype=torch.int32), None, {"camera": np.zeros((2, 4, 4, 3), np.float32)})
    moved = trainer.move_to_device(observation, torch.device("cpu"))
    assert isinstance(moved, Observation)
    assert moved.token_loss_mask is None
    assert moved.tokenized_prompt.dtype == torch.int32
    assert isinstance(moved.images["camera"], torch.Tensor)


def test_warmup_cosine_and_rsqrt_schedule():
    cosine = SimpleNamespace(warmup_steps=2, peak_lr=0.003, decay_steps=6, decay_lr=0.0003)
    assert trainer.learning_rate_at(cosine, 0) == pytest.approx(0.001)
    assert trainer.learning_rate_at(cosine, 1) == pytest.approx(0.002)
    assert trainer.learning_rate_at(cosine, 2) == pytest.approx(0.003)
    assert trainer.learning_rate_at(cosine, 6) == pytest.approx(0.0003)
    assert trainer.learning_rate_at(cosine, 100) == pytest.approx(0.0003)
    rsqrt = SimpleNamespace(warmup_steps=2, peak_lr=0.003, timescale=10)
    assert trainer.learning_rate_at(rsqrt, 2) == pytest.approx(0.003)
    assert trainer.learning_rate_at(rsqrt, 12) == pytest.approx(0.003 / 2**0.5)


def test_jax_is_restricted_before_shared_imports():
    import os

    assert os.environ["JAX_PLATFORMS"] == "cpu"


def _ddp_worker(rank, rendezvous, directory):
    import torch.distributed as distributed

    distributed.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(246)
        core = TinyJoint()
        core.always_unused = nn.Parameter(torch.zeros(2))
        core.checkpoint_encoder = True
        model = torch.nn.parallel.DistributedDataParallel(core, find_unused_parameters=True)
        optimizer = _optimizer(model)
        observation, actions = _batch(8)
        start = rank * 4
        local = ({key: value[start : start + 4] for key, value in observation.items()}, actions[start : start + 4])
        metrics = _update(model, optimizer, _split_batch(local), accumulation=2)
        config = _config(Path(directory), steps=1, interval=100, accumulation=2)
        trainer.save_checkpoint(model, optimizer, 1, config)
        restored = TinyJoint()
        restored.always_unused = nn.Parameter(torch.zeros(2))
        restored_optimizer = _optimizer(restored)
        assert trainer.load_checkpoint(restored, restored_optimizer, directory, torch.device("cpu"), config) == 1
        assert core.always_unused.grad is None
        assert core.always_unused not in optimizer.state
        for name, parameter in trainer.get_model_state_dict(model).items():
            torch.testing.assert_close(parameter, restored.state_dict()[name], rtol=0, atol=0)
        torch.save(
            {"model": trainer.get_model_state_dict(model), "metrics": metrics}, Path(directory) / f"rank_{rank}.pt"
        )
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="Torch distributed unavailable")
def test_two_rank_ddp_accumulation_and_checkpoint_matches_global_batch(tmp_path, monkeypatch):
    import os
    import socket

    import torch.multiprocessing as multiprocessing

    # Sandboxed developer environments can prohibit even local socket binding.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Local loopback sockets are prohibited by the execution sandbox.")
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    rendezvous = tmp_path / "rendezvous"
    multiprocessing.spawn(_ddp_worker, args=(str(rendezvous), str(tmp_path)), nprocs=2, join=True)
    torch.manual_seed(246)
    reference = TinyJoint()
    reference.always_unused = nn.Parameter(torch.zeros(2))
    expected = _update(reference, _optimizer(reference), [_batch(8)])
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for name, parameter in reference.state_dict().items():
            torch.testing.assert_close(parameter, actual["model"][name], rtol=1e-5, atol=1e-7)
        assert actual["metrics"]["loss"] == pytest.approx(expected["loss"], rel=1e-6)
    assert json.loads((tmp_path / "1" / "metadata.json").read_text())["world_size"] == 2

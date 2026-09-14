"""Checkpoint interoperability for the real patched HF model, kept small on CPU."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch_integration_test import observation
from openpi.models_pytorch.pi0_pytorch_integration_test import tiny_hf_model  # noqa: F401

_SPEC = importlib.util.spec_from_file_location(
    "_checkpoint_integration_trainer", Path(__file__).resolve().parents[1] / "train_pytorch.py"
)
trainer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trainer
_SPEC.loader.exec_module(trainer)


def _optimizer(model):
    return torch.optim.AdamW(trainer.get_model_parameters(model), lr=1e-4, foreach=False)


def _update(model, optimizer, batch):
    return trainer.train_one_update(
        model,
        optimizer,
        iter([batch]),
        device=torch.device("cpu"),
        accumulation_steps=1,
        precision="float32",
        clip_gradient_norm=1.0,
    )


@pytest.mark.parametrize("joint", [False, True])
# The imported pytest fixture is injected by name; it is not an unused Python import.
def test_real_hf_checkpoint_resumes_next_optimizer_update_with_tied_weights(tiny_hf_model, tmp_path, joint):  # noqa: F811
    model = tiny_hf_model
    model.train_subtask_prediction = joint
    model.gradient_checkpointing_enable()
    model.train()
    trainer.set_seed(612, 0)
    obs = observation()
    if not joint:
        obs.token_ar_mask = None
        obs.token_loss_mask = None
    batch = obs, torch.randn(1, 2, 3)
    optimizer = _optimizer(model)
    config = SimpleNamespace(
        checkpoint_dir=tmp_path,
        batch_size=1,
        pytorch_gradient_accumulation_steps=1,
        save_interval=1,
        num_train_steps=2,
        wandb_enabled=False,
    )
    _update(model, optimizer, batch)
    unused_head = model.paligemma_with_expert.gemma_expert.lm_head.weight
    assert not unused_head.requires_grad
    assert unused_head not in optimizer.state
    trainer.save_checkpoint(model, optimizer, 1, config)
    checkpoint_state = copy.deepcopy(model.state_dict())
    expected_metrics = _update(model, optimizer, batch)

    restored = type(model)(model.config)
    restored.train_subtask_prediction = joint
    restored.gradient_checkpointing_enable()
    restored.train()
    restored_optimizer = _optimizer(restored)
    assert trainer.load_checkpoint(restored, restored_optimizer, tmp_path, torch.device("cpu"), config) == 1
    assert restored.state_dict().keys() == checkpoint_state.keys()
    for name, saved in checkpoint_state.items():
        torch.testing.assert_close(restored.state_dict()[name], saved, rtol=0, atol=0)
    backbone = restored.paligemma_with_expert
    assert backbone.paligemma.lm_head.weight is backbone.paligemma.language_model.embed_tokens.weight
    assert not backbone.gemma_expert.lm_head.weight.requires_grad
    actual_metrics = _update(restored, restored_optimizer, batch)
    assert actual_metrics == expected_metrics
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    for parameter, state in restored_optimizer.state.items():
        assert parameter.dtype == state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    for expected_state, actual_state in zip(optimizer.state.values(), restored_optimizer.state.values(), strict=True):
        for key in expected_state:
            torch.testing.assert_close(expected_state[key], actual_state[key], rtol=0, atol=0)

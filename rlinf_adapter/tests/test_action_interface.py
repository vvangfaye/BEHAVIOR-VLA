"""Execution-boundary regressions, including preservation of raw PPO samples."""

import random

import numpy as np
import pytest
import torch

from b1k_rlinf.action_interface import ExecutionActionInterface
from test_model import model as model


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_only_near_open_grippers_change_and_inputs_are_preserved(backend):
    value = np.zeros((2, 5, 23), dtype=np.float32)
    value[..., 14] = [-1., .998, .999, .99999, 1.001]
    value[..., 22] = [.9995, .5, 0., -1.0001, 1.]
    value[..., 5] = .99999  # An arm coordinate near one must not be snapped.
    before = value.copy()
    source = torch.from_numpy(value) if backend == "torch" else value
    result = ExecutionActionInterface(.001)(source)
    actual = result.numpy() if backend == "torch" else result
    expected = before.copy()
    expected[..., 14] = [-1., .998, 1., 1., 1.001]
    expected[..., 22] = [1., .5, 0., -1.0001, 1.]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(value, before)
    assert result.dtype == source.dtype and result.shape == source.shape
    second = ExecutionActionInterface(.001)(result)
    np.testing.assert_array_equal(second.numpy() if backend == "torch" else second, actual)


def test_disabled_mapping_preserves_legacy_results():
    value = torch.full((1, 16, 23), .9999)
    assert ExecutionActionInterface()(value) is value


@pytest.mark.parametrize("value", [-.001, 1., float("nan"), float("inf"), True])
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError):
        ExecutionActionInterface(value)


def test_invalid_action_contract_is_rejected():
    interface = ExecutionActionInterface(.001)
    with pytest.raises(ValueError):
        interface(np.zeros((16, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        interface(torch.full((16, 23), float("nan")))


def test_public_train_eval_boundary_preserves_raw_samples_and_ppo_density(model, monkeypatch):
    # Use the real small Gemma/SDE model and its public sampling method. Only
    # language generation is fixed to avoid random untrained text failing EOS.
    token = model.joint_tokenizer.processor.encode("move hand", add_eos=True)
    def labels(obs):
        batch = obs.state.shape[0]
        return torch.tensor([token] * batch), torch.full((batch,), len(token), dtype=torch.long)
    monkeypatch.setattr(model, "generate_subtask", labels)
    env = {
        "states": np.zeros((1, 23), dtype=np.float32),
        "main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
        "task_descriptions": ["turn on radio"],
    }
    decode = model.decode_actions
    def near_open(*args, **kwargs):
        actions = decode(*args, **kwargs).clone()
        actions[..., 14] = .9998
        actions[..., 22] = .9997
        return actions
    monkeypatch.setattr(model, "decode_actions", near_open)
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    for mode in ("train", "eval"):
        random.seed(21)
        model.execution_interface = ExecutionActionInterface()
        noise = torch.zeros((1, 32, 32))
        raw, original = model.predict_action_batch(env, mode=mode, noise=noise,
                                                  rng=torch.Generator().manual_seed(17))
        random.seed(21)
        model.execution_interface = ExecutionActionInterface(.001)
        executed, mapped = model.predict_action_batch(env, mode=mode, noise=noise,
                                                     rng=torch.Generator().manual_seed(17))
        torch.testing.assert_close(mapped["model_actions"], original["model_actions"], atol=0, rtol=0)
        torch.testing.assert_close(mapped["policy_actions"], raw, atol=0, rtol=0)
        torch.testing.assert_close(mapped["forward_inputs"]["action"],
                                   original["forward_inputs"]["action"], atol=0, rtol=0)
        torch.testing.assert_close(executed[..., [14, 22]], torch.ones_like(executed[..., [14, 22]]))
        keep = [i for i in range(23) if i not in (14, 22)]
        torch.testing.assert_close(executed[..., keep], raw[..., keep], atol=0, rtol=0)
        if mode == "train":
            torch.testing.assert_close(mapped["prev_logprobs"], original["prev_logprobs"], atol=0, rtol=0)
            actual = model.default_forward(mapped["forward_inputs"])
            torch.testing.assert_close(actual["logprobs"], mapped["prev_logprobs"], atol=1e-5, rtol=1e-5)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)

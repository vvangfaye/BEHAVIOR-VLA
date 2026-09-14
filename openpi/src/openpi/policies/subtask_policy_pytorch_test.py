"""Exercise the Torch policy boundary with a local tokenizer and generated tokens."""

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.policies.subtask_policy import SubtaskPolicy

pytest_plugins = ["openpi.models.subtask_test"]


class TorchSubtaskModel(torch.nn.Module):
    action_dim = 4
    train_subtask_prediction = True

    def __init__(self, label):
        super().__init__()
        self.register_buffer("label", torch.as_tensor(label))
        self.generation_inputs = []
        self.action_inputs = []
        self.has_eos = True
        self.valid_prefix = True

    def generate_subtask(self, device, observation, **kwargs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert observation.state.device == torch.device(device)
        assert not observation.token_loss_mask.any()
        self.generation_inputs.append(observation)
        return {
            "tokens": self.label[None],
            "lengths": torch.tensor([len(self.label)], device=device),
            "has_eos": torch.tensor([self.has_eos], device=device),
            "valid_prefix": torch.tensor([self.valid_prefix], device=device),
        }

    def sample_actions(self, device, observation, **kwargs):
        assert observation.token_loss_mask.any()
        self.action_inputs.append(observation)
        return torch.ones(1, 2, self.action_dim, device=device)


def make_policy(tokenizer):
    arrays = tokenizer.tokenize_subtask_training("turn on radio", "press radio", max_subtask_len=8)
    model = TorchSubtaskModel(arrays[0][arrays[3]])
    policy = SubtaskPolicy(
        model,
        tokenizer_transform=transforms.TokenizeSubtaskPrompt(tokenizer, max_subtask_len=8),
        is_pytorch=True,
        pytorch_device="cpu",
        subtask_interval=2,
    )
    return policy, model


def source():
    return {
        "image": {"base_0_rgb": np.zeros((16, 16, 3), dtype=np.uint8)},
        "image_mask": {"base_0_rgb": np.asarray(1, dtype=np.bool_)},
        "state": np.zeros(4, dtype=np.float32),
        "prompt": "turn on radio",
    }


def test_torch_generation_cache_gt_and_reset(tokenizer):
    policy, model = make_policy(tokenizer)
    first = policy.infer(source())
    assert first["subtask"] == "press radio"
    assert first["subtask_source"] == "generated"
    assert first["actions"].shape == (2, 4)
    assert np.isfinite(first["actions"]).all()
    assert policy.infer(source())["subtask_source"] == "cached"
    assert len(model.generation_inputs) == 1
    assert policy.infer(source())["subtask_source"] == "generated"
    assert len(model.generation_inputs) == 2
    supplied = policy.infer({**source(), "subtask": "move to radio"})
    assert supplied["subtask"] == "move to radio"
    assert supplied["subtask_source"] == "provided"
    assert len(model.generation_inputs) == 2
    assert policy.infer(source())["subtask_source"] == "generated"
    assert len(model.generation_inputs) == 3
    policy.reset()
    assert policy.infer(source())["subtask_source"] == "generated"
    assert len(model.generation_inputs) == 4
    assert policy.infer({**source(), "prompt": "radio"})["subtask_source"] == "generated"
    assert len(model.generation_inputs) == 5


@pytest.mark.parametrize("invalid", ["has_eos", "valid_prefix"])
def test_torch_policy_never_acts_on_incomplete_generation(tokenizer, invalid):
    policy, model = make_policy(tokenizer)
    setattr(model, invalid, False)
    with pytest.raises(RuntimeError, match="EOS"):
        policy.infer(source())
    assert not model.action_inputs
    assert policy._cached_tokens is None  # noqa: SLF001 -- failed generation must not populate the cache.

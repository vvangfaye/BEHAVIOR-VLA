"""Small CPU checks for checkpoint conversion, without downloading model weights."""
# ruff: noqa: SLF001 -- exercise strict state loading independently of a full model.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from examples import convert_jax_model_to_pytorch as converter


@pytest.mark.parametrize(("pi05", "directory"), [(True, "/checkpoints/1000"), (False, "/tmp/pi05_in_directory_name")])
def test_expert_architecture_comes_from_config_not_checkpoint_path(pi05, directory):
    config = SimpleNamespace(vocab_size=8, hidden_size=4, num_hidden_layers=1, num_attention_heads=2, head_dim=2)
    params = {
        "llm/layers/attn/attn_vec_einsum_1/w": np.zeros((1, 2, 2, 4), np.float32),
        "llm/layers/attn/kv_einsum_1/w": np.zeros((1, 2, 1, 4, 2), np.float32),
        "llm/layers/attn/q_einsum_1/w": np.zeros((1, 2, 4, 2), np.float32),
        "llm/layers/mlp_1/gating_einsum": np.zeros((1, 2, 4, 6), np.float32),
        "llm/layers/mlp_1/linear": np.zeros((1, 6, 4), np.float32),
    }
    for norm in ["pre_attention_norm_1", "pre_ffw_norm_1"]:
        if pi05:
            params[f"llm/layers/{norm}/Dense_0/bias"] = np.zeros((1, 12), np.float32)
            params[f"llm/layers/{norm}/Dense_0/kernel"] = np.zeros((1, 4, 12), np.float32)
        else:
            params[f"llm/layers/{norm}/scale"] = np.zeros((1, 4), np.float32)
    if pi05:
        params["llm/final_norm_1/Dense_0/bias"] = np.zeros(12, np.float32)
        params["llm/final_norm_1/Dense_0/kernel"] = np.zeros((4, 12), np.float32)
    else:
        params["llm/final_norm_1/scale"] = np.zeros(4, np.float32)
    converted = converter.slice_gemma_state_dict(params, config, num_expert=1, checkpoint_dir=directory, pi05=pi05)
    norm_key = "dense.weight" if pi05 else "weight"
    assert f"paligemma_with_expert.gemma_expert.model.norm.{norm_key}" in converted
    assert all(isinstance(value, torch.Tensor) for value in converted.values())


def _tiny_meta_model():
    class PaliGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        @property
        def language_model(self):
            return self.model.language_model

    with torch.device("meta"):
        model = nn.Module()
        model.paligemma_with_expert = nn.Module()
        model.paligemma_with_expert.paligemma = PaliGemma()
        model.paligemma_with_expert.gemma_expert = nn.Module()
        model.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(4, 8, bias=False)
        model.action_in_proj = nn.Linear(4, 4)
    return model


def _source_params():
    return {
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight": torch.ones(8, 4),
        "action_in_proj.weight": torch.eye(4),
        "action_in_proj.bias": torch.zeros(4),
    }


def test_strict_conversion_materializes_meta_parameters_and_reties_embedding():
    model = _tiny_meta_model()
    converter._load_converted_state(model, _source_params(), torch.bfloat16)
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    paligemma = model.paligemma_with_expert.paligemma
    assert paligemma.lm_head.weight is paligemma.language_model.embed_tokens.weight
    assert torch.count_nonzero(model.paligemma_with_expert.gemma_expert.lm_head.weight) == 0


@pytest.mark.parametrize("error", ["missing", "unexpected", "shape"])
def test_strict_conversion_rejects_missing_unexpected_or_misshaped_trained_weights(error):
    params = _source_params()
    if error == "missing":
        params.pop("action_in_proj.bias")
    elif error == "unexpected":
        params["unknown.weight"] = torch.zeros(2)
    else:
        params["action_in_proj.weight"] = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="mismatch"):
        converter._load_converted_state(_tiny_meta_model(), params, torch.float32)

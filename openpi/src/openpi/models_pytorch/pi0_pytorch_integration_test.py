"""Exercise the installed Transformers extensions with small real HF backbones."""
# ruff: noqa: E402, SLF001 -- optional dependency gate and small-backbone test substitution.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers import GemmaConfig
from transformers import PaliGemmaConfig
from transformers import SiglipVisionConfig
from transformers.models.siglip import check

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch import pi0_pytorch


@pytest.fixture
def tiny_hf_model(monkeypatch):
    if not check.check_whether_transformers_replace_is_installed_correctly():
        pytest.fail("Install the repository Transformers extensions before running the HF integration tests.")
    small = SimpleNamespace(width=16, mlp_dim=32, num_heads=2, head_dim=8, depth=1, num_kv_heads=1)

    def pali_config():
        return PaliGemmaConfig(
            vision_config=SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=224,
                patch_size=112,
                vision_use_head=False,
            ),
            text_config=GemmaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            ),
        )

    monkeypatch.setattr(pi0_pytorch._gemma, "get_config", lambda variant: small)
    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": pali_config, "gemma": GemmaConfig})
    return pi0_pytorch.PI0Pytorch(
        Pi0Config(
            pi05=True,
            dtype="float32",
            max_token_len=12,
            max_subtask_len=4,
            subtask_loss_chunk_size=2,
            train_subtask_prediction=True,
            action_dim=3,
            action_horizon=2,
            pytorch_compile_mode=None,
        )
    )


def observation():
    image_keys = pi0_pytorch._preprocessing.IMAGE_KEYS
    return SimpleNamespace(
        images={key: torch.rand(1, 3, 224, 224) * 2 - 1 for key in image_keys},
        image_masks={key: torch.tensor([True]) for key in image_keys},
        tokenized_prompt=torch.tensor([[2, 3, 4, 5, 1, 6, 0, 0, 0, 0, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        token_ar_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        token_loss_mask=torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
        state=torch.randn(1, 3),
    )


@pytest.mark.parametrize("joint", [False, True])
def test_real_hf_train_backward_and_cached_inference(tiny_hf_model, joint):
    model = tiny_hf_model
    model.train_subtask_prediction = joint
    obs = observation()
    if joint:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
        obs.token_ar_mask = None
        obs.token_loss_mask = None
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=joint):
        loss, metrics = model(obs, torch.randn(1, 2, 3), return_metrics=True)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()
    assert ("subtask_loss" in metrics) == joint
    loss.mean().backward()
    assert model.action_out_proj.weight.grad is not None
    backbone = model.paligemma_with_expert
    assert backbone.paligemma.language_model.embed_tokens.weight.grad is not None
    assert backbone.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight.grad is not None
    assert not backbone.gemma_expert.lm_head.weight.requires_grad
    assert backbone.paligemma.lm_head.weight is backbone.paligemma.language_model.embed_tokens.weight
    model.eval()
    if joint:
        # Match create_trained_policy: inference stores the backbone in BF16
        # while selected norms/vision embeddings and action projections stay FP32.
        backbone.to_bfloat16_for_selected_params("bfloat16")
    actions = model.sample_actions("cpu", obs, num_steps=2)
    assert actions.shape == (1, 2, 3)
    assert torch.isfinite(actions).all()
    if joint:
        obs.tokenized_prompt_mask[:, 3:] = False
        obs.token_ar_mask[:] = False
        obs.token_loss_mask[:] = False
        result = model.generate_subtask("cpu", obs)
        assert result["valid_prefix"].item()
        length = result["lengths"].item()
        assert 2 <= length <= 4
        if result["has_eos"].item():
            assert result["tokens"][0, length - 1].item() == 1
        else:
            assert length == 4

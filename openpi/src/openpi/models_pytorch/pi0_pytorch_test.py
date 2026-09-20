"""Small CPU contract tests: real Torch autograd without a 3B model download."""
# ruff: noqa: SLF001 -- directly verify internal loss and checkpoint contracts.

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@pytest.fixture
def model_module(monkeypatch):
    # Only dependency boundaries are stubbed; all PI0Pytorch methods below are
    # loaded from the implementation, including masking, CE and checkpointing.
    importlib.import_module("openpi.models")
    importlib.import_module("openpi.models_pytorch")

    for name, values in {
        "openpi.models.gemma": {},
        "openpi.models_pytorch.gemma_pytorch": {"PaliGemmaWithExpertModel": object},
        "openpi.models_pytorch.preprocessing_pytorch": {
            "preprocess_observation_pytorch": lambda observation, **kwargs: observation
        },
    }.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
        parent_name, attribute = name.rsplit(".", 1)
        monkeypatch.setattr(sys.modules[parent_name], attribute, stub, raising=False)
    spec = importlib.util.spec_from_file_location("_torch_pi0_contract", Path(__file__).with_name("pi0_pytorch.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyBackbone(nn.Module):
    def __init__(self, width=8, vocab=19):
        super().__init__()
        self.paligemma = nn.Module()
        self.paligemma.language_model = nn.Module()
        self.paligemma.language_model.embed_tokens = nn.Embedding(vocab, width)
        self.paligemma.language_model.config = types.SimpleNamespace()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(width, width, bias=False)
        self.paligemma.language_model.layers = nn.ModuleList([layer])
        self.image_projection = nn.Linear(width, width)
        self.projection_sizes = []

    def embed_image(self, image):
        return self.image_projection(image)

    def embed_language_tokens(self, tokens):
        return self.paligemma.language_model.embed_tokens(tokens)

    def deembed_language_tokens(self, hidden):
        self.projection_sizes.append(hidden.shape[:-1])
        return F.linear(hidden, self.paligemma.language_model.embed_tokens.weight)

    def forward(self, *, inputs_embeds, attention_mask, **kwargs):
        prefix, suffix = inputs_embeds
        embeddings = prefix if suffix is None else torch.cat([prefix, suffix], dim=1)
        query = self.paligemma.language_model.layers[0].self_attn.q_proj(embeddings)
        attention = (query @ embeddings.transpose(-2, -1) / 8**0.5 + attention_mask[:, 0]).softmax(dim=-1)
        output = embeddings + attention @ embeddings
        return [output[:, : prefix.shape[1]], output[:, prefix.shape[1] :]], None


def make_model(module, *, subtask=True):
    model = module.PI0Pytorch.__new__(module.PI0Pytorch)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(action_dim=3, action_horizon=2)
    model.pi05 = True
    model.train_subtask_prediction = subtask
    model.subtask_action_conditioning = "task_and_subtask"
    model.train_subtask_executor = False
    model.max_subtask_len = 4
    model.subtask_loss_chunk_size = 2
    model.subtask_loss_weight = 0.7
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = TinyBackbone()
    model.action_in_proj = nn.Linear(3, 8)
    model.action_out_proj = nn.Linear(8, 3)
    model.time_mlp_in = nn.Linear(8, 8)
    model.time_mlp_out = nn.Linear(8, 8)
    return model


def make_observation():
    return types.SimpleNamespace(
        images={"camera": torch.randn(2, 2, 8)},
        image_masks={"camera": torch.tensor([True, True])},
        tokenized_prompt=torch.tensor([[2, 3, 4, 5, 1, 6, 0], [2, 3, 7, 1, 6, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        token_ar_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool),
        token_loss_mask=torch.tensor([[0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool),
        state=torch.randn(2, 3),
    )


def test_factorized_action_loss_and_gradients_do_not_depend_on_task(model_module):
    torch.manual_seed(17)
    model = make_model(model_module)
    model.subtask_action_conditioning = "subtask_only"
    obs = make_observation()
    obs.action_tokenized_prompt = torch.tensor([[2, 5, 1, 6], [2, 7, 1, 6]])
    obs.action_tokenized_prompt_mask = torch.ones(2, 4, dtype=torch.bool)
    actions, noise, time = torch.randn(2, 2, 3), torch.randn(2, 2, 3), torch.tensor([0.3, 0.7])
    _, metrics = model(obs, actions, noise=noise, time=time, return_metrics=True)
    params = tuple(model.parameters())
    grads = torch.autograd.grad(metrics["action_loss"], params, retain_graph=True, allow_unused=True)
    obs.tokenized_prompt[:, 1] = 18  # Change full task only; leave low-level context fixed.
    _, changed = model(obs, actions, noise=noise, time=time, return_metrics=True)
    torch.testing.assert_close(metrics["action_loss"], changed["action_loss"], rtol=0, atol=0)
    changed_grads = torch.autograd.grad(changed["action_loss"], params, allow_unused=True)
    for first, second in zip(grads, changed_grads, strict=True):
        # This fixture's tiny attention layer does not use adaptive time norms.
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.isclose(metrics["subtask_loss"], changed["subtask_loss"])
    # Both branches can backpropagate through the shared SigLIP computation.
    fresh_loss = model(obs, actions, noise=noise, time=time)
    fresh_loss.mean().backward()
    assert model.paligemma_with_expert.image_projection.weight.grad.abs().sum() > 0
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    obs.action_tokenized_prompt = None
    with pytest.raises(ValueError, match="action_tokenized_prompt"):
        model(obs, actions, noise=noise, time=time)


def test_prefix_masks_prevent_label_leakage_and_allow_action_conditioning(model_module):
    model = make_model(model_module)
    obs = make_observation()
    _, padding, ar = model.embed_prefix(
        list(obs.images.values()),
        list(obs.image_masks.values()),
        obs.tokenized_prompt,
        obs.tokenized_prompt_mask,
        obs.token_ar_mask,
    )
    # Append the bidirectional action block after the autoregressive text.
    padding = torch.cat([padding, torch.ones(2, 2, dtype=torch.bool)], dim=-1)
    ar = torch.cat([ar, torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)], dim=-1)
    mask = model_module.make_att_2d_masks(padding, ar)
    assert mask[0, 0, 4]  # Image can see the complete task.
    assert not mask[0, 4, 5]  # Last task token cannot see its label target.
    assert not mask[0, 5, 6]  # First label token cannot see the next token.
    assert not mask[0, 5, 9]  # Language cannot see noisy actions.
    assert mask[0, 9, 5]  # Actions see the complete label.
    assert mask[0, 9, 6]
    assert not mask[0, 9, 8]  # Padding remains invisible.


def test_chunked_ce_matches_full_projection_and_gradients(model_module):
    torch.manual_seed(3)
    model = make_model(model_module)
    obs = make_observation()
    obs.token_loss_mask[0, 2] = True  # Different label lengths, crossing chunk boundary.
    hidden = torch.randn(2, 9, 8, requires_grad=True)
    weight = model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight
    actual, accuracy, counts = model._compute_subtask_metrics(hidden, obs)
    text_hidden = hidden[:, -7:]
    logits = F.linear(text_hidden[:, :-1], weight)
    targets = obs.tokenized_prompt[:, 1:]
    mask = obs.token_loss_mask[:, 1:]
    full_ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
    expected = (full_ce * mask).sum(-1) / mask.sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(counts, torch.tensor([3, 2]))
    torch.testing.assert_close(accuracy, ((logits.argmax(-1) == targets) * mask).sum(-1) / mask.sum(-1))
    actual_grads = torch.autograd.grad(actual.sum(), (hidden, weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert max(size[1] for size in model.paligemma_with_expert.projection_sizes) <= 2
    assert len(model.paligemma_with_expert.projection_sizes) > 2  # Recomputed in backward.


@pytest.mark.parametrize("invalid", ["empty", "first", "padding", "too_long"])
def test_invalid_labels_fail_before_silent_zero_loss(model_module, invalid):
    model = make_model(model_module)
    obs = make_observation()
    if invalid == "empty":
        obs.token_loss_mask[0] = False
    elif invalid == "first":
        obs.token_loss_mask[0, 0] = True
    elif invalid == "padding":
        obs.token_loss_mask[0, -1] = True
    else:
        model.max_subtask_len = 1
    with pytest.raises(ValueError, match="Subtask labels"):
        model._compute_subtask_metrics(torch.randn(2, 9, 8), obs)


def test_masked_action_reduction_and_no_mask_regression(model_module):
    elementwise = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 200.0]],
            [[2.0, 4.0], [50.0, 60.0], [70.0, 80.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    masked, metrics = model_module.reduce_action_loss(elementwise, mask)

    expected = torch.tensor(
        [
            [[1.5, 4.5], [7.5, 10.5], [0.0, 0.0]],
            [[6.0, 12.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(masked, expected)
    torch.testing.assert_close(metrics["action_loss"], torch.tensor(3.5))
    torch.testing.assert_close(metrics["valid_action_fraction"], torch.tensor(0.5))
    unmasked, unmasked_metrics = model_module.reduce_action_loss(elementwise, None)
    torch.testing.assert_close(unmasked, elementwise)
    torch.testing.assert_close(unmasked_metrics["action_loss"], elementwise.mean())
    assert set(unmasked_metrics) == {"action_loss"}


def test_joint_forward_weights_losses_and_backpropagates(model_module):
    torch.manual_seed(8)
    model = make_model(model_module)
    obs = make_observation()
    actions = torch.randn(2, 2, 3)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.6])
    losses, metrics = model(obs, actions, noise=noise, time=time, return_metrics=True)
    torch.testing.assert_close(losses.mean(), metrics["action_loss"] + 0.7 * metrics["subtask_loss"])
    losses.mean().backward()
    for parameter in (
        model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight,
        model.paligemma_with_expert.image_projection.weight,
        model.action_out_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.train_subtask_prediction = False
    ordinary, ordinary_metrics = model(obs, actions, noise=noise, time=time, return_metrics=True)
    assert ordinary.shape == (2, 2, 3)
    torch.testing.assert_close(ordinary.mean(), metrics["action_loss"])
    assert set(ordinary_metrics) == {"action_loss"}


def test_executor_forward_reports_masked_action_metrics(model_module):
    torch.manual_seed(9)
    model = make_model(model_module, subtask=False)
    model.train_subtask_executor = True
    obs = make_observation()
    obs.token_ar_mask = None
    obs.token_loss_mask = None
    obs.action_loss_mask = torch.tensor([[True, False], [True, True]])
    losses, metrics = model(obs, torch.randn(2, 2, 3), return_metrics=True)

    assert losses.shape == (2, 2, 3)
    torch.testing.assert_close(losses[0, 1], torch.zeros(3))
    assert torch.count_nonzero(losses[1]) > 0
    assert set(metrics) == {"action_loss", "valid_action_fraction"}
    torch.testing.assert_close(metrics["valid_action_fraction"], torch.tensor(0.75))


def test_executor_valid_loss_is_independent_of_masked_future_targets(model_module):
    torch.manual_seed(10)
    model = make_model(model_module, subtask=False)
    model.train_subtask_executor = True
    obs = make_observation()
    obs.token_ar_mask = None
    obs.token_loss_mask = None
    obs.action_loss_mask = torch.tensor([[True, False], [True, False]])
    actions = torch.randn(2, 2, 3)
    changed_actions = actions.clone()
    changed_actions[:, 1] += 1000.0
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.6])

    original = model(obs, actions, noise=noise, time=time)
    changed = model(obs, changed_actions, noise=noise, time=time)

    torch.testing.assert_close(original[:, 0], changed[:, 0])
    torch.testing.assert_close(original[:, 1], torch.zeros_like(original[:, 1]))
    torch.testing.assert_close(changed[:, 1], torch.zeros_like(changed[:, 1]))


def test_checkpoint_keeps_embedding_gradients_with_integer_inputs(model_module):
    model = make_model(model_module)
    model.gradient_checkpointing_enabled = True
    tokens = torch.tensor([[2, 3]])
    output = model._apply_checkpoint(model.paligemma_with_expert.embed_language_tokens, tokens)
    output.sum().backward()
    grad = model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.grad
    assert grad[2:4].abs().sum() > 0


def test_bf16_autocast_keeps_master_parameters_and_loss_float32(model_module):
    model = make_model(model_module)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = model(make_observation(), torch.randn(2, 2, 3), return_metrics=True)
    assert loss.dtype == torch.float32
    assert metrics["subtask_loss"].dtype == torch.float32
    loss.mean().backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        if parameter.grad is not None:
            assert parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()


class GenerationBackbone(TinyBackbone):
    def __init__(self, *, finish=True):
        super().__init__()
        self.calls = []
        self.finish = finish

    def forward(self, *, inputs_embeds, attention_mask, position_ids, past_key_values=None, **kwargs):
        self.calls.append((attention_mask.clone(), position_ids.clone(), past_key_values))
        return [inputs_embeds[0], None], object()

    def deembed_language_tokens(self, hidden):
        logits = torch.zeros(hidden.shape[0], 19)
        logits[:, 0] = 100  # PAD must never be generated.
        logits[:, 2] = 99  # Nor BOS.
        logits[:, 1] = 90 if len(self.calls) == 1 or self.finish else 0
        logits[:, 4] = 80
        return logits


@pytest.mark.parametrize("finish", [True, False])
def test_generation_eos_limits_and_cache_masks(model_module, finish):
    model = make_model(model_module).eval()
    model.paligemma_with_expert = GenerationBackbone(finish=finish)
    obs = make_observation()
    obs.tokenized_prompt_mask[:] = False
    obs.tokenized_prompt_mask[0, :3] = True
    obs.tokenized_prompt_mask[1, :2] = True
    obs.token_ar_mask[:] = False
    obs.token_loss_mask[:] = False
    result = model.generate_subtask("cpu", obs)
    assert result["valid_prefix"].tolist() == [True, True]
    assert result["tokens"][:, 0].tolist() == [4, 4]  # First-token EOS is masked.
    assert result["has_eos"].tolist() == [finish, finish]
    assert result["lengths"].tolist() == ([2, 2] if finish else [4, 4])
    mask, positions, cache = model.paligemma_with_expert.calls[1]
    assert cache is not None
    assert positions.tolist() == [[5], [4]]  # Image + task, independent of right padding.
    assert mask.shape == (2, 1, 1, 10)
    assert mask[0, 0, 0, 9] == 0  # Appended generated token is visible.
    assert mask[0, 0, 0, 5] < -1e20  # Old text padding stays invisible.


def test_generation_rejects_teacher_forced_prefix(model_module):
    model = make_model(model_module).eval()
    model.paligemma_with_expert = GenerationBackbone()
    result = model.generate_subtask("cpu", make_observation())
    assert not result["valid_prefix"].any()
    assert not result["has_eos"].any()
    assert not result["lengths"].any()

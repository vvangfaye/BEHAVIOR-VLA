# ruff: noqa: SLF001 -- verify cached decoding against the real teacher-forced transformer.
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.shared import nnx_utils

pytest_plugins = ["openpi.models.subtask_test"]


def prefix_only(obs):
    lengths = np.argmax(np.asarray(obs.token_loss_mask), axis=1)
    mask = jnp.arange(obs.tokenized_prompt.shape[1])[None] < lengths[:, None]
    return dataclasses.replace(
        obs,
        tokenized_prompt=jnp.where(mask, obs.tokenized_prompt, 0),
        tokenized_prompt_mask=mask,
        token_ar_mask=jnp.zeros_like(obs.token_ar_mask),
        token_loss_mask=jnp.zeros_like(obs.token_loss_mask),
    )


def test_cached_token_logits_match_teacher_forcing_with_unequal_prefixes_and_masked_images(model_setup):
    model, obs, _, _ = model_setup
    obs = dataclasses.replace(obs, image_masks={**obs.image_masks, "left_wrist_0_rgb": jnp.array([False, True])})
    context = prefix_only(obs)
    embeddings, mask, ar = model.embed_prefix(context)
    (hidden, _), cache = model.PaliGemma.llm(
        [embeddings, None], positions=jnp.cumsum(mask, axis=1) - 1, mask=pi0.make_attn_mask(mask, ar)
    )
    rows = jnp.arange(2)
    starts = embeddings.shape[1] - context.tokenized_prompt.shape[1] + context.tokenized_prompt_mask.sum(axis=1)
    logits = model.PaliGemma.llm(hidden[rows, starts - 1][:, None], method="deembed")[:, 0]
    full, fullmask, fullar = model.embed_prefix(obs)
    (reference, _), _ = model.PaliGemma.llm(
        [full, None], positions=jnp.cumsum(fullmask, axis=1) - 1, mask=pi0.make_attn_mask(fullmask, fullar)
    )
    targets = [np.asarray(t)[np.asarray(m)] for t, m in zip(obs.tokenized_prompt, obs.token_loss_mask, strict=True)]
    shapes = [x.shape for x in cache]
    decode = nnx_utils.module_jit(model._decode_subtask_token)
    for i in range(max(map(len, targets))):
        active = jnp.array([i < len(t) for t in targets])
        expected = model.PaliGemma.llm(reference[rows, starts + i - 1][:, None], method="deembed")[:, 0]
        np.testing.assert_allclose(np.asarray(logits)[active], np.asarray(expected)[active], rtol=2e-4, atol=2e-5)
        tokens = jnp.array([t[i] if i < len(t) else 0 for t in targets])
        logits, cache, mask = decode(tokens, mask.sum(axis=1), starts + i, mask, cache, active)
        assert [x.shape for x in cache] == shapes


@pytest.mark.parametrize("emit_eos", [True, False])
def test_eos_and_budget_termination_under_jit(model_setup, monkeypatch, emit_eos):
    model, obs, _, _ = model_setup

    def constant_head(self, hidden):
        logits = jnp.zeros((*hidden.shape[:2], 64)).at[..., 4].set(5)
        return logits.at[..., 1].set(10 if emit_eos else -10)

    monkeypatch.setattr(gemma.Module, "deembed", constant_head)
    generate = nnx_utils.module_jit(model.generate_subtask, static_argnames=("max_decoding_steps",))
    result = generate(jax.random.key(0), prefix_only(obs), max_decoding_steps=4)
    np.testing.assert_array_equal(result["has_eos"], [emit_eos, emit_eos])
    np.testing.assert_array_equal(result["lengths"], [2 if emit_eos else 4] * 2)
    if emit_eos:
        np.testing.assert_array_equal(result["tokens"], [[4, 1, 0, 0]] * 2)
    invalid = generate(jax.random.key(0), obs, max_decoding_steps=4)
    assert not np.asarray(invalid["valid_prefix"]).any()
    assert not np.asarray(invalid["has_eos"]).any()


def test_training_to_free_generation_recovers_labels(model_setup, tmp_path):
    model, obs, actions, config = model_setup
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    @nnx.jit
    def step(model, optimizer):
        loss, grads = nnx.value_and_grad(lambda m: m.compute_loss(jax.random.key(7), obs, actions).mean())(model)
        optimizer.update(grads)
        return loss

    for _ in range(80):
        loss = step(model, optimizer)
    assert np.isfinite(loss)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / "params", {"params": nnx.state(model).to_pure_dict()})
    model = config.load(model_lib.restore_params(tmp_path / "params"))
    generated = nnx_utils.module_jit(model.generate_subtask)(jax.random.key(0), prefix_only(obs))
    assert np.asarray(generated["has_eos"]).all()
    for row in range(2):
        expected = np.asarray(obs.tokenized_prompt[row])[np.asarray(obs.token_loss_mask[row])]
        np.testing.assert_array_equal(generated["tokens"][row, : int(generated["lengths"][row])], expected)


def test_tokenizer_round_trip_matches_training_exactly_and_reserves_budget(tokenizer):
    full = tokenizer.tokenize_subtask_training("turn on radio", "press radio", np.zeros(4), max_subtask_len=8)
    ids = full[0][full[3]]
    repacked = tokenizer.tokenize_subtask_tokens("turn on radio", ids, np.zeros(4), max_subtask_len=8)
    for original, actual in zip(full, repacked, strict=True):
        np.testing.assert_array_equal(original, actual)
    prefix = tokenizer.tokenize_subtask_prefix("turn on radio", np.zeros(4), max_subtask_len=8)
    np.testing.assert_array_equal(prefix[0][prefix[1]], full[0][: np.flatnonzero(full[3])[0]])
    assert not prefix[2].any()
    assert not prefix[3].any()
    with pytest.raises(ValueError, match="max_token_len"):
        tokenizer.tokenize_subtask_prefix("radio " * 30, np.zeros(4), max_subtask_len=16)
    with pytest.raises(ValueError, match="final EOS"):
        tokenizer.tokenize_subtask_tokens("task", ids[:-1])


def test_bfloat16_parameters_and_cache_decode_under_jit(model_setup):
    _, obs, _, config = model_setup
    config = dataclasses.replace(config, dtype="bfloat16")
    model = config.create(jax.random.key(0))
    params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), nnx.state(model).to_pure_dict())
    model = config.load(params)
    result = nnx_utils.module_jit(model.generate_subtask)(jax.random.key(0), prefix_only(obs))
    assert np.asarray(result["valid_prefix"]).all()
    assert (np.asarray(result["lengths"]) >= 2).all()
    assert (np.asarray(result["lengths"]) <= config.max_subtask_len).all()

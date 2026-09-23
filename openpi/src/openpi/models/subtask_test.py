# ruff: noqa: SLF001 -- inspect internal masks/weights and use a local tokenizer fixture.
import dataclasses

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sentencepiece

from openpi import transforms
from openpi.configs import ROBOT_REGISTRY
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies.b1k_policy import B1KInputs
from openpi.shared import nnx_utils


@pytest.fixture
def tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Task: turn on radio, State: 128 255; Subtask: move to radio press radio Action: \n") * 20)
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "sp"),
        model_type="word",
        vocab_size=64,
        hard_vocab_limit=False,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        minloglevel=2,
    )
    result = tokenizer_lib.PaligemmaTokenizer.__new__(tokenizer_lib.PaligemmaTokenizer)
    result._max_len = 48
    result._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tmp_path / "sp.model"))
    return result


class TinyImageEncoder(nn.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        return nn.Dense(self.num_classes)(image.mean(axis=(1, 2)))[:, None], None


@pytest.fixture
def model_setup(monkeypatch, tokenizer):
    # Exercise the real Pi0/Gemma/ToNNX forward and backward, with a small
    # vocabulary and image encoder so this is a bounded CPU regression test.
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    monkeypatch.setattr(pi0._siglip, "Module", TinyImageEncoder)
    config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        max_token_len=48,
        train_subtask_prediction=True,
        max_subtask_len=8,
        subtask_loss_chunk_size=3,
    )
    arrays = [
        tokenizer.tokenize_subtask_training(task, label, np.zeros(4), max_subtask_len=8)
        for task, label in [("turn on radio", "move to radio"), ("radio", "press radio")]
    ]
    tokens, mask, ar, labels = [jnp.asarray(np.stack([a[i] for a in arrays])) for i in range(4)]
    obs = dataclasses.replace(
        config.fake_obs(2),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=mask,
        token_ar_mask=ar,
        token_loss_mask=labels,
    )
    model = config.create(jax.random.key(0))
    return model, obs, jnp.zeros((2, 2, 4)), config


def test_labels_include_first_word_and_eos_without_future_leakage(tokenizer):
    tokens, mask, ar, labels = tokenizer.tokenize_subtask_training("turn on radio", "move to radio", np.zeros(4))
    label_positions = np.flatnonzero(labels)
    assert tokens[label_positions[-1]] == tokenizer._tokenizer.eos_id()
    assert tokenizer._tokenizer.decode(tokens[labels].tolist()) == "move to radio"
    combined_ar = jnp.array([[0, 0, *ar, 1, 0]])
    combined_valid = jnp.array([[True, True, *mask, True, True]])
    attention = np.asarray(pi0.make_attn_mask(combined_valid, combined_ar))[0]
    for target in label_positions + 2:
        assert not attention[target - 1, target:].any()
    assert attention[-1, label_positions + 2].all()
    assert not attention[:-2, -2:].any()


def test_reject_empty_or_truncated_supervision(tokenizer):
    with pytest.raises(ValueError, match="non-empty"):
        tokenizer.tokenize_subtask_training("task", " ")
    with pytest.raises(ValueError, match="non-empty"):
        tokenizer.tokenize_subtask_training("task", "___")
    with pytest.raises(ValueError, match="max_subtask_len"):
        tokenizer.tokenize_subtask_training("task", "move to radio", max_subtask_len=2)
    with pytest.raises(ValueError, match="max_token_len"):
        tokenizer.tokenize_subtask_training("radio " * 50, "press radio")
    with pytest.raises(ValueError, match="finite"):
        tokenizer.tokenize_subtask_training("task", "press radio", np.array([np.nan]))
    with pytest.raises(ValueError, match="subtask is required"):
        transforms.TokenizeSubtaskPrompt(tokenizer)({"prompt": "task", "state": np.zeros(4)})


def test_b1k_input_adapter_preserves_separate_label(tokenizer):
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    state_dim = max(i for p in robot.proprio for i in p.indices) + 1
    raw = {
        "observation/state": np.zeros(state_dim),
        "prompt": "turn on radio",
        "subtask": "press radio",
        "actions": np.zeros((2, robot.action_dim)),
    }
    raw.update({f"observation/{key}": np.zeros((224, 224, 3), dtype=np.uint8) for key in robot.observations})
    data = B1KInputs(model_type=model_lib.ModelType.PI05, robot_config=robot)(raw)
    assert data["prompt"] == raw["prompt"]
    assert data["subtask"] == raw["subtask"]
    tokenizer._max_len = 128
    data = transforms.TokenizeSubtaskPrompt(tokenizer)(data)
    assert "prompt" not in data
    assert "subtask" not in data
    assert data["token_loss_mask"].sum() > 1


def test_chunked_ce_matches_dense_ce_and_gradients(model_setup):
    model, obs, _, config = model_setup
    hidden = jax.random.normal(jax.random.key(8), (2, config.max_token_len + 3, 64))

    def dense(h):
        logits = model.PaliGemma.llm(h[:, -config.max_token_len : -1], method="deembed").astype(jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(logp, obs.tokenized_prompt[:, 1:, None], axis=-1)[..., 0]
        mask = obs.token_loss_mask[:, 1:]
        return jnp.mean(jnp.sum(nll * mask, axis=-1) / jnp.sum(mask, axis=-1))

    def chunked(h):
        return jnp.mean(model._compute_subtask_metrics(h, obs)[0])

    np.testing.assert_allclose(chunked(hidden), dense(hidden), rtol=1e-5)
    np.testing.assert_allclose(jax.grad(chunked)(hidden), jax.grad(dense)(hidden), rtol=1e-5, atol=1e-6)
    bad_obs = dataclasses.replace(obs, token_loss_mask=jnp.zeros_like(obs.token_loss_mask))
    assert np.isnan(model._compute_subtask_metrics(hidden, bad_obs)[0]).all()


def with_action_context(obs, tokenizer):
    arrays = [
        tokenizer.tokenize_subtask_actions(np.asarray(tokens)[np.asarray(mask)], np.zeros(4), max_subtask_len=8)
        for tokens, mask in zip(obs.tokenized_prompt, obs.token_loss_mask, strict=True)
    ]
    return dataclasses.replace(
        obs,
        action_tokenized_prompt=jnp.asarray(np.stack([a[0] for a in arrays])),
        action_tokenized_prompt_mask=jnp.asarray(np.stack([a[1] for a in arrays])),
    )


@pytest.mark.parametrize("conditioning", ["task_and_subtask", "subtask_only"])
def test_joint_training_updates_both_experts_and_overfits(model_setup, tokenizer, conditioning):
    model, obs, actions, _ = model_setup
    model.subtask_action_conditioning = conditioning
    if conditioning == "subtask_only":
        obs = with_action_context(obs, tokenizer)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    @nnx.jit
    def step(model, optimizer):
        def objective(m):
            loss, metrics = m.compute_loss_with_metrics(jax.random.key(7), obs, actions)
            return loss.mean(), metrics

        (loss, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model)
        norms = (optax.global_norm(grads["PaliGemma"]["llm"]), optax.global_norm(grads["action_out_proj"]))
        optimizer.update(grads)
        return loss, metrics, norms

    initial, first_metrics, norms = step(model, optimizer)
    assert all(float(n) > 0 for n in norms)
    for _ in range(15):
        final, metrics, _ = step(model, optimizer)
    assert np.isfinite(final)
    assert float(final) < float(initial)
    assert float(metrics["subtask_loss"]) < float(first_metrics["subtask_loss"])
    assert float(metrics["subtask_token_accuracy"]) > float(first_metrics["subtask_token_accuracy"])


def test_factorized_actions_independent_of_task_in_train_and_inference(model_setup, tokenizer):
    model, obs, actions, config = model_setup
    model.subtask_action_conditioning = "subtask_only"
    new_config = dataclasses.replace(config, subtask_action_conditioning="subtask_only")
    assert new_config.inputs_spec()[0].action_tokenized_prompt is not None
    assert jax.tree.structure(nnx.state(model)) == jax.tree.structure(nnx.state(new_config.create(jax.random.key(0))))
    obs = with_action_context(obs, tokenizer)
    # Fresh adaRMS gates are zero, making the randomly initialized action expert
    # independent of *all* prefix tokens. Update once so the negative control
    # (changing the subtask must change actions) is meaningful.
    optimizer = nnx.Optimizer(model, optax.sgd(0.1))
    grads = nnx.grad(lambda m: m.compute_loss(jax.random.key(7), obs, actions).mean())(model)
    optimizer.update(grads)
    changed_task = dataclasses.replace(obs, tokenized_prompt=obs.tokenized_prompt.at[:, 1].set(42))
    loss_fn = nnx_utils.module_jit(model.compute_loss_with_metrics)
    _, metrics = loss_fn(jax.random.key(7), obs, actions)
    _, changed_metrics = loss_fn(jax.random.key(7), changed_task, actions)
    np.testing.assert_array_equal(metrics["action_loss"], changed_metrics["action_loss"])
    assert not np.isclose(metrics["subtask_loss"], changed_metrics["subtask_loss"])

    sample = nnx_utils.module_jit(model.sample_actions)
    prediction = sample(jax.random.key(3), obs, num_steps=2)
    np.testing.assert_array_equal(prediction, sample(jax.random.key(3), changed_task, num_steps=2))
    changed_subtask = dataclasses.replace(obs, action_tokenized_prompt=obs.action_tokenized_prompt.at[:, 3].set(41))
    assert not np.allclose(prediction, sample(jax.random.key(3), changed_subtask, num_steps=2))
    with pytest.raises(ValueError, match="action_tokenized_prompt"):
        model.sample_actions(jax.random.key(3), dataclasses.replace(obs, action_tokenized_prompt=None), num_steps=2)


def test_factorized_tokenizer_context_and_observation_round_trip(tokenizer):
    transform = transforms.TokenizeSubtaskPrompt(
        tokenizer, max_subtask_len=8, subtask_action_conditioning="subtask_only"
    )
    source = {"state": np.zeros(4), "prompt": "turn on radio", "subtask": "press radio"}
    first = transform(source)
    second = transform({**source, "prompt": "move to radio"})
    assert not np.array_equal(first["tokenized_prompt"], second["tokenized_prompt"])
    np.testing.assert_array_equal(first["action_tokenized_prompt"], second["action_tokenized_prompt"])
    text = tokenizer.decode_subtask(first["action_tokenized_prompt"][first["action_tokenized_prompt_mask"]])
    assert "Task:" not in text
    assert "press radio" in text
    first.update(image={}, image_mask={})
    observation = model_lib.Observation.from_dict(first)
    restored = model_lib.Observation.from_dict(observation.to_dict())
    np.testing.assert_array_equal(restored.for_subtask_actions().tokenized_prompt, first["action_tokenized_prompt"])


def test_action_only_api_and_parameter_shapes_are_compatible(model_setup):
    model, obs, actions, config = model_setup
    conditioned = nnx_utils.module_jit(model.sample_actions)(jax.random.key(1), obs, num_steps=2)
    assert conditioned.shape == (2, 2, 4)
    assert np.isfinite(conditioned).all()
    plain = dataclasses.replace(config, train_subtask_prediction=False).create(jax.random.key(0))
    assert jax.tree.structure(nnx.state(model)) == jax.tree.structure(nnx.state(plain))
    obs = dataclasses.replace(obs, token_ar_mask=None, token_loss_mask=None)
    with pytest.raises(ValueError, match="requires token_ar_mask"):
        model.compute_loss(jax.random.key(1), obs, actions)
    loss = nnx_utils.module_jit(plain.compute_loss)(jax.random.key(1), obs, actions)
    sampled = nnx_utils.module_jit(plain.sample_actions)(jax.random.key(1), obs, num_steps=2)
    assert loss.shape == (2, 2)
    assert np.isfinite(loss).all()
    assert sampled.shape == (2, 2, 4)
    assert np.isfinite(sampled).all()


def test_pi0_continuous_state_prefix_mask_regression(model_setup):
    _, obs, actions, config = model_setup
    plain = dataclasses.replace(config, pi05=False, discrete_state_input=False, train_subtask_prediction=False)
    model = plain.create(jax.random.key(0))
    obs = dataclasses.replace(obs, token_ar_mask=None, token_loss_mask=None)
    loss = nnx_utils.module_jit(model.compute_loss)(jax.random.key(1), obs, actions)
    sampled = nnx_utils.module_jit(model.sample_actions)(jax.random.key(1), obs, num_steps=2)
    assert loss.shape == (2, 2)
    assert np.isfinite(loss).all()
    assert sampled.shape == (2, 2, 4)
    assert np.isfinite(sampled).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pi05": False},
        {"subtask_loss_weight": 0},
        {"subtask_loss_weight": float("nan")},
        {"max_subtask_len": 1},
        {"max_subtask_len": 200},
        {"subtask_loss_chunk_size": 0},
    ],
)
def test_invalid_subtask_configs_fail_early(kwargs):
    with pytest.raises(ValueError, match=r"requires pi05|subtask_loss_weight|Require"):
        pi0_config.Pi0Config(**{"pi05": True, "train_subtask_prediction": True, **kwargs})

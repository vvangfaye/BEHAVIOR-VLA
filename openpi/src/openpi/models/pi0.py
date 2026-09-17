import logging
import math

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def reduce_action_loss(
    action_loss: at.Array, action_loss_mask: at.Array | None
) -> tuple[at.Array, dict[str, at.Array]]:
    """Mask invalid timesteps while giving every sample equal weight under an outer mean."""
    if action_loss_mask is None:
        return action_loss, {"action_loss": jnp.mean(action_loss)}
    if action_loss_mask.shape != action_loss.shape:
        raise ValueError(
            f"action_loss_mask must match per-timestep action loss shape {action_loss.shape}, "
            f"got {action_loss_mask.shape}."
        )
    mask = action_loss_mask.astype(jnp.bool_)
    valid_steps = jnp.sum(mask, axis=-1)
    scale = action_loss.shape[-1] / jnp.maximum(valid_steps, 1)
    normalized = jnp.where(mask, action_loss * scale[..., None], 0.0)
    normalized = jnp.where((valid_steps > 0)[..., None], normalized, jnp.nan)
    return normalized, {
        "action_loss": jnp.mean(normalized),
        "valid_action_fraction": jnp.mean(mask.astype(jnp.float32)),
    }


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.train_subtask_prediction = config.train_subtask_prediction
        self.train_subtask_executor = config.train_subtask_executor
        self.subtask_loss_weight = config.subtask_loss_weight
        self.max_subtask_len = config.max_subtask_len
        self.subtask_loss_chunk_size = config.subtask_loss_chunk_size
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Tasks/images are bidirectional; supervised subtask tokens are causal.
            ar_mask.append(
                jnp.zeros_like(obs.tokenized_prompt_mask)
                if obs.token_ar_mask is None
                else obs.token_ar_mask.astype(jnp.bool_)
            )
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_metrics(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_metrics(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Array, dict[str, at.Array]]:
        if self.train_subtask_prediction and (observation.token_ar_mask is None or observation.token_loss_mask is None):
            raise ValueError("Subtask training requires token_ar_mask and token_loss_mask from the subtask tokenizer.")
        if self.train_subtask_executor and observation.action_loss_mask is None:
            raise ValueError("Pure subtask executor training requires action_loss_mask.")
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        if observation.action_loss_mask is not None:
            valid_targets = observation.action_loss_mask.astype(jnp.bool_)[..., None]
            x_t = jnp.where(valid_targets, x_t, noise)
            u_t = jnp.where(valid_targets, u_t, 0.0)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.broadcast_to(suffix_ar_mask, suffix_mask.shape)], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        if observation.action_loss_mask is not None:
            attn_mask = attn_mask.at[:, :, -self.action_horizon :].set(
                attn_mask[:, :, -self.action_horizon :] & observation.action_loss_mask.astype(jnp.bool_)[:, None, :]
            )
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_timestep_action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        action_loss, metrics = reduce_action_loss(per_timestep_action_loss, observation.action_loss_mask)
        if not self.train_subtask_prediction:
            return action_loss, metrics
        subtask_loss, token_accuracy, counts = self._compute_subtask_metrics(prefix_out, observation)
        metrics.update(
            subtask_loss=jnp.mean(subtask_loss),
            subtask_token_accuracy=jnp.sum(token_accuracy * counts) / jnp.maximum(jnp.sum(counts), 1),
            subtask_tokens=jnp.mean(counts.astype(jnp.float32)),
        )
        return action_loss + self.subtask_loss_weight * subtask_loss[:, None], metrics

    def _compute_subtask_metrics(self, prefix_out: at.Array, observation: _model.Observation):
        """Next-token CE, gathering only label positions and rematerializing small vocab projections.

        Projecting every prompt/padding token to the 257k vocabulary can consume
        many GB per batch. A rematerialized scan bounds the live logits to
        [batch, subtask_loss_chunk_size, vocabulary], including during backward.
        """
        tokens = observation.tokenized_prompt
        loss_mask = observation.token_loss_mask.astype(jnp.bool_)
        counts = jnp.sum(loss_mask, axis=-1)
        positions = jax.vmap(lambda m: jnp.nonzero(m, size=self.max_subtask_len, fill_value=0)[0])(loss_mask)
        selected_mask = jnp.arange(self.max_subtask_len)[None, :] < counts[:, None]
        batch_indices = jnp.arange(tokens.shape[0])[:, None]
        text_hidden = prefix_out[:, -tokens.shape[1] :]
        hidden = text_hidden[batch_indices, jnp.maximum(positions - 1, 0)]
        targets = tokens[batch_indices, positions]

        chunk_size = self.subtask_loss_chunk_size
        padding = (-self.max_subtask_len) % chunk_size
        hidden = jnp.pad(hidden, ((0, 0), (0, padding), (0, 0)))
        targets = jnp.pad(targets, ((0, 0), (0, padding)))
        selected_mask = jnp.pad(selected_mask, ((0, 0), (0, padding)))
        chunks = (
            einops.rearrange(hidden, "b (n c) d -> n b c d", c=chunk_size),
            einops.rearrange(targets, "b (n c) -> n b c", c=chunk_size),
            einops.rearrange(selected_mask, "b (n c) -> n b c", c=chunk_size),
        )

        @jax.checkpoint
        def chunk_metrics(carry, chunk):
            h, target, mask = chunk
            logits = self.PaliGemma.llm(h, method="deembed").astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(logp, target[..., None], axis=-1).squeeze(-1)
            correct = jnp.argmax(logits, axis=-1) == target
            loss_sum, correct_sum = carry
            return (
                loss_sum + jnp.sum(jnp.where(mask, nll, 0), axis=-1),
                correct_sum + jnp.sum(mask & correct, axis=-1),
            ), None

        zeros = jnp.zeros((tokens.shape[0],), dtype=jnp.float32)
        (loss_sum, correct_sum), _ = jax.lax.scan(chunk_metrics, (zeros, zeros), chunks)
        # The data transform rejects malformed labels before JIT. Fail visibly
        # (instead of a plausible zero CE) for callers bypassing that transform.
        valid = (counts > 0) & (counts <= self.max_subtask_len) & ~loss_mask[:, 0]
        valid &= jnp.all(~loss_mask | observation.tokenized_prompt_mask, axis=-1)
        loss = jnp.where(valid, loss_sum / jnp.maximum(counts, 1), jnp.nan)
        return loss, correct_sum / jnp.maximum(counts, 1), counts

    def _decode_subtask_token(self, token, positions, write_indices, valid_mask, kv_cache, active):
        """Append one token using the existing Gemma cache API, then compact into fixed slots.

        Gemma returns [old cache, new KV]. Move that last entry into this row's
        reserved text slot so shapes stay constant inside the decoding loop.
        """
        embedded = self.PaliGemma.llm(token[:, None], method="embed")
        mask = jnp.concatenate([valid_mask, active[:, None]], axis=1) & active[:, None]
        (hidden, _), appended = self.PaliGemma.llm(
            [embedded, None], positions=positions[:, None], mask=mask[:, None, :], kv_cache=kv_cache
        )
        rows = jnp.arange(token.shape[0])

        def compact(cache):
            old = cache[:, :, :-1]
            entry = jnp.where(active[None, :, None, None], cache[:, :, -1], old[:, rows, write_indices])
            return old.at[:, rows, write_indices].set(entry)

        cache = jax.tree.map(compact, appended)
        valid_mask = valid_mask.at[rows, write_indices].set(active)
        logits = self.PaliGemma.llm(hidden, method="deembed")[:, 0].astype(jnp.float32)
        return logits, cache, valid_mask

    def generate_subtask(self, rng, observation, *, max_decoding_steps=None, temperature=0.0, eos_token_id=1):
        """Autoregressive text inference from task/images/state, with a fixed-size KV cache.

        Shape-changing decoding settings must be static when JIT-compiling.
        Returns tokens, lengths including EOS, and per-row EOS/prefix validity.
        Reaching the length limit does not fabricate EOS or a complete subtask.
        """
        if not self.train_subtask_prediction:
            raise ValueError("Subtask generation requires a model configured for subtask prediction.")
        limit = self.max_subtask_len if max_decoding_steps is None else max_decoding_steps
        if not 2 <= limit <= self.max_subtask_len:
            raise ValueError("Require 2 <= max_decoding_steps <= max_subtask_len (including EOS).")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Subtask generation requires a tokenized task/state prefix.")
        observation = _model.preprocess_observation(None, observation, train=False)
        text_mask = observation.tokenized_prompt_mask
        text_lengths = jnp.sum(text_mask, axis=1)
        valid_prefix = (text_lengths > 0) & (text_lengths + limit <= text_mask.shape[1])
        valid_prefix &= jnp.all(text_mask == (jnp.arange(text_mask.shape[1])[None, :] < text_lengths[:, None]), axis=1)
        if observation.token_ar_mask is not None:
            valid_prefix &= ~jnp.any(observation.token_ar_mask != 0, axis=1)
        if observation.token_loss_mask is not None:
            valid_prefix &= ~jnp.any(observation.token_loss_mask, axis=1)
        prefix, valid_mask, ar = self.embed_prefix(observation)
        positions = jnp.cumsum(valid_mask, axis=1) - 1
        (hidden, _), cache = self.PaliGemma.llm(
            [prefix, None], mask=make_attn_mask(valid_mask, ar), positions=positions
        )
        rows = jnp.arange(prefix.shape[0])
        starts = prefix.shape[1] - text_mask.shape[1] + text_lengths
        last_hidden = hidden[rows, starts - 1][:, None]
        logits = self.PaliGemma.llm(last_hidden, method="deembed")[:, 0].astype(jnp.float32)
        generated = jnp.zeros((prefix.shape[0], limit), dtype=jnp.int32)
        lengths = jnp.zeros((prefix.shape[0],), dtype=jnp.int32)

        def cond(carry):
            i, _, _, _, done, _, _, _ = carry
            return (i < limit) & ~jnp.all(done)

        def step(carry):
            i, cache, mask, logits, done, tokens, lengths, key = carry
            key, sample_key = jax.random.split(key)
            # PaliGemma's padding/unknown and BOS are not subtask text. Never
            # accept an empty EOS-only instruction.
            logits = logits.at[:, 0].set(-jnp.inf).at[:, 2].set(-jnp.inf)
            logits = logits.at[:, eos_token_id].set(jnp.where(i == 0, -jnp.inf, logits[:, eos_token_id]))
            token = (
                jnp.argmax(logits, axis=-1)
                if temperature == 0
                else jax.random.categorical(sample_key, logits / temperature)
            ).astype(jnp.int32)
            active = ~done
            token = jnp.where(active, token, 0)
            tokens = tokens.at[:, i].set(token)
            lengths = lengths + active.astype(jnp.int32)
            logits, cache, mask = self._decode_subtask_token(
                token, jnp.sum(mask, axis=1), jnp.minimum(starts + i, mask.shape[1] - 1), mask, cache, active
            )
            return i + 1, cache, mask, logits, done | (token == eos_token_id), tokens, lengths, key

        _, _, _, _, done, tokens, lengths, _ = jax.lax.while_loop(
            cond, step, (0, cache, valid_mask, logits, ~valid_prefix, generated, lengths, rng)
        )
        return {"tokens": tokens, "lengths": lengths, "has_eos": done & valid_prefix, "valid_prefix": valid_prefix}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

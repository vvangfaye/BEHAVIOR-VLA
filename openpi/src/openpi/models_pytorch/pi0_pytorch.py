import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        if getattr(config, "framesamp_modul", False):
            raise ValueError("FrameSamp + Modul is currently implemented for JAX only.")
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.train_subtask_prediction = config.train_subtask_prediction
        self.subtask_action_conditioning = config.subtask_action_conditioning
        self.subtask_loss_weight = config.subtask_loss_weight
        self.max_subtask_len = config.max_subtask_len
        self.subtask_loss_chunk_size = config.subtask_loss_chunk_size

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==5.5.4` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, lang_ar_masks=None, *, image_embeddings=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for index, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = (
                self._apply_checkpoint(image_embed_func, img) if image_embeddings is None else image_embeddings[index]
            )

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Images/task/state remain bidirectional; the subtask and action cue
        # use the same causal token boundaries as the JAX training format.
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks)).clone()
        if lang_ar_masks is not None:
            if lang_ar_masks.shape != lang_masks.shape:
                raise ValueError("token_ar_mask must have the same shape as tokenized_prompt_mask.")
            att_masks[:, -num_lang_embs:] = lang_ar_masks.to(dtype=torch.bool)

        return embs, pad_masks, att_masks

    @staticmethod
    def _subtask_action_tokens(observation):
        tokens = getattr(observation, "action_tokenized_prompt", None)
        mask = getattr(observation, "action_tokenized_prompt_mask", None)
        if tokens is None or mask is None:
            raise ValueError("subtask_only actions require action_tokenized_prompt and its mask.")
        return tokens, mask

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, *, return_metrics=False):
        """Return per-action loss, optionally with the action and language metrics.

        Subtask training teacher-forces only the subtask label with next-token
        CE. Its per-example loss is broadcast over actions so ``loss.mean()``
        has exactly the same action/CE weighting as the JAX trainer.
        """
        if self.train_subtask_prediction and (observation.token_ar_mask is None or observation.token_loss_mask is None):
            raise ValueError("Subtask training requires token_ar_mask and token_loss_mask from the subtask tokenizer.")
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        factorized = self.train_subtask_prediction and self.subtask_action_conditioning == "subtask_only"
        high_level_out = None
        image_embeddings = None
        lang_ar_masks = observation.token_ar_mask
        if factorized:
            action_tokens, action_masks = self._subtask_action_tokens(observation)
            # SigLIP has no language input. Its outputs can share an autograd
            # graph; task-conditioned LLM representations cannot be reused.
            image_embeddings = [self._apply_checkpoint(self.paligemma_with_expert.embed_image, img) for img in images]
            high_embs, high_mask, high_ar = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, lang_ar_masks, image_embeddings=image_embeddings
            )
            backbone_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            high_embs = high_embs.to(backbone_dtype)
            (high_level_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(high_mask, high_ar)),
                position_ids=torch.cumsum(high_mask, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[high_embs, None],
                use_cache=False,
            )
            lang_tokens, lang_masks, lang_ar_masks = action_tokens, action_masks, None

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_ar_masks, image_embeddings=image_embeddings
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        # The combined backbone checkpoints its individual layers. Wrapping the
        # whole backbone again adds a second full recomputation in backward.
        prefix_out, suffix_out = forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        action_loss = F.mse_loss(u_t.float(), v_t.float(), reduction="none")
        metrics = {"action_loss": action_loss.mean()}
        loss = action_loss
        if self.train_subtask_prediction:
            subtask_loss, token_accuracy, counts = self._compute_subtask_metrics(
                high_level_out if factorized else prefix_out, observation
            )
            metrics.update(
                subtask_loss=subtask_loss.mean(),
                subtask_token_accuracy=(token_accuracy * counts).sum() / counts.sum().clamp_min(1),
                subtask_tokens=counts.float().mean(),
            )
            loss = loss + self.subtask_loss_weight * subtask_loss[:, None, None]
        return (loss, metrics) if return_metrics else loss

    def _compute_subtask_metrics(self, prefix_out, observation):
        """Project only label predecessors, rematerializing each small vocab chunk.

        Checkpoint the projection and CE together even when backbone gradient
        checkpointing is disabled. Otherwise autograd retains all 257k-vocab
        softmax tensors until backward, defeating chunking's memory bound.
        """
        tokens = observation.tokenized_prompt
        loss_mask = observation.token_loss_mask.bool()
        if loss_mask.shape != tokens.shape or observation.tokenized_prompt_mask.shape != tokens.shape:
            raise ValueError("Subtask token masks must match tokenized_prompt.")
        counts = loss_mask.sum(dim=-1)
        valid = (counts > 0) & (counts <= self.max_subtask_len) & ~loss_mask[:, 0]
        valid &= (~loss_mask | observation.tokenized_prompt_mask.bool()).all(dim=-1)
        if not bool(valid.all()):
            raise ValueError(
                "Subtask labels must contain 1..max_subtask_len valid tokens with a preceding input token."
            )

        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        positions = positions.masked_fill(~loss_mask, tokens.shape[1]).sort(dim=-1).values[:, : self.max_subtask_len]
        selected_mask = positions < tokens.shape[1]
        positions = positions.clamp_max(tokens.shape[1] - 1)
        rows = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
        text_hidden = prefix_out[:, -tokens.shape[1] :]
        hidden = text_hidden[rows, (positions - 1).clamp_min(0)]
        targets = tokens[rows, positions].long()

        def chunk_metrics(chunk_hidden, chunk_targets, chunk_mask):
            logits = self.paligemma_with_expert.deembed_language_tokens(chunk_hidden).float()
            nll = F.cross_entropy(logits.flatten(0, 1), chunk_targets.flatten(), reduction="none")
            nll = nll.reshape_as(chunk_targets)
            correct = (logits.argmax(dim=-1) == chunk_targets) & chunk_mask
            return (nll * chunk_mask).sum(dim=-1), correct.sum(dim=-1).float()

        loss_sum = torch.zeros(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        correct_sum = torch.zeros_like(loss_sum)
        for start in range(0, hidden.shape[1], self.subtask_loss_chunk_size):
            stop = start + self.subtask_loss_chunk_size
            args = hidden[:, start:stop], targets[:, start:stop], selected_mask[:, start:stop]
            if torch.is_grad_enabled():
                chunk_loss, chunk_correct = checkpoint(chunk_metrics, *args, use_reentrant=False)
            else:
                chunk_loss, chunk_correct = chunk_metrics(*args)
            loss_sum = loss_sum + chunk_loss
            correct_sum = correct_sum + chunk_correct
        return loss_sum / counts, correct_sum / counts, counts

    @torch.no_grad()
    def generate_subtask(self, device, observation, *, max_decoding_steps=None, temperature=0.0, eos_token_id=1):
        """Decode a subtask from images and the complete task/state prefix.

        The returned lengths include EOS. Rows that exhaust the token budget
        retain ``has_eos=False`` so the policy cannot treat truncated text as a
        completed instruction. The KV cache is local to this generation call.
        """
        if not self.train_subtask_prediction:
            raise ValueError("Subtask generation requires a model configured for subtask prediction.")
        limit = self.max_subtask_len if max_decoding_steps is None else max_decoding_steps
        if not isinstance(limit, int) or not 2 <= limit <= self.max_subtask_len:
            raise ValueError("Require 2 <= max_decoding_steps <= max_subtask_len (including EOS).")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Subtask generation requires a tokenized task/state prefix.")

        images, img_masks, tokens, text_mask, _ = self._preprocess_observation(observation, train=False)
        text_mask = text_mask.bool()
        text_lengths = text_mask.sum(dim=-1)
        valid_prefix = (text_lengths > 0) & (text_lengths + limit <= text_mask.shape[1])
        valid_prefix &= (
            text_mask == (torch.arange(text_mask.shape[1], device=tokens.device)[None] < text_lengths[:, None])
        ).all(dim=-1)
        if observation.token_ar_mask is not None:
            valid_prefix &= ~(observation.token_ar_mask != 0).any(dim=-1)
        if observation.token_loss_mask is not None:
            valid_prefix &= ~observation.token_loss_mask.bool().any(dim=-1)

        prefix, valid_mask, ar_mask = self.embed_prefix(images, img_masks, tokens, text_mask)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (hidden, _), cache = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(valid_mask, ar_mask)),
            position_ids=valid_mask.long().cumsum(dim=-1) - 1,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        last_positions = prefix.shape[1] - text_mask.shape[1] + text_lengths - 1
        logits = self.paligemma_with_expert.deembed_language_tokens(hidden[rows, last_positions.clamp_min(0)]).float()
        if not 0 <= eos_token_id < logits.shape[-1] or eos_token_id in (0, 2):
            raise ValueError("eos_token_id must be a vocabulary token distinct from PAD/UNK and BOS.")
        generated = torch.zeros((tokens.shape[0], limit), dtype=torch.long, device=tokens.device)
        lengths = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        done = ~valid_prefix
        has_eos = torch.zeros_like(valid_prefix)

        for index in range(limit):
            if bool(done.all()):
                break
            logits = logits.clone()
            logits[:, 0] = -torch.inf
            logits[:, 2] = -torch.inf
            if index == 0:
                logits[:, eos_token_id] = -torch.inf
            if temperature == 0:
                token = logits.argmax(dim=-1)
            else:
                token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            active = ~done
            token = torch.where(active, token, 0)
            generated[:, index] = token
            lengths += active.long()
            has_eos |= active & (token == eos_token_id)
            done |= has_eos
            if index + 1 == limit or bool(done.all()):
                break

            position_ids = valid_mask.sum(dim=-1)[:, None]
            valid_mask = torch.cat([valid_mask, active[:, None]], dim=-1)
            embedded = self.paligemma_with_expert.embed_language_tokens(token[:, None])
            embedded = embedded * math.sqrt(embedded.shape[-1])
            (hidden, _), cache = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d((valid_mask & active[:, None])[:, None, :]),
                position_ids=position_ids,
                inputs_embeds=[embedded, None],
                past_key_values=cache,
                use_cache=True,
            )
            logits = self.paligemma_with_expert.deembed_language_tokens(hidden[:, 0]).float()
        return {"tokens": generated, "lengths": lengths, "has_eos": has_eos, "valid_prefix": valid_prefix}

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        lang_ar_masks = observation.token_ar_mask
        if self.train_subtask_prediction and self.subtask_action_conditioning == "subtask_only":
            lang_tokens, lang_masks = self._subtask_action_tokens(observation)
            lang_ar_masks = None
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, lang_ar_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

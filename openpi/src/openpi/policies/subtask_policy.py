"""Task -> generated subtask -> continuous actions, using the training token format."""

from collections.abc import Sequence
import contextlib
import logging
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi import transforms as transforms_lib
from openpi.models import model as model_lib
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils


class SubtaskPolicy(Policy):
    def __init__(
        self,
        model,
        *,
        tokenizer_transform: transforms_lib.TokenizeSubtaskPrompt,
        transforms: Sequence[transforms_lib.DataTransformFn] = (),
        subtask_interval: int = 1,
        subtask_temperature: float = 0.0,
        **kwargs,
    ):
        if not model.train_subtask_prediction:
            raise ValueError("SubtaskPolicy requires a subtask-prediction model.")
        if not isinstance(subtask_interval, int) or subtask_interval < 1:
            raise ValueError("subtask_interval must be a positive number of action-chunk requests.")
        if not math.isfinite(subtask_temperature) or subtask_temperature < 0:
            raise ValueError("subtask_temperature must be finite and nonnegative.")
        conditioning = getattr(model, "subtask_action_conditioning", "task_and_subtask")
        if conditioning != tokenizer_transform.subtask_action_conditioning:
            raise ValueError("Model and tokenizer subtask_action_conditioning must match.")
        super().__init__(model, **kwargs)
        self._observation_transform = transforms_lib.compose(transforms)
        self._tokenizer_transform = tokenizer_transform
        self._tokenizer = tokenizer_transform.tokenizer
        self._pad = transforms_lib.PadStatesAndActions(model.action_dim)
        self._interval = subtask_interval
        self._temperature = subtask_temperature
        self._generate = (
            model.generate_subtask
            if self._is_pytorch_model
            else nnx_utils.module_jit(
                model.generate_subtask, static_argnames=("max_decoding_steps", "temperature", "eos_token_id")
            )
        )
        self._metadata = {
            **self._metadata,
            "subtask_generation": True,
            "subtask_interval": subtask_interval,
            "subtask_action_conditioning": conditioning,
        }
        self.reset()

    def reset(self):
        super().reset()
        self.reset_subtask()

    def reset_subtask(self):
        self._cached_tokens = None
        self._last_prompt = None
        self._calls_since_generation = 0

    @staticmethod
    def _pack(inputs, arrays):
        result = {k: v for k, v in inputs.items() if k not in ("prompt", "subtask")}
        result.update(
            zip(("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"), arrays, strict=True)
        )
        return result

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        result = self._infer_candidates(obs, 1, noise=noise)
        result["actions"] = result["actions"][0]
        return result

    def infer_candidates(self, obs: dict, count: int) -> dict:
        """Sample independent action chunks under one subtask decision.

        Candidate sampling advances the action RNG for each chunk, while the
        subtask refresh interval advances once for the whole decision.
        """
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        return self._infer_candidates(obs, count)

    def _infer_candidates(self, obs: dict, count: int, *, noise: np.ndarray | None = None) -> dict:
        started = time.monotonic()
        inputs = self._observation_transform(jax.tree.map(lambda x: x, obs))
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str):
            prompt = np.asarray(prompt).item()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A non-empty task prompt is required for subtask inference.")
        if self._last_prompt is not None and prompt != self._last_prompt:
            self.reset()
        self._last_prompt = prompt
        state = inputs["state"] if self._tokenizer_transform.discrete_state_input else None
        limit = self._tokenizer_transform.max_subtask_len
        generation_ms = 0.0
        if "subtask" in inputs:
            # Keep explicit subtask control, including oracle evaluation, separate
            # from generation. It cannot seed a later autonomous cached subtask.
            self._cached_tokens = None
            prepared = self._tokenizer_transform(inputs)
            label_tokens = prepared["tokenized_prompt"][prepared["token_loss_mask"]]
            source = "provided"
        else:
            if self._cached_tokens is None or self._calls_since_generation >= self._interval:
                arrays = self._tokenizer.tokenize_subtask_prefix(prompt, state, max_subtask_len=limit)
                prefix = self._pad(self._pack(inputs, arrays))
                if self._is_pytorch_model:
                    prefix = model_lib.Observation.from_dict(
                        jax.tree.map(lambda x: torch.as_tensor(np.array(x), device=self._pytorch_device)[None], prefix)
                    )
                    key = self._pytorch_device
                else:
                    prefix = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], prefix))
                    self._rng, key = jax.random.split(self._rng)
                generation_start = time.monotonic()
                with torch.inference_mode() if self._is_pytorch_model else contextlib.nullcontext():
                    result = self._generate(
                        key,
                        prefix,
                        max_decoding_steps=limit,
                        temperature=self._temperature,
                        eos_token_id=self._tokenizer.eos_id,
                    )
                if self._is_pytorch_model:
                    result = jax.tree.map(lambda x: x.detach().cpu().numpy(), result)
                else:
                    result = jax.device_get(result)
                generation_ms = (time.monotonic() - generation_start) * 1000
                if not bool(result["valid_prefix"][0]) or not bool(result["has_eos"][0]):
                    self._cached_tokens = None
                    raise RuntimeError(
                        "Subtask generation did not finish with EOS within max_subtask_len; no actions were sampled. "
                        "Check the subtask-trained checkpoint and token budget."
                    )
                label_tokens = result["tokens"][0, : int(result["lengths"][0])]
                # Validate before committing any cache state.
                arrays = self._tokenizer.tokenize_subtask_tokens(prompt, label_tokens, state, max_subtask_len=limit)
                self._cached_tokens = label_tokens.copy()
                self._calls_since_generation = 0
                source = "generated"
                logging.info("Generated subtask: %s", self._tokenizer.decode_subtask(label_tokens))
            else:
                label_tokens = self._cached_tokens
                arrays = self._tokenizer.tokenize_subtask_tokens(prompt, label_tokens, state, max_subtask_len=limit)
                source = "cached"
            prepared = self._pack(inputs, arrays)
        # Rebuild from current state and the exact label on every action request,
        # including cached and externally provided subtasks. No high-level KV
        # cache is passed to sample_actions.
        prepared.update(self._tokenizer_transform.action_fields(label_tokens, state))
        outputs = [super(SubtaskPolicy, self).infer(self._pad(prepared), noise=noise) for _ in range(count)]
        self._calls_since_generation += 1
        result = outputs[0]
        result["actions"] = np.stack([item["actions"] for item in outputs])
        # Add text AFTER action-only output adapters so it survives B1KOutputs.
        result["subtask"] = self._tokenizer.decode_subtask(label_tokens)
        result["subtask_source"] = source
        result["policy_timing"]["infer_ms"] = sum(item["policy_timing"]["infer_ms"] for item in outputs)
        result["policy_timing"]["subtask_ms"] = generation_ms
        result["policy_timing"]["total_ms"] = (time.monotonic() - started) * 1000
        return result

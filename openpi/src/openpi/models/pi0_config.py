import dataclasses
import math
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Supervise intermediate language as well as continuous actions (JAX and Torch).
    train_subtask_prediction: bool = False
    # Keep the old default for checkpoint compatibility. Joint launchers select
    # subtask_only: p(s | observation, task) p(actions | observation, s).
    subtask_action_conditioning: Literal["task_and_subtask", "subtask_only"] = "task_and_subtask"
    # Train a pure current-subtask executor with temporal action-loss masking.
    train_subtask_executor: bool = False
    subtask_loss_weight: float = 1.0
    max_subtask_len: int = 64  # Includes EOS.
    subtask_loss_chunk_size: int = 8

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.subtask_action_conditioning not in ("task_and_subtask", "subtask_only"):
            raise ValueError("Unknown subtask_action_conditioning.")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.train_subtask_prediction and self.train_subtask_executor:
            raise ValueError("train_subtask_executor is mutually exclusive with train_subtask_prediction.")
        if self.train_subtask_executor and not self.pi05:
            raise ValueError("Pure subtask executor training requires pi05=True.")
        if self.train_subtask_prediction:
            if not self.pi05:
                raise ValueError("Subtask prediction training requires pi05=True.")
            if not math.isfinite(self.subtask_loss_weight) or self.subtask_loss_weight <= 0:
                raise ValueError("subtask_loss_weight must be finite and positive.")
            if not 1 < self.max_subtask_len < self.max_token_len:
                raise ValueError("Require 1 < max_subtask_len < max_token_len (including EOS).")
            if not 1 <= self.subtask_loss_chunk_size <= self.max_subtask_len:
                raise ValueError("Require 1 <= subtask_loss_chunk_size <= max_subtask_len.")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.train_subtask_prediction
                    else None
                ),
                token_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)
                    if self.train_subtask_prediction
                    else None
                ),
                action_tokenized_prompt=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.train_subtask_prediction and self.subtask_action_conditioning == "subtask_only"
                    else None
                ),
                action_tokenized_prompt_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool)
                    if self.train_subtask_prediction and self.subtask_action_conditioning == "subtask_only"
                    else None
                ),
                action_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon], bool)
                    if self.train_subtask_executor
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared.frame_memory import FIELDS as MEMORY_FIELDS

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._frame_memory = None
        self._external_memory_observations = False
        self._pending_memory_ms = 0.0
        if getattr(model, "framesamp_modul", False):
            if is_pytorch:
                raise ValueError("FrameSamp + Modul is supported only by the JAX policy.")
            from openpi.policies.frame_memory import FrameMemoryBuffer
            self._frame_memory = FrameMemoryBuffer(model)
            self._metadata = {**self._metadata, "framesamp_modul": True, "memory_frame_stride": model.memory_frame_stride}

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(0)

    def reset(self):
        if self._frame_memory is not None:
            self._frame_memory.reset()
        self._external_memory_observations = False
        self._pending_memory_ms = 0.0

    def observe_memory(self, obs, *, step):
        """B1K supplies every control observation, including between action requests."""
        if self._frame_memory is None:
            return
        camera_index = _model.IMAGE_KEYS.index(self._frame_memory.camera)
        started = time.monotonic()
        self._frame_memory.add(obs[f"observation/image_{camera_index}"], step)
        self._pending_memory_ms += (time.monotonic() - started) * 1000
        self._external_memory_observations = True

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        memory_started = time.monotonic()
        if self._frame_memory is not None:
            supplied = [key in inputs for key in MEMORY_FIELDS]
            if any(supplied) and not all(supplied):
                raise ValueError("Provide all memory fields together for offline inference.")
            if not any(supplied):
                if not self._external_memory_observations:
                    self._frame_memory.add(inputs["image"][self._frame_memory.camera], self._frame_memory.next_step)
                inputs.update(self._frame_memory.pack())
        memory_ms = (time.monotonic() - memory_started) * 1000 + self._pending_memory_ms
        self._pending_memory_ms = 0.0
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # Robot output transforms may modify actions in place. JAX's NumPy
            # view can be read-only, so materialize writable host arrays here.
            outputs = jax.tree.map(lambda x: np.array(x[0, ...], copy=True), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if self._frame_memory is not None:
            outputs["policy_timing"]["memory_ms"] = memory_ms
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @property
    def metadata(self):
        return self._policy.metadata

    def reset(self):
        self._policy.reset()

    def observe_memory(self, obs, *, step):
        self._policy.observe_memory(obs, step=step)

    def reset_subtask(self):
        if hasattr(self._policy, "reset_subtask"):
            self._policy.reset_subtask()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

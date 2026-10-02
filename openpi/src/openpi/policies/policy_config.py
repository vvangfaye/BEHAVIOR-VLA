import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
from openpi.policies.subtask_policy import SubtaskPolicy
from openpi.shared import frame_memory
from openpi.shared import subtask_contract
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    subtask_interval: int | None = None,
    subtask_temperature: float = 0.0,
    subtask_action_conditioning: str | None = None,
    subtask_vote_window: int | None = None,
    subtask_vote_threshold: int | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    if (subtask_vote_window is not None or subtask_vote_threshold is not None) and not getattr(
        train_config.model, "train_subtask_prediction", False
    ):
        raise ValueError("Temporal subtask voting requires a subtask-prediction model.")
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    memory_contract = frame_memory.load(checkpoint_dir / "assets")
    if memory_contract is not None:
        expected = frame_memory.model_contract(train_config.model)
        if expected is not None and any(memory_contract[key] != value for key, value in expected.items()):
            raise ValueError("Requested memory settings disagree with checkpoint metadata.")
        train_config = dataclasses.replace(train_config, model=dataclasses.replace(
            train_config.model, framesamp_modul=True, memory_num_frames=memory_contract["num_frames"],
            memory_grid_size=memory_contract["grid_size"], memory_frame_stride=memory_contract["frame_stride"],
            memory_camera=memory_contract["camera"],
        ))
    elif getattr(train_config.model, "framesamp_modul", False):
        raise ValueError("Memory inference requires a trained checkpoint with frame_memory.json metadata.")

    if getattr(train_config.model, "train_subtask_prediction", False):
        saved_mode = subtask_contract.load(checkpoint_dir / "assets")
        if (
            subtask_action_conditioning is not None
            and saved_mode is not None
            and subtask_action_conditioning != saved_mode
        ):
            raise ValueError("Requested subtask action conditioning disagrees with the checkpoint metadata.")
        mode = subtask_action_conditioning or saved_mode or train_config.model.subtask_action_conditioning
        if saved_mode is None:
            logging.warning("Checkpoint has no joint conditioning metadata; using %s.", mode)
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, subtask_action_conditioning=mode)
        )
        if subtask_interval is None:
            # Engineering default, not a paper-specified frequency. One request
            # corresponds to the action execution horizon configured by the client.
            subtask_interval = 2 if mode == subtask_contract.FACTORIZED else 1
        logging.info("Joint action conditioning=%s, subtask refresh every %s requests", mode, subtask_interval)

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    policy_cls = _policy.Policy
    subtask_kwargs = {}
    if getattr(train_config.model, "train_subtask_prediction", False):
        tokenizers = [t for t in input_transforms if isinstance(t, transforms.TokenizeSubtaskPrompt)]
        if len(tokenizers) != 1:
            raise ValueError("Subtask inference requires exactly one matching training tokenizer transform.")
        input_transforms = [
            t
            for t in input_transforms
            if not isinstance(t, transforms.TokenizeSubtaskPrompt | transforms.PadStatesAndActions)
        ]
        policy_cls = SubtaskPolicy
        subtask_kwargs = {
            "tokenizer_transform": tokenizers[0],
            "subtask_interval": subtask_interval,
            "subtask_temperature": subtask_temperature,
            "subtask_vote_window": subtask_vote_window,
            "subtask_vote_threshold": subtask_vote_threshold,
        }
    return policy_cls(
        model,
        transforms=input_transforms,
        **subtask_kwargs,
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

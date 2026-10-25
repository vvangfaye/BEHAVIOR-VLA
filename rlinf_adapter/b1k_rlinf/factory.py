"""Build a joint policy under RLinf's existing openpi model registration."""

import dataclasses
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .action_interface import ExecutionActionInterface


def build_action_interface(cfg):
    tolerance = OmegaConf.select(cfg, "b1k_joint.execution_gripper_open_tolerance", default=0.0)
    return ExecutionActionInterface(gripper_open_tolerance=tolerance)


def match_jax_action_head_precision(model):
    """Match JAX's BF16 checkpoint restore while retaining FP32 computation.

    RLinf already rounds the VLM to BF16 before restoring its FP32 islands.
    Its action/time projections are outside that module and otherwise keep
    the original FP32 checkpoint values. JAX restores those to BF16 as well.
    This is only for an initial base-checkpoint load, never trained weights.
    """
    prefixes = ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
    names = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes):
                parameter.copy_(parameter.to(torch.bfloat16).to(parameter.dtype))
                names.append(name)
    if len(names) != 8:
        raise ValueError(f"Expected eight pi0.5 action/time projection tensors, got {names}")
    return names


def build_rl_config(cfg):
    """Match the native builder's top-level value-head switch."""
    from rlinf.models.embodiment.openpi.tasks.rl import Pi0RLConfig

    names = {field.name for field in dataclasses.fields(Pi0RLConfig)}
    values = {k: v for k, v in OmegaConf.to_container(cfg.openpi, resolve=True).items() if k in names}
    values["add_value_head"] = bool(
        OmegaConf.select(cfg, "add_value_head", default=values.get("add_value_head", False))
    )
    return Pi0RLConfig(**values)


def build_model(cfg, torch_dtype=None):
    from rlinf.models.embodiment.openpi import (
        _apply_openpi_param_dtypes,
        get_model,
    )

    joint = OmegaConf.select(cfg, "b1k_joint", default=None)
    if joint is None or not joint.get("enabled", False):
        return get_model(cfg, torch_dtype)

    from rlinf.models.embodiment.openpi.checkpoint import (
        load_base_safetensors,
        load_full_weights,
        resolve_full_weights,
        resolve_model_safetensors,
    )
    from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    from .io import CheckpointIO
    from .model import JointPi0RL
    from .tokenizer import JointTokenizer

    if not cfg.pi05 or not cfg.openpi.train_expert_only:
        raise ValueError("Joint adapter requires pi05=True and train_expert_only=True")
    if joint.get("subtask_interval", 1) != 1 or joint.get("subtask_temperature", 0) != 0:
        raise ValueError("Initial adapter supports the baseline's interval=1, greedy subtasks")
    if cfg.openpi.noise_method != "flow_sde" or cfg.openpi.joint_logprob:
        raise ValueError("Initial adapter requires Flow-SDE with joint_logprob=False")
    expected = (
        int(cfg.openpi.model_action_dim),
        int(cfg.openpi.action_horizon),
        int(cfg.action_dim),
        int(cfg.num_action_chunks),
        int(cfg.openpi.max_token_len),
    )
    if expected != (32, 32, 23, 16, 512):
        raise ValueError(
            f"Checkpoint requires model/horizon/env/chunk/tokens=(32,32,23,16,512), got {expected}"
        )
    model_path = Path(cfg.model_path).expanduser()
    contract_path = Path(joint.get("contract_path", model_path / "b1k_checkpoint.json"))
    contract = json.loads(contract_path.read_text())
    if contract.get("action_conditioning") != "task_and_subtask":
        raise ValueError("Expected a verified task_and_subtask checkpoint contract")
    io = CheckpointIO(joint.norm_stats_path, state_layout=joint.state_layout)
    if io.sha256 != contract.get("norm_sha256"):
        raise ValueError("Normalization does not match the exported checkpoint")
    tokenizer = JointTokenizer(joint.tokenizer_path, max_len=512, max_subtask_len=64)
    set_torch_compile(False)
    embed_dtype = "float32" if torch_dtype == torch.float32 else "bfloat16"
    model_cfg = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=32,
        max_token_len=512,
        dtype=embed_dtype,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    rl_cfg = build_rl_config(cfg)
    model = JointPi0RL(
        model_cfg,
        num_steps=int(cfg.num_steps),
        action_chunk=16,
        action_env_dim=23,
        rl_cfg=rl_cfg,
        config_name="pi05_behavior",
    )
    model.install_joint_io(io, tokenizer, build_action_interface(cfg))
    full = resolve_full_weights(model_path)
    weights = resolve_model_safetensors(model_path)
    if full is not None:
        load_full_weights(model, full, expect_rlt=False)
    elif weights is not None:
        load_base_safetensors(model, weights)
        if joint.get("match_jax_restore_precision", False):
            if torch_dtype not in (None, torch.bfloat16):
                raise ValueError("JAX BF16 restore matching requires BF16 policy precision")
            model.jax_restore_rounded_heads = match_jax_action_head_precision(model)
    else:
        raise FileNotFoundError(f"No PyTorch policy weights under {model_path}")
    model.freeze_vlm()
    _apply_openpi_param_dtypes(model, torch_dtype)
    if joint.get("match_jax_compute", joint.get("match_jax_restore_precision", False)):
        from .precision import install_jax_primitives

        model.jax_precision_modules = install_jax_primitives(model)
        torch.set_float32_matmul_precision("highest")
    return model

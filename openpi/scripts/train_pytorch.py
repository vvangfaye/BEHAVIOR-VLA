"""PyTorch fine-tuning with FP32 weights, BF16 autocast, accumulation and DDP.

``batch_size`` is the global microbatch, divided across torchrun workers. An
optimizer update consumes ``batch_size * pytorch_gradient_accumulation_steps``
samples. Checkpoint step names count completed optimizer updates.

Single GPU:
    python scripts/b1k/train_b1k_pytorch.py pi05_b1k_pytorch --exp-name run \
        --pytorch-weight-path /path/to/converted_base
Multiple GPUs:
    torchrun --standalone --nproc-per-node=2 scripts/b1k/train_b1k_pytorch.py \
        pi05_b1k_pytorch --exp-name run --pytorch-weight-path /path/to/converted_base
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import TYPE_CHECKING

# Shared OpenPI configuration and transforms import JAX. Only Torch may own GPUs
# in this entrypoint, including spawned data workers.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist

from openpi.shared import subtask_contract

if TYPE_CHECKING:
    import openpi.training.config as _config


_CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "metadata.json", "rng.pt")
_RESUME_COMPATIBLE_DEFAULTS = {
    "model": (
        (("subtask_action_conditioning",), "task_and_subtask"),
    ),
}


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        # NCCL must know the local device before its first collective.
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    return dist.is_initialized(), local_rank, device


def cleanup_ddp():
    # A barrier in a finally block would hang if another rank raised earlier.
    if dist.is_initialized():
        dist.destroy_process_group()


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def _on_primary(operation):
    """Run a filesystem operation once and propagate failures to every DDP rank."""
    result = [None]
    if _rank() == 0:
        try:
            result[0] = (True, operation())
        except Exception as exc:  # All ranks must receive the failure before raising.
            result[0] = (False, f"{type(exc).__name__}: {exc}")
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    success, value = result[0]
    if not success:
        raise RuntimeError(value)
    return value


def set_seed(seed: int, rank: int):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def get_model_state_dict(model):
    return _unwrap(model).state_dict()


def get_model_parameters(model):
    return (parameter for parameter in _unwrap(model).parameters() if parameter.requires_grad)


def move_to_device(value, device):
    """Move Observation dataclasses without creating any JAX arrays or backend."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: move_to_device(getattr(value, field.name), device) for field in dataclasses.fields(value)},
        )
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_to_device(item, device) for item in value))
    if isinstance(value, tuple | list):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def learning_rate_at(schedule, completed_steps: int) -> float:
    """Evaluate the supported OpenPI schedules without calling JAX/Optax."""
    warmup = schedule.warmup_steps
    if warmup < 0 or schedule.peak_lr <= 0:
        raise ValueError("Learning rate warmup must be nonnegative and peak_lr positive.")
    if completed_steps < warmup:
        initial = schedule.peak_lr / (warmup + 1)
        return initial + (schedule.peak_lr - initial) * completed_steps / warmup
    if hasattr(schedule, "decay_steps"):
        if schedule.decay_steps <= warmup:
            raise ValueError("decay_steps must exceed warmup_steps.")
        progress = min(1.0, (completed_steps - warmup) / (schedule.decay_steps - warmup))
        return schedule.decay_lr + (schedule.peak_lr - schedule.decay_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    if hasattr(schedule, "timescale") and schedule.timescale > 0:
        return schedule.peak_lr / math.sqrt((schedule.timescale + completed_steps - warmup) / schedule.timescale)
    raise TypeError(f"Unsupported PyTorch LR schedule: {type(schedule).__name__}")


def _autocast(device, precision):
    if precision == "bfloat16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _loss_and_metrics(output):
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        losses, metrics = output
    elif isinstance(output, dict):
        losses, metrics = output["loss"], {key: value for key, value in output.items() if key != "loss"}
    else:
        losses, metrics = output, {}
    if not isinstance(losses, torch.Tensor) or not losses.requires_grad:
        raise TypeError("The model must return a differentiable loss tensor.")
    loss = losses.float().mean()
    return loss, {
        "loss": loss.detach(),
        **{
            key: value.detach().float().mean()
            if isinstance(value, torch.Tensor)
            else torch.tensor(float(value), device=loss.device)
            for key, value in metrics.items()
        },
    }


def train_one_update(
    model, optimizer, data_iter, *, device, accumulation_steps, precision, clip_gradient_norm, return_metrics=True
):
    """Consume microbatches and perform exactly one optimizer update."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    optimizer.zero_grad(set_to_none=True)
    totals = {}
    for microstep in range(accumulation_steps):
        try:
            observation, actions = next(data_iter)
        except StopIteration as exc:
            raise RuntimeError("Training data exhausted before the requested optimizer updates completed.") from exc
        observation = move_to_device(observation, device)
        actions = move_to_device(actions, device).float()
        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and microstep + 1 < accumulation_steps
            else contextlib.nullcontext()
        )
        with sync_context:
            with _autocast(device, precision):
                output = (
                    model(observation, actions, return_metrics=True) if return_metrics else model(observation, actions)
                )
                loss, metrics = _loss_and_metrics(output)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite training loss; optimizer update was not applied.")
            (loss / accumulation_steps).backward()
        # Token accuracy is a token-weighted statistic, unlike the per-example
        # CE/action losses. Keep its numerator until microbatches/ranks combine.
        if "subtask_token_accuracy" in metrics and "subtask_tokens" in metrics:
            metrics["subtask_token_accuracy"] = metrics["subtask_token_accuracy"] * metrics["subtask_tokens"]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + value / accumulation_steps
    parameters = list(get_model_parameters(model))
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_gradient_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    totals["grad_norm"] = grad_norm.detach()
    if dist.is_initialized():
        keys = sorted(totals)
        reduced = torch.stack([totals[key] for key in keys])
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        totals = dict(zip(keys, reduced / _world_size(), strict=True))
    if "subtask_token_accuracy" in totals and "subtask_tokens" in totals:
        totals["subtask_token_accuracy"] = totals["subtask_token_accuracy"] / totals["subtask_tokens"].clamp_min(1)
    return {key: value.item() for key, value in totals.items()}


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def checkpoint_due(completed_steps, config):
    return completed_steps > 0 and (
        completed_steps % config.save_interval == 0 or completed_steps == config.num_train_steps
    )


def get_latest_checkpoint_step(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    steps = [
        int(path.name)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and all((path / name).is_file() for name in _CHECKPOINT_FILES)
    ]
    return max(steps) if steps else None


def _config_summary(config):
    # JSON metadata never requires unpickling Python/Flax config classes.
    def encode(value):
        if callable(value) and hasattr(value, "__qualname__"):
            return f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    fields = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
    return json.loads(json.dumps(fields, default=encode))


def _add_resume_compatible_defaults(section, saved, current):
    migrated = json.loads(json.dumps(saved))
    for path, default in _RESUME_COMPATIBLE_DEFAULTS.get(section, ()):
        saved_parent = migrated
        current_parent = current
        for key in path[:-1]:
            if not isinstance(saved_parent, dict) or not isinstance(current_parent, dict):
                break
            saved_parent = saved_parent.get(key)
            current_parent = current_parent.get(key)
        else:
            key = path[-1]
            if (
                isinstance(saved_parent, dict)
                and isinstance(current_parent, dict)
                and key not in saved_parent
                and current_parent.get(key) == default
            ):
                saved_parent[key] = default
    return migrated


def save_checkpoint(model, optimizer, global_step, config, is_main=None, data_config=None):
    """Collect per-rank RNG, then atomically publish a complete, inference-ready checkpoint.

    All DDP ranks must call this function; ``is_main`` remains only for call-site
    compatibility. New step directories count completed optimizer updates.
    """
    if not checkpoint_due(global_step, config):
        return
    local_rng = _capture_rng_state()
    if dist.is_initialized():
        rng_states = [None] * _world_size()
        dist.all_gather_object(rng_states, local_rng)
    else:
        rng_states = [local_rng]

    def write():
        final_dir = config.checkpoint_dir / str(global_step)
        temporary_dir = config.checkpoint_dir / f".tmp_{global_step}"
        if final_dir.exists():
            raise FileExistsError(f"Refusing to replace an existing complete checkpoint: {final_dir}")
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
        safetensors.torch.save_model(_unwrap(model), temporary_dir / "model.safetensors")
        torch.save(optimizer.state_dict(), temporary_dir / "optimizer.pt")
        torch.save(rng_states, temporary_dir / "rng.pt")
        metadata = {
            "format_version": 2,
            "global_step": global_step,
            "completed_steps": global_step,
            "micro_batches_seen": global_step * config.pytorch_gradient_accumulation_steps,
            "batch_size": config.batch_size,
            "world_size": _world_size(),
            "gradient_accumulation_steps": config.pytorch_gradient_accumulation_steps,
            "config": _config_summary(config),
            "timestamp": time.time(),
        }
        (temporary_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if getattr(getattr(config, "model", None), "train_subtask_prediction", False):
            subtask_contract.save(temporary_dir / "assets", config.model.subtask_action_conditioning)
        if data_config is not None and data_config.norm_stats is not None and data_config.asset_id is not None:
            import openpi.shared.normalize as normalize

            normalize.save(temporary_dir / "assets" / data_config.asset_id, data_config.norm_stats)
        temporary_dir.rename(final_dir)
        logging.info("Saved checkpoint after %d optimizer updates: %s", global_step, final_dir)

    _on_primary(write)


def load_checkpoint(model, optimizer, checkpoint_dir, device, config=None):
    """Restore weights, AdamW moments and rank-local RNG without a second GPU copy."""
    latest = get_latest_checkpoint_step(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No complete Torch training checkpoints in {checkpoint_dir}")
    directory = Path(checkpoint_dir) / str(latest)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["format_version"] != 2 or metadata["completed_steps"] != latest:
        raise ValueError(f"Invalid checkpoint step/format metadata in {directory}")
    if metadata["world_size"] != _world_size():
        raise ValueError("Resume requires the original world size to restore sampler and per-rank RNG state.")
    if config is not None:
        if (
            metadata["batch_size"] != config.batch_size
            or metadata["gradient_accumulation_steps"] != config.pytorch_gradient_accumulation_steps
        ):
            raise ValueError("Resume requires the original batch size and gradient accumulation steps.")
        saved_config = metadata["config"]
        current_config = _config_summary(config)
        for name in ("model", "data", "optimizer", "lr_schedule", "seed", "pytorch_training_precision"):
            saved_section = _add_resume_compatible_defaults(name, saved_config.get(name), current_config.get(name))
            if name in saved_config and saved_section != current_config.get(name):
                raise ValueError(
                    f"Resume configuration changed {name}; use a new experiment for a new fine-tuning run."
                )
    # CPU staging also lets load_state_dict cast saved state into FP32 master parameters.
    safetensors.torch.load_model(_unwrap(model), directory / "model.safetensors", device="cpu", strict=True)
    optimizer_state = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state
    rng_states = torch.load(directory / "rng.pt", map_location="cpu", weights_only=False)
    _restore_rng_state(rng_states[_rank()])
    logging.info("Resumed after %d optimizer updates (%d microbatches).", latest, metadata["micro_batches_seen"])
    return latest


def _prepare_output(config):
    def prepare():
        directory = config.checkpoint_dir
        if config.resume:
            if get_latest_checkpoint_step(directory) is None:
                raise FileNotFoundError(f"No complete Torch training checkpoint in {directory}")
        else:
            if directory.exists():
                if not config.overwrite:
                    raise FileExistsError(
                        f"Experiment already exists: {directory}; choose --resume or another --exp-name."
                    )
                shutil.rmtree(directory)
            directory.mkdir(parents=True)

    _on_primary(prepare)


def _data_config_for_resume(config):
    """Restore normalization from the model's checkpoint, even on a new machine."""
    if not config.resume:
        return config
    latest = get_latest_checkpoint_step(config.checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"No complete Torch training checkpoint in {config.checkpoint_dir}")
    assets = config.checkpoint_dir / str(latest) / "assets"
    if config.data.repo_id != "fake" and not assets.is_dir():
        raise FileNotFoundError(f"Training checkpoint is missing normalization assets: {assets}")
    data_assets = dataclasses.replace(config.data.assets, assets_dir=str(assets))
    return dataclasses.replace(config, data=dataclasses.replace(config.data, assets=data_assets))


def _validate_config(config, device):
    if config.resume and config.overwrite:
        raise ValueError("Cannot resume and overwrite at the same time.")
    for name in (
        "batch_size",
        "pytorch_gradient_accumulation_steps",
        "num_train_steps",
        "save_interval",
        "log_interval",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if config.batch_size % _world_size():
        raise ValueError("batch_size must be divisible by the torchrun world size.")
    if config.ema_decay is not None:
        raise ValueError("Torch trainer does not implement EMA; use pi05_b1k_pytorch or --ema-decay None explicitly.")
    if config.fsdp_devices != 1:
        raise ValueError("This Torch trainer uses DDP, not FSDP; set --fsdp-devices 1.")
    if config.val_log_interval is not None:
        raise ValueError("Periodic validation is not implemented in the Torch trainer; use the evaluation scripts.")
    if config.pytorch_training_precision not in ("bfloat16", "float32"):
        raise ValueError("pytorch_training_precision must be bfloat16 or float32.")
    if device.type == "cuda" and config.pytorch_training_precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16; select --pytorch-training-precision float32.")
    if not config.resume and config.pytorch_weight_path is None and config.data.repo_id != "fake":
        raise ValueError(
            "BEHAVIOR fine-tuning requires --pytorch-weight-path pointing to a converted model.safetensors checkpoint."
        )
    if config.pytorch_weight_path is not None and not config.resume:
        path = Path(config.pytorch_weight_path)
        path = path / "model.safetensors" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(f"Converted PyTorch weights do not exist: {path}")
    if type(config.optimizer).__name__ != "AdamW":
        raise TypeError("Torch trainer currently supports the AdamW optimizer config.")
    if config.optimizer.clip_gradient_norm <= 0:
        raise ValueError("clip_gradient_norm must be positive.")
    learning_rate_at(config.lr_schedule, 0)


def log_memory_usage(device, step, phase="training"):
    if device.type == "cuda":
        logging.info(
            "step=%d %s CUDA allocated=%.2f GiB reserved=%.2f GiB peak=%.2f GiB",
            step,
            phase,
            torch.cuda.memory_allocated(device) / 2**30,
            torch.cuda.memory_reserved(device) / 2**30,
            torch.cuda.max_memory_allocated(device) / 2**30,
        )


def _init_wandb(config):
    import wandb

    run_id_file = config.checkpoint_dir / "wandb_id.txt"
    kwargs = {}
    if config.resume and run_id_file.exists():
        kwargs.update(id=run_id_file.read_text().strip(), resume="must")
    run = wandb.init(name=config.exp_name, project=config.project_name, config=_config_summary(config), **kwargs)
    run_id_file.write_text(run.id)
    return run


def train_loop(config: _config.TrainConfig):
    use_ddp, _, device = setup_ddp()
    run = None
    progress = None
    try:
        _validate_config(config, device)
        # Import the heavyweight shared stack only after JAX has been restricted to CPU.
        import flax.nnx as nnx
        import tqdm

        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        import openpi.training.data_loader as data_loader

        if not isinstance(config.model, Pi0Config):
            raise TypeError("Torch trainer requires a Pi0Config model.")
        if not isinstance(config.freeze_filter, nnx.Nothing):
            raise ValueError("JAX freeze_filter is not supported by this Torch full-finetuning entrypoint.")
        if "lora" in config.model.paligemma_variant or "lora" in config.model.action_expert_variant:
            raise ValueError("This Torch entrypoint supports full fine-tuning; JAX LoRA variants are not implemented.")
        _prepare_output(config)
        set_seed(config.seed, _rank())
        loader = data_loader.create_data_loader(_data_config_for_resume(config), framework="pytorch", shuffle=True)
        data_config = loader.data_config()
        # FP32 master weights prevent small fine-tuning updates from rounding to zero.
        # Autocast controls activation/matmul precision independently.
        model_config = dataclasses.replace(config.model, dtype="float32", pytorch_compile_mode=None)
        model = PI0Pytorch(model_config)
        if config.pytorch_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if config.pytorch_weight_path is not None and not config.resume:
            weight_path = Path(config.pytorch_weight_path)
            weight_path = weight_path / "model.safetensors" if weight_path.is_dir() else weight_path
            if not weight_path.is_file():
                raise FileNotFoundError(f"Converted PyTorch weights do not exist: {weight_path}")
            safetensors.torch.load_model(model, weight_path, device="cpu", strict=True)
        model.to(device=device, dtype=torch.float32)
        model.train()
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        optimizer_config = config.optimizer
        optimizer_kwargs = {"fused": True} if device.type == "cuda" else {"foreach": False}
        optimizer = torch.optim.AdamW(
            get_model_parameters(model),
            lr=learning_rate_at(config.lr_schedule, 0),
            betas=(optimizer_config.b1, optimizer_config.b2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
            **optimizer_kwargs,
        )
        completed_steps = (
            load_checkpoint(model, optimizer, config.checkpoint_dir, device, config) if config.resume else 0
        )
        if completed_steps > config.num_train_steps:
            raise ValueError("Checkpoint already exceeds num_train_steps.")
        if completed_steps:
            if not hasattr(loader, "set_start_batch"):
                raise TypeError("This data loader cannot restore the sampler position for Torch resume.")
            loader.set_start_batch(completed_steps * config.pytorch_gradient_accumulation_steps)
            logging.info(
                "Sampler order resumes at the saved microbatch; data-worker augmentation RNG is reinitialized."
            )
        # Iterating a DataLoader must not perturb restored model/noise RNG state.
        saved_rng = _capture_rng_state()
        iterator = iter(loader)
        _restore_rng_state(saved_rng)
        if config.wandb_enabled:

            def initialize_tracking():
                nonlocal run
                run = _init_wandb(config)

            _on_primary(initialize_tracking)
        logging.info(
            "Torch %s: world_size=%d microbatch/global=%d microbatch/GPU=%d accumulation=%d effective_batch=%d",
            device,
            _world_size(),
            config.batch_size,
            config.batch_size // _world_size(),
            config.pytorch_gradient_accumulation_steps,
            config.batch_size * config.pytorch_gradient_accumulation_steps,
        )
        logging.info(
            "Parameters/AdamW states=FP32; compute=%s; EMA=disabled; gradient_checkpointing=%s",
            config.pytorch_training_precision if device.type == "cuda" else "float32 (CPU)",
            config.pytorch_gradient_checkpointing,
        )
        log_memory_usage(device, completed_steps, "initialized")
        progress = tqdm.tqdm(total=config.num_train_steps, initial=completed_steps, disable=_rank() != 0)
        accumulated_metrics = {}
        logged_updates = 0
        started = time.monotonic()
        while completed_steps < config.num_train_steps:
            learning_rate = learning_rate_at(config.lr_schedule, completed_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            metrics = train_one_update(
                model,
                optimizer,
                iterator,
                device=device,
                accumulation_steps=config.pytorch_gradient_accumulation_steps,
                precision=config.pytorch_training_precision,
                clip_gradient_norm=optimizer_config.clip_gradient_norm,
            )
            completed_steps += 1
            logged_updates += 1
            metrics["learning_rate"] = learning_rate
            if "subtask_token_accuracy" in metrics and "subtask_tokens" in metrics:
                metrics["subtask_token_accuracy"] *= metrics["subtask_tokens"]
            for key, value in metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value
            progress.update(1)
            if completed_steps % config.log_interval == 0 or completed_steps == config.num_train_steps:
                average = {key: value / logged_updates for key, value in accumulated_metrics.items()}
                if "subtask_token_accuracy" in average and "subtask_tokens" in average:
                    average["subtask_token_accuracy"] /= max(average["subtask_tokens"], 1)
                average["time_per_update"] = (time.monotonic() - started) / logged_updates
                if _rank() == 0:
                    logging.info(
                        "step=%d %s", completed_steps, " ".join(f"{key}={value:.5g}" for key, value in average.items())
                    )
                    log_memory_usage(device, completed_steps)
                    if run is not None:
                        run.log(average, step=completed_steps)
                accumulated_metrics.clear()
                logged_updates = 0
                started = time.monotonic()
            save_checkpoint(model, optimizer, completed_steps, config, data_config=data_config)
    finally:
        if progress is not None:
            progress.close()
        if run is not None:
            run.finish()
        cleanup_ddp()


def main():
    import openpi.training.config as config

    init_logging()
    train_loop(config.cli())


if __name__ == "__main__":
    main()

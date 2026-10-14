from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import json
import logging
import os
from typing import List, Protocol

from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
from openpi.shared import frame_memory
from openpi.shared import subtask_contract
from openpi.shared import learning_rate_contract
from openpi.shared import training_progress
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool, read_only: bool = False
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if jax.process_count() > 1:
        if overwrite:
            raise ValueError("Multi-process checkpoint initialization does not support overwrite.")
        exists = np.asarray(
            multihost_utils.process_allgather(np.asarray(checkpoint_dir.exists(), dtype=np.int32))
        ).reshape(-1)
        if not np.all(exists == exists[0]):
            raise RuntimeError(f"Processes disagree about checkpoint directory existence: {checkpoint_dir}.")
        if resume:
            if not bool(exists[0]):
                raise FileNotFoundError(f"Cannot resume missing checkpoint directory: {checkpoint_dir}.")
            resuming = True
        else:
            if bool(exists[0]):
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --resume to continue it."
                )
            bootstrap_ok = np.zeros((), dtype=np.int32)
            if jax.process_index() == 0:
                try:
                    checkpoint_dir.mkdir(parents=True)
                    bootstrap_ok = np.ones((), dtype=np.int32)
                except Exception:
                    logging.exception("Rank 0 failed to create checkpoint directory %s", checkpoint_dir)
            status = int(multihost_utils.broadcast_one_to_all(bootstrap_ok))
            if status != 1:
                raise RuntimeError(f"Checkpoint bootstrap failed for {checkpoint_dir}.")
            multihost_utils.sync_global_devices("openpi_checkpoint_bootstrap")
    else:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            should_keep_fn=training_progress.checkpoint_milestone if os.environ.get("OPENPI_FORMAL_RUN_CONTRACT") else None,
            create=False,
            read_only=read_only,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    checkpoint_steps = tuple(mngr.all_steps())
    if jax.process_count() > 1:
        latest_step = max(checkpoint_steps, default=-1)
        latest_steps = np.asarray(
            multihost_utils.process_allgather(np.asarray(latest_step, dtype=np.int64))
        ).reshape(-1)
        if not np.all(latest_steps == latest_steps[0]):
            raise RuntimeError(
                f"Processes disagree about the latest checkpoint in {checkpoint_dir}: {latest_steps.tolist()}."
            )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. A multi-process retry must fail instead of creating divergent fresh states.
    if resuming and checkpoint_steps in [(), (0,)]:
        if jax.process_count() > 1:
            raise FileNotFoundError(f"No resumable checkpoint found in {checkpoint_dir}.")
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    mixture_plan = getattr(data_loader, "mixture_plan", None)
    completed_steps = int(state.step) if mixture_plan is not None else None
    source_counts = dict(getattr(data_loader, "realized_source_counts", {}))

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        subtask_contract.save(directory, subtask_contract.from_data_config(data_config))
        learning_rate_contract.save(directory, getattr(data_loader, "learning_rate_contract", None))
        frame_memory.save(directory, getattr(data_config, "memory_contract", None))
        if (contract := getattr(data_loader, "formal_run_contract", None)) is not None:
            (directory / "formal_run_contract.json").write_text(json.dumps(contract, sort_keys=True) + "\n")
        if (contract := getattr(data_loader, "training_rng_contract", None)) is not None:
            (directory / "training_rng_contract.json").write_text(json.dumps(contract, sort_keys=True) + "\n")
        if source_counts:
            (directory / "realized_source_counts.json").write_text(json.dumps(source_counts, sort_keys=True) + "\n")
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
            _normalize.save(directory / asset_id, norm_stats)
        if mixture_plan is not None:
            from openpi.training.subtask_mixture import save_consumed_contract
            save_consumed_contract(directory, mixture_plan, completed_steps)
            (directory / "subtask_mixture_execution.json").write_text(
                json.dumps(data_loader.mixture_execution, sort_keys=True) + "\n")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    restore_step = checkpoint_manager.latest_step() if step is None else step
    assets_directory = checkpoint_manager.directory / str(restore_step) / "assets"
    learning_rate_contract.check_resume(assets_directory, getattr(data_loader, "learning_rate_contract", None))
    for filename, expected in (("formal_run_contract.json", getattr(data_loader, "formal_run_contract", None)),
                               ("training_rng_contract.json", getattr(data_loader, "training_rng_contract", None))):
        if expected is not None and (not (path := assets_directory / filename).exists() or json.loads(path.read_text()) != expected):
            raise ValueError("Checkpoint provenance or training RNG changed: " + filename)
    mixture_plan = getattr(data_loader, "mixture_plan", None)
    mixture_contract_path = assets_directory / "subtask_mixture.json"
    if mixture_plan is None and mixture_contract_path.exists():
        raise ValueError("Cannot resume an indexed mixture checkpoint with a different data loading mode.")
    if mixture_plan is not None:
        from openpi.training.subtask_mixture import check_consumed_contract
        check_consumed_contract(assets_directory, mixture_plan)
        execution_path = assets_directory / "subtask_mixture_execution.json"
        if not execution_path.exists() or json.loads(execution_path.read_text()) != data_loader.mixture_execution:
            raise ValueError("Mixture resume requires unchanged normalization, seed, batch and process count.")
    frame_memory.check_resume(checkpoint_manager.directory / str(restore_step) / "assets",
                              getattr(data_loader.data_config(), "memory_contract", None))
    subtask_contract.check_resume(
        checkpoint_manager.directory / str(restore_step) / "assets",
        subtask_contract.from_data_config(data_loader.data_config()),
    )

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    merged = _merge_params(restored["train_state"], restored["params"])
    if getattr(data_loader, "learning_rate_contract", None) is not None:
        learning_rate_contract.check_optimizer_clock(merged.opt_state, merged.step)
    if mixture_plan is not None:
        check_consumed_contract(assets_directory, mixture_plan, int(merged.step))
        counts_path = assets_directory / "realized_source_counts.json"
        if counts_path.exists():
            counts = json.loads(counts_path.read_text())
            if sum(counts.values()) != int(merged.step) * data_loader.mixture_execution["global_batch_size"]:
                raise ValueError("Checkpoint source counts differ from consumed global samples")
            data_loader.realized_source_counts = counts
    return merged


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str | List[str]) -> dict[str, _normalize.NormStats] | None:
    if isinstance(asset_id, list):
        # Only load the first asset_id assuming that the datasets are similar
        asset_id = asset_id[0]
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])

from collections.abc import Iterator, Sequence
import itertools
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.lerobot_compat as _lerobot_compat
from openpi.training.subtask_data import SubtaskFromSegments
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    @property
    def batches_per_epoch(self) -> int:
        """Number of complete batches per process for an indexed dataset."""
        raise NotImplementedError("This data loader does not expose indexed epochs.")

    def set_start_batch(self, index: int) -> None:
        """Set the microbatch position before creating the next iterator."""
        raise NotImplementedError("This data loader does not support restoring its sample position.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class StridedDataset(Dataset[T_co]):
    """Exposes every ``stride``-th frame of an indexable dataset.

    B1K is 30 fps video with a multi-step action chunk, so adjacent frames carry
    near-identical inputs and heavily overlapping action targets. Subsampling by
    a fixed global stride lets a training budget cover ~``stride`` times more
    distinct frames per pass. The stride is global over the concatenated
    episodes (unbiased); episode-tail action padding is already handled upstream
    by ``delta_timestamps``.
    """

    def __init__(self, dataset: Dataset, stride: int, offset: int = 0):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._dataset = dataset
        self._stride = int(stride)
        self._offset = int(offset)
        self._len = max(0, (len(dataset) - self._offset + self._stride - 1) // self._stride)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[self._offset + int(index) * self._stride]

    def __len__(self) -> int:
        return self._len


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int, *, framework: str = "jax"):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
        self._framework = framework

    def __getitem__(self, index: SupportsIndex) -> dict:
        if self._framework == "pytorch":
            rng = np.random.default_rng(index.__index__())

            def make_numpy(spec):
                shape = spec.shape[1:]
                if spec.dtype == jnp.float32:
                    return rng.uniform(-1, 1, size=shape).astype(np.float32)
                if spec.dtype == jnp.int32:
                    return rng.integers(0, 2048, size=shape, dtype=np.int32)
                return np.zeros(shape, dtype=spec.dtype)

            # Tree operations only inspect Python structure; no JAX backend is initialized.
            observation = jax.tree.map(make_numpy, self._observation_spec)
            return {**observation.to_dict(), "actions": jax.tree.map(make_numpy, self._action_spec)}

        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_b1k_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create a behavior dataset for training."""
    if getattr(data_config, "subtask_mixture_path", None) is not None:
        from openpi.training.subtask_mixture_data import create_dataset
        return create_dataset(data_config, action_horizon)
    subtask_transform = None
    if data_config.subtask_prediction:
        if isinstance(data_config.repo_id, list):
            raise ValueError("Subtask supervision requires one dataset root with globally unique episode indices.")
        if not data_config.subtask_prompts_path:
            raise ValueError("Subtask prediction requires a subtask segment table.")
        if not data_config.prompt_from_task:
            raise ValueError("Subtask prediction requires task prompts.")
        if data_config.subtask_whole_task_prob != 0:
            raise ValueError("subtask_whole_task_prob must be 0 for joint subtask supervision.")
        subtask_transform = SubtaskFromSegments.from_parquet(
            data_config.subtask_prompts_path,
            episodes_index=data_config.episodes_index,
        )
    if isinstance(data_config.repo_id, list):
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id[0], root=os.path.join(data_config.dataset_root, data_config.repo_id[0])
        )
        dataset_kwargs = {"repo_ids": data_config.repo_id, **data_config.dataset_kwargs}
    else:
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id, root=data_config.dataset_root
        )
        dataset_kwargs = {"repo_id": data_config.repo_id, **data_config.dataset_kwargs}
    if data_config.episodes_index is not None:
        dataset_kwargs["episodes"] = data_config.episodes_index
    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )

    if data_config.frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {data_config.frame_stride}")
    if data_config.memory_contract is not None:
        from openpi.training.frame_memory_data import CachedFrameMemory
        if isinstance(data_config.repo_id, list) or not data_config.memory_cache_dir:
            raise ValueError("FrameSamp training requires one dataset root and --data.memory-cache-dir.")
        dataset = TransformedDataset(dataset, [CachedFrameMemory(
            data_config.memory_cache_dir, data_config.memory_contract, data_config.dataset_root, data_config.episodes_index
        )])
    if data_config.prompt_from_task:
        meta_root = (
            os.path.join(data_config.dataset_root, data_config.repo_id[0])
            if isinstance(data_config.repo_id, list)
            else data_config.dataset_root
        )
        tasks = _lerobot_compat.b1k_task_prompts(meta_root, _lerobot_compat.tasks_from_metadata(dataset_meta))
        if data_config.subtask_prediction:
            dataset = TransformedDataset(
                dataset, [_transforms.PromptFromLeRobotTask(tasks), typing.cast(SubtaskFromSegments, subtask_transform)]
            )
        elif data_config.subtask_prompts_path:
            prompt_tf = _transforms.PromptFromSubtask.from_parquet(
                data_config.subtask_prompts_path, tasks, data_config.subtask_whole_task_prob
            )
            dataset = TransformedDataset(dataset, [prompt_tf])
        else:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])

    if data_config.frame_stride > 1:
        dataset = StridedDataset(dataset, data_config.frame_stride)

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    framework: str = "jax",
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024, framework=framework)
    if data_config.dataset_root is not None or data_config.subtask_mixture_path is not None:
        return create_b1k_dataset(data_config, action_horizon)

    dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(repo_id)
    dataset = _lerobot_compat.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return TransformedDataset(dataset, transforms)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_b1k_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(data_config=data_config, action_horizon=config.model.action_horizon)
    return _create_map_data_loader(
        data_config,
        dataset,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, framework=framework)
    return _create_map_data_loader(
        data_config,
        dataset,
        batch_size=batch_size,
        sharding=sharding,
        skip_norm_stats=skip_norm_stats,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )


def _create_map_data_loader(
    data_config: _config.DataConfig,
    dataset: Dataset,
    *,
    batch_size: int,
    sharding: jax.sharding.Sharding | None,
    skip_norm_stats: bool,
    shuffle: bool,
    num_batches: int | None,
    num_workers: int,
    seed: int,
    framework: str,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    mixture_plan = getattr(dataset, "mixture_plan", None)
    if mixture_plan is not None:
        if framework != "jax":
            raise ValueError("Indexed mixture optimizer/data resume is currently supported by the JAX B1K trainer.")
        if batch_size != mixture_plan.global_batch_size or seed != mixture_plan.seed:
            raise ValueError("Training batch size and seed must match the frozen mixture plan.")
        # The global index encodes a phase and a randomized source/frame draw.
        # Shuffling those indices would mix future curriculum phases into the past.
        shuffle = False
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    if framework == "pytorch":
        process_count = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    elif framework == "jax":
        process_count = jax.process_count()
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    if batch_size < 1 or batch_size % process_count:
        raise ValueError(
            f"Global batch_size ({batch_size}) must be positive and divisible by process count ({process_count})."
        )
    local_batch_size = batch_size // process_count

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader, mixture_plan=mixture_plan,
                          process_count=process_count, batch_size=batch_size, seed=seed)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see the DROID training guide in upstream OpenPI

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class _ResumableBatchSampler:
    """Skip already consumed indices before the workers load or decode any samples."""

    def __init__(self, sampler, batch_size: int):
        self._batches = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
        self.start_batch = 0

    def __iter__(self):
        return itertools.islice(self._batches, self.start_batch, None)

    def __len__(self):
        return max(0, len(self._batches) - self.start_batch)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if framework not in {"jax", "pytorch"}:
            raise ValueError(f"Unsupported framework: {framework}")
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be positive.")
        if framework == "pytorch" and sharding is not None:
            raise ValueError("JAX sharding cannot be used with the PyTorch data loader.")
        if framework == "jax" and jax.process_count() > 1:
            if sampler is not None:
                raise ValueError("A custom sampler cannot be combined with JAX multi-process data loading.")
            sampler = torch.utils.data.distributed.DistributedSampler(
                typing.cast(torch.utils.data.Dataset, dataset),
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=shuffle,
                seed=seed,
                drop_last=True,
            )
            shuffle = False
        elif framework == "pytorch" and torch.distributed.is_initialized() and sampler is None:
            sampler = torch.utils.data.distributed.DistributedSampler(
                typing.cast(torch.utils.data.Dataset, dataset),
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                seed=seed,
                drop_last=True,
            )
            shuffle = False

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._seed = seed
        self._start_batch = 0
        self._sampler_generator = torch.Generator()
        if sampler is None:
            sampler = (
                torch.utils.data.RandomSampler(dataset, generator=self._sampler_generator)
                if shuffle
                else torch.utils.data.SequentialSampler(dataset)
            )
        self._sampler = sampler
        self._batch_sampler = _ResumableBatchSampler(sampler, local_batch_size)
        self._batches_per_epoch = len(self._batch_sampler)
        if self._batches_per_epoch == 0:
            raise ValueError(
                "The process shard has no complete batch. Reduce global batch_size or add training samples."
            )

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        if framework == "jax":
            process_index = jax.process_index()
        else:
            process_index = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        process_seed = seed + process_index
        generator.manual_seed(process_seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_sampler=self._batch_sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def batches_per_epoch(self) -> int:
        return self._batches_per_epoch

    def set_start_batch(self, index: int) -> None:
        """Resume sample ordering at a global microbatch offset, without reading skipped frames.

        Call before creating an iterator. Worker augmentation RNG is not restored.
        """
        if index < 0:
            raise ValueError("start batch must be nonnegative.")
        self._start_batch = index

    def __iter__(self):
        num_items = 0
        epoch, start_batch = divmod(self._start_batch, self._batches_per_epoch)
        while True:
            if isinstance(self._sampler, torch.utils.data.distributed.DistributedSampler):
                self._sampler.set_epoch(epoch)
            self._sampler_generator.manual_seed(self._seed + epoch)
            self._batch_sampler.start_batch = start_batch
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            epoch += 1
            start_batch = 0


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader,
                 *, mixture_plan=None, process_count=1, batch_size=None, seed=0):
        self._data_config = data_config
        self._data_loader = data_loader
        self.mixture_plan = mixture_plan
        self.mixture_execution = None
        if mixture_plan is not None:
            from openpi.training.subtask_mixture_data import execution_contract
            self.mixture_execution = execution_contract(
                batch_size, process_count, seed, data_config.norm_stats)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def batches_per_epoch(self) -> int:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise NotImplementedError("Epoch lengths are only available for indexed datasets.")
        return self._data_loader.batches_per_epoch

    def set_start_batch(self, index: int) -> None:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise NotImplementedError("Data position restoration is only supported for indexed datasets.")
        self._data_loader.set_start_batch(index)

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

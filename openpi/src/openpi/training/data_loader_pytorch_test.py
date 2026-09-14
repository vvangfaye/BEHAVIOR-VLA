"""CPU regressions for PyTorch batches, distributed sampling and data-position restore."""

import dataclasses

import jax
import numpy as np
import pytest
import torch

from openpi.training import config as config_lib
from openpi.training import data_loader


class IndexedFrames:
    def __init__(self, count=12):
        self.count = count
        self.read_indices = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.read_indices.append(index)
        return {"index": np.int64(index)}


def _distributed(monkeypatch, rank):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: rank is not None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("rank", [None, 0, 1])
def test_resume_preserves_sample_order_without_reading_skipped_frames(monkeypatch, rank, shuffle):
    _distributed(monkeypatch, rank)
    original = data_loader.TorchDataLoader(
        IndexedFrames(), 2, framework="pytorch", shuffle=shuffle, seed=41, num_batches=13
    )
    expected = [batch["index"] for batch in original]
    resumed_frames = IndexedFrames()
    resumed = data_loader.TorchDataLoader(
        resumed_frames, 2, framework="pytorch", shuffle=shuffle, seed=41, num_batches=5
    )
    resumed.set_start_batch(8)
    actual = list(resumed)
    assert len(resumed_frames.read_indices) == 10
    assert all(torch.equal(batch["index"], target) for batch, target in zip(actual, expected[8:], strict=True))


def test_distributed_shards_are_disjoint_and_epoch_shuffle_changes(monkeypatch):
    ranks = []
    for rank in range(2):
        _distributed(monkeypatch, rank)
        loader = data_loader.TorchDataLoader(
            IndexedFrames(), 2, framework="pytorch", shuffle=True, seed=41, num_batches=6
        )
        batches = [batch["index"].tolist() for batch in loader]
        ranks.append(
            [[index for batch in batches[:3] for index in batch], [index for batch in batches[3:] for index in batch]]
        )
    for epoch in range(2):
        assert set(ranks[0][epoch]).isdisjoint(ranks[1][epoch])
        assert sorted(ranks[0][epoch] + ranks[1][epoch]) == list(range(12))
    assert ranks[0][0] != ranks[0][1]


def test_too_small_distributed_shard_fails_instead_of_looping(monkeypatch):
    _distributed(monkeypatch, 0)
    with pytest.raises(ValueError, match="no complete batch"):
        data_loader.TorchDataLoader(IndexedFrames(3), 2, framework="pytorch")


def test_pytorch_workers_emit_cpu_tensor_batches():
    loader = data_loader.TorchDataLoader(IndexedFrames(), 3, framework="pytorch", num_workers=2, num_batches=2)
    assert [batch["index"].tolist() for batch in loader] == [[0, 1, 2], [3, 4, 5]]


def test_global_batch_must_be_divisible_by_world_size(monkeypatch):
    _distributed(monkeypatch, 0)
    config = dataclasses.replace(config_lib.get_config("debug"), batch_size=3, num_workers=0)
    with pytest.raises(ValueError, match="divisible by process count"):
        data_loader.create_data_loader(config, framework="pytorch", skip_norm_stats=True)


def test_fake_torch_loader_never_calls_jax_backend(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("A PyTorch loader initialized JAX")

    monkeypatch.setattr(jax, "devices", unexpected)
    monkeypatch.setattr(jax, "process_count", unexpected)
    monkeypatch.setattr(jax, "process_index", unexpected)
    monkeypatch.setattr(jax.random, "key", unexpected)
    config = dataclasses.replace(config_lib.get_config("debug"), batch_size=2, num_workers=0)
    observation, actions = next(
        iter(data_loader.create_data_loader(config, framework="pytorch", skip_norm_stats=True, num_batches=1))
    )
    assert all(
        isinstance(leaf, torch.Tensor) and leaf.device.type == "cpu" for leaf in jax.tree.leaves((observation, actions))
    )
    assert observation.tokenized_prompt.dtype == torch.int32
    assert observation.tokenized_prompt_mask.dtype == torch.bool

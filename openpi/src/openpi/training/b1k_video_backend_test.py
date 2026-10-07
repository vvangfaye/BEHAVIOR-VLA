"""Explicit decoder selection reaches B1K training while preserving defaults."""
import dataclasses
import sys

import pytest

from openpi import transforms
from openpi.training import config as config_lib


@pytest.mark.parametrize("backend", [None, "pyav", "torchcodec"])
def test_video_backend_preserves_other_reader_options(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(config_lib, "ModelTransformFactory", lambda: lambda model: transforms.Group())
    config = config_lib.get_config("pi05_b1k")
    original = dict(config.data.base_config.dataset_kwargs)
    factory = dataclasses.replace(config.data, video_backend=backend)
    result = factory.create(tmp_path, config.model)
    expected = dict(original)
    if backend is not None:
        expected["video_backend"] = backend
    assert result.dataset_kwargs == expected
    assert config.data.base_config.dataset_kwargs == original


def test_invalid_backend_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config_lib, "ModelTransformFactory", lambda: lambda model: transforms.Group())
    config = config_lib.get_config("pi05_b1k")
    with pytest.raises(ValueError, match="video_backend"):
        dataclasses.replace(config.data, video_backend="unknown").create(tmp_path, config.model)


def test_training_cli_accepts_explicit_pyav(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_b1k", "pi05_b1k", "--exp-name", "decoder-check",
                                     "--data.video-backend", "pyav"])
    config = config_lib.cli()
    assert config.data.video_backend == "pyav"
    assert config.model.action_horizon == 32

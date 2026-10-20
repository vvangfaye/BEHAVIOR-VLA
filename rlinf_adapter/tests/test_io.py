import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from b1k_rlinf.io import DELTA_INDICES, CheckpointIO, extract_state, resize_image


def test_raw_layouts_represent_identical_robot_state():
    raw57 = np.arange(114, dtype=np.float32).reshape(2, 57)
    raw25 = np.concatenate(
        [raw57[:, :3], raw57[:, 53:57], raw57[:, 3:10], raw57[:, 24:26], raw57[:, 28:35], raw57[:, 49:51]],
        axis=-1,
    )
    expected = np.concatenate(
        [
            raw57[:, :3],
            raw57[:, 53:57],
            raw57[:, 3:10],
            raw57[:, 24:26].sum(-1, keepdims=True),
            raw57[:, 28:35],
            raw57[:, 49:51].sum(-1, keepdims=True),
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(extract_state(raw25, "canonical25"), expected)
    np.testing.assert_array_equal(extract_state(raw57, "checkpoint57"), expected)
    with pytest.raises(ValueError):
        extract_state(raw57, "canonical25")


def test_action_coordinates_match_checkpoint(norm_path):
    io = CheckpointIO(norm_path, "policy23")
    physical = np.linspace(-1, 1, 46).reshape(2, 23).astype(np.float32)
    state = np.pad(io.normalize_state(physical), ((0, 0), (0, 9)))
    normalized_actions = torch.zeros(2, 32, 32)
    result = io.decode(normalized_actions, torch.from_numpy(state), action_chunk=16)
    expected = np.broadcast_to(io.stats["actions"]["mean"], (2, 16, 23)).copy()
    expected[..., DELTA_INDICES] += physical[:, None, DELTA_INDICES]
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
    assert result.shape == (2, 16, 23)


def test_image_resize_matches_existing_policy():
    path = (
        Path(__file__).resolve().parents[2] / "openpi/packages/openpi-client/src/openpi_client/image_tools.py"
    )
    spec = importlib.util.spec_from_file_location("original_image_tools", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    image = np.random.default_rng(7).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    np.testing.assert_array_equal(resize_image(image), original.resize_with_pad(image, 224, 224))

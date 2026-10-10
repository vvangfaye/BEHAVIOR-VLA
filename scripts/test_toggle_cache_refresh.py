"""Exercise the production Warp contact-timer kernel without launching Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import warp as wp


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    source_path = Path(__file__).resolve().parents[1] / "BEHAVIOR-1K/OmniGibson/omnigibson/object_states/toggle.py"
    source = source_path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_set_toggle_value_kernel")
    # Load the actual production function with its Warp decorator. Warp needs a
    # real source file for inspection; no copied implementation or fake runtime.
    path = tmp_path_factory.mktemp("toggle_kernel") / "production_toggle_kernel.py"
    path.write_text("import warp as wp\n\n@wp.kernel\n" + ast.get_source_segment(source, node) + "\n")
    spec = importlib.util.spec_from_file_location("production_toggle_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._set_toggle_value_kernel


def launch(kernel, values, timers, masks, dt):
    values = wp.array(np.asarray(values, dtype=np.uint8), dtype=wp.uint8, device="cpu")
    timers = wp.array(np.asarray(timers, dtype=np.float32), dtype=wp.float32, device="cpu")
    masks = wp.array(np.asarray(masks, dtype=np.int32).reshape(-1), dtype=wp.int32, device="cpu")
    elapsed = wp.array([dt], dtype=wp.float32, device="cpu")
    wp.launch(kernel, dim=values.shape, inputs=[values, masks, timers, wp.int32(values.shape[1]), wp.float32(0.15), elapsed], device="cpu")
    return values.numpy(), timers.numpy(), masks.numpy()


def test_zero_time_refresh_preserves_restored_values_and_contact_timers(kernel):
    values = [[1, 0, 1], [0, 1, 0]]
    timers = np.asarray([[3, 0.14, 0.1], [3, 0.14, 0]], dtype=np.float32)
    actual_values, actual_timers, masks = launch(kernel, values, timers, [[0, 1, 2], [2, 0, 1]], 0)
    np.testing.assert_array_equal(actual_values, values)
    np.testing.assert_array_equal(actual_timers, timers)
    np.testing.assert_array_equal(masks, np.zeros(6, dtype=np.int32))


def test_physics_steps_keep_contact_loss_and_threshold_behavior(kernel):
    values, timers, masks = launch(kernel, [[1, 0, 1, 0, 1]], [[3, 0.14, 0.1, 3, 0.14]], [[0, 2, 2, 2, 1]], 0.02)
    np.testing.assert_array_equal(values, [[1, 1, 1, 0, 1]])
    np.testing.assert_allclose(timers, [[0, 0.16, 0.12, 3.02, 0]], rtol=1e-6)
    np.testing.assert_array_equal(masks, np.zeros(5, dtype=np.int32))

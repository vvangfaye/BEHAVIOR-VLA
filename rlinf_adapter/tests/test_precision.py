"""Protect the JAX BF16 activation contract, not just the formula in FP32."""

import numpy as np
import pytest
import torch

from b1k_rlinf.precision import jax_gelu, jax_gelu_glu, jax_linear_forward


def test_bfloat16_gelu_matches_actual_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    torch.manual_seed(31)
    x = torch.randn(4096).bfloat16()
    y = torch.randn_like(x)
    jx = jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16)
    jy = jnp.asarray(y.float().numpy(), dtype=jnp.bfloat16)
    reference = jax.jit(lambda a, b: jax.nn.gelu(a) * b)(jx, jy)
    np.testing.assert_array_equal(
        jax_gelu_glu(x, y).float().numpy(), np.asarray(reference, dtype=np.float32)
    )


def test_gelu_keeps_finite_training_gradients():
    x = torch.linspace(-3, 3, 256, requires_grad=True)
    jax_gelu(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0


def test_bfloat16_bias_matches_jax_rounding():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    torch.manual_seed(3)
    layer = torch.nn.Linear(32, 24).bfloat16()
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.bias.copy_(torch.randn_like(layer.bias))
    x = torch.randn(4, 32).bfloat16()
    arrays = [jnp.asarray(t.detach().float().numpy(), dtype=jnp.bfloat16) for t in (x, layer.weight, layer.bias)]
    expected = jax.jit(lambda a, w, b: jnp.dot(a, w.T) + b)(*arrays)
    np.testing.assert_array_equal(
        jax_linear_forward(layer, x).detach().float().numpy(),
        np.asarray(expected, dtype=np.float32),
    )

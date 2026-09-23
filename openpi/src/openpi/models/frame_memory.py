"""FrameSamp + memory modulation for the action expert (RoboMME-inspired).

The VLM stream never attends to these tokens. Shapes remain fixed under JIT;
padding is masked even when the entire history is absent.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp


def pool_tokens(tokens, grid_size=4):
    side = math.isqrt(tokens.shape[-2])
    if side * side != tokens.shape[-2] or side % grid_size:
        raise ValueError("Vision tokens must form a square divisible by memory_grid_size.")
    cells = side // grid_size
    shape = (*tokens.shape[:-2], grid_size, cells, grid_size, cells, tokens.shape[-1])
    return (
        tokens.reshape(shape)
        .astype(jnp.float32)
        .mean(axis=(-4, -2))
        .reshape(*tokens.shape[:-2], grid_size * grid_size, tokens.shape[-1])
    )


def position_features(positions):
    """768-dimensional temporal/row/column sinusoidal encoding, 256 per axis."""
    angles = positions.astype(jnp.float32)[..., None] / (10000 ** (jnp.arange(128) / 128))
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(*positions.shape[:-1], 768)


def _rms(x):
    value = x.astype(jnp.float32)
    return (value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + 1e-6)).astype(x.dtype)


def _rope(x, positions):
    angles = positions[..., None, None] / (10000 ** (jnp.arange(x.shape[-1] // 2) * 2 / x.shape[-1]))
    a, b = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate(
        [a * jnp.cos(angles) - b * jnp.sin(angles), b * jnp.cos(angles) + a * jnp.sin(angles)], -1
    ).astype(x.dtype)


class MemoryModulator(nn.Module):
    """Action queries -> grouped-query memory attention -> adaptive RMSNorm.

    Four Q heads, one KV head; at production width 1024 each head is 256.
    Small nonzero initialization lets gradients reach memory attention on step 1.
    """

    width: int

    @nn.compact
    def __call__(self, x, memory, mask):
        dtype = x.dtype
        if self.width % 8:
            raise ValueError("Memory modulation width must be divisible by 8.")
        head_dim = self.width // 4
        memory = jnp.where(mask[..., None], memory, 0).astype(dtype)
        q = nn.Dense(self.width, use_bias=False, dtype=dtype, name="query")(_rms(x))
        kv = nn.Dense(2 * head_dim, use_bias=False, dtype=dtype, name="key_value")(_rms(memory))
        q = q.reshape(*x.shape[:2], 4, head_dim)
        k, v = jnp.split(kv, 2, axis=-1)
        q = _rope(q, jnp.broadcast_to(jnp.arange(x.shape[1]) + memory.shape[1], x.shape[:2]))
        k = _rope(k[:, :, None], jnp.broadcast_to(jnp.arange(memory.shape[1]), memory.shape[:2]))[:, :, 0]
        logits = jnp.einsum("bthd,bsd->bhts", q, k, preferred_element_type=jnp.float32) / math.sqrt(head_dim)
        weights = jax.nn.softmax(jnp.where(mask[:, None, None, :], logits, -2.3819763e38), axis=-1)
        weights = jnp.where(mask[:, None, None, :], weights, 0).astype(dtype)
        read = jnp.einsum("bhts,bsd->bthd", weights, v).reshape(*x.shape[:2], self.width)
        read = nn.Dense(self.width, use_bias=False, dtype=dtype, name="output")(read)
        affine = nn.Dense(2 * self.width, dtype=dtype, kernel_init=nn.initializers.normal(1e-3), name="affine")(read)
        scale, shift = jnp.split(affine, 2, axis=-1)
        modulated = _rms(x) * (1 + scale) + shift
        return jnp.where(jnp.any(mask, axis=-1)[:, None, None], modulated, x).astype(dtype)

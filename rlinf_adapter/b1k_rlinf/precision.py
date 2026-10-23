"""Numerical contracts of the original JAX inference path."""

import math
from types import MethodType

import torch
from torch.nn import functional as F


def jax_gelu(x):
    """JAX/Flax GELU, including constants rounded to the activation dtype.

    F.gelu defaults to erf; even its tanh approximation uses FP32 constants
    internally. JAX casts the constants to x.dtype, which matters for BF16.
    Keep the individual BF16 operations, as in the source JAX expression.
    """
    root = x.new_tensor(math.sqrt(2 / math.pi))
    cubic = x.new_tensor(0.044715)
    half = x.new_tensor(0.5)
    one = x.new_tensor(1.0)
    cdf = half * (one + torch.tanh(root * (x + cubic * ((x * x) * x))))
    return x * cdf


def jax_gelu_glu(gate, value):
    return jax_gelu(gate) * value


def jax_linear_forward(self, x):
    # A fused BF16 addmm rounds once; JAX Dense rounds dot and bias separately.
    return F.linear(x, self.weight, None) + self.bias


def jax_adaptive_norm(self, x, cond=None):
    if cond is None:
        raise ValueError("Adaptive RMSNorm requires a conditioning vector")
    dtype = x.dtype
    xf = x.float()
    normed = xf * torch.rsqrt(torch.mean(xf.square(), dim=-1, keepdim=True) + 1e-6)
    dense = self.ada_modulation
    modulation = F.linear(cond.to(dtype), dense.weight.to(dtype), None) + dense.bias.to(dtype)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    if x.ndim == 3:
        scale, shift, gate = (value.unsqueeze(-2) for value in (scale, shift, gate))
    return (normed * (1 + scale.float()) + shift.float()).to(dtype), gate


def jax_mlp_forward(self, x):
    return self.fc2(self.dropout(jax_gelu(self.fc1(x))))


def jax_feedforward(self, x):
    dtype = x.dtype
    a = torch.matmul(x, self.w_gating[0].to(dtype))
    b = torch.matmul(x, self.w_gating[1].to(dtype))
    return torch.matmul(jax_gelu_glu(a, b), self.w_linear.to(dtype))


def install_jax_primitives(model):
    """Apply only to this model; leave upstream modules and other models intact."""
    from rlinf.models.embodiment.openpi.modules.gemma import FeedForward, RMSNorm
    from rlinf.models.embodiment.openpi.modules.siglip import MlpBlock

    counts = {"feedforwards": 0, "vision_mlps": 0, "bf16_biases": 0, "adaptive_norms": 0}
    for module in model.modules():
        if isinstance(module, FeedForward):
            module.forward = MethodType(jax_feedforward, module)
            counts["feedforwards"] += 1
        elif isinstance(module, MlpBlock):
            module.forward = MethodType(jax_mlp_forward, module)
            counts["vision_mlps"] += 1
        elif isinstance(module, torch.nn.Linear) and module.bias is not None and module.weight.dtype == torch.bfloat16:
            module.forward = MethodType(jax_linear_forward, module)
            counts["bf16_biases"] += 1
        elif isinstance(module, RMSNorm) and module.adaptive:
            module.forward = MethodType(jax_adaptive_norm, module)
            counts["adaptive_norms"] += 1
    return counts

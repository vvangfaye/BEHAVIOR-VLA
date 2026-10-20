"""Tiny JAX loop preflight, not an actual VLA sampler equivalence test."""
import json
import os
from pathlib import Path

if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import jax
import jax.numpy as jnp


def dynamic(parameter):
    dt = -0.1

    def step(carry):
        x, time = carry
        return x + dt * parameter, time + dt

    def condition(carry):
        return carry[1] >= -dt / 2

    return jax.lax.while_loop(condition, step, (jnp.array(0.0), jnp.array(1.0)))[0]


def fixed(parameter):
    def step(_, carry):
        x, time = carry
        return x - 0.1 * parameter, time - 0.1

    return jax.lax.fori_loop(0, 10, step, (jnp.array(0.0), jnp.array(1.0)))[0]


parameter = jnp.array(0.7)
error = None
try:
    jax.grad(dynamic)(parameter)
except ValueError as exc:
    error = str(exc)
assert error is not None and "Reverse-mode differentiation does not work" in error
value_dynamic, value_fixed = float(dynamic(parameter)), float(fixed(parameter))
gradient = float(jax.grad(fixed)(parameter))
assert value_dynamic == value_fixed
assert abs(gradient + 1) < 1e-6
result = {
    "status": "complete",
    "jax_version": jax.__version__,
    "dynamic_output": value_dynamic,
    "fixed_output": value_fixed,
    "fixed_reverse_gradient": gradient,
    "dynamic_reverse_gradient_error": error,
    "scope": "A tiny scalar loop reproduces the dynamic-control-flow limitation. No VLA model or weights are loaded or changed; actual-model output/gradient parity remains untested.",
}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

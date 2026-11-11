"""Compare real JAX/Torch RMSNorm source with identical BF16 parameters."""

import ast
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import linen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "BEHAVIOR-VLAs/rlinf_adapter"))
from b1k_rlinf.precision import jax_adaptive_norm


def load_class(path, namespace, remove_decorator=False):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "RMSNorm")
    if remove_decorator:
        node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["RMSNorm"]


JaxNorm = load_class(ROOT / "BEHAVIOR-VLAs/openpi/src/openpi/models/gemma.py", {"nn": linen, "jnp": jnp}, True)
TorchNorm = load_class(HERE.parent / "upstream/rlinf/models/embodiment/openpi/modules/gemma.py", {"nn": torch.nn, "torch": torch})
torch.manual_seed(7)
torch.set_num_threads(2)
report = {}
for adaptive in (False, True):
    module = TorchNorm(32, adaptive=adaptive)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_((torch.randn_like(p) * 0.08).bfloat16().float())
    x = torch.randn(2, 11, 32).bfloat16()
    cond = torch.randn(2, 32) if adaptive else None
    if adaptive:
        params = {"Dense_0": {"kernel": jnp.asarray(module.ada_modulation.weight.detach().T.numpy(), dtype=jnp.bfloat16), "bias": jnp.asarray(module.ada_modulation.bias.detach().numpy(), dtype=jnp.bfloat16)}}
    else:
        params = {"scale": jnp.asarray(module.scale.detach().numpy(), dtype=jnp.bfloat16)}
    reference, gate = jax.jit(JaxNorm().apply)({"params": params}, jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16), None if cond is None else jnp.asarray(cond.numpy()))
    reference = torch.from_numpy(np.asarray(reference, dtype=np.float32).copy())
    old, _ = module(x, cond)
    fixed, _ = jax_adaptive_norm(module, x, cond) if adaptive else module(x, cond)
    old_error = (old.float() - reference).abs()
    fixed_error = (fixed.float() - reference).abs()
    fixed.float().square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    print(json.dumps({"adaptive": adaptive, "old_mean": float(old_error.mean()), "old_max": float(old_error.max()), "fixed_mean": float(fixed_error.mean()), "fixed_max": float(fixed_error.max())}), flush=True)
    report[str(adaptive)] = {"old_mean_error": float(old_error.mean()), "old_max_error": float(old_error.max()), "fixed_mean_error": float(fixed_error.mean()), "fixed_max_error": float(fixed_error.max()), "finite_gradients": True}
(HERE / "adaptive_contract.json").write_text(json.dumps({"passed": all(x["fixed_max_error"] <= x["old_max_error"] for x in report.values()), "cases": report}, indent=2) + "\n")
print(json.dumps(report))

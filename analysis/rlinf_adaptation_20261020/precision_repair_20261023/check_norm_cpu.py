"""Compare the captured fixed inputs before/after matching JAX head quantization."""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / "cloud_smoke_retry04_20261023"
sys.path[:0] = [str(ROOT / "BEHAVIOR-VLAs/rlinf_adapter"), str(HERE.parent / "upstream"), str(BASE)]
from common import env_observation, label_text

tree = ast.parse((BASE / "common.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "torch_model")
source = ast.unparse(node).replace(".cuda().eval()", ".eval()")
namespace = {}
exec(source, namespace)
plan = json.loads((BASE / "plan.json").read_text())
plan["converted"] = str(HERE.parent / "checkpoint1688977_openpi")
plan["tokenizer"] = str(ROOT / "analysis/retraining_20261016_quality_taskbalanced_v1/smoke/bundle/inputs/paligemma_tokenizer.model")
torch.set_num_threads(16)
started = time.monotonic()
model = namespace["torch_model"](plan)
probes = sorted((BASE / "retrieved/jax").glob("*/probe_*.npz"))
assert len(probes) == 12
from b1k_rlinf.precision import install_jax_rmsnorm
install_jax_rmsnorm(model)
torch.set_float32_matmul_precision("highest")
report = {"scope": "CPU numerical diagnosis on the existing 12 captured inputs; GPU acceptance remains required.", "variants": {}}
heads = ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
for variant in ["jax_norms_and_heads"]:
    if variant == "jax_norms_and_heads":
        changed = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.startswith(heads):
                    before = parameter.clone()
                    parameter.copy_(parameter.to(torch.bfloat16).to(parameter.dtype))
                    changed.append({"name": name, "max_rounding": float((before-parameter).abs().max())})
        report["rounded_heads"] = changed
        assert len(changed) == 8
    results = []
    for path in probes:
        with np.load(path, allow_pickle=False) as data:
            raw = {"state": data["state"], "images": list(data["images"]), "prompt": str(data["prompt"])}
            actions, result = model.predict_action_batch(env_observation(raw), mode="eval", noise=torch.from_numpy(data["noise"]))
            text = label_text(model, result)
            error = np.abs(actions[0].numpy() - data["actions"]) / (model.checkpoint_io.stats["actions"]["std"][None] + 1e-6)
            item = {"instance": path.parent.name, "probe": path.name, "subtask_equal": text == str(data["subtask"]), "mean_error": float(error.mean()), "max_error": float(error.max())}
            results.append(item)
            print(json.dumps({"variant": variant, **item}), flush=True)
    report["variants"][variant] = results
    report["elapsed_seconds"] = time.monotonic() - started
    (HERE / "norm_result.json").write_text(json.dumps(report, indent=2) + "\n")

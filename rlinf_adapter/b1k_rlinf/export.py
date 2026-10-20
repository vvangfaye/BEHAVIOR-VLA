"""Read the checkpoint's OCDBT arrays without restoring its training topology."""

import json
from pathlib import Path

import numpy as np
import tensorstore as ts


def read_params(checkpoint):
    """Restore all array leaves and remove Orbax's params / NNX value wrappers."""
    directory = Path(checkpoint).resolve() / "params"
    metadata = json.loads((directory / "_METADATA").read_text())
    if metadata.get("use_zarr3"):
        raise ValueError("This checkpoint reader expects Orbax Zarr v2 arrays")
    kv_spec = {"driver": "ocdbt", "base": {"driver": "file", "path": str(directory) + "/"}}
    store = ts.KvStore.open(kv_spec).result(timeout=30)
    keys = {key.decode() for key in store.list().result(timeout=30)}
    entries = list(metadata["tree_metadata"].values())
    expected = []
    for entry in entries:
        value_meta = entry["value_metadata"]
        if value_meta.get("skip_deserialize", False):
            raise ValueError("Unexpected non-array leaf in inference checkpoint")
        parts = [item["key"] for item in entry["key_metadata"]]
        key = ".".join(parts)
        if key + "/.zarray" not in keys:
            raise ValueError(f"Missing stored array metadata: {key}")
        expected.append((key, parts))
    actual = {key[: -len("/.zarray")] for key in keys if key.endswith("/.zarray")}
    if actual != {key for key, _ in expected}:
        raise ValueError("OCDBT arrays differ from the checkpoint metadata inventory")
    result = {}
    for index, (key, parts) in enumerate(expected, 1):
        array = ts.open({"driver": "zarr", "kvstore": {**kv_spec, "path": key + "/"}}).result(timeout=30)
        value = np.asarray(array.read().result(timeout=300), dtype=np.float32)
        if parts[0] == "params":
            parts = parts[1:]
        if parts[-1] == "value":
            parts = parts[:-1]
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        if parts[-1] in target:
            raise ValueError(f"Duplicate unwrapped checkpoint leaf: {parts}")
        target[parts[-1]] = value
        print(f"Read array {index}/{len(expected)}: {key} {value.shape}", flush=True)
    return result


def export_checkpoint(checkpoint, norm_path, output):
    """Use pinned RLinf weight transforms after topology-independent restore."""
    from rlinf.utils.ckpt_convertor.openpi._core import (
        copy_norm_stats,
        save_safetensors,
        write_config_json,
    )
    from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi import (
        convert_llm,
        convert_projections,
        convert_siglip,
    )

    params = read_params(checkpoint)
    weights = {}
    for part in (convert_siglip(params), convert_llm(params, True), convert_projections(params, True)):
        for key, tensor in part.items():
            if key in weights:
                raise ValueError(f"Duplicate converted weight: {key}")
            weights[key] = tensor.contiguous()
    output = Path(output)
    save_safetensors(weights, output / "model.safetensors")
    write_config_json(
        {
            "pi05": True,
            "action_dim": 32,
            "action_horizon": 32,
            "max_token_len": 512,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "dtype": "bfloat16",
            "pcd": False,
        },
        output,
    )
    copy_norm_stats(norm_path, output / "norm_stats.json")

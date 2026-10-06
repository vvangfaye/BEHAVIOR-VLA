#!/usr/bin/env python3
"""Run a small Warp CUDA kernel and verify the bundled evaluation imports."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from eval_runtime import VERSION


@wp.kernel
def increment(values: wp.array(dtype=wp.float32)):
    index = wp.tid()
    values[index] += 1.0


def check(cache_dir):
    if wp.__version__ != VERSION:
        raise RuntimeError(f"Warp version must be {VERSION}, found {wp.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation Torch cannot access CUDA")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache_dir)
    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("Warp CUDA device is unavailable")
    values = wp.zeros(4, dtype=wp.float32, device=device)
    wp.launch(increment, dim=4, inputs=[values], device=device)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(values.numpy(), np.ones(4, dtype=np.float32))
    tensor = wp.to_torch(values)
    if not tensor.is_cuda or tensor.sum().item() != 4.0:
        raise RuntimeError("Warp/Torch CUDA interoperability failed")
    # Import the actual bundled source before copying checkpoints or assets.
    import bddl
    import hydra
    import isaacsim
    import msgpack
    import omnigibson
    import av
    if omnigibson.__version__ != "3.9.3":
        raise RuntimeError("Expected bundled OmniGibson 3.9.3")
    return {"eval_runtime": "passed", "warp_version": wp.__version__,
            "omnigibson_version": omnigibson.__version__, "cuda_kernel": "passed",
            "torch_interop": "passed", "hdf5_version": h5py.__version__,
            "gpu": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.cache_dir)))


if __name__ == "__main__":
    main()

"""Validate a complete FrameSamp cache without video decoding or GPU allocation."""

import argparse
import json

import numpy as np

from openpi.training.frame_memory_data import CachedFrameMemory


def main(args):
    contract = {
        "num_frames": args.num_frames,
        "grid_size": args.grid_size,
        "frame_stride": args.frame_stride,
        "camera": args.camera,
    }
    reader = CachedFrameMemory(args.cache_dir, contract, args.dataset_root, args.episodes_index)
    if reader.manifest["feature_width"] != 2048:
        raise ValueError("Production pi05 requires 2048-dimensional cached SigLIP features.")
    size = 0
    for episode, length in reader.lengths.items():
        path = reader.root / f"episode_{episode:08d}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = ((length + args.frame_stride - 1) // args.frame_stride, args.grid_size**2, 2048)
        if array.shape != expected or array.dtype != np.float16:
            raise ValueError(f"Invalid memory file: {path}; expected {expected}, float16.")
        size += path.stat().st_size
    print(
        json.dumps(
            {
                "memory_episodes": len(reader.lengths),
                "cache_bytes": size,
                "memory_tokens": args.num_frames * args.grid_size**2,
                "encoder_sha256": reader.manifest["encoder_sha256"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--episodes-index", type=int, nargs="+")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--camera", default="base_0_rgb")
    main(parser.parse_args())

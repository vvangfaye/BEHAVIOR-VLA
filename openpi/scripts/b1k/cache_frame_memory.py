"""Cache frozen SigLIP features for JAX FrameSamp training, one atomic file/episode.

Uses only the camera named by the model contract. Can be sharded across GPU
processes using disjoint episode assignments, without initializing JAX distributed.
"""

import argparse
import gc
import json
import os
from pathlib import Path

from filelock import FileLock
from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools

from openpi.configs import ROBOT_REGISTRY
from openpi.models import frame_memory
from openpi.models import model as model_lib
from openpi.models import siglip
from openpi.policies.b1k_policy import _parse_image
from openpi.shared.download import maybe_download
from openpi.shared.frame_memory import encoder_fingerprint
from openpi.training import lerobot_compat
from openpi.training.frame_memory_data import dataset_fingerprint
from openpi.training.subtask_data import episode_lengths_in_dataset_order


def load_vision(params_path):
    # Restore on host; instantiate only the vision module on the accelerator.
    params = model_lib.restore_params(maybe_download(str(params_path)), restore_type=np.ndarray)
    weights = params["PaliGemma"]["img"]
    fingerprint = encoder_fingerprint(weights)
    del params
    gc.collect()

    def create():
        encoder = bridge.ToNNX(
            siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="bfloat16")
        )
        encoder.lazy_init(jnp.zeros((1, 224, 224, 3)), train=False, rngs=nnx.Rngs(0))
        return encoder

    graph, state = nnx.split(nnx.eval_shape(create))
    state.replace_by_pure_dict(jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.bfloat16), weights))
    return nnx.merge(graph, state), fingerprint


def write_episode(root, episode, length, manifest, dataset, encode, batch_size):
    output = root / f"episode_{episode:08d}.npy"
    complete = root / f"episode_{episode:08d}.json"
    expected = {"episode_index": episode, "length": length, "encoder_sha256": manifest["encoder_sha256"]}
    stride = manifest["frame_stride"]
    frames = np.arange(0, length, stride)
    shape = (len(frames), manifest["grid_size"] ** 2, manifest["feature_width"])
    with FileLock(str(root / f"episode_{episode:08d}.lock")):
        if complete.exists():
            if json.loads(complete.read_text()) != expected:
                raise ValueError(f"Conflicting completed cache episode {episode}.")
            array = np.load(output, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.float16:
                raise ValueError(f"Invalid completed cache episode {episode}.")
            return
        temporary = output.with_suffix(f".tmp.{os.getpid()}.npy")
        try:
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
            for start in range(0, len(frames), batch_size):
                selected = frames[start : start + batch_size]
                images = []
                for frame in selected:
                    row = dataset[int(frame)]
                    if int(row["episode_index"]) != episode or int(row["frame_index"]) != frame:
                        raise ValueError("Dataset subset changed original episode/frame indexing.")
                    image = _parse_image(row[manifest["dataset_camera"]])
                    images.append(image_tools.resize_with_pad(image, 224, 224))
                # Fixed batch shape prevents a compile for every episode tail length.
                images += [images[-1]] * (batch_size - len(images))
                features = np.asarray(encode(np.stack(images)))[: len(selected)].astype(np.float16)
                if not np.isfinite(features).all():
                    raise ValueError("Vision cache contains non-finite features.")
                array[start : start + len(selected)] = features
            array.flush()
            del array
            os.replace(temporary, output)
            marker = complete.with_suffix(f".tmp.{os.getpid()}.json")
            marker.write_text(json.dumps(expected) + "\n")
            os.replace(marker, complete)
        finally:
            temporary.unlink(missing_ok=True)


def main(args):
    if args.batch_size < 1 or args.frame_stride < 1 or args.grid_size not in (1, 2, 4, 8, 16):
        raise ValueError("Invalid cache batch size, stride or pooling grid.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards.")
    episodes = None if args.episodes_file is None else [int(x) for x in args.episodes_file.read_text().split()]
    lengths = episode_lengths_in_dataset_order(args.dataset_root, episodes)
    camera_index = model_lib.IMAGE_KEYS.index(args.camera)
    camera = ROBOT_REGISTRY[args.robot].observations[f"image_{camera_index}"].dataset_key
    encoder, fingerprint = load_vision(args.init_params)
    manifest = {
        "version": 1,
        "pool": "mean",
        "dtype": "float16",
        "feature_width": 2048,
        "grid_size": args.grid_size,
        "frame_stride": args.frame_stride,
        "camera": args.camera,
        "dataset_camera": camera,
        "encoder_sha256": fingerprint,
        "dataset_sha256": dataset_fingerprint(args.dataset_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_dir / "manifest.lock")):
        manifest_path = args.output_dir / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing cache uses different weights/data/settings; use a new output directory.")
        else:
            temporary = args.output_dir / f"manifest.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, manifest_path)

    def encode(module, images):
        features, _ = module(images.astype(jnp.float32) / 127.5 - 1, train=False)
        return frame_memory.pool_tokens(features, args.grid_size)

    encode_jit = nnx.jit(encode)
    for ordinal, (episode, length) in enumerate(lengths.items()):
        if ordinal % args.num_shards != args.shard_index:
            continue
        # One episode at a time avoids loading an entire full100 frame index into RAM.
        dataset = lerobot_compat.LeRobotDataset(
            repo_id=args.repo_id, root=args.dataset_root, episodes=[episode], tolerance_s=5e-3
        )
        write_episode(
            args.output_dir,
            episode,
            length,
            manifest,
            dataset,
            lambda images: encode_jit(encoder, images),
            args.batch_size,
        )
        print(f"cached episode={episode} frames={(length + args.frame_stride - 1) // args.frame_stride}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--init-params", required=True, help="Same SigLIP weights as training INIT_PARAMS.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-file", type=Path)
    parser.add_argument("--robot", default="b1k/R1Pro")
    parser.add_argument("--camera", choices=model_lib.IMAGE_KEYS, default="base_0_rgb")
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    main(parser.parse_args())
